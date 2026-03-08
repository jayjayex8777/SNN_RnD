"""
Phase 3: Firing Rate Regularization (Sparsification)

KD 학습 완료된 Student 모델을 로드하여,
발화율 제약(firing rate regularization)으로 fine-tuning.

목표: 발화율 80~95% → 5~10% 로 낮추면서 정확도 유지

전략:
  1. KD 완료된 Student 가중치 로드
  2. threshold를 learnable parameter로 전환
  3. firing_rate_loss 추가 (target_rate sweep)
  4. 점진적으로 lambda_sparse 증가 (curriculum)
  5. 각 variant × T × target_rate 조합 실험
"""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.mobile_optimizer import optimize_for_mobile

from dataset import FileLevelSpikeDataset, rate_code_zscore_sigmoid, pad_collate
from lif_module import LIFNode
from model import HardLIFNode, _HardSpikeSTE

# ========================= Config =========================
DATA_DIR = "./data/2_ffilled_data"
NUM_CLASSES = 4
EPOCHS = 40
LR = 5e-4
BATCH_SIZE = 4
GRAD_CLIP = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = min(16, os.cpu_count() or 2)
PIN_MEMORY = DEVICE.type == "cuda"
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

MODEL_DIR = "./models"
RESULT_DIR = "./result"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

C1, C2 = 32, 64
VARIANT_NAMES = {3: "smallest", 5: "small", 7: "medium", 9: "large", 11: "largest"}
KERNEL_SIZES = [3, 5, 7, 9, 11]
T_VALUES = [3, 5, 10, 15]
TARGET_RATES = [0.3, 0.2, 0.1, 0.05]
LAMBDA_SPARSE = 5.0


# ========================= Learnable Threshold HardLIF =========================
class _HardSpikeSTE_LearnableThresh(torch.autograd.Function):
    """STE that passes gradient to BOTH membrane potential AND threshold."""
    @staticmethod
    def forward(ctx, mem, threshold, width):
        ctx.save_for_backward(mem, threshold)
        ctx.width = width
        return (mem >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        mem, threshold = ctx.saved_tensors
        delta = mem - threshold
        mask = (delta.abs() <= ctx.width).float()
        surrogate = mask / (2.0 * ctx.width + 1e-6)
        # grad w.r.t. mem: positive direction
        grad_mem = grad_output * surrogate
        # grad w.r.t. threshold: negative direction (higher thresh → less spikes)
        grad_thresh = -grad_output * surrogate
        # Sum over batch/spatial dims to get scalar grad for threshold
        grad_thresh = grad_thresh.sum()
        return grad_mem, grad_thresh, None


class LearnableHardLIFNode(nn.Module):
    """HardLIFNode with learnable threshold for sparsification.
    Gradient flows to threshold via custom STE."""
    def __init__(self, tau=2.0, threshold=0.5, surrogate_width=0.5, reset_value=0.0):
        super().__init__()
        self.tau = tau
        self.threshold = nn.Parameter(torch.tensor(float(threshold)))
        self.surrogate_width = surrogate_width
        self.reset_value = reset_value
        self.register_buffer("v", torch.zeros(0))

    def forward(self, x):
        if self.v.numel() == 0 or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)

        self.v = self.v + (x - self.v) / self.tau
        if torch.jit.is_scripting():
            spike = (self.v >= self.threshold).float()
        else:
            spike = _HardSpikeSTE_LearnableThresh.apply(
                self.v, self.threshold, self.surrogate_width
            )
        self.v = torch.where(spike > 0, torch.full_like(self.v, self.reset_value), self.v)
        return spike

    def reset(self):
        self.v = torch.zeros(0, device=self.v.device if isinstance(self.v, torch.Tensor) else None)


# ========================= Sparse Student Model =========================
class SparseStudentSNN1d(nn.Module):
    """Student with learnable thresholds and spike collection for FR loss."""
    def __init__(self, c1, c2, kernel_size=3, num_classes=NUM_CLASSES, T=20,
                 thresh1=0.02, thresh2=0.02, surrogate_width=5.0, gain=3.0):
        super().__init__()
        self.T = T
        self.gain = nn.Parameter(torch.tensor(float(gain)))
        self.conv1 = nn.Conv1d(12, c1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.lif1 = LearnableHardLIFNode(tau=1.0, threshold=thresh1, surrogate_width=surrogate_width)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=kernel_size // 2)
        self.lif2 = LearnableHardLIFNode(tau=0.9, threshold=thresh2, surrogate_width=surrogate_width)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c2, num_classes)

    def forward(self, x, return_spikes=False):
        all_spk1, all_spk2 = [], []

        for t in range(self.T):
            xt = x[:, :, :, t]
            spk1 = self.lif1(self.conv1(xt) * self.gain)
            spk2 = self.lif2(self.conv2(spk1) * self.gain)

            if return_spikes:
                all_spk1.append(spk1)
                all_spk2.append(spk2)

            if t == 0:
                acc = spk2
            else:
                acc = acc + spk2

        out = acc / float(self.T)
        out = self.pool(out).squeeze(-1)
        logits = self.fc(out)

        if return_spikes:
            return logits, all_spk1, all_spk2
        return logits

    def reset(self):
        self.lif1.reset()
        self.lif2.reset()


# ========================= Firing Rate Loss =========================
def firing_rate_loss(spikes_list, target_rate):
    """
    Penalize deviation from target firing rate.
    spikes_list: list of spike tensors (B, C, N)
    target_rate: desired average firing rate (e.g. 0.05)
    """
    loss = 0.0
    for spk in spikes_list:
        actual_rate = spk.mean()
        loss += (actual_rate - target_rate) ** 2
    return loss / len(spikes_list)


def measure_firing_rates(model, loader, device):
    """Measure average firing rates on validation set."""
    model.eval()
    fr1_sum, fr2_sum, count = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            model.reset()
            _, spk1_list, spk2_list = model(x, return_spikes=True)
            fr1 = torch.stack(spk1_list).mean().item()
            fr2 = torch.stack(spk2_list).mean().item()
            fr1_sum += fr1
            fr2_sum += fr2
            count += 1
    return fr1_sum / count, fr2_sum / count


# ========================= Load KD Student weights =========================
def load_kd_student(variant_name, kernel_size, T, device):
    """Load pre-trained KD student weights into SparseStudentSNN1d."""
    # Try T-sweep naming first, then default naming
    candidates = [
        os.path.join(MODEL_DIR, f"snn1d_student_{variant_name}_T{T}.pt"),
        os.path.join(MODEL_DIR, f"student_kd1d_{variant_name}_sensor.pt"),
    ]

    pt_path = None
    for c in candidates:
        if os.path.exists(c):
            pt_path = c
            break

    if pt_path is None:
        print(f"  [SKIP] No KD student found for {variant_name} T={T}")
        return None

    model = SparseStudentSNN1d(C1, C2, kernel_size=kernel_size, T=T).to(device)

    # Load weights (skip membrane potential buffers)
    state = torch.load(pt_path, map_location=device, weights_only=True)
    filtered = {}
    for k, v in state.items():
        new_k = k
        # threshold was scalar, now it's nn.Parameter — compatible
        if k.endswith(".v"):
            continue
        filtered[new_k] = v

    # gain: if saved as scalar attribute, load into nn.Parameter
    if "gain" in filtered and not isinstance(filtered["gain"], torch.Tensor):
        filtered["gain"] = torch.tensor(float(filtered["gain"]))

    model.load_state_dict(filtered, strict=False)
    print(f"  Loaded KD student from: {pt_path}")
    return model


# ========================= Export =========================
class ExportableHardLIFNode(nn.Module):
    """TorchScript-compatible HardLIF for export (no custom autograd)."""
    def __init__(self, tau: float, threshold: float, reset_value: float = 0.0):
        super().__init__()
        self.tau = tau
        self.threshold = threshold
        self.reset_value = reset_value
        self.v: torch.Tensor = torch.zeros(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.v.numel() == 0 or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
        self.v = self.v + (x - self.v) / self.tau
        spike = (self.v >= self.threshold).float()
        self.v = torch.where(spike > 0, torch.full_like(self.v, self.reset_value), self.v)
        return spike

    def reset(self) -> None:
        self.v = torch.zeros(0)


class ExportableSparseStudent(nn.Module):
    """TorchScript-compatible model for mobile export."""
    def __init__(self, c1: int, c2: int, kernel_size: int, T: int,
                 thresh1: float, thresh2: float, gain: float,
                 num_classes: int = 4):
        super().__init__()
        self.T = T
        self.gain = gain
        self.conv1 = nn.Conv1d(12, c1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.lif1 = ExportableHardLIFNode(tau=1.0, threshold=thresh1)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=kernel_size // 2)
        self.lif2 = ExportableHardLIFNode(tau=0.9, threshold=thresh2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acc = torch.zeros(0, device=x.device)
        for t in range(self.T):
            xt = x[:, :, :, t]
            spk1 = self.lif1(self.conv1(xt) * self.gain)
            spk2 = self.lif2(self.conv2(spk1) * self.gain)
            if acc.numel() == 0:
                acc = spk2
            else:
                acc = acc + spk2
        out = acc / float(self.T)
        out = self.pool(out).squeeze(-1)
        return self.fc(out)

    def reset(self) -> None:
        self.lif1.reset()
        self.lif2.reset()


def export_sparse_ptl(model, T, kernel_size, filename):
    """Export sparse student to TorchScript Lite via script-compatible model."""
    try:
        # Extract learned parameters
        thresh1 = model.lif1.threshold.item()
        thresh2 = model.lif2.threshold.item()
        gain = model.gain.item()

        # Build exportable model
        export_model = ExportableSparseStudent(
            C1, C2, kernel_size, T, thresh1, thresh2, gain
        ).cpu()

        # Copy conv/fc weights
        state = model.cpu().state_dict()
        export_state = {}
        for k, v in state.items():
            if k.endswith(".v") or k == "gain":
                continue
            # Map threshold Parameter → scalar (already baked into constructor)
            if "threshold" in k:
                continue
            export_state[k] = v

        export_model.load_state_dict(export_state, strict=False)
        export_model.eval()
        export_model.reset()

        scripted = torch.jit.script(export_model)
        opt = optimize_for_mobile(scripted)
        path = os.path.join(MODEL_DIR, filename)
        opt._save_for_lite_interpreter(path)
        print(f"  Exported: {filename} (thresh={thresh1:.3f}/{thresh2:.3f}, gain={gain:.2f})")
    except Exception as e:
        print(f"  Export failed for {filename}: {e}")
        import traceback; traceback.print_exc()


# ========================= Phase 3: Sparsification Training =========================
def train_sparse(model, train_loader, val_loader, T, target_rate, device):
    """Fine-tune KD student with firing rate regularization."""
    # Separate param groups: higher LR for threshold and gain
    thresh_params = [model.lif1.threshold, model.lif2.threshold, model.gain]
    thresh_ids = {id(p) for p in thresh_params}
    other_params = [p for p in model.parameters() if id(p) not in thresh_ids]

    optimizer = torch.optim.Adam([
        {"params": other_params, "lr": LR},
        {"params": thresh_params, "lr": LR * 10},  # 10x LR for threshold/gain
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_score = -1.0
    best_state = None

    # Curriculum: gradually increase lambda_sparse
    for epoch in range(EPOCHS):
        # Ramp up lambda over first 10 epochs
        progress = min(1.0, (epoch + 1) / 10.0)
        current_lambda = LAMBDA_SPARSE * progress

        model.train()
        correct, total, total_loss = 0, 0, 0.0
        epoch_fr1, epoch_fr2 = 0.0, 0.0
        batch_count = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            model.reset()
            optimizer.zero_grad()

            logits, spk1_list, spk2_list = model(x, return_spikes=True)

            # Classification loss
            loss_ce = F.cross_entropy(logits, y)

            # Firing rate regularization
            all_spikes = spk1_list + spk2_list
            loss_fr = firing_rate_loss(all_spikes, target_rate)

            # Threshold regularization: prevent threshold from going negative
            thresh_penalty = 0.0
            for lif in [model.lif1, model.lif2]:
                if lif.threshold < 0:
                    thresh_penalty += lif.threshold ** 2

            # Gain regularization: penalize gain increase (higher gain → more spikes)
            gain_penalty = F.relu(model.gain - 3.0) ** 2  # penalize gain > initial value

            loss = loss_ce + current_lambda * loss_fr + 10.0 * thresh_penalty + 0.5 * gain_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            # Clamp threshold to positive
            with torch.no_grad():
                model.lif1.threshold.clamp_(min=0.01)
                model.lif2.threshold.clamp_(min=0.01)

            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
            total_loss += loss_ce.item() * y.size(0)

            # Track firing rates
            with torch.no_grad():
                epoch_fr1 += torch.stack(spk1_list).mean().item()
                epoch_fr2 += torch.stack(spk2_list).mean().item()
                batch_count += 1

        scheduler.step()
        train_acc = correct / total
        avg_fr1 = epoch_fr1 / batch_count
        avg_fr2 = epoch_fr2 / batch_count

        # Validation
        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                model.reset()
                out = model(x)
                vc += (out.argmax(1) == y).sum().item()
                vt += y.size(0)
        val_acc = vc / max(vt, 1)
        score = train_acc + val_acc

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | train={train_acc:.4f} | val={val_acc:.4f} | "
                  f"FR1={avg_fr1:.3f} FR2={avg_fr2:.3f} | "
                  f"thresh=({model.lif1.threshold.item():.3f}, {model.lif2.threshold.item():.3f}) | "
                  f"gain={model.gain.item():.2f} | lambda={current_lambda:.2f}")

        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_state, best_score


# ========================= Main =========================
def main():
    print(f"Device: {DEVICE}")
    print(f"Phase 3: Sparsification with Firing Rate Regularization")
    print(f"T values: {T_VALUES}")
    print(f"Target rates: {TARGET_RATES}")
    print(f"Lambda sparse: {LAMBDA_SPARSE}")
    print()

    all_results = []

    for T in T_VALUES:
        # Dataset for this T
        dataset = FileLevelSpikeDataset(DATA_DIR, rate_code_zscore_sigmoid, T=T)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                         generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=pad_collate, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=pad_collate, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        for ks in KERNEL_SIZES:
            vname = VARIANT_NAMES[ks]
            print(f"\n{'='*70}")
            print(f"  Variant={vname} (k={ks}), T={T}")
            print(f"{'='*70}")

            for target_rate in TARGET_RATES:
                print(f"\n--- target_rate={target_rate} ---")

                # Load KD student
                model = load_kd_student(vname, ks, T, DEVICE)
                if model is None:
                    continue

                # Measure initial firing rates
                fr1_init, fr2_init = measure_firing_rates(model, val_loader, DEVICE)
                print(f"  Initial firing rates: FR1={fr1_init:.3f}, FR2={fr2_init:.3f}")

                # Train with FR regularization
                best_state, best_score = train_sparse(
                    model, train_loader, val_loader, T, target_rate, DEVICE
                )

                # Load best and measure final firing rates
                model.load_state_dict(best_state)
                model.to(DEVICE)
                fr1_final, fr2_final = measure_firing_rates(model, val_loader, DEVICE)
                print(f"  Final firing rates: FR1={fr1_final:.3f}, FR2={fr2_final:.3f}")
                print(f"  Best score: {best_score:.4f}")

                # Save model
                pt_name = f"sparse_{vname}_T{T}_fr{int(target_rate*100):02d}.pt"
                torch.save(best_state, os.path.join(MODEL_DIR, pt_name))

                # Export ptl
                ptl_name = f"sparse_{vname}_T{T}_fr{int(target_rate*100):02d}.ptl"
                model.load_state_dict(best_state)
                export_sparse_ptl(model, T, ks, ptl_name)

                result = {
                    "T": T,
                    "variant": vname,
                    "kernel_size": ks,
                    "target_rate": target_rate,
                    "type": "sparse_student",
                    "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "best_score": round(best_score, 4),
                    "fr1_init": round(fr1_init, 4),
                    "fr2_init": round(fr2_init, 4),
                    "fr1_final": round(fr1_final, 4),
                    "fr2_final": round(fr2_final, 4),
                    "final_thresh1": round(model.lif1.threshold.item(), 4),
                    "final_thresh2": round(model.lif2.threshold.item(), 4),
                    "final_gain": round(model.gain.item(), 4),
                }
                all_results.append(result)

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Save all results
    result_path = os.path.join(RESULT_DIR, "sparse_results.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: Sparsification Results")
    print(f"{'='*80}")
    print(f"{'Variant':<10} {'T':>3} {'Target':>7} {'Score':>7} {'FR1_i':>7} {'FR1_f':>7} {'FR2_i':>7} {'FR2_f':>7} {'Th1':>7} {'Th2':>7} {'Gain':>6}")
    print("-" * 85)
    for r in all_results:
        print(f"{r['variant']:<10} {r['T']:>3} {r['target_rate']:>7.2f} {r['best_score']:>7.4f} "
              f"{r['fr1_init']:>7.3f} {r['fr1_final']:>7.3f} "
              f"{r['fr2_init']:>7.3f} {r['fr2_final']:>7.3f} "
              f"{r['final_thresh1']:>7.3f} {r['final_thresh2']:>7.3f} "
              f"{r['final_gain']:>6.2f}")

    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    main()
