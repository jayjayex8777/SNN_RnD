"""
Phase 4: INT8 Static Quantization for Sparse SNN Models

기존 학습 완료된 sparse_*.pt 모델을 로드하여,
Conv1d 가중치를 INT8로 정적 양자화 (재학습 없음).

Conv layers: INT8 (QuantStub/DeQuantStub 경계)
LIF neurons: FP32 유지 (막전위 동역학)
FC layer: FP32 유지

입력: models/sparse_{variant}_T{T}_fr{rate}.pt
출력: models/qsparse_{variant}_T{T}_fr{rate}.pt
결과: result/qsparse_results.json
"""

import copy
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.mobile_optimizer import optimize_for_mobile

from dataset import FileLevelSpikeDataset, rate_code_zscore_sigmoid, pad_collate

# ========================= Config =========================
DATA_DIR = "./data/2_ffilled_data"
MODEL_DIR = "./models"
RESULT_DIR = "./result"
NUM_CLASSES = 4
BATCH_SIZE = 4
SEED = 42

C1, C2 = 32, 64
VARIANT_NAMES = {3: "smallest", 5: "small", 7: "medium", 9: "large", 11: "largest"}
KERNEL_SIZES = [3, 5, 7, 9, 11]
T_VALUES = [3, 5, 10, 15]
TARGET_RATES = [0.30, 0.20, 0.10, 0.05]

LOG_FILE = "train_sparse_quant.log"


# ========================= Logger =========================
class Logger:
    """Tee stdout to both terminal and log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ========================= LIF Node (FP32, no custom autograd) =========================
class QuantFriendlyLIFNode(nn.Module):
    """LIF neuron that stays in FP32. No custom autograd, no learnable params."""
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


# ========================= Quantizable Sparse Student =========================
class QuantizableSparseStudent(nn.Module):
    """Sparse Student with QuantStub/DeQuantStub for INT8 static quantization.

    Conv1d layers run in INT8 between QuantStub/DeQuantStub boundaries.
    LIF neurons, pool, and FC stay in FP32.
    """

    def __init__(self, c1: int, c2: int, kernel_size: int, T: int,
                 thresh1: float, thresh2: float, gain: float,
                 num_classes: int = 4):
        super().__init__()
        self.T = T
        self.gain = gain
        pad = kernel_size // 2

        # Conv1 quantization boundary
        self.quant1 = torch.quantization.QuantStub()
        self.conv1 = nn.Conv1d(12, c1, kernel_size=kernel_size, padding=pad)
        self.dequant1 = torch.quantization.DeQuantStub()

        # LIF1 (FP32)
        self.lif1 = QuantFriendlyLIFNode(tau=1.0, threshold=thresh1)

        # Conv2 quantization boundary
        self.quant2 = torch.quantization.QuantStub()
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=pad)
        self.dequant2 = torch.quantization.DeQuantStub()

        # LIF2 (FP32)
        self.lif2 = QuantFriendlyLIFNode(tau=0.9, threshold=thresh2)

        # Output (FP32)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acc = torch.zeros(0, device=x.device)
        for t in range(self.T):
            xt = x[:, :, :, t]

            # Conv1 (INT8)
            xt = self.quant1(xt)
            xt = self.conv1(xt)
            xt = self.dequant1(xt)

            # LIF1 (FP32)
            spk1 = self.lif1(xt * self.gain)

            # Conv2 (INT8)
            spk1 = self.quant2(spk1)
            spk2_pre = self.conv2(spk1)
            spk2_pre = self.dequant2(spk2_pre)

            # LIF2 (FP32)
            spk2 = self.lif2(spk2_pre * self.gain)

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


# ========================= Weight Loading =========================
def load_sparse_weights(variant_name, kernel_size, T, target_rate):
    """Load sparse model weights and build quantizable model."""
    fr_int = int(target_rate * 100)
    pt_path = os.path.join(MODEL_DIR, f"sparse_{variant_name}_T{T}_fr{fr_int:02d}.pt")

    if not os.path.exists(pt_path):
        return None, None

    state = torch.load(pt_path, map_location="cpu", weights_only=True)

    # Extract learned threshold and gain values
    thresh1 = state.get("lif1.threshold", torch.tensor(0.5))
    if isinstance(thresh1, torch.Tensor):
        thresh1 = thresh1.item()
    thresh2 = state.get("lif2.threshold", torch.tensor(0.5))
    if isinstance(thresh2, torch.Tensor):
        thresh2 = thresh2.item()
    gain = state.get("gain", torch.tensor(3.0))
    if isinstance(gain, torch.Tensor):
        gain = gain.item()

    # Build quantizable model
    model = QuantizableSparseStudent(
        C1, C2, kernel_size, T, thresh1, thresh2, gain
    )

    # Map conv/fc weights from sparse state dict
    model_sd = model.state_dict()
    for k, v in state.items():
        # Skip membrane potential, threshold params, gain param
        if k.endswith(".v") or k == "gain" or "threshold" in k:
            continue
        if k in model_sd:
            model_sd[k] = v

    model.load_state_dict(model_sd, strict=True)
    return model, pt_path


# ========================= Quantization =========================
def calibrate_and_quantize(model, calibration_loader):
    """Apply INT8 static quantization with calibration.

    1. Set qconfig (qnnpack for mobile)
    2. Prepare: insert observers on conv layers
    3. Calibrate: forward pass to collect activation statistics
    4. Convert: replace FP32 conv with INT8 quantized conv
    """
    model_q = copy.deepcopy(model)
    model_q.eval()

    # Set quantization config for conv layers only
    model_q.qconfig = torch.quantization.get_default_qconfig("qnnpack")

    # Exclude LIF, pool, fc from quantization
    model_q.lif1.qconfig = None
    model_q.lif2.qconfig = None
    model_q.pool.qconfig = None
    model_q.fc.qconfig = None

    # Prepare: insert observers
    torch.quantization.prepare(model_q, inplace=True)

    # Calibration: forward pass over training data
    with torch.no_grad():
        for x, _ in calibration_loader:
            model_q.reset()
            model_q(x)

    # Convert: FP32 → INT8
    torch.quantization.convert(model_q, inplace=True)

    return model_q


# ========================= Evaluation =========================
def evaluate(model, val_loader):
    """Measure validation accuracy."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            model.reset()
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


# ========================= Export =========================
def export_qsparse_ptl(model, filename):
    """Export quantized model to .ptl for mobile deployment."""
    path = os.path.join(MODEL_DIR, filename)
    try:
        model.eval()
        model.reset()
        scripted = torch.jit.script(model)
        opt = optimize_for_mobile(scripted)
        opt._save_for_lite_interpreter(path)
        return os.path.getsize(path) / 1024
    except Exception as e:
        print(f"  [WARN] PTL export failed: {e}", flush=True)
        return None


def model_size_kb(model):
    """Calculate model size in KB (parameters + buffers)."""
    size = sum(p.nelement() * p.element_size() for p in model.parameters())
    size += sum(b.nelement() * b.element_size() for b in model.buffers())
    return size / 1024.0


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ========================= Main =========================
def main():
    torch.backends.quantized.engine = "qnnpack"

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    sys.stdout = Logger(LOG_FILE)

    total_configs = len(T_VALUES) * len(KERNEL_SIZES) * len(TARGET_RATES)

    print("=" * 80, flush=True)
    print("  Phase 4: INT8 Static Quantization for Sparse SNN", flush=True)
    print("=" * 80, flush=True)
    print(f"  T values:      {T_VALUES}", flush=True)
    print(f"  Kernel sizes:  {KERNEL_SIZES}", flush=True)
    print(f"  Target rates:  {TARGET_RATES}", flush=True)
    print(f"  Total configs: {total_configs}", flush=True)
    print(f"  Channels:      C1={C1}, C2={C2}", flush=True)
    print(f"  Backend:       qnnpack", flush=True)
    print(flush=True)

    all_results = []
    idx = 0
    skipped = 0
    failed = 0
    start_time = time.time()

    for T in T_VALUES:
        print(f"\n{'='*70}", flush=True)
        print(f"  Loading dataset for T={T}...", flush=True)
        print(f"{'='*70}", flush=True)

        dataset = FileLevelSpikeDataset(DATA_DIR, rate_code_zscore_sigmoid, T=T)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(SEED)
        )

        cal_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=pad_collate, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=pad_collate, num_workers=0)

        print(f"  Dataset: {len(train_ds)} train / {len(val_ds)} val", flush=True)

        for ks in KERNEL_SIZES:
            vname = VARIANT_NAMES[ks]

            for target_rate in TARGET_RATES:
                idx += 1
                fr_int = int(target_rate * 100)
                tag = f"sparse_{vname}_T{T}_fr{fr_int:02d}"

                elapsed = time.time() - start_time
                eta = (elapsed / idx * (total_configs - idx)) if idx > 0 else 0
                print(f"\n[{idx:3d}/{total_configs}] {tag}  "
                      f"(elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s)", flush=True)

                # --- Load sparse model ---
                model, pt_path = load_sparse_weights(vname, ks, T, target_rate)
                if model is None:
                    print(f"  [SKIP] Model not found", flush=True)
                    skipped += 1
                    continue

                print(f"  Loaded: {os.path.basename(pt_path)}", flush=True)
                print(f"  thresh=({model.lif1.threshold:.4f}, {model.lif2.threshold:.4f})  "
                      f"gain={model.gain:.4f}  params={count_params(model):,}", flush=True)

                # --- FP32 baseline accuracy ---
                model.eval()
                fp32_acc = evaluate(model, val_loader)
                fp32_kb = model_size_kb(model)
                print(f"  FP32  → Acc: {fp32_acc:.4f}  Size: {fp32_kb:.1f} KB", flush=True)

                # --- INT8 quantization ---
                try:
                    model_q = calibrate_and_quantize(model, cal_loader)
                except Exception as e:
                    print(f"  [FAIL] Quantization error: {e}", flush=True)
                    import traceback; traceback.print_exc()
                    failed += 1
                    continue

                # --- INT8 accuracy ---
                int8_acc = evaluate(model_q, val_loader)
                int8_kb = model_size_kb(model_q)
                acc_drop = fp32_acc - int8_acc
                compress = fp32_kb / int8_kb if int8_kb > 0 else 0

                status = "OK" if abs(acc_drop) < 0.05 else "WARN"
                print(f"  INT8  → Acc: {int8_acc:.4f}  Size: {int8_kb:.1f} KB  "
                      f"Drop: {acc_drop:+.4f}  Compress: {compress:.1f}x  [{status}]",
                      flush=True)

                # --- Save quantized model ---
                qt_name = f"qsparse_{vname}_T{T}_fr{fr_int:02d}.pt"
                qt_path = os.path.join(MODEL_DIR, qt_name)
                torch.save(model_q.state_dict(), qt_path)

                # --- Export PTL ---
                ptl_name = f"qsparse_{vname}_T{T}_fr{fr_int:02d}.ptl"
                ptl_kb = export_qsparse_ptl(model_q, ptl_name)
                if ptl_kb:
                    print(f"  Export → {ptl_name} ({ptl_kb:.1f} KB)", flush=True)

                # --- Record result ---
                result = {
                    "T": T,
                    "variant": vname,
                    "kernel_size": ks,
                    "target_rate": target_rate,
                    "type": "qsparse",
                    "params": count_params(model),
                    "fp32_val_acc": round(fp32_acc, 4),
                    "int8_val_acc": round(int8_acc, 4),
                    "acc_drop": round(acc_drop, 4),
                    "fp32_size_kb": round(fp32_kb, 1),
                    "int8_size_kb": round(int8_kb, 1),
                    "ptl_size_kb": round(ptl_kb, 1) if ptl_kb else None,
                    "compression_ratio": round(compress, 2),
                    "thresh1": round(model.lif1.threshold, 4),
                    "thresh2": round(model.lif2.threshold, 4),
                    "gain": round(model.gain, 4),
                }
                all_results.append(result)

                del model, model_q

    # ========================= Save Results =========================
    result_path = os.path.join(RESULT_DIR, "qsparse_results.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # ========================= Summary =========================
    total_time = time.time() - start_time
    print(f"\n\n{'='*80}", flush=True)
    print(f"  SUMMARY: INT8 Quantization Complete", flush=True)
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)
    print(f"  Processed: {len(all_results)}  Skipped: {skipped}  Failed: {failed}", flush=True)
    print(f"{'='*80}", flush=True)

    if all_results:
        print(f"\n{'Variant':<10} {'T':>3} {'FR%':>5} {'FP32':>8} {'INT8':>8} "
              f"{'Drop':>7} {'FP32KB':>7} {'INT8KB':>7} {'Ratio':>6}", flush=True)
        print("-" * 72, flush=True)

        for r in all_results:
            fr_pct = int(r['target_rate'] * 100)
            print(f"{r['variant']:<10} {r['T']:>3} {fr_pct:>5} "
                  f"{r['fp32_val_acc']:>8.4f} {r['int8_val_acc']:>8.4f} "
                  f"{r['acc_drop']:>+7.4f} "
                  f"{r['fp32_size_kb']:>7.1f} {r['int8_size_kb']:>7.1f} "
                  f"{r['compression_ratio']:>5.1f}x", flush=True)

    # --- Best configs ---
    if all_results:
        print(f"\n  Top 5 by INT8 Accuracy:", flush=True)
        sorted_by_acc = sorted(all_results, key=lambda r: r['int8_val_acc'], reverse=True)
        for r in sorted_by_acc[:5]:
            fr_pct = int(r['target_rate'] * 100)
            print(f"    {r['variant']} T={r['T']} fr={fr_pct}%  "
                  f"INT8={r['int8_val_acc']:.4f}  drop={r['acc_drop']:+.4f}  "
                  f"size={r['int8_size_kb']:.1f}KB", flush=True)

        print(f"\n  Top 5 smallest INT8 models (acc >= 0.95):", flush=True)
        valid = [r for r in all_results if r['int8_val_acc'] >= 0.95]
        sorted_by_size = sorted(valid, key=lambda r: r['int8_size_kb'])
        for r in sorted_by_size[:5]:
            fr_pct = int(r['target_rate'] * 100)
            print(f"    {r['variant']} T={r['T']} fr={fr_pct}%  "
                  f"INT8={r['int8_val_acc']:.4f}  size={r['int8_size_kb']:.1f}KB",
                  flush=True)

    print(f"\nResults saved to {result_path}", flush=True)


if __name__ == "__main__":
    main()
