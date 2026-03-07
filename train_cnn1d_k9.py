"""
Conv1d CNN (kernel=9) training for 4-class gesture recognition.
Conv2d → Conv1d 전환, kernel_size=9.
기존 train_cnn_4class.py의 Conv2d CNN을 Conv1d로 변환.

5 size variants: smallest~largest
+ Quantized (dynamic) export
"""

import os
import json
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.mobile_optimizer import optimize_for_mobile
from scipy.special import expit


# ============================================================
# Dataset: spike-encoded (12, N, T) — SNN과 동일
# ============================================================
class SpikeDataset(Dataset):
    def __init__(self, folder, T=20):
        self.T = T
        self.probs = []
        self.labels = []
        count = 0
        for file in sorted(os.listdir(folder)):
            if not file.endswith(".csv"):
                continue
            path = os.path.join(folder, file)
            lower = file.lower()
            if "swipe_up" in lower and "sensor" in lower:
                label = 0
            elif "swipe_down" in lower and "sensor" in lower:
                label = 1
            elif "flick_up" in lower and "sensor" in lower:
                label = 2
            elif "flick_down" in lower and "sensor" in lower:
                label = 3
            else:
                continue
            df = pd.read_csv(path)
            prob = self._compute_probs(df)
            self.probs.append(prob)
            self.labels.append(label)
            count += 1
        print(f"  Loaded {count} samples from {folder}")

    def _compute_probs(self, df):
        channels = []
        for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
            signal = df[col].to_numpy()
            pos = np.clip(signal, 0, None)
            neg = np.clip(-signal, 0, None)
            for s in [pos, neg]:
                mean = np.mean(s)
                std = np.std(s) + 1e-8
                z = (s - mean) / std
                prob = expit(z)
                channels.append(prob)
        return np.stack(channels, axis=0)  # (12, N)

    def __len__(self):
        return len(self.probs)

    def __getitem__(self, idx):
        prob = self.probs[idx]  # (12, N)
        C, N = prob.shape
        spike = (np.random.rand(C, N, self.T) < prob[:, :, None]).astype(np.float32)
        tensor = torch.from_numpy(spike)  # (12, N, T)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor, label


def pad_collate(batch):
    """Pad to max sequence length. Output: (B, 12, N_max, T)."""
    xs, ys = zip(*batch)
    max_n = max(x.shape[1] for x in xs)
    c, _, t = xs[0].shape
    out = torch.zeros(len(xs), c, max_n, t, dtype=xs[0].dtype)
    for i, x in enumerate(xs):
        n = x.shape[1]
        out[i, :, :n, :] = x
    return out, torch.stack(ys)


# ============================================================
# Conv1d CNN with configurable kernel size
# 입력: (B, 12, N, T) → T timesteps를 평균하여 (B, 12, N) → Conv1d
# ============================================================
class Conv1dCNN(nn.Module):
    def __init__(self, ch1, ch2, kernel_size=9, num_classes=4, dropout=0.5):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(12, ch1, kernel_size=kernel_size, padding=pad)
        self.gn1 = nn.GroupNorm(min(8, ch1), ch1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(ch1, ch2, kernel_size=kernel_size, padding=pad)
        self.gn2 = nn.GroupNorm(min(8, ch2), ch2)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(ch2, num_classes)

    def forward(self, x):
        # x: (B, 12, N, T) → average over T → (B, 12, N)
        if x.dim() == 4:
            x = x.mean(dim=-1)
        x = self.relu1(self.gn1(self.conv1(x)))
        x = self.relu2(self.gn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


class QuantizableConv1dCNN(nn.Module):
    """Quantization-compatible Conv1d CNN (BN for fusion)."""
    def __init__(self, ch1, ch2, kernel_size=9, num_classes=4):
        super().__init__()
        pad = kernel_size // 2
        self.quant = torch.quantization.QuantStub()
        self.conv1 = nn.Conv1d(12, ch1, kernel_size=kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(ch1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(ch1, ch2, kernel_size=kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(ch2)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(ch2, num_classes)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        if x.dim() == 4:
            x = x.mean(dim=-1)
        x = self.quant(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(
            self, [["conv1", "bn1", "relu1"], ["conv2", "bn2", "relu2"]], inplace=True
        )


SIZE_VARIANTS = {
    "smallest": (16, 32),
    "small":    (24, 48),
    "medium":   (32, 64),
    "large":    (40, 80),
    "largest":  (48, 96),
}

KERNEL_SIZE = 9


# ============================================================
# Training
# ============================================================
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_kb(model):
    size = sum(p.nelement() * p.element_size() for p in model.parameters())
    size += sum(b.nelement() * b.element_size() for b in model.buffers())
    return size / 1024.0


def get_file_size_kb(path):
    return os.path.getsize(path) / 1024 if os.path.exists(path) else 0


def export_ptl(model, path):
    model.eval()
    scripted = torch.jit.script(model)
    optimized = optimize_for_mobile(scripted)
    optimized._save_for_lite_interpreter(path)
    return get_file_size_kb(path)


def load_gn_to_bn(gn_state_dict, q_model):
    bn_sd = q_model.state_dict()
    for key, val in gn_state_dict.items():
        mapped = key.replace("gn1.", "bn1.").replace("gn2.", "bn2.")
        if mapped in bn_sd and "dropout" not in mapped:
            bn_sd[mapped] = val
    for name in bn_sd:
        if "running_mean" in name:
            bn_sd[name].zero_()
        elif "running_var" in name:
            bn_sd[name].fill_(1.0)
        elif "num_batches_tracked" in name:
            bn_sd[name].zero_()
    q_model.load_state_dict(bn_sd)
    return q_model


def quantize_model(model_fp32, calibration_loader):
    model_q = copy.deepcopy(model_fp32)
    model_q.eval()
    model_q.fuse_model()
    model_q.qconfig = torch.quantization.get_default_qconfig("qnnpack")
    torch.quantization.prepare(model_q, inplace=True)
    with torch.no_grad():
        for x, _ in calibration_loader:
            model_q(x)
    torch.quantization.convert(model_q, inplace=True)
    return model_q


def train_model(model, train_loader, val_loader, epochs=200, lr=1e-3,
                weight_decay=1e-4, patience=30, accum_steps=16, device="cpu"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2
    )
    best_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        correct, total, total_loss = 0, 0, 0.0
        optimizer.zero_grad()

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y) / accum_steps
            loss.backward()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
            total_loss += loss.item() * accum_steps

            if (i + 1) % accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        if (i + 1) % accum_steps != 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step(epoch + 1)
        train_acc = correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += y.size(0)
        val_acc = val_correct / val_total

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"  [Epoch {epoch+1:3d}] Train: {train_acc:.4f}  Val: {val_acc:.4f}  "
                  f"Loss: {total_loss/total:.4f}  LR: {cur_lr:.1e}  Best: {best_acc:.4f}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model.cpu(), best_acc


# ============================================================
# Main
# ============================================================
def main():
    torch.backends.quantized.engine = "qnnpack"

    data_dir = "./data/2_ffilled_data"
    out_dir = "./models"
    result_dir = "./result"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    dataset = SpikeDataset(data_dir, T=20)
    total = len(dataset)
    print(f"Total samples: {total}")
    train_size = int(0.8 * total)
    val_size = total - train_size
    gen = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=gen)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                              collate_fn=pad_collate)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            collate_fn=pad_collate)
    calibration_loader = DataLoader(train_dataset, batch_size=1, shuffle=False,
                                    collate_fn=pad_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Kernel size: {KERNEL_SIZE}")
    print(f"Training: Conv1d, GN+ReLU, grad_accum=16, label_smooth=0.1, cosine_LR, patience=30\n")

    all_results = []

    for variant, (ch1, ch2) in SIZE_VARIANTS.items():
        print(f"{'='*60}")
        print(f"Conv1d CNN {variant} (ch={ch1},{ch2}, k={KERNEL_SIZE})")
        print(f"{'='*60}")

        # --- FP32 CNN ---
        model = Conv1dCNN(ch1, ch2, kernel_size=KERNEL_SIZE, dropout=0.5)
        params = count_params(model)
        print(f"  Parameters: {params:,}  Size: {model_size_kb(model):.1f}KB")

        model, best_acc = train_model(
            model, train_loader, val_loader,
            epochs=200, lr=1e-3, weight_decay=1e-4, patience=30,
            accum_steps=16, device=device,
        )

        pt_path = os.path.join(out_dir, f"cnn1d_{variant}_sensor.pt")
        torch.save(model.state_dict(), pt_path)

        ptl_path = os.path.join(out_dir, f"cnn1d_{variant}_sensor.ptl")
        ptl_kb = export_ptl(model, ptl_path)
        print(f"  FP32 PTL: {ptl_path} ({ptl_kb:.1f} KB)")

        result = {
            "variant": variant,
            "type": "Conv1dCNN_k9",
            "kernel_size": KERNEL_SIZE,
            "channels": [ch1, ch2],
            "params": params,
            "size_kb": round(model_size_kb(model), 1),
            "best_val_acc": best_acc,
            "file_kb_ptl": round(ptl_kb, 1),
            "pt_path": pt_path,
            "ptl_path": ptl_path,
        }

        # --- Quantized CNN ---
        print(f"\n  Quantizing {variant}...")
        sd = model.state_dict()
        try:
            q_model = QuantizableConv1dCNN(ch1, ch2, kernel_size=KERNEL_SIZE)
            load_gn_to_bn(sd, q_model)

            q_model.eval()
            q_model.bn1.reset_running_stats()
            q_model.bn2.reset_running_stats()
            q_model.bn1.training = True
            q_model.bn2.training = True
            with torch.no_grad():
                for x, _ in calibration_loader:
                    q_model(x)
            q_model.bn1.training = False
            q_model.bn2.training = False
            q_model.eval()

            q_model_int8 = quantize_model(q_model, calibration_loader)

            q_correct, q_total = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    out = q_model_int8(x)
                    q_correct += (out.argmax(1) == y).sum().item()
                    q_total += y.size(0)
            q_acc = q_correct / q_total
            print(f"  Quantized Val Acc: {q_acc:.4f} (FP32 best: {best_acc:.4f})")

            qptl_path = os.path.join(out_dir, f"qcnn1d_{variant}_sensor.ptl")
            model_for_export = Conv1dCNN(ch1, ch2, kernel_size=KERNEL_SIZE, dropout=0.0)
            model_for_export.load_state_dict(sd)
            model_for_export.eval()
            model_dq = torch.quantization.quantize_dynamic(
                model_for_export, {nn.Linear}, dtype=torch.qint8
            )
            qptl_kb = export_ptl(model_dq, qptl_path)
            print(f"  Quantized PTL: {qptl_path} ({qptl_kb:.1f} KB)")

            result["q_val_acc"] = q_acc
            result["q_file_kb_ptl"] = round(qptl_kb, 1)
            result["q_ptl_path"] = qptl_path
        except Exception as e:
            print(f"  [WARN] Quantization failed: {e}")
            import traceback; traceback.print_exc()
            result["q_val_acc"] = None
            result["q_file_kb_ptl"] = None

        all_results.append(result)
        print()

    results_path = os.path.join(result_dir, "cnn1d_k9_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"SUMMARY: Conv1d CNN (kernel={KERNEL_SIZE})")
    print(f"{'='*80}")
    print(f"{'Variant':<12} {'Params':<10} {'FP32 Best':<11} {'Q-INT8':<10} {'FP32 KB':<9} {'Q KB':<9}")
    print("-" * 65)
    for r in all_results:
        q_acc = f"{r.get('q_val_acc', 0):.4f}" if r.get("q_val_acc") else "N/A"
        q_kb = f"{r.get('q_file_kb_ptl', 0):.1f}" if r.get("q_file_kb_ptl") else "N/A"
        print(f"{r['variant']:<12} {r['params']:<10,} {r['best_val_acc']:<11.4f} "
              f"{q_acc:<10} {r['file_kb_ptl']:<9.1f} {q_kb:<9}")

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
