"""
CNN Static Quantization (Conv1d + FC 전부 INT8) — QSparse와 동일한 방식
기존 FP32 CNN .pt 파일을 로드하여 Static Quantization 적용 후 PTL export
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
# Dataset (train_cnn1d_k9.py와 동일)
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
        print(f"  Loaded {count} samples")

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
        return np.stack(channels, axis=0)

    def __len__(self):
        return len(self.probs)

    def __getitem__(self, idx):
        prob = self.probs[idx]
        C, N = prob.shape
        spike = (np.random.rand(C, N, self.T) < prob[:, :, None]).astype(np.float32)
        tensor = torch.from_numpy(spike)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor, label


def pad_collate(batch):
    xs, ys = zip(*batch)
    max_n = max(x.shape[1] for x in xs)
    c, _, t = xs[0].shape
    out = torch.zeros(len(xs), c, max_n, t, dtype=xs[0].dtype)
    for i, x in enumerate(xs):
        n = x.shape[1]
        out[i, :, :n, :] = x
    return out, torch.stack(ys)


# ============================================================
# FP32 CNN (원본과 동일)
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
        if x.dim() == 4:
            x = x.mean(dim=-1)
        x = self.relu1(self.gn1(self.conv1(x)))
        x = self.relu2(self.gn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


# ============================================================
# Quantizable CNN — Static Quantization용
# ============================================================
class QuantizableConv1dCNN(nn.Module):
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


def get_file_size_kb(path):
    return os.path.getsize(path) / 1024 if os.path.exists(path) else 0


def export_ptl(model, path):
    model.eval()
    scripted = torch.jit.script(model)
    optimized = optimize_for_mobile(scripted)
    optimized._save_for_lite_interpreter(path)
    return get_file_size_kb(path)


def main():
    torch.backends.quantized.engine = "qnnpack"

    data_dir = "./data/2_ffilled_data"
    src_dir = "./models"
    out_dir = "./models_cnn"
    result_dir = "./result"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # Load dataset
    dataset = SpikeDataset(data_dir, T=20)
    total = len(dataset)
    train_size = int(0.8 * total)
    val_size = total - train_size
    gen = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=gen)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=pad_collate)
    calibration_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=pad_collate)

    all_results = []

    for variant, (ch1, ch2) in SIZE_VARIANTS.items():
        print(f"\n{'='*60}")
        print(f"QCNN Static: {variant} (ch={ch1},{ch2}, k={KERNEL_SIZE})")
        print(f"{'='*60}")

        # 1. Load FP32 model
        pt_path = os.path.join(src_dir, f"cnn1d_{variant}_sensor.pt")
        fp32_model = Conv1dCNN(ch1, ch2, kernel_size=KERNEL_SIZE, dropout=0.0)
        fp32_model.load_state_dict(torch.load(pt_path, map_location="cpu"))
        fp32_model.eval()

        # FP32 accuracy
        fp32_correct, fp32_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                out = fp32_model(x)
                fp32_correct += (out.argmax(1) == y).sum().item()
                fp32_total += y.size(0)
        fp32_acc = fp32_correct / fp32_total
        print(f"  FP32 Val Acc: {fp32_acc:.4f}")

        # FP32 PTL export
        fp32_ptl_path = os.path.join(out_dir, f"cnn1d_{variant}.ptl")
        fp32_ptl_kb = export_ptl(fp32_model, fp32_ptl_path)
        print(f"  FP32 PTL: {fp32_ptl_kb:.1f} KB")

        # 2. Create QuantizableConv1dCNN and load weights (GN → BN)
        q_model = QuantizableConv1dCNN(ch1, ch2, kernel_size=KERNEL_SIZE)
        sd = fp32_model.state_dict()
        bn_sd = q_model.state_dict()
        for key, val in sd.items():
            mapped = key.replace("gn1.", "bn1.").replace("gn2.", "bn2.")
            if mapped in bn_sd and "dropout" not in key:
                bn_sd[mapped] = val
        # Initialize BN running stats
        for name in bn_sd:
            if "running_mean" in name:
                bn_sd[name].zero_()
            elif "running_var" in name:
                bn_sd[name].fill_(1.0)
            elif "num_batches_tracked" in name:
                bn_sd[name].zero_()
        q_model.load_state_dict(bn_sd)

        # 3. Calibrate BN running stats
        q_model.eval()
        q_model.bn1.reset_running_stats()
        q_model.bn2.reset_running_stats()
        q_model.bn1.training = True
        q_model.bn2.training = True
        print("  Calibrating BN stats...")
        with torch.no_grad():
            for x, _ in calibration_loader:
                q_model(x)
        q_model.bn1.training = False
        q_model.bn2.training = False
        q_model.eval()

        # 4. Static Quantization (Conv1d + FC 전부 INT8)
        q_model_int8 = copy.deepcopy(q_model)
        q_model_int8.eval()
        q_model_int8.fuse_model()
        q_model_int8.qconfig = torch.quantization.get_default_qconfig("qnnpack")
        torch.quantization.prepare(q_model_int8, inplace=True)

        print("  Calibrating quantization observers...")
        with torch.no_grad():
            for x, _ in calibration_loader:
                q_model_int8(x)

        torch.quantization.convert(q_model_int8, inplace=True)

        # 5. INT8 accuracy
        int8_correct, int8_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                out = q_model_int8(x)
                int8_correct += (out.argmax(1) == y).sum().item()
                int8_total += y.size(0)
        int8_acc = int8_correct / int8_total
        acc_drop = fp32_acc - int8_acc
        print(f"  INT8 Val Acc: {int8_acc:.4f} (drop: {acc_drop:+.4f})")

        # 6. Export INT8 PTL
        int8_ptl_path = os.path.join(out_dir, f"qcnn1d_{variant}_static.ptl")
        int8_ptl_kb = export_ptl(q_model_int8, int8_ptl_path)
        print(f"  INT8 PTL: {int8_ptl_kb:.1f} KB")

        print(f"  Size reduction: {fp32_ptl_kb:.1f}KB → {int8_ptl_kb:.1f}KB "
              f"({fp32_ptl_kb/int8_ptl_kb:.1f}x compression)")

        result = {
            "variant": variant,
            "channels": [ch1, ch2],
            "kernel_size": KERNEL_SIZE,
            "fp32_val_acc": fp32_acc,
            "int8_val_acc": int8_acc,
            "acc_drop": round(acc_drop, 4),
            "fp32_ptl_kb": round(fp32_ptl_kb, 1),
            "int8_ptl_kb": round(int8_ptl_kb, 1),
            "compression_ratio": round(fp32_ptl_kb / int8_ptl_kb, 2) if int8_ptl_kb > 0 else 0,
        }
        all_results.append(result)

    # Save results
    results_path = os.path.join(result_dir, "qcnn_static_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: QCNN Static Quantization (Conv1d + FC 전부 INT8)")
    print(f"{'='*80}")
    print(f"{'Variant':<10} {'Ch':>10} | {'FP32 Acc':>9} {'FP32 PTL':>9} | {'INT8 Acc':>9} {'INT8 PTL':>9} | {'Drop':>7} {'Ratio':>6}")
    print("-" * 80)
    for r in all_results:
        ch = f"({r['channels'][0]},{r['channels'][1]})"
        print(f"{r['variant']:<10} {ch:>10} | {r['fp32_val_acc']:>9.4f} {r['fp32_ptl_kb']:>8.1f}KB | "
              f"{r['int8_val_acc']:>9.4f} {r['int8_ptl_kb']:>8.1f}KB | {r['acc_drop']:>+7.4f} {r['compression_ratio']:>5.1f}x")

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
