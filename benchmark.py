"""
Benchmark: Teacher SNN / Student SNN / Sparse SNN / CNN / QCNN
Measures: Validation Accuracy, CPU Latency, Theoretical Energy
"""
import json
import os
import time
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import FileLevelSpikeDataset, rate_code_zscore_sigmoid, pad_collate
from lif_module import LIFNode
from model import HardLIFNode

# ========================= Config =========================
DATA_DIR = "./data/2_ffilled_data"
MODEL_DIR = "./models"
RESULT_DIR = "./result"
NUM_CLASSES = 4
BATCH_SIZE = 1          # single-sample latency
DEVICE = torch.device("cpu")
WARMUP = 20
REPEATS = 100
SEED = 42

# Energy constants (45nm process, pJ per operation)
ENERGY_FP32_MAC = 4.6
ENERGY_FP32_AC = 0.9
ENERGY_INT8_MAC = 0.2
ENERGY_SRAM_ACCESS = 5.0    # per access (cache hit)
ENERGY_DRAM_ACCESS = 640.0  # per access (cache miss)

# Size variants
VARIANT_CHANNELS = {
    "smallest": (16, 32),
    "small":    (24, 48),
    "medium":   (32, 64),
    "large":    (40, 80),
    "largest":  (48, 96),
}

# Kernel sweep variant mapping (used by T-sweep / kernel-sweep models)
KERNEL_VARIANT = {3: "smallest", 5: "small", 7: "medium", 9: "large", 11: "largest"}
VARIANT_KERNEL = {v: k for k, v in KERNEL_VARIANT.items()}


# ========================= Model Definitions =========================
# (Self-contained so benchmark doesn't depend on training scripts)

class TeacherSNN1d(nn.Module):
    def __init__(self, c1, c2, kernel_size=3, T=20):
        super().__init__()
        self.T = T
        self.conv1 = nn.Conv1d(12, c1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.lif1 = LIFNode(tau=1.0, threshold=0.2)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=kernel_size // 2)
        self.lif2 = LIFNode(tau=0.9, threshold=0.15)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c2, NUM_CLASSES)

    def forward(self, x):
        xt = x[:, :, :, 0]
        xt = self.lif1(self.conv1(xt))
        xt = self.lif2(self.conv2(xt))
        acc = xt.clone()
        for t in range(1, self.T):
            xt = x[:, :, :, t]
            xt = self.lif1(self.conv1(xt))
            xt = self.lif2(self.conv2(xt))
            acc = acc + xt
        out = acc / float(self.T)
        out = self.pool(out).squeeze(-1)
        return self.fc(out)

    def reset(self):
        self.lif1.reset()
        self.lif2.reset()


class StudentSNN1d(nn.Module):
    def __init__(self, c1, c2, kernel_size=3, T=20,
                 thresh1=0.02, thresh2=0.02, surrogate_width=5.0, gain=3.0):
        super().__init__()
        self.T = T
        self.gain = gain
        self.conv1 = nn.Conv1d(12, c1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.lif1 = HardLIFNode(tau=1.0, threshold=thresh1, surrogate_width=surrogate_width)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=kernel_size // 2)
        self.lif2 = HardLIFNode(tau=0.9, threshold=thresh2, surrogate_width=surrogate_width)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c2, NUM_CLASSES)

    def forward(self, x):
        xt = x[:, :, :, 0]
        xt = self.lif1(self.conv1(xt) * self.gain)
        xt = self.lif2(self.conv2(xt) * self.gain)
        acc = xt
        for t in range(1, self.T):
            xt = x[:, :, :, t]
            xt = self.lif1(self.conv1(xt) * self.gain)
            xt = self.lif2(self.conv2(xt) * self.gain)
            acc = acc + xt
        out = acc / float(self.T)
        out = self.pool(out).squeeze(-1)
        return self.fc(out)

    def reset(self):
        self.lif1.reset()
        self.lif2.reset()


class SparseStudentSNN1d(nn.Module):
    def __init__(self, c1, c2, kernel_size=3, T=20,
                 thresh1=0.02, thresh2=0.02, surrogate_width=5.0, gain=3.0):
        super().__init__()
        self.T = T
        self.gain = nn.Parameter(torch.tensor(float(gain)))
        self.conv1 = nn.Conv1d(12, c1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.lif1 = _BenchmarkLIF(tau=1.0, threshold=thresh1, surrogate_width=surrogate_width)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=kernel_size // 2)
        self.lif2 = _BenchmarkLIF(tau=0.9, threshold=thresh2, surrogate_width=surrogate_width)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c2, NUM_CLASSES)

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


class _BenchmarkLIF(nn.Module):
    """HardLIF with learnable threshold for sparse model loading."""
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
        spike = (self.v >= self.threshold).float()
        self.v = torch.where(spike > 0, torch.full_like(self.v, self.reset_value), self.v)
        return spike

    def reset(self):
        self.v = torch.zeros(0, device=self.v.device if isinstance(self.v, torch.Tensor) else None)


class Conv1dCNN(nn.Module):
    def __init__(self, ch1, ch2, kernel_size=9, num_classes=4, dropout=0.0):
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


# ========================= Utility Functions =========================

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_kb(model):
    size = sum(p.nelement() * p.element_size() for p in model.parameters())
    size += sum(b.nelement() * b.element_size() for b in model.buffers())
    return size / 1024.0


def load_state_filtered(model, state_dict):
    """Load state dict, skipping membrane potential buffers."""
    filtered = {k: v for k, v in state_dict.items() if not k.endswith(".v")}
    model.load_state_dict(filtered, strict=False)


def measure_accuracy(model, loader, device, is_snn=True):
    """Measure validation accuracy."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if is_snn and hasattr(model, "reset"):
                model.reset()
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def measure_latency(model, sample_input, device, is_snn=True, warmup=WARMUP, repeats=REPEATS):
    """Measure single-sample latency in ms (GPU-aware with cuda.synchronize)."""
    model.eval()
    x = sample_input.to(device)
    use_cuda = device.type == "cuda"

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            if is_snn and hasattr(model, "reset"):
                model.reset()
            model(x)
            if use_cuda:
                torch.cuda.synchronize()

    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(repeats):
            if is_snn and hasattr(model, "reset"):
                model.reset()
            if use_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter_ns()
            model(x)
            if use_cuda:
                torch.cuda.synchronize()
            end = time.perf_counter_ns()
            latencies.append((end - start) / 1e6)  # ms

    return {
        "mean_ms": np.mean(latencies),
        "median_ms": np.median(latencies),
        "p95_ms": np.percentile(latencies, 95),
        "std_ms": np.std(latencies),
    }


def measure_firing_rates(model, loader, device):
    """Measure average firing rates for sparse SNN."""
    model.eval()
    fr1_sum, fr2_sum, count = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            model.reset()
            _, spk1_list, spk2_list = model(x, return_spikes=True)
            fr1_sum += torch.stack(spk1_list).mean().item()
            fr2_sum += torch.stack(spk2_list).mean().item()
            count += 1
    if count == 0:
        return 0.0, 0.0
    return fr1_sum / count, fr2_sum / count


def count_conv1d_macs(in_ch, out_ch, kernel_size, seq_len):
    """Count MAC operations for one Conv1d layer."""
    return in_ch * out_ch * kernel_size * seq_len


def count_linear_macs(in_features, out_features):
    return in_features * out_features


def estimate_energy(model_type, c1, c2, kernel_size, seq_len, T,
                    firing_rate_1=1.0, firing_rate_2=1.0, is_int8=False):
    """
    Estimate theoretical energy consumption.

    CNN:      conv1(MAC) + conv2(MAC) + fc(MAC)               x 1
    Teacher:  conv1(MAC) + conv2(MAC) + fc(MAC)               x T  (soft spike = still MAC)
    Student:  conv1(MAC) + conv2(AC * fr1) + fc(MAC)          x T  (layer1: analog in, layer2+: binary spike)
    Sparse:   conv1(MAC) + conv2(AC * fr1) + fc(MAC)          x T  (lower firing rate)
    QCNN:     conv1(INT8 MAC) + conv2(INT8 MAC) + fc(INT8 MAC) x 1

    Note: Layer 1 always receives analog/rate-coded input, so it's MAC.
          Layer 2 receives binary spikes from LIF1, so it can be AC (if exploited).
          Currently NOT exploited, but we estimate the *potential* AC energy too.
    """
    e_mac = ENERGY_INT8_MAC if is_int8 else ENERGY_FP32_MAC
    e_ac = ENERGY_FP32_AC  # AC is always float addition

    # Conv1: input is analog (rate-coded or spike probability) → MAC
    macs_conv1 = count_conv1d_macs(12, c1, kernel_size, seq_len)

    # Conv2: input could be binary spike → AC (if exploited)
    macs_conv2_full = count_conv1d_macs(c1, c2, kernel_size, seq_len)

    # FC: input is pooled float → MAC
    macs_fc = count_linear_macs(c2, NUM_CLASSES)

    if model_type == "CNN":
        # Single forward pass, all MAC
        total_macs = macs_conv1 + macs_conv2_full + macs_fc
        energy_actual = total_macs * e_mac
        energy_potential = energy_actual  # no AC potential
        return energy_actual, energy_potential, total_macs, 0

    elif model_type == "QCNN":
        total_macs = macs_conv1 + macs_conv2_full + macs_fc
        energy_actual = total_macs * ENERGY_INT8_MAC
        energy_potential = energy_actual
        return energy_actual, energy_potential, total_macs, 0

    elif model_type == "Teacher":
        # T timesteps, all MAC (soft LIF output is continuous, not binary)
        total_macs = (macs_conv1 + macs_conv2_full + macs_fc) * T
        energy_actual = total_macs * e_mac
        energy_potential = energy_actual
        return energy_actual, energy_potential, total_macs, 0

    else:
        # Student / Sparse: hard binary spikes
        # ACTUAL (current code): still dense Conv1d → all MAC
        total_macs_actual = (macs_conv1 + macs_conv2_full + macs_fc) * T
        energy_actual = total_macs_actual * e_mac

        # POTENTIAL (if AC exploited):
        # Conv1: still MAC (analog input from spike encoding)
        # Conv2: AC, only active when LIF1 fires (firing_rate_1)
        # FC: MAC (pooled continuous input)
        acs_conv2 = macs_conv2_full * firing_rate_1  # only fr1 fraction active
        energy_per_step = (macs_conv1 * e_mac +
                           acs_conv2 * e_ac +
                           macs_fc * e_mac)
        energy_potential = energy_per_step * T
        total_acs = int(acs_conv2 * T)

        return energy_actual, energy_potential, total_macs_actual, total_acs


# ========================= Model Registry =========================

def discover_models():
    """Discover all available trained models and return a list of configs."""
    models = []

    # --- CNN (k=9, channel variants) ---
    for vname, (c1, c2) in VARIANT_CHANNELS.items():
        pt = os.path.join(MODEL_DIR, f"cnn1d_{vname}_sensor.pt")
        if os.path.exists(pt):
            models.append({
                "type": "CNN", "variant": vname, "kernel_size": 9,
                "c1": c1, "c2": c2, "T": 1, "pt": pt,
                "firing_rate": None, "target_fr": None,
            })

    # --- QCNN (k=9, channel variants) - ptl only ---
    for vname, (c1, c2) in VARIANT_CHANNELS.items():
        ptl = os.path.join(MODEL_DIR, f"qcnn1d_{vname}_sensor.ptl")
        pt_fp32 = os.path.join(MODEL_DIR, f"cnn1d_{vname}_sensor.pt")
        if os.path.exists(ptl) and os.path.exists(pt_fp32):
            models.append({
                "type": "QCNN", "variant": vname, "kernel_size": 9,
                "c1": c1, "c2": c2, "T": 1, "ptl": ptl, "pt_fp32": pt_fp32,
                "firing_rate": None, "target_fr": None,
            })

    # --- Teacher SNN (T-sweep: T=3,5,10,15 × k=3,5,7,9,11) ---
    for T in [3, 5, 10, 15]:
        for ks, vname in KERNEL_VARIANT.items():
            pt = os.path.join(MODEL_DIR, f"snn1d_teacher_{vname}_T{T}.pt")
            if os.path.exists(pt):
                models.append({
                    "type": "Teacher", "variant": vname, "kernel_size": ks,
                    "c1": 32, "c2": 64, "T": T, "pt": pt,
                    "firing_rate": None, "target_fr": None,
                })

    # --- Teacher SNN (kernel-sweep: T=20, k=3,5,7,9) ---
    for ks in [3, 5, 7, 9]:
        pt = os.path.join(MODEL_DIR, f"snn1d_teacher_k{ks}.pt")
        if os.path.exists(pt):
            vname = KERNEL_VARIANT[ks]
            models.append({
                "type": "Teacher", "variant": vname, "kernel_size": ks,
                "c1": 32, "c2": 64, "T": 20, "pt": pt,
                "firing_rate": None, "target_fr": None,
            })

    # --- Teacher k=11 T=20 ---
    pt = os.path.join(MODEL_DIR, "snn1d_teacher_largest.pt")
    if os.path.exists(pt):
        models.append({
            "type": "Teacher", "variant": "largest", "kernel_size": 11,
            "c1": 32, "c2": 64, "T": 20, "pt": pt,
            "firing_rate": None, "target_fr": None,
        })

    # --- Student SNN (T-sweep: T=3,5,10,15 × k=3,5,7,9,11) ---
    for T in [3, 5, 10, 15]:
        for ks, vname in KERNEL_VARIANT.items():
            pt = os.path.join(MODEL_DIR, f"snn1d_student_{vname}_T{T}.pt")
            if os.path.exists(pt):
                models.append({
                    "type": "Student", "variant": vname, "kernel_size": ks,
                    "c1": 32, "c2": 64, "T": T, "pt": pt,
                    "firing_rate": None, "target_fr": None,
                })

    # --- Student SNN (kernel-sweep: T=20, k=3,5,7,9) ---
    for ks in [3, 5, 7, 9]:
        pt = os.path.join(MODEL_DIR, f"snn1d_student_k{ks}.pt")
        if os.path.exists(pt):
            vname = KERNEL_VARIANT[ks]
            models.append({
                "type": "Student", "variant": vname, "kernel_size": ks,
                "c1": 32, "c2": 64, "T": 20, "pt": pt,
                "firing_rate": None, "target_fr": None,
            })

    # --- Student k=11 T=20 ---
    pt = os.path.join(MODEL_DIR, "snn1d_student_largest.pt")
    if os.path.exists(pt):
        models.append({
            "type": "Student", "variant": "largest", "kernel_size": 11,
            "c1": 32, "c2": 64, "T": 20, "pt": pt,
            "firing_rate": None, "target_fr": None,
        })

    # --- Sparse Student (T=3,5 × k=3,5,7,9,11 × fr=05,10,20,30) ---
    for T in [3, 5, 10, 15]:
        for ks, vname in KERNEL_VARIANT.items():
            for fr in [5, 10, 20, 30]:
                pt = os.path.join(MODEL_DIR, f"sparse_{vname}_T{T}_fr{fr:02d}.pt")
                if os.path.exists(pt):
                    models.append({
                        "type": "Sparse", "variant": vname, "kernel_size": ks,
                        "c1": 32, "c2": 64, "T": T, "pt": pt,
                        "firing_rate": None, "target_fr": fr / 100.0,
                    })

    return models


def build_and_load_model(cfg):
    """Build model from config and load weights."""
    mtype = cfg["type"]
    c1, c2, ks, T = cfg["c1"], cfg["c2"], cfg["kernel_size"], cfg["T"]

    if mtype == "CNN":
        model = Conv1dCNN(c1, c2, kernel_size=ks, dropout=0.0)
        state = torch.load(cfg["pt"], map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model

    elif mtype == "QCNN":
        # Load the ptl (TorchScript Lite) directly for latency
        # But for accuracy, load FP32 and quantize dynamically
        model_fp32 = Conv1dCNN(c1, c2, kernel_size=ks, dropout=0.0)
        state = torch.load(cfg["pt_fp32"], map_location="cpu", weights_only=True)
        model_fp32.load_state_dict(state)
        model_fp32.eval()
        model_q = torch.quantization.quantize_dynamic(
            model_fp32, {nn.Linear}, dtype=torch.qint8
        )
        return model_q

    elif mtype == "Teacher":
        model = TeacherSNN1d(c1, c2, kernel_size=ks, T=T)
        state = torch.load(cfg["pt"], map_location="cpu", weights_only=True)
        load_state_filtered(model, state)
        return model

    elif mtype == "Student":
        model = StudentSNN1d(c1, c2, kernel_size=ks, T=T)
        state = torch.load(cfg["pt"], map_location="cpu", weights_only=True)
        load_state_filtered(model, state)
        return model

    elif mtype == "Sparse":
        model = SparseStudentSNN1d(c1, c2, kernel_size=ks, T=T)
        state = torch.load(cfg["pt"], map_location="cpu", weights_only=True)
        load_state_filtered(model, state)
        return model

    raise ValueError(f"Unknown model type: {mtype}")


# ========================= Main Benchmark =========================

def main():
    print("=" * 100)
    print("  BENCHMARK: SNN vs CNN — Accuracy / Latency / Energy")
    print("=" * 100)
    print(f"  Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA: {torch.version.cuda}")

    # --- Dataset ---
    print("\nLoading dataset...")
    dataset = FileLevelSpikeDataset(DATA_DIR, rate_code_zscore_sigmoid, T=20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_ds = random_split(dataset, [train_size, val_size],
                              generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=pad_collate, num_workers=0)
    print(f"Validation samples: {len(val_ds)}")

    # Get a sample input for latency measurement
    sample_x, _ = next(iter(val_loader))
    seq_len = sample_x.shape[2]  # N dimension
    print(f"Sample input shape: {sample_x.shape} (B, 12, N={seq_len}, T=20)")

    # --- Discover models ---
    all_configs = discover_models()
    print(f"\nDiscovered {len(all_configs)} models")

    # --- Filter for representative subset ---
    # To keep output manageable, select key comparisons
    selected = []
    for cfg in all_configs:
        # CNN/QCNN: all variants
        if cfg["type"] in ("CNN", "QCNN"):
            selected.append(cfg)
        # SNN: focus on T=3 and T=20 for all kernel sizes
        elif cfg["type"] in ("Teacher", "Student") and cfg["T"] in (3, 20):
            selected.append(cfg)
        # Sparse: T=3, fr=05 and fr=30 (extremes)
        elif cfg["type"] == "Sparse" and cfg["T"] == 3 and cfg.get("target_fr") in (0.05, 0.30):
            selected.append(cfg)

    print(f"Selected {len(selected)} models for benchmarking\n")

    # --- Run benchmark ---
    results = []
    for i, cfg in enumerate(selected):
        tag = f"{cfg['type']}"
        if cfg["type"] in ("CNN", "QCNN"):
            tag += f" {cfg['variant']} k={cfg['kernel_size']}"
        else:
            tag += f" {cfg['variant']} k={cfg['kernel_size']} T={cfg['T']}"
        if cfg.get("target_fr") is not None:
            tag += f" fr={cfg['target_fr']}"

        print(f"[{i+1}/{len(selected)}] {tag}...")

        try:
            model = build_and_load_model(cfg)
            model.to(DEVICE).eval()

            is_snn = cfg["type"] not in ("CNN", "QCNN")

            # QCNN quantized models only work on CPU
            dev = torch.device("cpu") if cfg["type"] == "QCNN" else DEVICE
            model.to(dev)

            # --- Accuracy ---
            # For SNN models with T != 20, rebuild dataset with matching T
            if is_snn and cfg["T"] != 20:
                ds_t = FileLevelSpikeDataset(DATA_DIR, rate_code_zscore_sigmoid, T=cfg["T"])
                _, val_ds_t = random_split(ds_t, [int(0.8 * len(ds_t)), len(ds_t) - int(0.8 * len(ds_t))],
                                            generator=torch.Generator().manual_seed(SEED))
                loader_t = DataLoader(val_ds_t, batch_size=BATCH_SIZE, shuffle=False,
                                      collate_fn=pad_collate, num_workers=0)
                acc = measure_accuracy(model, loader_t, dev, is_snn=is_snn)
                sample_for_latency, _ = next(iter(loader_t))
                seq_len_t = sample_for_latency.shape[2]
            else:
                acc = measure_accuracy(model, val_loader, dev, is_snn=is_snn)
                sample_for_latency = sample_x
                seq_len_t = seq_len

            # --- Latency ---
            lat = measure_latency(model, sample_for_latency, dev, is_snn=is_snn)

            # --- Firing rates (Sparse only) ---
            fr1, fr2 = 0.0, 0.0
            if cfg["type"] == "Sparse":
                loader_for_fr = loader_t if cfg["T"] != 20 else val_loader
                fr1, fr2 = measure_firing_rates(model, loader_for_fr, dev)

            # For Student models, measure firing rate too (to estimate AC potential)
            if cfg["type"] == "Student":
                # Wrap in a temporary sparse model to measure FR
                try:
                    sparse_tmp = SparseStudentSNN1d(cfg["c1"], cfg["c2"],
                                                     kernel_size=cfg["kernel_size"], T=cfg["T"])
                    state = torch.load(cfg["pt"], map_location="cpu", weights_only=True)
                    load_state_filtered(sparse_tmp, state)
                    sparse_tmp.to(dev).eval()
                    loader_for_fr = loader_t if cfg["T"] != 20 else val_loader
                    fr1, fr2 = measure_firing_rates(sparse_tmp, loader_for_fr, dev)
                    del sparse_tmp
                except Exception:
                    fr1, fr2 = 0.5, 0.5  # fallback estimate

            # Default firing rates for energy estimation
            if cfg["type"] == "Teacher":
                fr1, fr2 = 1.0, 1.0  # soft spike = continuous = treat as 100%
            elif cfg["type"] in ("CNN", "QCNN"):
                fr1, fr2 = 1.0, 1.0

            # --- Energy ---
            e_actual, e_potential, total_macs, total_acs = estimate_energy(
                cfg["type"], cfg["c1"], cfg["c2"], cfg["kernel_size"],
                seq_len_t, cfg["T"],
                firing_rate_1=fr1, firing_rate_2=fr2,
                is_int8=(cfg["type"] == "QCNN"),
            )

            # --- Record ---
            result = {
                "type": cfg["type"],
                "variant": cfg["variant"],
                "kernel_size": cfg["kernel_size"],
                "T": cfg["T"],
                "target_fr": cfg.get("target_fr"),
                "params": count_params(model),
                "size_kb": round(model_size_kb(model), 1),
                "val_acc": round(acc, 4),
                "latency_mean_ms": round(lat["mean_ms"], 3),
                "latency_p95_ms": round(lat["p95_ms"], 3),
                "fr1": round(fr1, 4),
                "fr2": round(fr2, 4),
                "total_macs": total_macs,
                "total_acs": total_acs,
                "energy_actual_nJ": round(e_actual / 1000, 2),      # pJ → nJ
                "energy_potential_nJ": round(e_potential / 1000, 2),
            }
            results.append(result)

            print(f"    Acc={acc:.4f}  Lat={lat['mean_ms']:.2f}ms  "
                  f"E_actual={e_actual/1000:.1f}nJ  E_potential={e_potential/1000:.1f}nJ  "
                  f"FR=({fr1:.3f},{fr2:.3f})")

            del model
        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback; traceback.print_exc()

    # --- Save JSON ---
    json_path = os.path.join(RESULT_DIR, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # --- Print Summary Table ---
    print_summary_table(results)

    print(f"\nResults saved to {json_path}")


def print_summary_table(results):
    """Print formatted comparison table."""

    # Find CNN medium as baseline for relative comparison
    cnn_baseline = None
    for r in results:
        if r["type"] == "CNN" and r["variant"] == "medium":
            cnn_baseline = r
            break
    if cnn_baseline is None:
        for r in results:
            if r["type"] == "CNN":
                cnn_baseline = r
                break

    print("\n")
    print("=" * 140)
    print("  BENCHMARK RESULTS")
    print("=" * 140)
    header = (f"{'Type':<10} {'Variant':<10} {'k':>3} {'T':>3} {'FR%':>5} "
              f"{'Params':>8} {'KB':>7} {'Val Acc':>8} "
              f"{'Lat(ms)':>9} {'Lat_p95':>9} "
              f"{'E_actual':>10} {'E_poten':>10} "
              f"{'vs CNN':>8}")
    print(header)
    print("-" * 140)

    # Group by type
    type_order = ["CNN", "QCNN", "Teacher", "Student", "Sparse"]
    grouped = {t: [] for t in type_order}
    for r in results:
        if r["type"] in grouped:
            grouped[r["type"]].append(r)

    for mtype in type_order:
        group = grouped[mtype]
        if not group:
            continue

        # Sort within group
        group.sort(key=lambda r: (r["variant"], r["T"], r.get("target_fr") or 0))

        for r in group:
            fr_str = f"{r['target_fr']*100:.0f}" if r.get("target_fr") else "-"

            # Relative energy vs CNN baseline
            if cnn_baseline:
                rel = r["energy_actual_nJ"] / cnn_baseline["energy_actual_nJ"]
                rel_str = f"{rel:.2f}x"
            else:
                rel_str = "-"

            line = (f"{r['type']:<10} {r['variant']:<10} {r['kernel_size']:>3} "
                    f"{r['T']:>3} {fr_str:>5} "
                    f"{r['params']:>8,} {r['size_kb']:>7.1f} {r['val_acc']:>8.4f} "
                    f"{r['latency_mean_ms']:>9.2f} {r['latency_p95_ms']:>9.2f} "
                    f"{r['energy_actual_nJ']:>9.1f}nJ {r['energy_potential_nJ']:>9.1f}nJ "
                    f"{rel_str:>8}")
            print(line)

        print()  # blank line between types

    # --- Energy comparison summary ---
    if cnn_baseline:
        print("-" * 140)
        print("  ENERGY COMPARISON vs CNN medium baseline")
        print("-" * 140)
        print(f"  {'Model':<45} {'Actual (now)':>15} {'Potential (AC)':>15} {'Actual vs CNN':>15} {'Potential vs CNN':>15}")
        print(f"  {'':45} {'(nJ)':>15} {'(nJ)':>15} {'(ratio)':>15} {'(ratio)':>15}")
        print("  " + "-" * 105)

        baseline_e = cnn_baseline["energy_actual_nJ"]
        for r in results:
            tag = f"{r['type']} {r['variant']} k={r['kernel_size']} T={r['T']}"
            if r.get("target_fr"):
                tag += f" fr={r['target_fr']*100:.0f}%"

            ratio_actual = r["energy_actual_nJ"] / baseline_e if baseline_e > 0 else 0
            ratio_potential = r["energy_potential_nJ"] / baseline_e if baseline_e > 0 else 0

            marker = ""
            if ratio_potential < 0.5:
                marker = " <<< ENERGY WIN"
            elif ratio_actual > 2.0:
                marker = " (worse)"

            print(f"  {tag:<45} {r['energy_actual_nJ']:>13.1f}nJ {r['energy_potential_nJ']:>13.1f}nJ "
                  f"{ratio_actual:>14.2f}x {ratio_potential:>14.2f}x{marker}")


if __name__ == "__main__":
    main()
