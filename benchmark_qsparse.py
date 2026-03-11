"""
Benchmark: QSparse SNN (INT8 Static Quantized Sparse SNN)
Measures: Validation Accuracy, CPU Latency, Theoretical Energy, Firing Rate

Runs independently from benchmark.py — QSparse models only.
Results saved to result/qsparse_benchmark_results.json

Monitor progress:
    tail -f benchmark_qsparse.log
"""
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import FileLevelSpikeDataset, rate_code_zscore_sigmoid, pad_collate

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

LOG_FILE = "benchmark_qsparse.log"

# Energy constants (45nm process, pJ per operation)
ENERGY_FP32_MAC = 4.6
ENERGY_FP32_AC = 0.9
ENERGY_INT8_MAC = 0.2
ENERGY_INT8_AC = 0.03

C1, C2 = 32, 64
VARIANT_NAMES = {3: "smallest", 5: "small", 7: "medium", 9: "large", 11: "largest"}
KERNEL_SIZES = [3, 5, 7, 9, 11]
T_VALUES = [3, 5, 10, 15]
TARGET_RATES = [0.30, 0.20, 0.10, 0.05]


# ========================= Logger =========================
class Logger:
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


# ========================= LIF Node (FP32) =========================
class QuantFriendlyLIFNode(nn.Module):
    def __init__(self, tau, threshold, reset_value=0.0):
        super().__init__()
        self.tau = tau
        self.threshold = threshold
        self.reset_value = reset_value
        self.v = torch.zeros(0)

    def forward(self, x):
        if self.v.numel() == 0 or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
        self.v = self.v + (x - self.v) / self.tau
        spike = (self.v >= self.threshold).float()
        self.v = torch.where(spike > 0, torch.full_like(self.v, self.reset_value), self.v)
        return spike

    def reset(self):
        self.v = torch.zeros(0)


# ========================= Quantizable Sparse Student =========================
class QuantizableSparseStudent(nn.Module):
    def __init__(self, c1, c2, kernel_size, T,
                 thresh1, thresh2, gain, num_classes=4):
        super().__init__()
        self.T = T
        self.gain = gain
        pad = kernel_size // 2

        self.quant1 = torch.quantization.QuantStub()
        self.conv1 = nn.Conv1d(12, c1, kernel_size=kernel_size, padding=pad)
        self.dequant1 = torch.quantization.DeQuantStub()
        self.lif1 = QuantFriendlyLIFNode(tau=1.0, threshold=thresh1)

        self.quant2 = torch.quantization.QuantStub()
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=pad)
        self.dequant2 = torch.quantization.DeQuantStub()
        self.lif2 = QuantFriendlyLIFNode(tau=0.9, threshold=thresh2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c2, num_classes)

    def forward(self, x, return_spikes=False):
        all_spk1, all_spk2 = [], []
        acc = torch.zeros(0, device=x.device)
        for t in range(self.T):
            xt = x[:, :, :, t]

            xt = self.quant1(xt)
            xt = self.conv1(xt)
            xt = self.dequant1(xt)
            spk1 = self.lif1(xt * self.gain)

            spk1_q = self.quant2(spk1)
            spk2_pre = self.conv2(spk1_q)
            spk2_pre = self.dequant2(spk2_pre)
            spk2 = self.lif2(spk2_pre * self.gain)

            if return_spikes:
                all_spk1.append(spk1)
                all_spk2.append(spk2)

            if acc.numel() == 0:
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


# ========================= Weight Loading =========================
def load_sparse_weights(variant_name, kernel_size, T, target_rate):
    """Load sparse model weights and build quantizable model."""
    fr_int = int(target_rate * 100)
    pt_path = os.path.join(MODEL_DIR, f"sparse_{variant_name}_T{T}_fr{fr_int:02d}.pt")

    if not os.path.exists(pt_path):
        return None

    state = torch.load(pt_path, map_location="cpu", weights_only=True)

    thresh1 = state.get("lif1.threshold", torch.tensor(0.5))
    if isinstance(thresh1, torch.Tensor):
        thresh1 = thresh1.item()
    thresh2 = state.get("lif2.threshold", torch.tensor(0.5))
    if isinstance(thresh2, torch.Tensor):
        thresh2 = thresh2.item()
    gain = state.get("gain", torch.tensor(3.0))
    if isinstance(gain, torch.Tensor):
        gain = gain.item()

    model = QuantizableSparseStudent(C1, C2, kernel_size, T, thresh1, thresh2, gain)

    model_sd = model.state_dict()
    for k, v in state.items():
        if k.endswith(".v") or k == "gain" or "threshold" in k:
            continue
        if k in model_sd:
            model_sd[k] = v

    model.load_state_dict(model_sd, strict=True)
    return model


def calibrate_and_quantize(model, calibration_loader):
    """Apply INT8 static quantization."""
    model_q = copy.deepcopy(model)
    model_q.eval()

    model_q.qconfig = torch.quantization.get_default_qconfig("qnnpack")
    model_q.lif1.qconfig = None
    model_q.lif2.qconfig = None
    model_q.pool.qconfig = None
    model_q.fc.qconfig = None

    torch.quantization.prepare(model_q, inplace=True)

    with torch.no_grad():
        for i, (x, _) in enumerate(calibration_loader):
            if i >= 50:  # 50 batches (200 samples) sufficient for observer stats
                break
            model_q.reset()
            model_q(x)

    torch.quantization.convert(model_q, inplace=True)
    return model_q


# ========================= Measurement Functions =========================
def measure_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            model.reset()
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def measure_latency(model, sample_input, warmup=WARMUP, repeats=REPEATS):
    model.eval()
    x = sample_input

    with torch.no_grad():
        for _ in range(warmup):
            model.reset()
            model(x)

    latencies = []
    with torch.no_grad():
        for _ in range(repeats):
            model.reset()
            start = time.perf_counter_ns()
            model(x)
            end = time.perf_counter_ns()
            latencies.append((end - start) / 1e6)

    return {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "std_ms": float(np.std(latencies)),
    }


def measure_firing_rates(model, loader, max_batches=50):
    model.eval()
    fr1_sum, fr2_sum, count = 0.0, 0.0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            model.reset()
            _, spk1_list, spk2_list = model(x, return_spikes=True)
            fr1_sum += torch.stack(spk1_list).mean().item()
            fr2_sum += torch.stack(spk2_list).mean().item()
            count += 1
    if count == 0:
        return 0.0, 0.0
    return fr1_sum / count, fr2_sum / count


def count_conv1d_macs(in_ch, out_ch, kernel_size, seq_len):
    return in_ch * out_ch * kernel_size * seq_len


def count_linear_macs(in_features, out_features):
    return in_features * out_features


def estimate_energy(c1, c2, kernel_size, seq_len, T, fr1, fr2):
    """
    QSparse energy estimation:
    - Conv1d layers: INT8 precision
    - Conv1: input is rate-coded analog → MAC
    - Conv2: input is binary spike from LIF1 → fr1 fraction MAC, rest AC
    - FC: MAC (pooled continuous input)
    - AC uses INT8 AC energy (0.03 pJ) since weights are INT8
    """
    macs_conv1 = count_conv1d_macs(12, c1, kernel_size, seq_len)
    macs_conv2 = count_conv1d_macs(c1, c2, kernel_size, seq_len)
    macs_fc = count_linear_macs(c2, NUM_CLASSES)

    # Actual energy: all ops as INT8 MAC (current code doesn't skip zero spikes)
    total_macs = (macs_conv1 + macs_conv2 + macs_fc) * T
    energy_actual = total_macs * ENERGY_INT8_MAC

    # Potential energy: Conv2 exploits spike sparsity (AC for zero spikes)
    energy_per_step = (
        macs_conv1 * ENERGY_INT8_MAC +                          # Conv1: always MAC
        macs_conv2 * fr1 * ENERGY_INT8_MAC +                    # Conv2: MAC when spike=1
        macs_conv2 * (1 - fr1) * ENERGY_INT8_AC +               # Conv2: AC when spike=0
        macs_fc * ENERGY_INT8_MAC                                # FC: always MAC
    )
    energy_potential = energy_per_step * T

    total_acs = int(macs_conv2 * (1 - fr1) * T)

    return {
        "energy_actual_nJ": round(energy_actual / 1000, 2),
        "energy_potential_nJ": round(energy_potential / 1000, 2),
        "total_macs": total_macs,
        "total_acs": total_acs,
    }


# ========================= Main =========================
def main():
    torch.backends.quantized.engine = "qnnpack"

    os.makedirs(RESULT_DIR, exist_ok=True)
    sys.stdout = Logger(LOG_FILE)

    total_configs = len(T_VALUES) * len(KERNEL_SIZES) * len(TARGET_RATES)

    print("=" * 90, flush=True)
    print("  QSparse Benchmark: INT8 Quantized Sparse SNN", flush=True)
    print("=" * 90, flush=True)
    print(f"  T values:      {T_VALUES}", flush=True)
    print(f"  Kernel sizes:  {KERNEL_SIZES}", flush=True)
    print(f"  Target rates:  {TARGET_RATES}", flush=True)
    print(f"  Total configs: {total_configs}", flush=True)
    print(f"  Latency:       warmup={WARMUP}, repeats={REPEATS}", flush=True)
    print(f"  Device:        {DEVICE}", flush=True)
    print(flush=True)

    all_results = []
    idx = 0
    skipped = 0
    failed = 0
    start_time = time.time()

    for T in T_VALUES:
        print(f"\n{'='*80}", flush=True)
        print(f"  Loading dataset for T={T}...", flush=True)
        print(f"{'='*80}", flush=True)

        dataset = FileLevelSpikeDataset(DATA_DIR, rate_code_zscore_sigmoid, T=T)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(SEED)
        )

        # Calibration loader (batch=4 for calibration)
        cal_loader = DataLoader(train_ds, batch_size=4, shuffle=False,
                                collate_fn=pad_collate, num_workers=0)
        # Validation loader (batch=16 for accuracy/FR, batch=1 for latency)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False,
                                collate_fn=pad_collate, num_workers=0)
        lat_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                                collate_fn=pad_collate, num_workers=0)

        # Get sample input for latency measurement (batch=1)
        sample_x, _ = next(iter(lat_loader))
        seq_len = sample_x.shape[2]

        print(f"  Dataset: {len(train_ds)} train / {len(val_ds)} val", flush=True)
        print(f"  Sample shape: {sample_x.shape} (seq_len={seq_len})", flush=True)

        for ks in KERNEL_SIZES:
            vname = VARIANT_NAMES[ks]

            for target_rate in TARGET_RATES:
                idx += 1
                fr_int = int(target_rate * 100)
                tag = f"qsparse_{vname}_T{T}_fr{fr_int:02d}"

                elapsed = time.time() - start_time
                eta = (elapsed / idx * (total_configs - idx)) if idx > 0 else 0
                print(f"\n[{idx:3d}/{total_configs}] {tag}  "
                      f"(elapsed: {elapsed:.0f}s, ETA: {eta:.0f}s)", flush=True)

                # --- Load & quantize ---
                model_fp32 = load_sparse_weights(vname, ks, T, target_rate)
                if model_fp32 is None:
                    print(f"  [SKIP] Sparse model not found", flush=True)
                    skipped += 1
                    continue

                try:
                    model_q = calibrate_and_quantize(model_fp32, cal_loader)
                except Exception as e:
                    print(f"  [FAIL] Quantization error: {e}", flush=True)
                    failed += 1
                    continue

                # --- Accuracy ---
                int8_acc = measure_accuracy(model_q, val_loader)

                # --- Firing Rates (use FP32 model for spike measurement) ---
                fr1, fr2 = measure_firing_rates(model_fp32, val_loader)

                # --- Latency ---
                lat = measure_latency(model_q, sample_x)

                # --- Energy ---
                energy = estimate_energy(C1, C2, ks, seq_len, T, fr1, fr2)

                # --- Model size ---
                ptl_path = os.path.join(MODEL_DIR, f"qsparse_{vname}_T{T}_fr{fr_int:02d}.ptl")
                ptl_kb = os.path.getsize(ptl_path) / 1024 if os.path.exists(ptl_path) else None

                params = sum(p.numel() for p in model_fp32.parameters())

                result = {
                    "type": "QSparse",
                    "variant": vname,
                    "kernel_size": ks,
                    "T": T,
                    "target_fr": target_rate,
                    "params": params,
                    "int8_val_acc": round(int8_acc, 4),
                    "ptl_size_kb": round(ptl_kb, 1) if ptl_kb else None,
                    "latency_mean_ms": round(lat["mean_ms"], 3),
                    "latency_median_ms": round(lat["median_ms"], 3),
                    "latency_p95_ms": round(lat["p95_ms"], 3),
                    "latency_std_ms": round(lat["std_ms"], 3),
                    "fr1": round(fr1, 4),
                    "fr2": round(fr2, 4),
                    "energy_actual_nJ": energy["energy_actual_nJ"],
                    "energy_potential_nJ": energy["energy_potential_nJ"],
                    "total_macs": energy["total_macs"],
                    "total_acs": energy["total_acs"],
                }
                all_results.append(result)

                print(f"  Acc={int8_acc:.4f}  Lat={lat['mean_ms']:.3f}ms  "
                      f"FR=({fr1:.3f},{fr2:.3f})  "
                      f"E_act={energy['energy_actual_nJ']:.1f}nJ  "
                      f"E_pot={energy['energy_potential_nJ']:.1f}nJ",
                      flush=True)

                del model_fp32, model_q

    # ========================= Save Results =========================
    result_path = os.path.join(RESULT_DIR, "qsparse_benchmark_results.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # ========================= Summary =========================
    total_time = time.time() - start_time
    print(f"\n\n{'='*90}", flush=True)
    print(f"  QSparse BENCHMARK COMPLETE", flush=True)
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)
    print(f"  Processed: {len(all_results)}  Skipped: {skipped}  Failed: {failed}", flush=True)
    print(f"{'='*90}", flush=True)

    # --- Summary table ---
    if all_results:
        print(f"\n{'Variant':<10} {'T':>3} {'FR%':>5} {'Acc':>8} {'Lat(ms)':>9} "
              f"{'Lat_p95':>9} {'FR1':>6} {'FR2':>6} "
              f"{'E_act(nJ)':>10} {'E_pot(nJ)':>10} {'PTL(KB)':>8}",
              flush=True)
        print("-" * 100, flush=True)

        for r in all_results:
            fr_pct = int(r['target_fr'] * 100)
            ptl_str = f"{r['ptl_size_kb']:.1f}" if r['ptl_size_kb'] else "N/A"
            print(f"{r['variant']:<10} {r['T']:>3} {fr_pct:>5} "
                  f"{r['int8_val_acc']:>8.4f} {r['latency_mean_ms']:>9.3f} "
                  f"{r['latency_p95_ms']:>9.3f} {r['fr1']:>6.3f} {r['fr2']:>6.3f} "
                  f"{r['energy_actual_nJ']:>10.1f} {r['energy_potential_nJ']:>10.1f} "
                  f"{ptl_str:>8}",
                  flush=True)

        # --- Top 5 by accuracy ---
        print(f"\n  Top 5 by INT8 Accuracy:", flush=True)
        sorted_acc = sorted(all_results, key=lambda r: r['int8_val_acc'], reverse=True)
        for r in sorted_acc[:5]:
            fr_pct = int(r['target_fr'] * 100)
            print(f"    {r['variant']} T={r['T']} fr={fr_pct}%  "
                  f"Acc={r['int8_val_acc']:.4f}  Lat={r['latency_mean_ms']:.3f}ms  "
                  f"E_pot={r['energy_potential_nJ']:.1f}nJ", flush=True)

        # --- Top 5 by energy efficiency ---
        print(f"\n  Top 5 by Energy Potential (acc >= 95%):", flush=True)
        valid = [r for r in all_results if r['int8_val_acc'] >= 0.95]
        sorted_energy = sorted(valid, key=lambda r: r['energy_potential_nJ'])
        for r in sorted_energy[:5]:
            fr_pct = int(r['target_fr'] * 100)
            print(f"    {r['variant']} T={r['T']} fr={fr_pct}%  "
                  f"Acc={r['int8_val_acc']:.4f}  E_pot={r['energy_potential_nJ']:.1f}nJ  "
                  f"Lat={r['latency_mean_ms']:.3f}ms", flush=True)

        # --- Top 5 by latency ---
        print(f"\n  Top 5 by Latency (acc >= 95%):", flush=True)
        sorted_lat = sorted(valid, key=lambda r: r['latency_mean_ms'])
        for r in sorted_lat[:5]:
            fr_pct = int(r['target_fr'] * 100)
            print(f"    {r['variant']} T={r['T']} fr={fr_pct}%  "
                  f"Acc={r['int8_val_acc']:.4f}  Lat={r['latency_mean_ms']:.3f}ms  "
                  f"E_pot={r['energy_potential_nJ']:.1f}nJ", flush=True)

    print(f"\nResults saved to {result_path}", flush=True)


if __name__ == "__main__":
    main()
