"""
QSparse T2 FR05 with Early Exit — channel variant
T=2에서 timestep 1 후 confidence가 높으면 즉시 반환, 아니면 timestep 2까지 실행
"""
import copy
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.mobile_optimizer import optimize_for_mobile

from dataset import FileLevelSpikeDataset, rate_code_zscore_sigmoid, pad_collate

DATA_DIR = "./data/2_ffilled_data"
MODEL_DIR = "./models_channel_variant"
RESULT_DIR = "./result"
KERNEL_SIZE = 9
SEED = 42
BATCH_SIZE = 4

VARIANT_CHANNELS = {
    "smallest": (16, 32),
    "small":    (24, 48),
    "medium":   (32, 64),
    "large":    (40, 80),
    "largest":  (48, 96),
}


# ========================= LIF Node =========================
class QuantFriendlyLIFNode(nn.Module):
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


# ========================= QSparse with Early Exit =========================
class EarlyExitQSparse(nn.Module):
    """QSparse T=2 with early exit after first timestep if confidence > threshold."""

    def __init__(self, c1: int, c2: int, kernel_size: int,
                 thresh1: float, thresh2: float, gain: float,
                 confidence_threshold: float = 0.9,
                 num_classes: int = 4):
        super().__init__()
        self.gain = gain
        self.confidence_threshold = confidence_threshold
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Timestep 1 ---
        xt = x[:, :, :, 0]
        xt = self.quant1(xt)
        xt = self.conv1(xt)
        xt = self.dequant1(xt)
        spk1 = self.lif1(xt * self.gain)
        spk1 = self.quant2(spk1)
        spk2_pre = self.conv2(spk1)
        spk2_pre = self.dequant2(spk2_pre)
        spk2 = self.lif2(spk2_pre * self.gain)
        acc = spk2

        # Early exit check
        out_early = self.pool(acc).squeeze(-1)
        logits_early = self.fc(out_early)
        confidence = torch.softmax(logits_early, dim=1).max(dim=1)[0]

        if confidence.item() >= self.confidence_threshold:
            return logits_early

        # --- Timestep 2 ---
        xt = x[:, :, :, 1]
        xt = self.quant1(xt)
        xt = self.conv1(xt)
        xt = self.dequant1(xt)
        spk1 = self.lif1(xt * self.gain)
        spk1 = self.quant2(spk1)
        spk2_pre = self.conv2(spk1)
        spk2_pre = self.dequant2(spk2_pre)
        spk2 = self.lif2(spk2_pre * self.gain)
        acc = acc + spk2

        out = acc / 2.0
        out = self.pool(out).squeeze(-1)
        return self.fc(out)

    def reset(self) -> None:
        self.lif1.reset()
        self.lif2.reset()


# ========================= Non-quantized version for calibration =========================
class EarlyExitQSparseCalib(nn.Module):
    """Same structure but without early exit for calibration (need all paths exercised)."""

    def __init__(self, c1: int, c2: int, kernel_size: int,
                 thresh1: float, thresh2: float, gain: float,
                 num_classes: int = 4):
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acc = torch.zeros(0, device=x.device)
        for t in range(2):
            xt = x[:, :, :, t]
            xt = self.quant1(xt)
            xt = self.conv1(xt)
            xt = self.dequant1(xt)
            spk1 = self.lif1(xt * self.gain)
            spk1 = self.quant2(spk1)
            spk2_pre = self.conv2(spk1)
            spk2_pre = self.dequant2(spk2_pre)
            spk2 = self.lif2(spk2_pre * self.gain)
            if acc.numel() == 0:
                acc = spk2
            else:
                acc = acc + spk2
        out = acc / 2.0
        out = self.pool(out).squeeze(-1)
        return self.fc(out)

    def reset(self) -> None:
        self.lif1.reset()
        self.lif2.reset()


def load_sparse_weights(variant_name):
    pt_path = os.path.join(MODEL_DIR, f"sparse_{variant_name}_T2_fr05.pt")
    if not os.path.exists(pt_path):
        return None
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    thresh1 = state.get("lif1.threshold", torch.tensor(0.5))
    if isinstance(thresh1, torch.Tensor): thresh1 = thresh1.item()
    thresh2 = state.get("lif2.threshold", torch.tensor(0.5))
    if isinstance(thresh2, torch.Tensor): thresh2 = thresh2.item()
    gain = state.get("gain", torch.tensor(3.0))
    if isinstance(gain, torch.Tensor): gain = gain.item()
    return state, thresh1, thresh2, gain


def load_weights_into(model, state):
    model_sd = model.state_dict()
    for k, v in state.items():
        if k.endswith(".v") or k == "gain" or "threshold" in k:
            continue
        if k in model_sd:
            model_sd[k] = v
    model.load_state_dict(model_sd, strict=True)


def evaluate_early_exit(model, val_loader):
    """Evaluate with early exit, also count early exits."""
    model.eval()
    correct, total, early_exits = 0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            for i in range(x.size(0)):
                xi = x[i:i+1]
                yi = y[i:i+1]
                model.reset()

                # Timestep 1
                xt = xi[:, :, :, 0]
                xt = model.quant1(xt)
                xt = model.conv1(xt)
                xt = model.dequant1(xt)
                spk1 = model.lif1(xt * model.gain)
                spk1 = model.quant2(spk1)
                spk2_pre = model.conv2(spk1)
                spk2_pre = model.dequant2(spk2_pre)
                spk2 = model.lif2(spk2_pre * model.gain)
                acc = spk2

                out_early = model.pool(acc).squeeze(-1)
                logits_early = model.fc(out_early)
                confidence = torch.softmax(logits_early, dim=1).max(dim=1)[0]

                if confidence.item() >= model.confidence_threshold:
                    pred = logits_early.argmax(1)
                    early_exits += 1
                else:
                    # Timestep 2
                    xt = xi[:, :, :, 1]
                    xt = model.quant1(xt)
                    xt = model.conv1(xt)
                    xt = model.dequant1(xt)
                    spk1 = model.lif1(xt * model.gain)
                    spk1 = model.quant2(spk1)
                    spk2_pre = model.conv2(spk1)
                    spk2_pre = model.dequant2(spk2_pre)
                    spk2 = model.lif2(spk2_pre * model.gain)
                    acc = acc + spk2
                    out = acc / 2.0
                    out = model.pool(out).squeeze(-1)
                    logits = model.fc(out)
                    pred = logits.argmax(1)

                correct += (pred == yi).sum().item()
                total += 1

    return correct / max(total, 1), early_exits, total


def export_ptl(model, filename):
    path = os.path.join(MODEL_DIR, filename)
    try:
        model.eval()
        model.reset()
        scripted = torch.jit.script(model)
        opt = optimize_for_mobile(scripted)
        opt._save_for_lite_interpreter(path)
        return os.path.getsize(path) / 1024
    except Exception as e:
        print(f"  [WARN] PTL export failed: {e}")
        return None


def main():
    torch.backends.quantized.engine = "qnnpack"
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    start_time = time.time()

    dataset = FileLevelSpikeDataset(DATA_DIR, rate_code_zscore_sigmoid, T=2)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(SEED))
    cal_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=pad_collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=pad_collate, num_workers=0)

    # Try multiple confidence thresholds to find optimal
    CONF_THRESHOLDS = [0.8, 0.85, 0.9, 0.95]

    all_results = []

    for vname, (c1, c2) in VARIANT_CHANNELS.items():
        print(f"\n{'='*60}")
        print(f"  {vname} (c1={c1}, c2={c2})")
        print(f"{'='*60}")

        loaded = load_sparse_weights(vname)
        if loaded is None:
            print(f"  [SKIP] sparse_{vname}_T2_fr05.pt not found")
            continue

        state, thresh1, thresh2, gain = loaded
        print(f"  thresh=({thresh1:.4f}, {thresh2:.4f}), gain={gain:.4f}")

        # 1. Calibrate using non-early-exit version
        print(f"  Calibrating...")
        calib_model = EarlyExitQSparseCalib(c1, c2, KERNEL_SIZE, thresh1, thresh2, gain)
        load_weights_into(calib_model, state)
        calib_model.eval()

        calib_q = copy.deepcopy(calib_model)
        calib_q.eval()
        calib_q.qconfig = torch.quantization.get_default_qconfig("qnnpack")
        calib_q.lif1.qconfig = None
        calib_q.lif2.qconfig = None
        calib_q.pool.qconfig = None
        calib_q.fc.qconfig = None
        torch.quantization.prepare(calib_q, inplace=True)
        with torch.no_grad():
            for x, _ in cal_loader:
                calib_q.reset()
                calib_q(x)
        torch.quantization.convert(calib_q, inplace=True)

        # Extract quantized weights
        q_state = calib_q.state_dict()

        # 2. Test different confidence thresholds
        best_conf = 0.9
        best_acc = 0.0
        best_early_pct = 0.0

        print(f"\n  Testing confidence thresholds:")
        for conf_thresh in CONF_THRESHOLDS:
            ee_model = EarlyExitQSparse(c1, c2, KERNEL_SIZE, thresh1, thresh2, gain,
                                        confidence_threshold=conf_thresh)
            load_weights_into(ee_model, state)
            ee_model.eval()

            # Quantize
            ee_q = copy.deepcopy(ee_model)
            ee_q.eval()
            ee_q.qconfig = torch.quantization.get_default_qconfig("qnnpack")
            ee_q.lif1.qconfig = None
            ee_q.lif2.qconfig = None
            ee_q.pool.qconfig = None
            ee_q.fc.qconfig = None
            torch.quantization.prepare(ee_q, inplace=True)
            with torch.no_grad():
                for x, _ in cal_loader:
                    ee_q.reset()
                    # Run full forward for calibration (ignore early exit)
                    acc_t = torch.zeros(0, device=x.device)
                    for t in range(2):
                        xt = x[:, :, :, t]
                        xt = ee_q.quant1(xt)
                        xt = ee_q.conv1(xt)
                        xt = ee_q.dequant1(xt)
                        s1 = ee_q.lif1(xt * ee_q.gain)
                        s1 = ee_q.quant2(s1)
                        s2p = ee_q.conv2(s1)
                        s2p = ee_q.dequant2(s2p)
                        s2 = ee_q.lif2(s2p * ee_q.gain)
                        if acc_t.numel() == 0:
                            acc_t = s2
                        else:
                            acc_t = acc_t + s2
            torch.quantization.convert(ee_q, inplace=True)

            acc, early_count, total_count = evaluate_early_exit(ee_q, val_loader)
            early_pct = early_count / total_count * 100

            print(f"    conf={conf_thresh:.2f}: acc={acc:.4f}, early_exit={early_pct:.1f}% ({early_count}/{total_count})")

            if acc > best_acc:
                best_acc = acc
                best_conf = conf_thresh
                best_early_pct = early_pct

            del ee_model, ee_q

        # 3. Build final model with best threshold
        print(f"\n  Best: conf={best_conf}, acc={best_acc:.4f}, early_exit={best_early_pct:.1f}%")

        final_model = EarlyExitQSparse(c1, c2, KERNEL_SIZE, thresh1, thresh2, gain,
                                       confidence_threshold=best_conf)
        load_weights_into(final_model, state)
        final_model.eval()

        final_q = copy.deepcopy(final_model)
        final_q.eval()
        final_q.qconfig = torch.quantization.get_default_qconfig("qnnpack")
        final_q.lif1.qconfig = None
        final_q.lif2.qconfig = None
        final_q.pool.qconfig = None
        final_q.fc.qconfig = None
        torch.quantization.prepare(final_q, inplace=True)
        with torch.no_grad():
            for x, _ in cal_loader:
                final_q.reset()
                acc_t = torch.zeros(0, device=x.device)
                for t in range(2):
                    xt = x[:, :, :, t]
                    xt = final_q.quant1(xt)
                    xt = final_q.conv1(xt)
                    xt = final_q.dequant1(xt)
                    s1 = final_q.lif1(xt * final_q.gain)
                    s1 = final_q.quant2(s1)
                    s2p = final_q.conv2(s1)
                    s2p = final_q.dequant2(s2p)
                    s2 = final_q.lif2(s2p * final_q.gain)
                    if acc_t.numel() == 0:
                        acc_t = s2
                    else:
                        acc_t = acc_t + s2
        torch.quantization.convert(final_q, inplace=True)

        # Save .pt
        pt_name = f"ee_qsparse_{vname}_T2_fr05.pt"
        torch.save(final_q.state_dict(), os.path.join(MODEL_DIR, pt_name))

        # Export .ptl
        ptl_name = f"ee_qsparse_{vname}_T2_fr05.ptl"
        ptl_kb = export_ptl(final_q, ptl_name)
        if ptl_kb:
            print(f"  Exported: {ptl_name} ({ptl_kb:.1f} KB)")

        result = {
            "variant": vname,
            "channels": [c1, c2],
            "T": 2,
            "target_rate": 0.05,
            "confidence_threshold": best_conf,
            "int8_val_acc": round(best_acc, 4),
            "early_exit_pct": round(best_early_pct, 1),
            "ptl_size_kb": round(ptl_kb, 1) if ptl_kb else None,
            "thresh1": round(thresh1, 4),
            "thresh2": round(thresh2, 4),
            "gain": round(gain, 4),
        }
        all_results.append(result)

        del final_model, final_q

    # Save results
    result_path = os.path.join(RESULT_DIR, "early_exit_qsparse_T2_results.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"  DONE! Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*80}")
    for r in all_results:
        ptl = f"{r['ptl_size_kb']:.1f}KB" if r['ptl_size_kb'] else "N/A"
        print(f"  {r['variant']:<10} acc={r['int8_val_acc']:.4f}  "
              f"early_exit={r['early_exit_pct']:.1f}%  conf={r['confidence_threshold']}  PTL={ptl}")


if __name__ == "__main__":
    main()
