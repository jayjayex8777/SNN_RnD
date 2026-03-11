"""Export channel-variant Student KD models to TorchScript Lite (.ptl).
Replaces HardLIFNode with TraceFriendlyLIF before tracing.

k=9 고정, 채널 가변: CNN과 동일한 variant 구조
  smallest: (16,32), small: (24,48), medium: (32,64), large: (40,80), largest: (48,96)
T = [3, 5, 10, 15, 20]
"""
import os
import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

NUM_CLASSES = 4
KERNEL_SIZE = 9
MODEL_DIR = "./models_channel_variant"

VARIANT_CHANNELS = {
    "smallest": (16, 32),
    "small":    (24, 48),
    "medium":   (32, 64),
    "large":    (40, 80),
    "largest":  (48, 96),
}
T_VALUES = [3, 5, 10, 15, 20]


class TraceFriendlyLIF(nn.Module):
    """LIF node that works with torch.jit.trace (no custom autograd)."""
    def __init__(self, tau=2.0, threshold=0.5, reset_value=0.0):
        super().__init__()
        self.tau = tau
        self.threshold = threshold
        self.reset_value = reset_value
        self.v = None

    def forward(self, x):
        if self.v is None:
            self.v = torch.zeros_like(x)
        self.v = self.v + (x - self.v) / self.tau
        spike = (self.v >= self.threshold).float()
        self.v = torch.where(spike > 0, torch.full_like(self.v, self.reset_value), self.v)
        return spike

    def reset(self):
        self.v = None


class ExportableStudentSNN1d(nn.Module):
    """Student with TraceFriendlyLIF for export (hard threshold, no STE needed)."""
    def __init__(self, c1, c2, kernel_size=9, num_classes=NUM_CLASSES, T=20, gain=3.0,
                 thresh1=0.02, thresh2=0.02):
        super().__init__()
        self.T = T
        self.gain = gain
        self.conv1 = nn.Conv1d(12, c1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.lif1 = TraceFriendlyLIF(tau=1.0, threshold=thresh1, reset_value=0.0)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=kernel_size // 2)
        self.lif2 = TraceFriendlyLIF(tau=0.9, threshold=thresh2, reset_value=0.0)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c2, num_classes)

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


def export_model(model, T, filename):
    model.eval()
    model.reset()
    dummy = torch.randn(1, 12, 100, T)
    with torch.no_grad():
        model(dummy)
    model.reset()
    traced = torch.jit.trace(model, dummy, check_trace=False)
    opt = optimize_for_mobile(traced)
    out_path = os.path.join(MODEL_DIR, filename)
    opt._save_for_lite_interpreter(out_path)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"  Exported: {filename} ({size_kb:.1f} KB)")


def main():
    print("Exporting channel-variant Student KD models to .ptl")
    print(f"Kernel size: {KERNEL_SIZE} (fixed)")
    print(f"T values: {T_VALUES}")
    print(f"Variants: {list(VARIANT_CHANNELS.keys())}")
    print()

    exported = 0
    failed = 0

    for T in T_VALUES:
        for vname, (c1, c2) in VARIANT_CHANNELS.items():
            student_pt = os.path.join(MODEL_DIR, f"snn1d_student_{vname}_T{T}.pt")
            if not os.path.exists(student_pt):
                print(f"  [SKIP] Not found: {student_pt}")
                continue

            state = torch.load(student_pt, map_location="cpu", weights_only=True)

            student_exp = ExportableStudentSNN1d(c1, c2, kernel_size=KERNEL_SIZE, T=T)

            new_state = {}
            for k, v in state.items():
                if '.v' in k and 'conv' not in k:
                    continue
                new_state[k] = v
            student_exp.load_state_dict(new_state, strict=False)

            try:
                export_model(student_exp, T, f"student_kd_{vname}_T{T}.ptl")
                exported += 1
            except Exception as e:
                print(f"  [FAIL] student_kd_{vname}_T{T}.ptl: {e}")
                failed += 1

    print(f"\nDone. Exported: {exported}, Failed: {failed}")


if __name__ == "__main__":
    main()
