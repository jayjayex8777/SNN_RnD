"""Export T-sweep models to TorchScript Lite (.ptl) for mobile inference.
Replaces HardLIFNode with a trace-friendly version before tracing.
"""
import os
import torch
import torch.nn as nn

from lif_module import LIFNode
from model import HardLIFNode

NUM_CLASSES = 4
C1, C2 = 32, 64
KERNEL_SIZES = [3, 5, 7, 9, 11]
VARIANT_NAMES = {3: "smallest", 5: "small", 7: "medium", 9: "large", 11: "largest"}
T_VALUES = [5, 10, 15]
MODEL_DIR = "./models"


class TraceFriendlyLIF(nn.Module):
    """LIF node that works with torch.jit.trace (no dynamic shape checks)."""
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


class TeacherSNN1d(nn.Module):
    def __init__(self, c1, c2, kernel_size=3, num_classes=NUM_CLASSES, T=20):
        super().__init__()
        self.T = T
        self.conv1 = nn.Conv1d(12, c1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.lif1 = LIFNode(tau=1.0, threshold=0.2)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=kernel_size // 2)
        self.lif2 = LIFNode(tau=0.9, threshold=0.15)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c2, num_classes)

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
    def __init__(self, c1, c2, kernel_size=3, num_classes=NUM_CLASSES, T=20,
                 thresh1=0.02, thresh2=0.02, surrogate_width=5.0, gain=3.0):
        super().__init__()
        self.T = T
        self.gain = gain
        self.conv1 = nn.Conv1d(12, c1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.lif1 = HardLIFNode(tau=1.0, threshold=thresh1, surrogate_width=surrogate_width)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=kernel_size // 2)
        self.lif2 = HardLIFNode(tau=0.9, threshold=thresh2, surrogate_width=surrogate_width)
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


class ExportableStudentSNN1d(nn.Module):
    """Student with TraceFriendlyLIF for export (hard threshold, no STE needed)."""
    def __init__(self, c1, c2, kernel_size=3, num_classes=NUM_CLASSES, T=20, gain=3.0,
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
    # Warm up to initialize v buffers
    with torch.no_grad():
        model(dummy)
    model.reset()
    traced = torch.jit.trace(model, dummy, check_trace=False)
    from torch.utils.mobile_optimizer import optimize_for_mobile
    opt = optimize_for_mobile(traced)
    out_path = os.path.join(MODEL_DIR, filename)
    opt._save_for_lite_interpreter(out_path)
    print(f"  Exported: {filename}")


def main():
    print("Exporting T-sweep models to .ptl")
    exported = 0

    for T in T_VALUES:
        for ks in KERNEL_SIZES:
            vname = VARIANT_NAMES[ks]

            # Teacher
            teacher_pt = os.path.join(MODEL_DIR, f"snn1d_teacher_{vname}_T{T}.pt")
            if os.path.exists(teacher_pt):
                teacher = TeacherSNN1d(C1, C2, kernel_size=ks, T=T)
                teacher.load_state_dict(torch.load(teacher_pt, map_location="cpu"), strict=False)
                try:
                    export_model(teacher, T, f"snn_{vname}_T{T}.ptl")
                    exported += 1
                except Exception as e:
                    print(f"  Teacher export failed {vname} T={T}: {e}")

            # Student - load weights into ExportableStudentSNN1d
            student_pt = os.path.join(MODEL_DIR, f"snn1d_student_{vname}_T{T}.pt")
            if os.path.exists(student_pt):
                # Load original student to get state dict
                student_orig = StudentSNN1d(C1, C2, kernel_size=ks, T=T)
                state = torch.load(student_pt, map_location="cpu")

                # Create exportable version and transfer weights
                student_exp = ExportableStudentSNN1d(C1, C2, kernel_size=ks, T=T)
                # Map weights: conv1, conv2, fc are the same
                # LIF modules have no learnable params, just buffers
                new_state = {}
                for k, v in state.items():
                    # Skip lif buffer 'v' (it's reset anyway)
                    if '.v' in k and 'conv' not in k:
                        continue
                    new_state[k] = v
                student_exp.load_state_dict(new_state, strict=False)

                try:
                    export_model(student_exp, T, f"student_kd_{vname}_T{T}.ptl")
                    exported += 1
                except Exception as e:
                    print(f"  Student export failed {vname} T={T}: {e}")

    print(f"\nDone. Exported {exported} models.")


if __name__ == "__main__":
    main()
