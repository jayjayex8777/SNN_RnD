"""
Export Teacher & Student .pt → .ptl (TorchScript Lite)

문제: Teacher의 LIFNode는 buffer=None 초기화, Student의 HardLIFNode는
커스텀 autograd(_HardSpikeSTE)를 사용하여 TorchScript가 직렬화 불가.

해결: Exportable 버전에서는
- Teacher: None 체크 제거, 고정 크기 텐서로 대체
- Student: 커스텀 autograd 제거, (v >= threshold).float()로 대체 (추론 시 STE 불필요)
"""
import os
import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

MODEL_DIR = "./models_channel_variant"
NUM_CLASSES = 4
KERNEL_SIZE = 9

VARIANT_CHANNELS = {
    "smallest": (16, 32),
    "small":    (24, 48),
    "medium":   (32, 64),
    "large":    (40, 80),
    "largest":  (48, 96),
}

# T=1,2 are from train_channel_T1T2.py, T=3,5,10,15,20 from train_channel_teacher_student.py
T_VALUES = [1, 2, 3, 5, 10, 15, 20]


# ─── Exportable LIF (Teacher용) ───
class ExportableLIFNode(nn.Module):
    """TorchScript 호환 Soft LIF. buffer를 None 대신 빈 텐서로 초기화."""
    def __init__(self, tau=2.0, threshold=0.5):
        super().__init__()
        self.tau = tau
        self.threshold = threshold
        self.register_buffer('v', torch.zeros(1))

    def forward(self, x):
        if self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
        self.v = self.v + (x - self.v) / self.tau
        scale = 10.0
        sig = torch.sigmoid(scale * (self.v - self.threshold))
        out = scale * sig * (1.0 - sig)
        self.v = torch.where(self.v >= self.threshold,
                             torch.zeros_like(self.v), self.v)
        return out

    def reset(self):
        self.v = torch.zeros(1)


# ─── Exportable Hard LIF (Student용) ───
class ExportableHardLIFNode(nn.Module):
    """TorchScript 호환 Hard LIF. 커스텀 autograd 제거, 추론 전용."""
    def __init__(self, tau=1.0, threshold=0.02):
        super().__init__()
        self.tau = tau
        self.threshold = threshold
        self.register_buffer('v', torch.zeros(1))

    def forward(self, x):
        if self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
        self.v = self.v + (x - self.v) / self.tau
        spike = (self.v >= self.threshold).float()
        self.v = torch.where(self.v >= self.threshold,
                             torch.zeros_like(self.v), self.v)
        return spike

    def reset(self):
        self.v = torch.zeros(1)


# ─── Exportable Teacher ───
class ExportableTeacherSNN1d(nn.Module):
    def __init__(self, c1, c2, kernel_size=9, num_classes=NUM_CLASSES, T=20):
        super().__init__()
        self.T = T
        self.conv1 = nn.Conv1d(12, c1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.lif1 = ExportableLIFNode(tau=1.0, threshold=0.2)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=kernel_size // 2)
        self.lif2 = ExportableLIFNode(tau=0.9, threshold=0.15)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c2, num_classes)

    def forward(self, x):
        self.lif1.reset()
        self.lif2.reset()
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


# ─── Exportable Student ───
class ExportableStudentSNN1d(nn.Module):
    def __init__(self, c1, c2, kernel_size=9, num_classes=NUM_CLASSES, T=20, gain=3.0):
        super().__init__()
        self.T = T
        self.gain = gain
        self.conv1 = nn.Conv1d(12, c1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.lif1 = ExportableHardLIFNode(tau=1.0, threshold=0.02)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=kernel_size // 2)
        self.lif2 = ExportableHardLIFNode(tau=0.9, threshold=0.02)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c2, num_classes)

    def forward(self, x):
        self.lif1.reset()
        self.lif2.reset()
        xt = x[:, :, :, 0]
        xt = self.lif1(self.conv1(xt) * self.gain)
        xt = self.lif2(self.conv2(xt) * self.gain)
        acc = xt.clone()
        for t in range(1, self.T):
            xt = x[:, :, :, t]
            xt = self.lif1(self.conv1(xt) * self.gain)
            xt = self.lif2(self.conv2(xt) * self.gain)
            acc = acc + xt
        out = acc / float(self.T)
        out = self.pool(out).squeeze(-1)
        return self.fc(out)


def export_ptl(model, T, filename):
    """TorchScript Lite export."""
    model.eval()
    dummy = torch.randn(1, 12, 100, T)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy, check_trace=False)
    opt = optimize_for_mobile(traced)
    out_path = os.path.join(MODEL_DIR, filename)
    opt._save_for_lite_interpreter(out_path)
    size_kb = os.path.getsize(out_path) / 1024
    return size_kb


def load_and_export(model_type, vname, c1, c2, T):
    """Load .pt weights → Exportable model → .ptl"""
    pt_name = f"snn1d_{model_type}_{vname}_T{T}.pt"
    pt_path = os.path.join(MODEL_DIR, pt_name)

    if not os.path.exists(pt_path):
        return None

    state = torch.load(pt_path, map_location="cpu", weights_only=True)

    if model_type == "teacher":
        model = ExportableTeacherSNN1d(c1, c2, kernel_size=KERNEL_SIZE, T=T)
        ptl_name = f"snn1d_teacher_{vname}_T{T}.ptl"
    else:
        model = ExportableStudentSNN1d(c1, c2, kernel_size=KERNEL_SIZE, T=T)
        ptl_name = f"snn1d_student_{vname}_T{T}.ptl"

    # Map weights: conv1, conv2, fc are identical keys
    # LIF buffers (v) are skipped (they're runtime state)
    model_state = model.state_dict()
    mapped = {}
    for k, v in state.items():
        # Original keys: conv1.*, conv2.*, fc.*, lif1.v, lif2.v, lif1.threshold_param, etc.
        if k in model_state and model_state[k].shape == v.shape:
            mapped[k] = v
        elif 'lif' in k:
            # Skip LIF buffers/params that don't match
            continue
        else:
            # Try direct mapping
            if k in model_state:
                mapped[k] = v

    model.load_state_dict(mapped, strict=False)

    try:
        size_kb = export_ptl(model, T, ptl_name)
        return ptl_name, size_kb
    except Exception as e:
        print(f"  FAILED: {ptl_name}: {e}")
        return None


def main():
    print("=" * 60)
    print("Export Teacher & Student .pt → .ptl")
    print("=" * 60)

    success = 0
    failed = 0

    for model_type in ["teacher", "student"]:
        print(f"\n--- {model_type.upper()} ---")
        for vname, (c1, c2) in VARIANT_CHANNELS.items():
            for T in T_VALUES:
                result = load_and_export(model_type, vname, c1, c2, T)
                if result is None:
                    pt_name = f"snn1d_{model_type}_{vname}_T{T}.pt"
                    if os.path.exists(os.path.join(MODEL_DIR, pt_name)):
                        print(f"  FAILED: {pt_name}")
                        failed += 1
                    # else: .pt doesn't exist, skip silently
                else:
                    ptl_name, size_kb = result
                    print(f"  OK: {ptl_name} ({size_kb:.1f} KB)")
                    success += 1

    print(f"\n{'=' * 60}")
    print(f"Done: {success} exported, {failed} failed")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
