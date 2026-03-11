"""
T=1, T=2 전용: Phase 1~4 통합 (Teacher → Student KD → Sparse FR5% → QSparse INT8)
k=9 고정, 채널 가변
"""
import copy
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.mobile_optimizer import optimize_for_mobile

from dataset import FileLevelSpikeDataset, rate_code_zscore_sigmoid, pad_collate
from lif_module import LIFNode
from model import HardLIFNode

# ========================= Config =========================
DATA_DIR = "./data/2_ffilled_data"
MODEL_DIR = "./models_channel_variant"
RESULT_DIR = "./result"
NUM_CLASSES = 4
BATCH_SIZE = 4
SEED = 42
KERNEL_SIZE = 9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = min(16, os.cpu_count() or 2)
PIN_MEMORY = DEVICE.type == "cuda"

VARIANT_CHANNELS = {
    "smallest": (16, 32),
    "small":    (24, 48),
    "medium":   (32, 64),
    "large":    (40, 80),
    "largest":  (48, 96),
}
T_VALUES = [1, 2]
TARGET_RATES = [0.05]

# Phase 1+2
EPOCHS_TEACHER = 50
EPOCHS_STUDENT = 50
LR_TS = 1e-3
TEMP = 2.0
ALPHA_CE = 0.9
GRAD_CLIP = 1.0

# Phase 3
EPOCHS_SPARSE = 40
LR_SPARSE = 5e-4
LAMBDA_SPARSE = 5.0

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

LOG_FILE = "train_channel_T1T2.log"


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


# ========================= Phase 1: Teacher =========================
class TeacherSNN1d(nn.Module):
    def __init__(self, c1, c2, kernel_size=9, num_classes=NUM_CLASSES, T=1):
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


# ========================= Phase 2: Student =========================
class StudentSNN1d(nn.Module):
    def __init__(self, c1, c2, kernel_size=9, num_classes=NUM_CLASSES, T=1,
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


# ========================= Phase 3: Sparse =========================
class _HardSpikeSTE_LearnableThresh(torch.autograd.Function):
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
        grad_mem = grad_output * surrogate
        grad_thresh = -grad_output * surrogate
        grad_thresh = grad_thresh.sum()
        return grad_mem, grad_thresh, None


class LearnableHardLIFNode(nn.Module):
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
            spike = _HardSpikeSTE_LearnableThresh.apply(self.v, self.threshold, self.surrogate_width)
        self.v = torch.where(spike > 0, torch.full_like(self.v, self.reset_value), self.v)
        return spike

    def reset(self):
        self.v = torch.zeros(0, device=self.v.device if isinstance(self.v, torch.Tensor) else None)


class SparseStudentSNN1d(nn.Module):
    def __init__(self, c1, c2, kernel_size=9, num_classes=NUM_CLASSES, T=1,
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


# ========================= Phase 4: QSparse =========================
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


class QuantizableSparseStudent(nn.Module):
    def __init__(self, c1: int, c2: int, kernel_size: int, T: int,
                 thresh1: float, thresh2: float, gain: float, num_classes: int = 4):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acc = torch.zeros(0, device=x.device)
        for t in range(self.T):
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
        out = acc / float(self.T)
        out = self.pool(out).squeeze(-1)
        return self.fc(out)

    def reset(self) -> None:
        self.lif1.reset()
        self.lif2.reset()


# ========================= Export helpers =========================
class ExportableHardLIFNode(nn.Module):
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
    def __init__(self, c1: int, c2: int, kernel_size: int, T: int,
                 thresh1: float, thresh2: float, gain: float, num_classes: int = 4):
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


def export_ptl(model, T, filename):
    try:
        model.eval()
        model.reset()
        dummy = torch.randn(1, 12, 100, T)
        traced = torch.jit.trace(model, dummy, check_trace=False)
        opt = optimize_for_mobile(traced)
        path = os.path.join(MODEL_DIR, filename)
        opt._save_for_lite_interpreter(path)
        return os.path.getsize(path) / 1024
    except Exception as e:
        print(f"  Export failed for {filename}: {e}", flush=True)
        return None


def export_sparse_ptl(model, c1, c2, T, filename):
    try:
        thresh1 = model.lif1.threshold.item()
        thresh2 = model.lif2.threshold.item()
        gain = model.gain.item()
        export_model = ExportableSparseStudent(c1, c2, KERNEL_SIZE, T, thresh1, thresh2, gain).cpu()
        state = model.cpu().state_dict()
        export_state = {k: v for k, v in state.items() if not k.endswith(".v") and k != "gain" and "threshold" not in k}
        export_model.load_state_dict(export_state, strict=False)
        export_model.eval()
        export_model.reset()
        scripted = torch.jit.script(export_model)
        opt = optimize_for_mobile(scripted)
        path = os.path.join(MODEL_DIR, filename)
        opt._save_for_lite_interpreter(path)
        return os.path.getsize(path) / 1024
    except Exception as e:
        print(f"  Export failed for {filename}: {e}", flush=True)
        return None


def export_qsparse_ptl(model, filename):
    try:
        model.eval()
        model.reset()
        scripted = torch.jit.script(model)
        opt = optimize_for_mobile(scripted)
        path = os.path.join(MODEL_DIR, filename)
        opt._save_for_lite_interpreter(path)
        return os.path.getsize(path) / 1024
    except Exception as e:
        print(f"  Export failed for {filename}: {e}", flush=True)
        return None


def kd_loss(student_logits, teacher_logits, T=2.0):
    log_p = F.log_softmax(student_logits / T, dim=1)
    q = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_p, q, reduction="batchmean") * (T * T)


def firing_rate_loss(spikes_list, target_rate):
    loss = 0.0
    for spk in spikes_list:
        loss += (spk.mean() - target_rate) ** 2
    return loss / len(spikes_list)


def evaluate(model, val_loader, device="cpu"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            model.reset()
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


# ========================= Main =========================
def main():
    torch.backends.quantized.engine = "qnnpack"
    sys.stdout = Logger(LOG_FILE)

    start_time = time.time()
    print("=" * 80, flush=True)
    print("  T=1, T=2 Channel-Variant: Phase 1~4 통합", flush=True)
    print("=" * 80, flush=True)
    print(f"  Device: {DEVICE}", flush=True)
    print(f"  T values: {T_VALUES}", flush=True)
    print(f"  FR targets: {TARGET_RATES}", flush=True)
    print(flush=True)

    all_ts_results = []
    all_sparse_results = []
    all_qsparse_results = []

    for T in T_VALUES:
        print(f"\n{'='*70}", flush=True)
        print(f"  T = {T}", flush=True)
        print(f"{'='*70}", flush=True)

        dataset = FileLevelSpikeDataset(DATA_DIR, rate_code_zscore_sigmoid, T=T)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                         generator=torch.Generator().manual_seed(SEED))
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=pad_collate, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=pad_collate, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        cal_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=pad_collate, num_workers=0)

        for vname, (c1, c2) in VARIANT_CHANNELS.items():
            print(f"\n--- {vname} (c1={c1}, c2={c2}, k={KERNEL_SIZE}, T={T}) ---", flush=True)

            # ========== Phase 1: Teacher ==========
            print(f"\n  [Phase 1] Teacher training...", flush=True)
            teacher = TeacherSNN1d(c1, c2, kernel_size=KERNEL_SIZE, T=T).to(DEVICE)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(teacher.parameters(), lr=LR_TS)
            best_score, best_state = -1.0, None

            for epoch in range(EPOCHS_TEACHER):
                teacher.train()
                correct, total = 0, 0
                for x, y in train_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    teacher.reset(); optimizer.zero_grad()
                    out = teacher(x)
                    loss = criterion(out, y); loss.backward(); optimizer.step()
                    correct += (out.argmax(1) == y).sum().item(); total += y.size(0)
                train_acc = correct / total
                teacher.eval()
                val_acc = evaluate(teacher, val_loader, DEVICE)
                score = train_acc + val_acc
                if epoch % 10 == 0 or epoch == EPOCHS_TEACHER - 1:
                    print(f"    Epoch {epoch+1}/{EPOCHS_TEACHER} | train={train_acc:.4f} | val={val_acc:.4f}", flush=True)
                if score > best_score:
                    best_score = score
                    best_state = {k: v.cpu().clone() for k, v in teacher.state_dict().items()}

            torch.save(best_state, os.path.join(MODEL_DIR, f"snn1d_teacher_{vname}_T{T}.pt"))
            teacher.load_state_dict(best_state); teacher.cpu().eval()
            export_ptl(teacher, T, f"snn_{vname}_T{T}.ptl")
            teacher.to(DEVICE)

            t_val_acc = best_score - int(best_score) if best_score > 1 else best_score
            all_ts_results.append({
                "T": T, "variant": vname, "type": "teacher",
                "channels": [c1, c2], "val_acc": round(t_val_acc, 4), "best_score": round(best_score, 4),
            })
            print(f"    Teacher best: score={best_score:.4f}, val_acc~{t_val_acc:.4f}", flush=True)

            # ========== Phase 2: Student KD ==========
            print(f"\n  [Phase 2] Student KD training...", flush=True)
            teacher.eval()
            student = StudentSNN1d(c1, c2, kernel_size=KERNEL_SIZE, T=T).to(DEVICE)
            optimizer = torch.optim.Adam(student.parameters(), lr=LR_TS)
            best_score_s, best_state_s = -1.0, None

            for epoch in range(EPOCHS_STUDENT):
                student.train()
                correct, total = 0, 0
                for x, y in train_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    student.reset(); teacher.reset(); optimizer.zero_grad()
                    s_logits = student(x)
                    with torch.no_grad():
                        teacher.reset(); t_logits = teacher(x)
                    loss = ALPHA_CE * F.cross_entropy(s_logits, y) + \
                           (1 - ALPHA_CE) * kd_loss(s_logits, t_logits, T=TEMP)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
                    optimizer.step()
                    correct += (s_logits.argmax(1) == y).sum().item(); total += y.size(0)
                train_acc = correct / total
                student.eval()
                val_acc = evaluate(student, val_loader, DEVICE)
                score = train_acc + val_acc
                if epoch % 10 == 0 or epoch == EPOCHS_STUDENT - 1:
                    print(f"    Epoch {epoch+1}/{EPOCHS_STUDENT} | train={train_acc:.4f} | val={val_acc:.4f}", flush=True)
                if score > best_score_s:
                    best_score_s = score
                    best_state_s = {k: v.cpu().clone() for k, v in student.state_dict().items()}

            torch.save(best_state_s, os.path.join(MODEL_DIR, f"snn1d_student_{vname}_T{T}.pt"))
            student.load_state_dict(best_state_s); student.cpu().eval()
            export_ptl(student, T, f"student_kd_{vname}_T{T}.ptl")

            all_ts_results.append({
                "T": T, "variant": vname, "type": "student",
                "channels": [c1, c2], "best_score": round(best_score_s, 4),
            })
            print(f"    Student best: score={best_score_s:.4f}", flush=True)

            # ========== Phase 3: Sparse FR5% ==========
            print(f"\n  [Phase 3] Sparse training (FR=5%)...", flush=True)
            target_rate = 0.05
            sparse_model = SparseStudentSNN1d(c1, c2, kernel_size=KERNEL_SIZE, T=T).to(DEVICE)

            # Load student weights
            filtered = {k: v for k, v in best_state_s.items() if not k.endswith(".v")}
            if "gain" in filtered and not isinstance(filtered["gain"], torch.Tensor):
                filtered["gain"] = torch.tensor(float(filtered["gain"]))
            sparse_model.load_state_dict(filtered, strict=False)

            thresh_params = [sparse_model.lif1.threshold, sparse_model.lif2.threshold, sparse_model.gain]
            thresh_ids = {id(p) for p in thresh_params}
            other_params = [p for p in sparse_model.parameters() if id(p) not in thresh_ids]
            optimizer = torch.optim.Adam([
                {"params": other_params, "lr": LR_SPARSE},
                {"params": thresh_params, "lr": LR_SPARSE * 10},
            ])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_SPARSE)

            best_score_sp, best_state_sp = -1.0, None
            for epoch in range(EPOCHS_SPARSE):
                progress = min(1.0, (epoch + 1) / 10.0)
                current_lambda = LAMBDA_SPARSE * progress
                sparse_model.train()
                correct, total = 0, 0
                for x, y in train_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    sparse_model.reset(); optimizer.zero_grad()
                    logits, spk1_list, spk2_list = sparse_model(x, return_spikes=True)
                    loss_ce = F.cross_entropy(logits, y)
                    loss_fr = firing_rate_loss(spk1_list + spk2_list, target_rate)
                    thresh_penalty = sum(l.threshold ** 2 for l in [sparse_model.lif1, sparse_model.lif2] if l.threshold < 0)
                    gain_penalty = F.relu(sparse_model.gain - 3.0) ** 2
                    loss = loss_ce + current_lambda * loss_fr + 10.0 * thresh_penalty + 0.5 * gain_penalty
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(sparse_model.parameters(), GRAD_CLIP)
                    optimizer.step()
                    with torch.no_grad():
                        sparse_model.lif1.threshold.clamp_(min=0.01)
                        sparse_model.lif2.threshold.clamp_(min=0.01)
                    correct += (logits.argmax(1) == y).sum().item(); total += y.size(0)
                scheduler.step()
                train_acc = correct / total
                sparse_model.eval()
                val_acc = evaluate(sparse_model, val_loader, DEVICE)
                score = train_acc + val_acc
                if epoch % 10 == 0 or epoch == EPOCHS_SPARSE - 1:
                    print(f"    Epoch {epoch+1}/{EPOCHS_SPARSE} | train={train_acc:.4f} | val={val_acc:.4f} | "
                          f"thresh=({sparse_model.lif1.threshold.item():.3f},{sparse_model.lif2.threshold.item():.3f}) "
                          f"gain={sparse_model.gain.item():.2f}", flush=True)
                if score > best_score_sp:
                    best_score_sp = score
                    best_state_sp = {k: v.cpu().clone() for k, v in sparse_model.state_dict().items()}

            fr_int = int(target_rate * 100)
            torch.save(best_state_sp, os.path.join(MODEL_DIR, f"sparse_{vname}_T{T}_fr{fr_int:02d}.pt"))
            sparse_model.load_state_dict(best_state_sp); sparse_model.to(DEVICE)

            # Measure final FR
            sparse_model.eval()
            fr1_sum, fr2_sum, cnt = 0.0, 0.0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(DEVICE); sparse_model.reset()
                    _, s1, s2 = sparse_model(x, return_spikes=True)
                    fr1_sum += torch.stack(s1).mean().item()
                    fr2_sum += torch.stack(s2).mean().item()
                    cnt += 1
            fr1_f, fr2_f = fr1_sum/cnt, fr2_sum/cnt

            export_sparse_ptl(sparse_model, c1, c2, T, f"sparse_{vname}_T{T}_fr{fr_int:02d}.ptl")
            sparse_model.to(DEVICE)

            all_sparse_results.append({
                "T": T, "variant": vname, "channels": [c1, c2], "target_rate": target_rate,
                "best_score": round(best_score_sp, 4),
                "fr1_final": round(fr1_f, 4), "fr2_final": round(fr2_f, 4),
                "final_thresh1": round(sparse_model.lif1.threshold.item(), 4),
                "final_thresh2": round(sparse_model.lif2.threshold.item(), 4),
                "final_gain": round(sparse_model.gain.item(), 4),
            })
            print(f"    Sparse best: score={best_score_sp:.4f}, FR1={fr1_f:.3f}, FR2={fr2_f:.3f}", flush=True)

            # ========== Phase 4: QSparse INT8 ==========
            print(f"\n  [Phase 4] QSparse INT8 quantization...", flush=True)
            state_sp = best_state_sp
            thresh1 = state_sp.get("lif1.threshold", torch.tensor(0.5))
            if isinstance(thresh1, torch.Tensor): thresh1 = thresh1.item()
            thresh2 = state_sp.get("lif2.threshold", torch.tensor(0.5))
            if isinstance(thresh2, torch.Tensor): thresh2 = thresh2.item()
            gain_val = state_sp.get("gain", torch.tensor(3.0))
            if isinstance(gain_val, torch.Tensor): gain_val = gain_val.item()

            q_model = QuantizableSparseStudent(c1, c2, KERNEL_SIZE, T, thresh1, thresh2, gain_val)
            model_sd = q_model.state_dict()
            for k, v in state_sp.items():
                if k.endswith(".v") or k == "gain" or "threshold" in k:
                    continue
                if k in model_sd:
                    model_sd[k] = v
            q_model.load_state_dict(model_sd, strict=True)
            q_model.eval()

            fp32_acc = evaluate(q_model, val_loader)

            # Quantize
            q_int8 = copy.deepcopy(q_model)
            q_int8.eval()
            q_int8.qconfig = torch.quantization.get_default_qconfig("qnnpack")
            q_int8.lif1.qconfig = None
            q_int8.lif2.qconfig = None
            q_int8.pool.qconfig = None
            q_int8.fc.qconfig = None
            torch.quantization.prepare(q_int8, inplace=True)
            with torch.no_grad():
                for x, _ in cal_loader:
                    q_int8.reset(); q_int8(x)
            torch.quantization.convert(q_int8, inplace=True)

            int8_acc = evaluate(q_int8, val_loader)
            acc_drop = fp32_acc - int8_acc

            qt_name = f"qsparse_{vname}_T{T}_fr{fr_int:02d}.pt"
            torch.save(q_int8.state_dict(), os.path.join(MODEL_DIR, qt_name))
            ptl_name = f"qsparse_{vname}_T{T}_fr{fr_int:02d}.ptl"
            ptl_kb = export_qsparse_ptl(q_int8, ptl_name)

            status = "OK" if abs(acc_drop) < 0.05 else "WARN"
            print(f"    FP32={fp32_acc:.4f} → INT8={int8_acc:.4f} (drop={acc_drop:+.4f}) [{status}]", flush=True)
            if ptl_kb:
                print(f"    PTL: {ptl_name} ({ptl_kb:.1f} KB)", flush=True)

            all_qsparse_results.append({
                "T": T, "variant": vname, "channels": [c1, c2], "target_rate": target_rate,
                "type": "qsparse",
                "fp32_val_acc": round(fp32_acc, 4), "int8_val_acc": round(int8_acc, 4),
                "acc_drop": round(acc_drop, 4),
                "ptl_size_kb": round(ptl_kb, 1) if ptl_kb else None,
                "thresh1": round(thresh1, 4), "thresh2": round(thresh2, 4), "gain": round(gain_val, 4),
            })

            del teacher, student, sparse_model, q_model, q_int8
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save all results
    # Append to existing results
    ts_path = os.path.join(RESULT_DIR, "channel_variant_teacher_student_results.json")
    if os.path.exists(ts_path):
        with open(ts_path) as f:
            existing = json.load(f)
        existing.extend(all_ts_results)
        with open(ts_path, "w") as f:
            json.dump(existing, f, indent=2)
    else:
        with open(ts_path, "w") as f:
            json.dump(all_ts_results, f, indent=2)

    sp_path = os.path.join(RESULT_DIR, "channel_variant_sparse_results.json")
    if os.path.exists(sp_path):
        with open(sp_path) as f:
            existing = json.load(f)
        existing.extend(all_sparse_results)
        with open(sp_path, "w") as f:
            json.dump(existing, f, indent=2)
    else:
        with open(sp_path, "w") as f:
            json.dump(all_sparse_results, f, indent=2)

    qs_path = os.path.join(RESULT_DIR, "channel_variant_qsparse_results.json")
    if os.path.exists(qs_path):
        with open(qs_path) as f:
            existing = json.load(f)
        existing.extend(all_qsparse_results)
        with open(qs_path, "w") as f:
            json.dump(existing, f, indent=2)
    else:
        with open(qs_path, "w") as f:
            json.dump(all_qsparse_results, f, indent=2)

    # T1T2 only results
    t1t2_path = os.path.join(RESULT_DIR, "channel_variant_T1T2_results.json")
    with open(t1t2_path, "w") as f:
        json.dump({
            "teacher_student": all_ts_results,
            "sparse": all_sparse_results,
            "qsparse": all_qsparse_results,
        }, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n{'='*80}", flush=True)
    print(f"  DONE! Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"{'='*80}", flush=True)

    # Summary
    print(f"\n  QSparse T1/T2 Results:", flush=True)
    for r in all_qsparse_results:
        ch = f"({r['channels'][0]},{r['channels'][1]})"
        ptl = f"{r['ptl_size_kb']:.1f}KB" if r['ptl_size_kb'] else "N/A"
        print(f"    {r['variant']:<10} T={r['T']} FR={int(r['target_rate']*100)}%  "
              f"INT8={r['int8_val_acc']:.4f}  drop={r['acc_drop']:+.4f}  PTL={ptl}", flush=True)


if __name__ == "__main__":
    main()
