"""
Phase 1+2: Channel-Variant SNN Teacher + Student (KD)

k=9 고정, 채널 가변: CNN과 동일한 variant 구조
  smallest: (16,32), small: (24,48), medium: (32,64), large: (40,80), largest: (48,96)

T = [3, 5, 10, 15, 20]
산출물: 25 teacher .pt + 25 teacher .ptl + 25 student .pt + 25 student_kd .ptl = 100 files
"""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset import FileLevelSpikeDataset, rate_code_zscore_sigmoid, pad_collate
from lif_module import LIFNode
from model import HardLIFNode

DATA_DIR = "./data/2_ffilled_data"
NUM_CLASSES = 4
EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 4
TEMP = 2.0
ALPHA_CE = 0.9
GRAD_CLIP = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = min(16, os.cpu_count() or 2)
PIN_MEMORY = DEVICE.type == "cuda"
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

MODEL_DIR = "./models_channel_variant"
RESULT_DIR = "./result"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

KERNEL_SIZE = 9
VARIANT_CHANNELS = {
    "smallest": (16, 32),
    "small":    (24, 48),
    "medium":   (32, 64),
    "large":    (40, 80),
    "largest":  (48, 96),
}
T_VALUES = [3, 5, 10, 15, 20]


class TeacherSNN1d(nn.Module):
    def __init__(self, c1, c2, kernel_size=9, num_classes=NUM_CLASSES, T=20):
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
    def __init__(self, c1, c2, kernel_size=9, num_classes=NUM_CLASSES, T=20,
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


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_size_kb(model):
    size = sum(p.nelement() * p.element_size() for p in model.parameters())
    size += sum(b.nelement() * b.element_size() for b in model.buffers())
    return size / 1024.0

def kd_loss(student_logits, teacher_logits, T=2.0):
    log_p = F.log_softmax(student_logits / T, dim=1)
    q = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_p, q, reduction="batchmean") * (T * T)


def export_ptl(model, T, dummy_n=100, filename="model.ptl"):
    """Export model to TorchScript Lite. Skips on error."""
    try:
        model.eval()
        model.reset()
        dummy = torch.randn(1, 12, dummy_n, T)
        traced = torch.jit.trace(model, dummy, check_trace=False)
        from torch.utils.mobile_optimizer import optimize_for_mobile
        opt = optimize_for_mobile(traced)
        opt._save_for_lite_interpreter(os.path.join(MODEL_DIR, filename))
        print(f"  Exported: {filename}")
    except Exception as e:
        print(f"  Export failed for {filename}: {e}")


def main():
    print(f"Device: {DEVICE}")
    print(f"Phase 1+2: Channel-Variant Teacher + Student (KD)")
    print(f"T values: {T_VALUES}")
    print(f"Kernel size: {KERNEL_SIZE} (fixed)")
    print(f"Variants: {list(VARIANT_CHANNELS.keys())}")
    print()

    all_results = []

    for T in T_VALUES:
        print(f"\n{'='*60}")
        print(f"  T = {T}")
        print(f"{'='*60}")

        dataset = FileLevelSpikeDataset(DATA_DIR, rate_code_zscore_sigmoid, T=T)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                         generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=pad_collate, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=pad_collate, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        for vname, (c1, c2) in VARIANT_CHANNELS.items():
            # --- Teacher ---
            print(f"\n--- Teacher {vname} (c1={c1}, c2={c2}, k={KERNEL_SIZE}, T={T}) ---")
            teacher = TeacherSNN1d(c1, c2, kernel_size=KERNEL_SIZE, T=T).to(DEVICE)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(teacher.parameters(), lr=LR)
            best_score, best_state = -1.0, None

            for epoch in range(EPOCHS):
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
                vc, vt = 0, 0
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(DEVICE), y.to(DEVICE)
                        teacher.reset(); out = teacher(x)
                        vc += (out.argmax(1) == y).sum().item(); vt += y.size(0)
                val_acc = vc / max(vt, 1)
                score = train_acc + val_acc
                if epoch % 10 == 0 or epoch == EPOCHS - 1:
                    print(f"  Epoch {epoch+1}/{EPOCHS} | train={train_acc:.4f} | val={val_acc:.4f}")
                if score > best_score:
                    best_score = score
                    best_state = {k: v.cpu().clone() for k, v in teacher.state_dict().items()}

            pt_path = os.path.join(MODEL_DIR, f"snn1d_teacher_{vname}_T{T}.pt")
            torch.save(best_state, pt_path)
            teacher.load_state_dict(best_state); teacher.to(DEVICE)

            # Export teacher ptl
            teacher.cpu().eval()
            export_ptl(teacher, T, filename=f"snn_{vname}_T{T}.ptl")
            teacher.to(DEVICE)

            teacher_result = {
                "T": T, "variant": vname, "kernel_size": KERNEL_SIZE, "type": "teacher",
                "channels": [c1, c2], "params": count_params(teacher),
                "size_kb": round(model_size_kb(teacher), 1),
                "train_acc": round((best_score - (best_score % 1)) if best_score > 1 else best_score, 4),
                "val_acc": round(best_score - int(best_score), 4) if best_score > 1 else round(best_score, 4),
                "best_score": round(best_score, 4),
            }
            all_results.append(teacher_result)
            print(f"  Best score: {best_score:.4f}")

            # --- Student KD ---
            print(f"\n--- Student {vname} (c1={c1}, c2={c2}, k={KERNEL_SIZE}, T={T}) ---")
            teacher.eval()
            student = StudentSNN1d(c1, c2, kernel_size=KERNEL_SIZE, T=T).to(DEVICE)
            optimizer = torch.optim.Adam(student.parameters(), lr=LR)
            best_score_s, best_state_s = -1.0, None

            for epoch in range(EPOCHS):
                student.train()
                correct, total = 0, 0
                for x, y in train_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    student.reset(); teacher.reset(); optimizer.zero_grad()
                    s_logits = student(x)
                    with torch.no_grad():
                        teacher.reset()
                        t_logits = teacher(x)
                    loss = ALPHA_CE * F.cross_entropy(s_logits, y) + \
                           (1 - ALPHA_CE) * kd_loss(s_logits, t_logits, T=TEMP)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
                    optimizer.step()
                    correct += (s_logits.argmax(1) == y).sum().item(); total += y.size(0)
                train_acc = correct / total

                student.eval()
                vc, vt = 0, 0
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(DEVICE), y.to(DEVICE)
                        student.reset(); out = student(x)
                        vc += (out.argmax(1) == y).sum().item(); vt += y.size(0)
                val_acc = vc / max(vt, 1)
                score = train_acc + val_acc
                if epoch % 10 == 0 or epoch == EPOCHS - 1:
                    print(f"  Epoch {epoch+1}/{EPOCHS} | train={train_acc:.4f} | val={val_acc:.4f}")
                if score > best_score_s:
                    best_score_s = score
                    best_state_s = {k: v.cpu().clone() for k, v in student.state_dict().items()}

            pt_path = os.path.join(MODEL_DIR, f"snn1d_student_{vname}_T{T}.pt")
            torch.save(best_state_s, pt_path)

            # Export student ptl
            student.load_state_dict(best_state_s)
            student.cpu().eval()
            export_ptl(student, T, filename=f"student_kd_{vname}_T{T}.ptl")

            student_result = {
                "T": T, "variant": vname, "kernel_size": KERNEL_SIZE, "type": "student",
                "channels": [c1, c2], "params": count_params(student),
                "size_kb": round(model_size_kb(student), 1),
                "best_score": round(best_score_s, 4),
            }
            all_results.append(student_result)
            print(f"  Best score: {best_score_s:.4f}")

    # Save results
    result_path = os.path.join(RESULT_DIR, "channel_variant_teacher_student_results.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {result_path}")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  T={r['T']} {r['type']:8s} {r['variant']:8s} (ch={r['channels']}): "
              f"score={r['best_score']:.4f}")


if __name__ == "__main__":
    main()
