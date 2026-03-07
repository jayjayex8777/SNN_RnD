"""Train only k=11 (largest) and merge with existing results."""
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
T_STEPS = 20
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

MODEL_DIR = "./models"
RESULT_DIR = "./result"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

C1, C2 = 32, 64
KS = 11

VARIANT_NAMES = {3: "smallest", 5: "small", 7: "medium", 9: "large", 11: "largest"}


class TeacherSNN1d(nn.Module):
    def __init__(self, c1, c2, kernel_size=11, num_classes=NUM_CLASSES, T=T_STEPS):
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
    def __init__(self, c1, c2, kernel_size=11, num_classes=NUM_CLASSES, T=T_STEPS,
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


def main():
    print(f"Device: {DEVICE}, Training k=11 (largest)")

    dataset = FileLevelSpikeDataset(DATA_DIR, rate_code_zscore_sigmoid, T=T_STEPS)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=pad_collate, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=pad_collate, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # --- Teacher k=11 ---
    print("\n=== Training Teacher largest (k=11) ===")
    teacher = TeacherSNN1d(C1, C2, kernel_size=KS).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(teacher.parameters(), lr=LR)
    best_score, best_state = -1.0, None
    teacher_pt = os.path.join(MODEL_DIR, "snn1d_teacher_largest.pt")

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
        print(f"[teacher_largest] Epoch {epoch+1}/{EPOCHS} | train={train_acc:.4f} | val={val_acc:.4f}")
        if score > best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in teacher.state_dict().items()}

    torch.save(best_state, teacher_pt)
    teacher_result = {
        "variant": "largest", "kernel_size": KS, "type": "teacher_1d",
        "channels": [C1, C2], "params": count_params(teacher),
        "size_kb": round(model_size_kb(teacher), 1), "best_score": best_score, "pt": teacher_pt,
    }
    print(f"  Saved: {teacher_pt} (params={count_params(teacher)}, {model_size_kb(teacher):.1f}KB)")

    # --- Student k=11 ---
    print("\n=== Training Student largest (k=11) ===")
    teacher.load_state_dict(best_state); teacher.to(DEVICE); teacher.eval()

    student = StudentSNN1d(C1, C2, kernel_size=KS).to(DEVICE)
    optimizer = torch.optim.Adam(student.parameters(), lr=LR)
    best_score, best_state_s = -1.0, None
    student_pt = os.path.join(MODEL_DIR, "snn1d_student_largest.pt")

    for epoch in range(EPOCHS):
        student.train()
        correct, total = 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            student.reset(); teacher.reset(); optimizer.zero_grad()
            s_logits = student(x)
            with torch.no_grad(): t_logits = teacher(x)
            loss = ALPHA_CE * F.cross_entropy(s_logits, y) + (1 - ALPHA_CE) * kd_loss(s_logits, t_logits, T=TEMP)
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
        print(f"[student_largest] Epoch {epoch+1}/{EPOCHS} | train={train_acc:.4f} | val={val_acc:.4f}")
        if score > best_score:
            best_score = score
            best_state_s = {k: v.cpu().clone() for k, v in student.state_dict().items()}

    torch.save(best_state_s, student_pt)
    student_result = {
        "variant": "largest", "kernel_size": KS, "type": "student_kd_1d",
        "channels": [C1, C2], "params": count_params(student),
        "size_kb": round(model_size_kb(student), 1), "best_score": best_score, "pt": student_pt,
    }
    print(f"  Saved: {student_pt} (params={count_params(student)}, {model_size_kb(student):.1f}KB)")

    # --- Merge with existing results ---
    result_path = os.path.join(RESULT_DIR, "kernel_sweep_results.json")
    existing = []
    if os.path.exists(result_path):
        with open(result_path) as f:
            existing = json.load(f)

    # Rename old entries with variant names and add new ones
    for r in existing:
        if "variant" not in r:
            r["variant"] = VARIANT_NAMES.get(r["kernel_size"], f"k{r['kernel_size']}")

    # Remove any old largest entries
    existing = [r for r in existing if r.get("variant") != "largest"]
    existing.append(teacher_result)
    existing.append(student_result)

    # Sort: teachers first (by kernel_size), then students (by kernel_size)
    teachers = sorted([r for r in existing if r["type"] == "teacher_1d"], key=lambda r: r["kernel_size"])
    students = sorted([r for r in existing if r["type"] == "student_kd_1d"], key=lambda r: r["kernel_size"])
    all_results = teachers + students

    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=== Done. Results saved to {result_path} ===")
    for r in all_results:
        print(f"  {r['type']} {r['variant']} (k={r['kernel_size']}): params={r['params']}, "
              f"size={r['size_kb']}KB, score={r['best_score']:.4f}")


if __name__ == "__main__":
    main()
