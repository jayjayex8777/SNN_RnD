import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.mobile_optimizer import optimize_for_mobile

from dataset import FileLevelSpikeDataset, rate_code_zscore_sigmoid, pad_collate
from lif_module import LIFNode
from model import SimpleSNN1d, HardLIFNode, StdpSNN1d


# ------------------------- config -------------------------
DATA_DIR = "../data/2_ffilled_data"
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


# ------------------------- variant configs -------------------------
@dataclass
class VariantConfig:
    name: str
    channels: Tuple[int, int]

VARIANTS: List[VariantConfig] = [
    VariantConfig("smallest", (16, 32)),
    VariantConfig("small",    (24, 48)),
    VariantConfig("medium",   (32, 64)),
    VariantConfig("large",    (40, 80)),
    VariantConfig("largest",  (48, 96)),
]


# ------------------------- Conv1d teacher (soft LIF) -------------------------
class TeacherSNN1d(nn.Module):
    def __init__(self, c1: int, c2: int, num_classes: int = NUM_CLASSES, T: int = T_STEPS):
        super().__init__()
        self.T = T
        self.conv1 = nn.Conv1d(12, c1, kernel_size=3, padding=1)
        self.lif1 = LIFNode(tau=1.0, threshold=0.2)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=3, padding=1)
        self.lif2 = LIFNode(tau=0.9, threshold=0.15)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c2, num_classes)

    def forward(self, x):
        # x: (B, 12, N, T)
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


# ------------------------- Conv1d student (hard LIF + KD) -------------------------
class StudentSNN1d(nn.Module):
    def __init__(self, c1: int, c2: int, num_classes: int = NUM_CLASSES, T: int = T_STEPS,
                 thresh1: float = 0.02, thresh2: float = 0.02,
                 surrogate_width: float = 5.0, gain: float = 3.0):
        super().__init__()
        self.T = T
        self.gain = gain
        self.conv1 = nn.Conv1d(12, c1, kernel_size=3, padding=1)
        self.lif1 = HardLIFNode(tau=1.0, threshold=thresh1, surrogate_width=surrogate_width)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=3, padding=1)
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


# ------------------------- utils -------------------------
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_kb(model: nn.Module) -> float:
    size = sum(p.nelement() * p.element_size() for p in model.parameters())
    size += sum(b.nelement() * b.element_size() for b in model.buffers())
    return size / 1024.0


def kd_loss(student_logits, teacher_logits, T=2.0):
    log_p = F.log_softmax(student_logits / T, dim=1)
    q = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_p, q, reduction="batchmean") * (T * T)


def export_lite(builder, state_dict, ptl_path: str):
    model_cpu = builder().cpu()
    state = {k: v for k, v in state_dict.items() if not k.endswith(".v")}
    model_cpu.load_state_dict(state, strict=False)
    model_cpu.eval()
    with torch.no_grad():
        model_cpu.reset()
        dummy = torch.zeros(1, 12, 1, T_STEPS)
        model_cpu(dummy)
    scripted = torch.jit.script(model_cpu)
    optimized = optimize_for_mobile(scripted)
    optimized._save_for_lite_interpreter(ptl_path)


# ========================= Phase 1: Train teachers =========================
def train_teachers(train_loader, val_loader):
    print("\n========== Phase 1: Training Conv1d Teachers (Soft LIF) ==========")
    results = []

    for cfg in VARIANTS:
        builder = lambda cfg=cfg: TeacherSNN1d(cfg.channels[0], cfg.channels[1])
        model = builder().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_score = -1.0
        best_state = None
        pt_path = os.path.join(MODEL_DIR, f"snn1d_{cfg.name}_sensor.pt")
        ptl_path = os.path.join(MODEL_DIR, f"snn1d_{cfg.name}_sensor.ptl")

        for epoch in range(EPOCHS):
            model.train()
            correct, total = 0, 0
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                model.reset()
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
            train_acc = correct / total

            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    model.reset()
                    out = model(x)
                    val_correct += (out.argmax(1) == y).sum().item()
                    val_total += y.size(0)
            val_acc = val_correct / max(val_total, 1)

            score = train_acc + val_acc
            print(f"[Teacher {cfg.name}] Epoch {epoch+1}/{EPOCHS} | train={train_acc:.4f} | val={val_acc:.4f}")

            if score > best_score:
                best_score = score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        torch.save(best_state, pt_path)
        export_lite(builder, best_state, ptl_path)
        print(f"  Saved: {pt_path} (params={count_params(model)}, {model_size_kb(model):.1f}KB)")

        results.append({
            "variant": cfg.name, "type": "teacher_1d",
            "channels": cfg.channels, "params": count_params(model),
            "best_score": best_score, "pt": pt_path, "ptl": ptl_path,
        })
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# ========================= Phase 2: KD students =========================
def train_students(train_loader, val_loader, teacher_results):
    print("\n========== Phase 2: Training Conv1d Students (HardLIF + KD) ==========")
    results = []

    for cfg in VARIANTS:
        teacher_pt = os.path.join(MODEL_DIR, f"snn1d_{cfg.name}_sensor.pt")
        if not os.path.exists(teacher_pt):
            print(f"  [SKIP] Teacher not found: {teacher_pt}")
            continue

        # load teacher
        teacher = TeacherSNN1d(cfg.channels[0], cfg.channels[1]).to(DEVICE)
        state = torch.load(teacher_pt, map_location=DEVICE)
        state = {k: v for k, v in state.items() if not k.endswith(".v")}
        teacher.load_state_dict(state, strict=False)
        teacher.eval()

        # build student
        student_builder = lambda cfg=cfg: StudentSNN1d(cfg.channels[0], cfg.channels[1])
        student = student_builder().to(DEVICE)
        optimizer = torch.optim.Adam(student.parameters(), lr=LR)

        # class weights
        best_score = -1.0
        best_state = None
        pt_path = os.path.join(MODEL_DIR, f"student_kd1d_{cfg.name}_sensor.pt")
        ptl_path = os.path.join(MODEL_DIR, f"student_kd1d_{cfg.name}_sensor.ptl")

        for epoch in range(EPOCHS):
            student.train()
            correct, total, total_loss = 0, 0, 0.0
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                student.reset()
                teacher.reset()
                optimizer.zero_grad()

                s_logits = student(x)
                with torch.no_grad():
                    t_logits = teacher(x)

                loss_ce = F.cross_entropy(s_logits, y)
                loss_kd = kd_loss(s_logits, t_logits, T=TEMP)
                loss = ALPHA_CE * loss_ce + (1 - ALPHA_CE) * loss_kd
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
                optimizer.step()

                total_loss += loss.item() * y.size(0)
                correct += (s_logits.argmax(1) == y).sum().item()
                total += y.size(0)

            train_acc = correct / total

            student.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    student.reset()
                    out = student(x)
                    val_correct += (out.argmax(1) == y).sum().item()
                    val_total += y.size(0)
            val_acc = val_correct / max(val_total, 1)

            score = train_acc + val_acc
            print(f"[Student {cfg.name}] Epoch {epoch+1}/{EPOCHS} | train={train_acc:.4f} | val={val_acc:.4f}")

            if score > best_score:
                best_score = score
                best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}

        torch.save(best_state, pt_path)
        export_lite(student_builder, best_state, ptl_path)
        print(f"  Saved: {pt_path} (params={count_params(student)}, {model_size_kb(student):.1f}KB)")

        results.append({
            "variant": cfg.name, "type": "student_kd_1d",
            "channels": cfg.channels, "params": count_params(student),
            "best_score": best_score, "pt": pt_path, "ptl": ptl_path,
        })
        del teacher, student
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# ========================= main =========================
def main():
    print(f"Device: {DEVICE}, workers={NUM_WORKERS}")

    dataset = FileLevelSpikeDataset(DATA_DIR, rate_code_zscore_sigmoid, T=T_STEPS)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=pad_collate, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=pad_collate, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    # Phase 1: teacher (soft LIF)
    teacher_results = train_teachers(train_loader, val_loader)

    # Phase 2: student (hard LIF + KD)
    student_results = train_students(train_loader, val_loader, teacher_results)

    all_results = teacher_results + student_results
    result_path = os.path.join(RESULT_DIR, "conv1d_results.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=== Done. Results saved to {result_path} ===")
    for r in all_results:
        print(f"  {r['type']} {r['variant']}: params={r['params']}, score={r['best_score']:.4f}")


if __name__ == "__main__":
    main()
