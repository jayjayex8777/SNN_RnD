# SNN 4-Phase 파이프라인 구현 상세 (Teacher → Student KD → Sparse → QSparse)

## 0. 데이터 준비 — Rate Coding

### 목적
IMU 센서의 아날로그 신호를 SNN이 처리할 수 있는 **바이너리 스파이크(0/1)**로 변환

### 입력 데이터
- 경로: `data/2_ffilled_data/`
- 파일: `{제스처}_sensor_{번호}_6axis_ffill.csv` (2,532개)
- 컬럼: `Timestamp, ax, ay, az, gx, gy, gz`
- 4클래스: swipe_up(0), swipe_down(1), flick_up(2), flick_down(3)

### 변환 과정 (dataset.py:9-32)

```
ax, ay, az, gx, gy, gz (6축)
    ↓ 양수/음수 분리
ax_pos, ax_neg, ay_pos, ay_neg, ... (12채널)
    ↓ Z-score 정규화
z = (signal - mean) / (std + 1e-8)
    ↓ Sigmoid (발화 확률로 변환)
prob = sigmoid(z)     — 값 범위 [0, 1]
    ↓ Bernoulli 샘플링 (T 타임스텝 반복)
spike = (rand < prob) → 0 또는 1
```

**최종 텐서 형태**: `(B, 12, N, T)` — Batch × 12채널 × 시퀀스길이 × 타임스텝

### 배치 처리 (dataset.py:70-79)
- CSV 파일마다 시퀀스 길이(N)가 다름
- `pad_collate`로 배치 내 최대 길이에 맞춰 zero-padding

### 채널 가변형 (Channel Variant) — CNN과 동일 구조

| Variant | Conv1 채널 | Conv2 채널 |
|---------|:--:|:--:|
| smallest | 16 | 32 |
| small | 24 | 48 |
| medium | 32 | 64 |
| large | 40 | 80 |
| largest | 48 | 96 |

- **kernel_size = 9 고정** (CNN, SNN 모두 동일)
- 채널만 가변하여 CNN과의 공정한 비교 기반 확보

---

## Phase 1: SNN Teacher (Soft LIF)

### 목적
**연속적 membrane potential**을 사용하는 고성능 SNN을 먼저 학습. 이후 Student에게 "지식"을 전달할 Teacher 역할.

### 왜 Teacher가 필요한가?
- 최종 목표는 바이너리 스파이크(0/1)만 쓰는 Student 모델
- 하지만 바이너리 스파이크는 0 아니면 1이라 정보량이 적고, 직접 학습하면 기울기가 불연속
- Teacher는 연속값을 출력하므로 학습이 안정적이고 정확도가 높음
- 이 Teacher의 출력 분포(soft logits)를 Student가 모방하도록 유도

### 아키텍처 (train_channel_teacher_student.py:52-79)

```
입력 (B, 12, N, T)
    ↓ 타임스텝별 반복 (t = 0 ~ T-1)
    x[:,:,:,t] → Conv1d(12→c1, k=9) → Soft LIF1 → Conv1d(c1→c2, k=9) → Soft LIF2
    ↓ T개 출력을 누적
    acc = Σ(output_t) / T
    ↓
    AdaptiveAvgPool1d(1) → FC(c2→4)
```

### Soft LIF 뉴런 (lif_module.py:4-26)

```python
# Membrane potential 업데이트
v = v + (x - v) / tau

# Soft surrogate spike — 연속적 출력 (0~1 사이 실수)
scale = 10.0
out = scale * sigmoid(scale*(v - threshold)) * (1 - sigmoid(scale*(v - threshold)))

# Reset: threshold 넘으면 v를 0으로
v = where(v >= threshold, 0, v)
```

**핵심**: 출력이 0과 1 사이의 **연속값**. 미분 가능하므로 역전파 학습이 자연스러움.

### LIF 파라미터
- **LIF1**: tau=1.0, threshold=0.2
- **LIF2**: tau=0.9, threshold=0.15

### 학습 설정
- Optimizer: Adam (lr=1e-3)
- Loss: CrossEntropyLoss
- Epochs: 50
- Batch: 4
- Train/Val 분할: 80%/20% (seed=42)
- T 값: [3, 5, 10, 15, 20]

### 산출물
- `snn1d_teacher_{variant}_T{T}.pt` — PyTorch 가중치 (25개)
- `snn1d_teacher_{variant}_T{T}.ptl` — TorchScript Lite (25개)
- 저장 위치: `models_channel_variant/`

### 스크립트
- `train_channel_teacher_student.py` (Phase 1+2 통합)
- `export_teacher_student_ptl.py` (PTL export 전용)

---

## Phase 2: Student KD (Hard LIF + STE)

### 목적
Teacher의 연속 출력을 **바이너리 스파이크(0/1)**로 전환. Knowledge Distillation으로 Teacher의 지식을 전달받아 정확도 손실 최소화.

### 왜 Hard Spike가 필요한가?
- 뉴로모픽 하드웨어는 바이너리 스파이크만 처리
- 스파이크가 0일 때 연산을 스킵하여 에너지 절감 (AC 연산)
- Teacher의 soft 출력을 그대로 쓰면 일반 CNN과 다를 바 없음

### 아키텍처 (train_channel_teacher_student.py:82-111)

```
입력 (B, 12, N, T)
    ↓ 타임스텝별 반복 (t = 0 ~ T-1)
    x[:,:,:,t] → Conv1d(12→c1, k=9) × gain → Hard LIF1
              → Conv1d(c1→c2, k=9) × gain → Hard LIF2
    ↓ T개 출력을 누적
    acc = Σ(spike_t) / T
    ↓
    AdaptiveAvgPool1d(1) → FC(c2→4)
```

### Hard LIF 뉴런 + STE (model.py:51-88)

```python
# Forward: 바이너리 스파이크 생성
spike = (v >= threshold).float()   # 0 또는 1

# Backward: Straight-Through Estimator (STE)
# 실제 미분이 0인 구간에서 surrogate gradient 사용
delta = mem - threshold
mask = (|delta| <= width)          # threshold 근처만 기울기 전달
grad = grad_output * mask / (2 * width)
```

**핵심**: forward에서는 0/1 바이너리, backward에서는 STE로 기울기를 근사하여 학습 가능하게 만듦.

### Knowledge Distillation Loss (train_channel_teacher_student.py:122-125)

```python
loss = 0.9 × CrossEntropy(student, label) + 0.1 × KL_div(student, teacher)

# KL Divergence (temperature scaling)
KL = KL_div(log_softmax(student_logits/T), softmax(teacher_logits/T)) × T²
```

- **Temperature T=2.0**: softmax를 부드럽게 만들어 Teacher의 "dark knowledge" 전달
- **α=0.9**: 정답 라벨 비중 90%, Teacher 모방 비중 10%
- Gradient Clipping: 1.0 (Hard spike의 불안정한 기울기 제어)

### Student 추가 파라미터
- **gain=3.0**: Conv 출력에 곱하여 membrane potential을 키움 → 더 잘 발화
- **threshold=0.02**: 매우 낮은 임계값 → 학습 초기에 쉽게 발화하도록
- **surrogate_width=5.0**: STE 기울기 전달 범위 확대

### 산출물
- `snn1d_student_{variant}_T{T}.pt` — PyTorch 가중치 (25개)
- `snn1d_student_{variant}_T{T}.ptl` — TorchScript Lite (25개)
- 저장 위치: `models_channel_variant/`

### PTL Export 문제와 해결
- **문제**: `_HardSpikeSTE` 커스텀 autograd가 TorchScript 직렬화 불가
- **해결**: `export_teacher_student_ptl.py`에서 `ExportableHardLIFNode` 생성
  - 커스텀 autograd 제거, `(v >= threshold).float()` 사용 (추론 시 STE 불필요)
  - 기존 .pt 가중치 로드 → Exportable 모델에 매핑 → TorchScript trace → PTL

---

## Phase 3: Sparse Student (Learnable Threshold + FR Regularization)

### 목적
Student의 뉴런 발화율을 **목표치(5~30%)로 억제**하여 희소성(Sparsity) 확보. 스파이크가 적을수록 뉴로모픽 HW에서 연산 스킵이 많아져 에너지 절감.

### 왜 Sparsity가 필요한가?
- Student KD의 자연 발화율은 약 30~50% (뉴런의 절반이 발화)
- 발화율 5%면 95%의 시냅스 연산을 스킵 가능
- 또한 Phase 4(양자화)의 전 단계로, threshold 학습이 양자화 안정성에 기여

### 핵심 기법 1: Learnable Threshold (train_channel_sparse.py:75-100)

```python
class LearnableHardLIFNode(nn.Module):
    def __init__(self, ...):
        self.threshold = nn.Parameter(torch.tensor(float(threshold)))  # 학습 가능!
```

- Student KD의 고정 threshold(0.02)를 **학습 가능한 파라미터**로 변환
- STE도 threshold에 기울기를 전달하도록 수정:
  - `grad_thresh = -grad_output × surrogate` (반대 방향)
- 학습 중 threshold가 올라가면 → 발화 조건이 까다로워짐 → 발화율 감소

### 핵심 기법 2: Firing Rate Regularization (train_channel_sparse.py:148-154)

```python
def firing_rate_loss(spikes_list, target_rate):
    for spk in spikes_list:
        actual_rate = spk.mean()
        loss += (actual_rate - target_rate) ** 2
    return loss / len(spikes_list)
```

### 전체 Loss (train_channel_sparse.py:319-330)

```python
loss = CE + λ × FR_loss + 10.0 × thresh_penalty + 0.5 × gain_penalty
```

| 항목 | 역할 |
|------|------|
| CE | CrossEntropy — 분류 정확도 유지 |
| λ × FR_loss | 발화율을 target_rate로 유도 |
| thresh_penalty | threshold < 0 방지 (clamp도 적용) |
| gain_penalty | gain > 3.0 방지 |

### Curriculum Lambda (점진적 강화)

```python
progress = min(1.0, (epoch + 1) / 10.0)
current_lambda = 5.0 × progress
```

- 처음 10 에포크에 걸쳐 λ를 0 → 5.0으로 서서히 증가
- 갑자기 발화율을 억제하면 정확도가 급락하므로 점진적 적용

### Optimizer 설정

```python
optimizer = Adam([
    {"params": other_params, "lr": 5e-4},         # Conv, FC
    {"params": thresh_params, "lr": 5e-3},         # threshold, gain (10배 높은 lr)
])
scheduler = CosineAnnealingLR(optimizer, T_max=40)
```

- threshold와 gain은 빠르게 학습해야 하므로 lr 10배
- CosineAnnealing으로 후반부에 lr 감소

### KD Student 가중치 로드 (train_channel_sparse.py:174-196)

```python
# Phase 2의 student .pt 파일에서 가중치 로드
state = torch.load(f"snn1d_student_{variant}_T{T}.pt")
# LIF buffer(.v)는 skip, 나머지는 SparseStudentSNN1d에 매핑
model.load_state_dict(filtered, strict=False)
```

### Export: ExportableSparseStudent (train_channel_sparse.py:200-283)

학습용 모델의 커스텀 autograd를 제거한 export 전용 모델:
- `ExportableHardLIFNode`: `(v >= threshold).float()` — 추론 시 STE 불필요
- 학습된 threshold, gain을 **상수로 주입**
- `torch.jit.script()` → `optimize_for_mobile()` → `.ptl`

### 산출물
- `sparse_{variant}_T{T}_fr{rate}.pt` — PyTorch 가중치 (80개)
- `sparse_{variant}_T{T}_fr{rate}.ptl` — TorchScript Lite (80개)
- T: [3, 5, 10, 15], FR: [30%, 20%, 10%, 5%]
- 5 variants × 4 T × 4 FR = 80개
- 저장 위치: `models_channel_variant/`
- 결과: `result/channel_variant_sparse_results.json`

### 스크립트
- `train_channel_sparse.py`

---

## Phase 4: QSparse (INT8 Static Quantization)

### 목적
Sparse Student의 Conv1d 가중치를 **FP32 → INT8**로 양자화하여 모델 크기 축소 및 연산 가속. 모바일/엣지 배포를 위한 최종 단계.

### 왜 INT8 Static Quantization인가?
- Dynamic Quantization은 가중치만 INT8, 활성화는 FP32 → 크기 감소 미미
- Static Quantization은 **가중치 + 활성화 모두 INT8** → 실질적 크기 감소
- QSparse는 Conv1d만 INT8, LIF 뉴런은 FP32 유지 (바이너리 스파이크 특성상 양자화 불필요)

### 핵심 구조: QuantizableSparseStudent (train_channel_qsparse.py:86-141)

```
타임스텝 반복:
    x[:,:,:,t]
        ↓ QuantStub1 (FP32 → INT8)
        ↓ Conv1d(12→c1, k=9)    ← INT8 연산
        ↓ DeQuantStub1 (INT8 → FP32)
        ↓ × gain
        ↓ LIF1 (FP32) → spike (0 or 1)
        ↓ QuantStub2 (FP32 → INT8)
        ↓ Conv1d(c1→c2, k=9)    ← INT8 연산
        ↓ DeQuantStub2 (INT8 → FP32)
        ↓ × gain
        ↓ LIF2 (FP32) → spike (0 or 1)
        ↓ 누적
    acc / T → Pool → FC (FP32)
```

### QuantFriendlyLIFNode (train_channel_qsparse.py:64-82)

```python
class QuantFriendlyLIFNode(nn.Module):
    def forward(self, x):
        self.v = self.v + (x - self.v) / self.tau
        spike = (self.v >= self.threshold).float()  # 순수 텐서 연산만
        self.v = where(spike > 0, 0, self.v)
        return spike
```

- 커스텀 autograd 없음 (STE도, LearnableThreshold도 없음)
- TorchScript/양자화 완전 호환
- threshold, gain은 Sparse에서 학습된 값을 **상수로 주입**

### 양자화 과정 (train_channel_qsparse.py:180-198)

```python
# 1. qconfig 설정 — Conv1d만 양자화, LIF/Pool/FC는 제외
model.qconfig = get_default_qconfig("qnnpack")
model.lif1.qconfig = None   # LIF는 양자화 제외
model.lif2.qconfig = None
model.pool.qconfig = None
model.fc.qconfig = None

# 2. Observer 삽입 (활성화 범위 측정 준비)
torch.quantization.prepare(model, inplace=True)

# 3. Calibration (학습 데이터로 활성화 범위 수집)
for x, _ in calibration_loader:
    model(x)

# 4. 실제 양자화 변환 (FP32 → INT8)
torch.quantization.convert(model, inplace=True)
```

### Sparse 가중치 로드 (train_channel_qsparse.py:144-176)

```python
# Phase 3의 sparse .pt 파일에서 가중치 + 학습된 threshold/gain 로드
state = torch.load(f"sparse_{variant}_T{T}_fr{rate}.pt")
thresh1 = state["lif1.threshold"].item()   # 학습된 threshold
thresh2 = state["lif2.threshold"].item()
gain = state["gain"].item()

# QuantizableSparseStudent에 상수로 주입
model = QuantizableSparseStudent(c1, c2, K, T, thresh1, thresh2, gain)
# Conv, FC 가중치만 로드 (LIF buffer, threshold, gain은 skip)
```

### 양자화 실패 기준 (train_channel_qsparse.py:326)

```python
status = "OK" if abs(acc_drop) < 0.05 else "WARN"
```

- 양자화 후 정확도 하락 > 5%p이면 WARN
- 실제로 80개 중 18개가 WARN (medium이 가장 불안정, large/largest가 가장 안정)

### Pure Spike Forward Only

양자화 후에도 레이어 간 전달되는 것은 **오직 바이너리 스파이크(0/1)**:
1. 입력 스파이크(0/1) → QuantStub → INT8 Conv1d → DeQuantStub → × gain
2. → LIF → **spike(0/1)** → QuantStub → INT8 Conv1d → DeQuantStub → × gain
3. → LIF → **spike(0/1)** → 누적 → Pool → FC → logits

### Export (train_channel_qsparse.py:215-226)

```python
scripted = torch.jit.script(model)    # TorchScript 변환
opt = optimize_for_mobile(scripted)    # 모바일 최적화
opt._save_for_lite_interpreter(path)   # .ptl 저장
```

### 산출물
- `qsparse_{variant}_T{T}_fr{rate}.pt` — INT8 가중치 (80개)
- `qsparse_{variant}_T{T}_fr{rate}.ptl` — TorchScript Lite (80개)
- T: [3, 5, 10, 15], FR: [30%, 20%, 10%, 5%]
- 저장 위치: `models_channel_variant/`
- 결과: `result/channel_variant_qsparse_results.json`

### 스크립트
- `train_channel_qsparse.py`

---

## 전체 파이프라인 요약

```
Phase 1: Teacher (Soft LIF)
  목적: 연속값 출력으로 안정적 학습, 높은 정확도 확보
  뉴런: Soft surrogate spike (연속 0~1)
  Loss: CrossEntropy
  산출: snn1d_teacher_*.pt

      ↓ Knowledge Distillation (soft logits 전달)

Phase 2: Student KD (Hard LIF + STE)
  목적: 바이너리 스파이크(0/1)로 전환, Teacher 지식 계승
  뉴런: Hard spike + STE (forward: 0/1, backward: surrogate gradient)
  Loss: 0.9×CE + 0.1×KL_div
  산출: snn1d_student_*.pt

      ↓ 가중치 로드 + Learnable threshold 초기화

Phase 3: Sparse Student
  목적: 발화율 억제 → 스파이크 희소성 확보 → 뉴로모픽 에너지 절감
  뉴런: Learnable threshold Hard LIF + STE
  Loss: CE + λ×FR_loss + penalties
  산출: sparse_*_fr{rate}.pt, .ptl

      ↓ 학습된 threshold/gain을 상수로 주입

Phase 4: QSparse (INT8)
  목적: Conv1d를 INT8로 양자화 → 모델 크기 축소 + 모바일 배포
  뉴런: QuantFriendlyLIF (커스텀 autograd 제거, FP32 유지)
  양자화: QuantStub → Conv1d(INT8) → DeQuantStub → LIF(FP32)
  산출: qsparse_*_fr{rate}.pt, .ptl
```

## 각 Phase에서 변하는 것과 유지되는 것

| 항목 | Phase 1 (Teacher) | Phase 2 (Student KD) | Phase 3 (Sparse) | Phase 4 (QSparse) |
|------|:--:|:--:|:--:|:--:|
| Conv1d 가중치 | 새 학습 | 새 학습 | Fine-tune | 양자화(INT8) |
| LIF 뉴런 | Soft(연속) | Hard(0/1)+STE | Learnable thresh | QuantFriendly |
| 스파이크 출력 | 연속값 | **0/1** | **0/1** | **0/1** |
| Threshold | 고정(0.2/0.15) | 고정(0.02) | **학습 가능** | 상수 주입 |
| Gain | 없음 | 3.0 고정 | **학습 가능** | 상수 주입 |
| 발화율 제어 | 없음 | 없음 | **FR Reg** | 계승 |
| 양자화 | FP32 | FP32 | FP32 | **INT8** |
| Export 가능 | △ (buffer 문제) | × (STE) | ○ (Exportable) | ○ (Script) |

## 관련 파일 목록

### 스크립트
| 파일 | 역할 |
|------|------|
| `dataset.py` | Rate Coding + Dataset + pad_collate |
| `lif_module.py` | Soft LIF 뉴런 (Teacher용) |
| `model.py` | Hard LIF + STE 뉴런 (Student용) |
| `train_channel_teacher_student.py` | Phase 1+2: Teacher + Student KD 학습 |
| `train_channel_sparse.py` | Phase 3: Sparse Student 학습 |
| `train_channel_qsparse.py` | Phase 4: INT8 Static Quantization |
| `train_channel_T1T2.py` | T=1,2 전용 Phase 1~4 통합 |
| `export_teacher_student_ptl.py` | Teacher/Student PTL export |
| `export_early_exit_qsparse.py` | Early Exit QSparse 생성 |
| `export_qcnn_static.py` | QCNN Static Quantization |

### 결과
| 파일 | 내용 |
|------|------|
| `result/channel_variant_teacher_student_results.json` | Phase 1+2 결과 |
| `result/channel_variant_sparse_results.json` | Phase 3 결과 |
| `result/channel_variant_qsparse_results.json` | Phase 4 결과 |
| `result/channel_variant_T1T2_results.json` | T=1,2 결과 |
| `result/early_exit_qsparse_T2_results.json` | Early Exit 결과 |
| `result/qcnn_static_results.json` | QCNN Static 결과 |

### 모델 저장 위치
| 폴더 | 내용 |
|------|------|
| `models_channel_variant/` | 채널 가변형 SNN 전체 (Teacher, Student, Sparse, QSparse, EarlyExit) |
| `models_k_variable/` | 커널 가변형 SNN 전체 (4단계) |
| `models_cnn/` | CNN FP32 + QCNN Static INT8 |
| `Final_Models/` | Best 모델 (cv_, kv_ prefix) |
