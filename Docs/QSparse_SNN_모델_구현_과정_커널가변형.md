# QSparse SNN 모델 구현 과정 (커널 가변형 기준)

## Phase 1: Teacher SNN 학습

**목적**: Soft LIF 뉴런으로 높은 정확도의 기준 모델 생성

- **아키텍처**: `TeacherSNN1d` — Conv1d 2층 + Soft LIF 뉴런
  - `Conv1d(12, 32, k) → LIFNode(tau=1.0, thresh=0.2) → Conv1d(32, 64, k) → LIFNode(tau=0.9, thresh=0.15) → AdaptiveAvgPool1d → FC(64, 4)`
- **LIF 뉴런**: `LIFNode` — 부드러운 미분 가능한 뉴런 (sigmoid surrogate gradient)
  - 막전위: `v = v + (x - v) / tau`
  - 스파이크: `sigmoid((v - threshold) * slope)` → 연속값 (0~1)
- **입력**: 6축 IMU → 12채널 (pos/neg split) → z-score → sigmoid → 확률적 이진 스파이크 (rate coding)
- **시간 축**: T timestep 동안 스파이크 누적 후 평균 → pooling → FC
- **설정**: epochs=50, lr=1e-3, Adam, k∈{3,5,7,9,11}, T∈{3,5,10,15}, 채널 고정 (32,64)
- **산출물**: `snn1d_teacher_{variant}_T{T}.pt` + `.ptl` (25 + 25 = 50개)

---

## Phase 2: Student SNN 학습 (Knowledge Distillation)

**목적**: Soft LIF → Hard LIF로 전환하면서 정확도 유지

- **아키텍처**: `StudentSNN1d` — 동일 구조이나 LIF가 다름
  - `HardLIFNode` 사용: 이진 스파이크 (0 또는 1) + STE(Surrogate Transfer Estimation) 역전파
  - `spike = (v >= threshold)` → forward는 step function, backward는 rectangular surrogate
- **핵심 변경**:
  - threshold: 0.2/0.15 → 0.02/0.02 (낮춰서 발화 촉진)
  - gain=3.0: conv 출력에 곱해서 신호 증폭 (`conv(x) * gain`)
  - surrogate_width=5.0: STE 구간 넓혀서 gradient flow 보장
- **KD Loss**: Teacher의 soft logits를 따라가도록 학습
  ```
  loss = 0.9 × CE(student, label) + 0.1 × KL_div(student/T, teacher/T) × T²
  ```
  - TEMP=2.0으로 softmax를 부드럽게 → Teacher의 dark knowledge 전달
- **산출물**: `snn1d_student_{variant}_T{T}.pt` + `student_kd_{variant}_T{T}.ptl` (25 + 25 = 50개)

---

## Phase 3: Sparse Student 학습 (Firing Rate Regularization)

**목적**: 발화율을 80~95% → 5~30%로 낮추면서 정확도 유지

- **아키텍처**: `SparseStudentSNN1d` — Student 기반 + 학습 가능한 threshold
  - `LearnableHardLIFNode`: threshold가 `nn.Parameter`로 전환
  - gain도 `nn.Parameter`로 전환
  - 커스텀 STE: threshold에도 gradient 흐름 (∂spike/∂threshold = -surrogate)
- **학습 전략**:
  1. KD Student 가중치 로드 (Phase 2 결과)
  2. **Firing Rate Loss**: `Σ(actual_rate - target_rate)²` — 스파이크 평균이 목표에 가까워지도록
  3. **Curriculum**: lambda_sparse를 0 → 5.0으로 10 epoch에 걸쳐 점진 증가
  4. **Regularization**: threshold < 0 방지 (clamp 0.01), gain 증가 페널티
  5. **Dual LR**: threshold/gain은 10배 높은 LR
- **설정**: epochs=40, lr=5e-4, target_rate∈{0.3, 0.2, 0.1, 0.05}
- **산출물**: `sparse_{variant}_T{T}_fr{rate}.pt` + `.ptl` (80 + 80 = 160개)
  - export 시 `ExportableSparseStudent`로 변환 (custom autograd 제거, TorchScript 호환)

---

## Phase 4: QSparse (INT8 Static Quantization)

**목적**: Conv1d 가중치를 INT8로 양자화하여 모델 크기/에너지 절감

- **아키텍처**: `QuantizableSparseStudent` — Sparse Student 재구성
  - **Conv1d만 INT8**: `QuantStub → Conv1d → DeQuantStub` 경계로 감싸기
  - **LIF는 FP32 유지**: 막전위 동역학은 정밀도 필요
  - **Pool, FC도 FP32 유지**
  - `QuantFriendlyLIFNode`: custom autograd 없는 순수 forward LIF
- **양자화 과정** (재학습 없음, Post-Training Quantization):
  1. Sparse `.pt`에서 가중치 로드 (threshold, gain 값 추출)
  2. `qconfig = get_default_qconfig("qnnpack")` 설정
  3. `prepare()`: Conv1d에 observer 삽입 (activation 통계 수집용)
  4. **Calibration**: 학습 데이터 전체를 forward pass → min/max 통계 수집
  5. `convert()`: FP32 Conv1d → INT8 QuantizedConv1d 변환
- **Forward 흐름** (timestep마다):
  ```
  입력(FP32) → QuantStub(→INT8) → Conv1d(INT8) → DeQuantStub(→FP32)
  → LIF1(FP32, spike) → QuantStub(→INT8) → Conv2d(INT8) → DeQuantStub(→FP32)
  → LIF2(FP32, spike) → 누적
  ```
- **결과**: FP32 대비 약 1.8~2x 압축, 정확도 drop < 5% 대부분
- **산출물**: `qsparse_{variant}_T{T}_fr{rate}.pt` + `.ptl` (80 + 80 = 160개)

---

## 전체 파이프라인 요약

```
Phase 1 (Teacher)     Phase 2 (Student KD)     Phase 3 (Sparse)        Phase 4 (QSparse)
Soft LIF, 고정확도  →  Hard LIF, KD로 유지  →  FR 정규화, 발화↓  →  INT8 양자화, 크기↓

  .pt 로드 →              .pt 로드 →               .pt 로드 →
  50 files                 50 files                 160 files               160 files
```

**총 산출물**: 50 + 50 + 160 + 160 = **410개 모델 파일** (커널 가변형, `models_k_variable/`)
