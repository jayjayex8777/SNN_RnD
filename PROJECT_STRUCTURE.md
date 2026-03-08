# Thumbthing_easy_intae2 프로젝트 전체 구조 분석

## 프로젝트 개요

엄지 제스처 인식을 위한 SNN(Spiking Neural Network) 최적화 파이프라인.
6축 IMU 센서(가속도계 + 자이로스코프) 데이터로 4가지 제스처를 분류하며, 모바일 배포를 목표로 모델 경량화(Knowledge Distillation, Sparsification, Quantization)를 체계적으로 실험한다.

---

## 1. 데이터 (`data/2_ffilled_data/`)

| 항목 | 내용 |
|------|------|
| **총 파일 수** | 7,596개 CSV |
| **제스처 4종** | `swipe_up`, `swipe_down`, `flick_up`, `flick_down` |
| **센서 소스 3종** | `sensor` (원본), `ois_2m`, `ois_20m` (OIS 보정) |
| **제스처별 샘플** | 각 633개 × 12종 = 7,596개 |
| **CSV 컬럼** | `Timestamp, ax, ay, az, gx, gy, gz` (6축) |
| **시퀀스 길이** | ~24 timestep (파일별 가변) |
| **전처리** | Forward-fill 완료 (`ffill` 접미사) |

> 학습에는 `sensor` 파일만 사용 (2,532개, 클래스당 633개). `ois_*` 파일은 현재 무시됨.

---

## 2. 핵심 모듈 (공통)

### `lif_module.py` — LIF 뉴런 (Soft Surrogate)

- `LIFNode`: 막전위 기반 뉴런, sigmoid surrogate gradient로 미분 가능
- 파라미터: `tau`(시정수), `threshold`(발화 임계값), `reset_value`

### `model.py` — SNN 모델 정의

- **`SimpleSNN1d`**: Conv1d(12→32→64) + Soft LIF, 기본 SNN
- **`HardLIFNode`**: 이진 스파이크(0/1) + STE(Straight-Through Estimator) 역전파
- **`_HardSpikeSTE`**: 커스텀 autograd — 임계값 주변 rectangular window로 gradient 전파
- **`StdpSNN1d`**: Hard spike SNN 변형

### `dataset.py` — 데이터 파이프라인

- **`rate_code_zscore_sigmoid`**: 6축 신호 → 12채널(양/음 분리) → Z-score → Sigmoid → 확률적 스파이크 인코딩
- **`FileLevelSpikeDataset`**: 파일별 lazy loading, 라벨 자동 매핑
- **`pad_collate`**: 가변 길이 시퀀스를 배치 내 최대 길이로 패딩 → `(B, 12, N_max, T)`

---

## 3. 학습 파이프라인 (4단계)

### Phase 1+2: `train.py` — 기본 Teacher/Student 학습

- **Teacher (Soft LIF)**: `TeacherSNN1d` — Conv1d(k=3) + Soft LIF, 5가지 크기 변형
  - `smallest(16,32)`, `small(24,48)`, `medium(32,64)`, `large(40,80)`, `largest(48,96)`
- **Student (Hard LIF + KD)**: `StudentSNN1d` — 동일 구조 + Hard spike + Knowledge Distillation
  - KD Loss: `α×CE + (1-α)×KL_div` (α=0.9, T_distill=2.0)
- T=20, 50 epochs, Adam, LR=1e-3
- 출력: `.pt` (PyTorch) + `.ptl` (TorchScript Lite for mobile)

### Phase 1+2 변형: `train_kernel_sweep.py` — 커널 크기 탐색

- 채널 고정 (32, 64), 커널 크기 k=3,5,7,9 sweep
- Teacher + Student(KD) 동일 파이프라인

### Phase 1+2 변형: `train_k11_only.py` — k=11 추가 학습

- k=11(largest) 단독 학습, 기존 결과에 병합

### Phase 1+2 변형: `train_t3.py` — T-step sweep

- T=3 (timestep 감소 실험)
- k=3,5,7,9,11 × Teacher/Student 전부 학습
- 커널 크기 → variant 매핑: `{3:smallest, 5:small, 7:medium, 9:large, 11:largest}`

### Phase 2.5: `export_t_sweep.py` — 모바일 export 전용

- `TraceFriendlyLIF`: `torch.jit.trace` 호환 LIF
- `ExportableStudentSNN1d`: STE 제거한 export 전용 Student
- T=5,10,15 모델들을 `.ptl`로 변환

### Phase 3: `train_sparse.py` — 발화율 희소화

- **`LearnableHardLIFNode`**: threshold가 `nn.Parameter` → 학습 가능
- **`SparseStudentSNN1d`**: gain도 learnable, 스파이크 수집 기능
- **Firing Rate Regularization**: `(actual_rate - target_rate)²` 손실
- Curriculum: λ_sparse를 10 epoch에 걸쳐 점진적 증가 (0→5.0)
- **Sweep**: T={3,5,10,15} × k={3,5,7,9,11} × target_rate={0.30,0.20,0.10,0.05}
- threshold 음수 방지 clamp, gain 증가 페널티
- `ExportableSparseStudent`: TorchScript 호환 export 모델

### CNN 베이스라인: `train_cnn1d_k9.py`

- **`Conv1dCNN`**: Conv1d(k=9) + GroupNorm + ReLU (비SNN 베이스라인)
- **`QuantizableConv1dCNN`**: BatchNorm + static INT8 quantization (qnnpack)
- 5가지 크기 변형, 200 epochs, Cosine Annealing + Early Stopping
- FP32 `.pt` + `.ptl` + Quantized `.ptl` 모두 export

---

## 4. 벤치마크: `benchmark.py`

5가지 모델 유형 종합 비교:

| 모델 | 설명 |
|------|------|
| **CNN** | Conv1d(k=9), 단일 forward pass |
| **QCNN** | INT8 양자화 CNN |
| **Teacher SNN** | Soft LIF, T timesteps |
| **Student SNN** | Hard LIF + KD, T timesteps |
| **Sparse SNN** | 발화율 억제 Student |

측정 항목:

- **Validation Accuracy**: 80/20 split
- **CPU Latency**: warmup 20회 후 100회 측정 (mean, median, p95)
- **이론적 에너지**: 45nm 공정 기준 MAC(4.6pJ) vs AC(0.9pJ) 계산
  - SNN의 이진 스파이크 → Accumulate-only(AC) 연산으로 에너지 절감 가능성 추정

---

## 5. 저장된 모델 (`models/`) — 187개

| 패턴 | 의미 | 대략적 수량 |
|------|------|------------|
| `cnn1d_*_sensor.pt/ptl` | CNN 베이스라인 (FP32) | ~10 |
| `qcnn1d_*_sensor.ptl` | 양자화 CNN (INT8) | ~5 |
| `snn1d_teacher_*` | Teacher SNN (다양한 T, k) | ~30+ |
| `snn1d_student_*` | Student SNN (KD 완료) | ~30+ |
| `student_kd_*` | Student KD export (.ptl) | ~20+ |
| `snn_*_T*.ptl` | Teacher SNN export | ~20+ |
| `sparse_*_T*_fr*.pt/ptl` | 희소화 Student (다양한 FR target) | ~60+ |

---

## 6. 결과 파일 (`result/`)

| 파일 | 내용 |
|------|------|
| `cnn1d_k9_results.json` | CNN 베이스라인 결과 |
| `kernel_sweep_results.json` | 커널 크기별 SNN 성능 |
| `t_sweep_results.json` | T-step별 SNN 성능 |
| `benchmark_results.json` | 종합 벤치마크 (정확도, 지연시간, 에너지) |

---

## 7. 전체 데이터 흐름

```
CSV (6축 IMU)
    ↓ rate_code_zscore_sigmoid()
Spike Tensor (12, N, T) — 확률적 이진 스파이크
    ↓ pad_collate()
Batch (B, 12, N_max, T)
    ↓
┌─────────────────────────────────────────────┐
│  Phase 1: Teacher (Soft LIF, 연속 출력)       │
│  Phase 2: Student (Hard LIF, KD 학습)         │
│  Phase 3: Sparse (발화율 제약, threshold 학습)  │
│  Baseline: CNN / QCNN (비SNN 대조군)           │
└─────────────────────────────────────────────┘
    ↓
.pt (학습 체크포인트) + .ptl (모바일 배포용 TorchScript Lite)
    ↓
benchmark.py → 정확도/지연시간/에너지 종합 비교
```

---

## 8. 핵심 설계 특징

1. **SNN의 에너지 효율성 검증**: MAC → AC 전환 시 에너지 5.1배 절감 가능성 분석 (4.6pJ → 0.9pJ)
2. **3단계 점진적 최적화**: Soft LIF Teacher → Hard spike Student (KD) → Firing Rate 희소화
3. **체계적 하이퍼파라미터 sweep**: 커널 크기(3~11), T-step(3~20), 발화율 목표(5~30%)
4. **모바일 배포 파이프라인**: `torch.jit.script/trace` → `optimize_for_mobile` → `.ptl` (Lite Interpreter)

---

## 9. 디렉토리 트리

```
Thumbthing_easy_intae2/
├── data/
│   └── 2_ffilled_data/          # 7,596개 CSV (6축 IMU, forward-filled)
├── models/                      # 187개 학습된 모델 (.pt, .ptl)
├── result/                      # 실험 결과 JSON
│   ├── benchmark_results.json
│   ├── cnn1d_k9_results.json
│   ├── kernel_sweep_results.json
│   └── t_sweep_results.json
├── lif_module.py                # LIF 뉴런 (Soft Surrogate)
├── model.py                     # SNN 모델 (SimpleSNN1d, HardLIFNode, StdpSNN1d)
├── dataset.py                   # 스파이크 인코딩 + Dataset + Collate
├── train.py                     # Phase 1+2: Teacher/Student (크기 변형)
├── train_kernel_sweep.py        # Phase 1+2: 커널 크기 sweep (k=3,5,7,9)
├── train_k11_only.py            # Phase 1+2: k=11 추가 학습
├── train_t3.py                  # Phase 1+2: T=3 sweep
├── train_t_sweep.py             # Phase 1+2: T-step sweep
├── train_sparse.py              # Phase 3: 발화율 희소화
├── train_cnn1d_k9.py            # CNN 베이스라인 + INT8 양자화
├── export_t_sweep.py            # T-sweep 모델 .ptl export
├── benchmark.py                 # 종합 벤치마크 (정확도/지연/에너지)
├── *.log                        # 학습 로그 파일
└── PROJECT_STRUCTURE.md         # 이 문서
```
