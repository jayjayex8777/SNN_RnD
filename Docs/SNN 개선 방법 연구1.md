# SNN 개선 방법 연구1

---

## 1. 프로젝트 전체 구조 분석

### 프로젝트 개요

엄지 제스처 인식을 위한 SNN(Spiking Neural Network) 최적화 파이프라인.
6축 IMU 센서(가속도계 + 자이로스코프) 데이터로 4가지 제스처를 분류하며, 모바일 배포를 목표로 모델 경량화(Knowledge Distillation, Sparsification, Quantization)를 체계적으로 실험한다.

### 1.1 데이터 (`data/2_ffilled_data/`)

| 항목 | 내용 |
|------|------|
| **총 파일 수** | 7,596개 CSV |
| **제스처 4종** | `swipe_up`, `swipe_down`, `flick_up`, `flick_down` |
| **센서 소스 3종** | `sensor` (원본), `ois_2m`, `ois_20m` (OIS 보정) |
| **제스처별 샘플** | 각 633개 x 12종 = 7,596개 |
| **CSV 컬럼** | `Timestamp, ax, ay, az, gx, gy, gz` (6축) |
| **시퀀스 길이** | ~24 timestep (파일별 가변) |
| **전처리** | Forward-fill 완료 (`ffill` 접미사) |

> 학습에는 `sensor` 파일만 사용 (2,532개, 클래스당 633개). `ois_*` 파일은 현재 무시됨.

### 1.2 핵심 모듈 (공통)

#### `lif_module.py` — LIF 뉴런 (Soft Surrogate)

- `LIFNode`: 막전위 기반 뉴런, sigmoid surrogate gradient로 미분 가능
- 파라미터: `tau`(시정수), `threshold`(발화 임계값), `reset_value`

#### `model.py` — SNN 모델 정의

- **`SimpleSNN1d`**: Conv1d(12→32→64) + Soft LIF, 기본 SNN
- **`HardLIFNode`**: 이진 스파이크(0/1) + STE(Straight-Through Estimator) 역전파
- **`_HardSpikeSTE`**: 커스텀 autograd — 임계값 주변 rectangular window로 gradient 전파
- **`StdpSNN1d`**: Hard spike SNN 변형

#### `dataset.py` — 데이터 파이프라인

- **`rate_code_zscore_sigmoid`**: 6축 신호 → 12채널(양/음 분리) → Z-score → Sigmoid → 확률적 스파이크 인코딩
- **`FileLevelSpikeDataset`**: 파일별 lazy loading, 라벨 자동 매핑
- **`pad_collate`**: 가변 길이 시퀀스를 배치 내 최대 길이로 패딩 → `(B, 12, N_max, T)`

### 1.3 학습 파이프라인 (4단계)

#### Phase 1+2: `train.py` — 기본 Teacher/Student 학습

- **Teacher (Soft LIF)**: `TeacherSNN1d` — Conv1d(k=3) + Soft LIF, 5가지 크기 변형
  - `smallest(16,32)`, `small(24,48)`, `medium(32,64)`, `large(40,80)`, `largest(48,96)`
- **Student (Hard LIF + KD)**: `StudentSNN1d` — 동일 구조 + Hard spike + Knowledge Distillation
  - KD Loss: `a x CE + (1-a) x KL_div` (a=0.9, T_distill=2.0)
- T=20, 50 epochs, Adam, LR=1e-3
- 출력: `.pt` (PyTorch) + `.ptl` (TorchScript Lite for mobile)

#### Phase 1+2 변형: `train_kernel_sweep.py` — 커널 크기 탐색

- 채널 고정 (32, 64), 커널 크기 k=3,5,7,9 sweep
- Teacher + Student(KD) 동일 파이프라인

#### Phase 1+2 변형: `train_k11_only.py` — k=11 추가 학습

- k=11(largest) 단독 학습, 기존 결과에 병합

#### Phase 1+2 변형: `train_t3.py` — T-step sweep

- T=3 (timestep 감소 실험)
- k=3,5,7,9,11 x Teacher/Student 전부 학습
- 커널 크기 → variant 매핑: `{3:smallest, 5:small, 7:medium, 9:large, 11:largest}`

#### Phase 2.5: `export_t_sweep.py` — 모바일 export 전용

- `TraceFriendlyLIF`: `torch.jit.trace` 호환 LIF
- `ExportableStudentSNN1d`: STE 제거한 export 전용 Student
- T=5,10,15 모델들을 `.ptl`로 변환

#### Phase 3: `train_sparse.py` — 발화율 희소화

- **`LearnableHardLIFNode`**: threshold가 `nn.Parameter` → 학습 가능
- **`SparseStudentSNN1d`**: gain도 learnable, 스파이크 수집 기능
- **Firing Rate Regularization**: `(actual_rate - target_rate)^2` 손실
- Curriculum: lambda_sparse를 10 epoch에 걸쳐 점진적 증가 (0→5.0)
- **Sweep**: T={3,5,10,15} x k={3,5,7,9,11} x target_rate={0.30,0.20,0.10,0.05}
- threshold 음수 방지 clamp, gain 증가 페널티
- `ExportableSparseStudent`: TorchScript 호환 export 모델

#### CNN 베이스라인: `train_cnn1d_k9.py`

- **`Conv1dCNN`**: Conv1d(k=9) + GroupNorm + ReLU (비SNN 베이스라인)
- **`QuantizableConv1dCNN`**: BatchNorm + static INT8 quantization (qnnpack)
- 5가지 크기 변형, 200 epochs, Cosine Annealing + Early Stopping
- FP32 `.pt` + `.ptl` + Quantized `.ptl` 모두 export

### 1.4 벤치마크: `benchmark.py`

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

### 1.5 저장된 모델 (`models/`) — 187개

| 패턴 | 의미 | 대략적 수량 |
|------|------|------------|
| `cnn1d_*_sensor.pt/ptl` | CNN 베이스라인 (FP32) | ~10 |
| `qcnn1d_*_sensor.ptl` | 양자화 CNN (INT8) | ~5 |
| `snn1d_teacher_*` | Teacher SNN (다양한 T, k) | ~30+ |
| `snn1d_student_*` | Student SNN (KD 완료) | ~30+ |
| `student_kd_*` | Student KD export (.ptl) | ~20+ |
| `snn_*_T*.ptl` | Teacher SNN export | ~20+ |
| `sparse_*_T*_fr*.pt/ptl` | 희소화 Student (다양한 FR target) | ~60+ |

### 1.6 결과 파일 (`result/`)

| 파일 | 내용 |
|------|------|
| `cnn1d_k9_results.json` | CNN 베이스라인 결과 |
| `kernel_sweep_results.json` | 커널 크기별 SNN 성능 |
| `t_sweep_results.json` | T-step별 SNN 성능 |
| `benchmark_results.json` | 종합 벤치마크 (정확도, 지연시간, 에너지) |

### 1.7 전체 데이터 흐름

```
CSV (6축 IMU)
    | rate_code_zscore_sigmoid()
Spike Tensor (12, N, T) -- 확률적 이진 스파이크
    | pad_collate()
Batch (B, 12, N_max, T)
    |
+---------------------------------------------+
|  Phase 1: Teacher (Soft LIF, 연속 출력)       |
|  Phase 2: Student (Hard LIF, KD 학습)         |
|  Phase 3: Sparse (발화율 제약, threshold 학습)  |
|  Baseline: CNN / QCNN (비SNN 대조군)           |
+---------------------------------------------+
    |
.pt (학습 체크포인트) + .ptl (모바일 배포용 TorchScript Lite)
    |
benchmark.py -> 정확도/지연시간/에너지 종합 비교
```

### 1.8 핵심 설계 특징

1. **SNN의 에너지 효율성 검증**: MAC → AC 전환 시 에너지 5.1배 절감 가능성 분석 (4.6pJ → 0.9pJ)
2. **3단계 점진적 최적화**: Soft LIF Teacher → Hard spike Student (KD) → Firing Rate 희소화
3. **체계적 하이퍼파라미터 sweep**: 커널 크기(3~11), T-step(3~20), 발화율 목표(5~30%)
4. **모바일 배포 파이프라인**: `torch.jit.script/trace` → `optimize_for_mobile` → `.ptl` (Lite Interpreter)

### 1.9 디렉토리 트리

```
Thumbthing_easy_intae2/
+-- data/
|   +-- 2_ffilled_data/          # 7,596개 CSV (6축 IMU, forward-filled)
+-- models/                      # 187개 학습된 모델 (.pt, .ptl)
+-- result/                      # 실험 결과 JSON
|   +-- benchmark_results.json
|   +-- cnn1d_k9_results.json
|   +-- kernel_sweep_results.json
|   +-- t_sweep_results.json
+-- lif_module.py                # LIF 뉴런 (Soft Surrogate)
+-- model.py                     # SNN 모델 (SimpleSNN1d, HardLIFNode, StdpSNN1d)
+-- dataset.py                   # 스파이크 인코딩 + Dataset + Collate
+-- train.py                     # Phase 1+2: Teacher/Student (크기 변형)
+-- train_kernel_sweep.py        # Phase 1+2: 커널 크기 sweep (k=3,5,7,9)
+-- train_k11_only.py            # Phase 1+2: k=11 추가 학습
+-- train_t3.py                  # Phase 1+2: T=3 sweep
+-- train_t_sweep.py             # Phase 1+2: T-step sweep
+-- train_sparse.py              # Phase 3: 발화율 희소화
+-- train_cnn1d_k9.py            # CNN 베이스라인 + INT8 양자화
+-- export_t_sweep.py            # T-sweep 모델 .ptl export
+-- benchmark.py                 # 종합 벤치마크 (정확도/지연/에너지)
+-- *.log                        # 학습 로그 파일
```

---

## 2. dataset.py 상세 분석

### 2.1 `rate_code_zscore_sigmoid()` — 스파이크 인코딩 함수

```
원본 신호 (6축) -> 12채널 분리 -> Z-score 정규화 -> Sigmoid -> 확률적 스파이크 생성
```

#### 단계별 동작

**1) 6축 → 12채널 분리**

각 축(`ax, ay, az, gx, gy, gz`)의 신호를 양수/음수로 분리:
- `pos = clip(signal, 0, inf)` — 양의 활성만 추출
- `neg = clip(-signal, 0, inf)` — 음의 활성만 추출

→ 6축 x 2(양/음) = **12채널** (`ax_pos, ax_neg, ay_pos, ...`)

이유: 생물학적 뉴런은 음수 값을 직접 표현할 수 없어서, ON/OFF 채널로 분리하는 것이 SNN의 표준적인 인코딩 방식.

**2) Z-score 정규화**

```python
z = (s - mean) / (std + 1e-8)
```

각 채널별로 평균 0, 표준편차 1로 정규화. `1e-8`은 분모가 0이 되는 것을 방지.

**3) Sigmoid → 발화 확률**

```python
prob = sigmoid(z)  # scipy.special.expit
```

Z-score를 0~1 범위의 확률로 변환:
- z가 크면(신호가 강하면) → prob ≈ 1 (높은 발화 확률)
- z가 작으면(신호가 약하면) → prob ≈ 0 (낮은 발화 확률)
- z = 0 (평균) → prob = 0.5

**4) 확률적 스파이크 생성**

```python
spike_seq = (rand(len(s), T) < prob[:, None]).astype(uint8)
```

각 시점(N)에서 T번의 독립적인 베르누이 시행을 수행:
- `rand(N, T)` — 0~1 균일 난수 생성
- `< prob` — 확률보다 작으면 1(발화), 크면 0(비발화)

같은 입력이라도 **매 호출마다 다른 스파이크 패턴**이 생성됨 (Rate Coding의 핵심).

**5) 최종 출력**

```python
spike_tensor = np.stack(spikes, axis=1)  # (N, 12, T)
```

- `N`: 시퀀스 길이 (시계열 샘플 수)
- `12`: 채널 수 (6축 x 양/음)
- `T`: 타임스텝 수 (스파이크 반복 횟수)

### 2.2 `FileLevelSpikeDataset` — PyTorch Dataset

#### `__init__()`

- `data/2_ffilled_data/` 폴더를 스캔하여 CSV 파일 목록을 구성
- **파일명 기반 라벨 매핑** — `sensor`가 포함된 파일만 사용:

| 파일명 패턴 | 라벨 |
|------------|------|
| `swipe_up` + `sensor` | 0 |
| `swipe_down` + `sensor` | 1 |
| `flick_up` + `sensor` | 2 |
| `flick_down` + `sensor` | 3 |

- `ois_2m`, `ois_20m` 파일은 `sensor`가 아니므로 **자동으로 제외**됨
- 이 시점에서는 CSV를 읽지 않고, **(경로, 라벨)** 튜플만 저장 → **Lazy Loading**

#### `__getitem__()`

호출될 때마다:

```python
df = pd.read_csv(path)                          # CSV 로드
spike_tensor, _ = self.spike_func(df, T=self.T)  # (N, 12, T) 스파이크 생성
spike_tensor = spike_tensor.transpose(1, 0, 2)   # (12, N, T) 로 전치
return torch.tensor(spike_tensor), torch.tensor(label)
```

1. CSV 파일을 그때그때 읽음 (메모리 절약)
2. `rate_code_zscore_sigmoid()`로 스파이크 인코딩 수행
3. 차원 순서를 `(N, 12, T)` → `(12, N, T)`로 변경 — Conv1d의 입력 형식 `(C, L, ...)`에 맞춤
4. **매 호출마다 난수가 다르므로**, 같은 샘플도 다른 스파이크 패턴 → **자연스러운 Data Augmentation 효과**

### 2.3 `pad_collate()` — 배치 패딩 함수

각 CSV 파일의 시퀀스 길이(N)가 다르므로, 배치 내 가장 긴 시퀀스에 맞춰 나머지를 **0으로 패딩**:

```python
def pad_collate(batch):
    xs, ys = zip(*batch)
    max_n = max(x.shape[1] for x in xs)          # 배치 내 최대 시퀀스 길이
    out = torch.zeros(B, 12, max_n, T)            # 0으로 채운 텐서
    for i, x in enumerate(xs):
        out[i, :, :n, :] = x                      # 실제 데이터를 앞쪽에 배치
    return out, torch.tensor(ys)
```

- 출력 형태: `(B, 12, N_max, T)` — 4D 텐서
- 0 패딩은 스파이크 "비발화"와 동일하므로 모델에 자연스럽게 호환됨

### 2.4 dataset.py 전체 흐름 요약

```
CSV 파일 (Timestamp, ax, ay, az, gx, gy, gz)
    |
    +- FileLevelSpikeDataset.__init__()
    |   -> 파일명에서 라벨 추출, (경로, 라벨) 목록 저장
    |
    +- __getitem__(idx) 호출 시:
    |   -> pd.read_csv()로 CSV 로드
    |   -> rate_code_zscore_sigmoid():
    |       6축 -> 12채널(양/음) -> Z-score -> Sigmoid -> 확률적 스파이크
    |   -> (12, N, T) 텐서 반환
    |
    +- DataLoader + pad_collate():
        -> 배치 내 최대 길이로 0-패딩
        -> (B, 12, N_max, T) 텐서로 배치 구성
            -> 이후 SNN/CNN 모델의 입력으로 사용
```

---

## 3. 학습 모델 종류

총 5가지 모델이 학습/변환됨:

| # | 모델 | 학습 방식 | 학습 스크립트 |
|---|------|----------|-------------|
| 1 | **Teacher SNN (SG-SNN)** | Soft LIF, CE Loss 직접 학습 | `train.py`, `train_kernel_sweep.py`, `train_t3.py` |
| 2 | **Student SNN (BS-SNN)** | Hard LIF, Teacher로부터 KD 학습 | 위와 동일 (Phase 2) |
| 3 | **Sparse Student SNN** | Student 가중치 로드 후, 발화율 제약으로 fine-tuning | `train_sparse.py` |
| 4 | **CNN** | Conv1d + GroupNorm + ReLU, CE Loss | `train_cnn1d_k9.py` |
| 5 | **QCNN** | CNN 학습 완료 후 INT8 양자화 (post-training) | `train_cnn1d_k9.py` 내에서 변환 |

QCNN은 별도 학습이 아니라 CNN 가중치를 양자화한 것이고, Sparse Student도 Student 가중치를 로드해서 fine-tuning하는 것이므로, **처음부터 독립적으로 학습되는 모델은 3개**(Teacher, Student, CNN).

```
Teacher SNN --KD--> Student SNN --FR regularization--> Sparse Student SNN
CNN --quantization--> QCNN
```

---

## 4. train_sparse.py 실행 흐름

### 4.1 설정

```
T_VALUES = [3, 5, 10, 15]          -> 4개
KERNEL_SIZES = [3, 5, 7, 9, 11]    -> 5개 (variant: smallest~largest)
TARGET_RATES = [0.3, 0.2, 0.1, 0.05] -> 4개
```

**총 조합: 4 x 5 x 4 = 80개 실험**

### 4.2 T값별 루프

각 T값마다:

**1) 데이터셋 생성** — 해당 T값으로 스파이크 인코딩

```python
dataset = FileLevelSpikeDataset(DATA_DIR, rate_code_zscore_sigmoid, T=T)
```

- 80/20 train/val split (seed=42 고정)
- DataLoader 구성 (batch_size=4)

**2) 커널 크기별 루프**

**3) target_rate별 루프**

### 4.3 개별 실험 수행 과정 (80회 반복)

#### Step A: KD Student 가중치 로드

```
models/snn1d_student_{variant}_T{T}.pt  <- 우선 탐색
models/student_kd1d_{variant}_sensor.pt <- fallback
```

#### Step B: 초기 발화율 측정

```python
fr1_init, fr2_init = measure_firing_rates(model, val_loader, DEVICE)
```

#### Step C: 발화율 억제 학습 — 40 epochs

**Optimizer 설정:**
- conv/fc 가중치: LR = 5e-4
- threshold/gain: LR = 5e-3 (10배 높음)
- CosineAnnealingLR 스케줄러

**매 epoch마다:**

```
Loss = CE_loss
     + lambda_sparse x firing_rate_loss    <- 핵심: 발화율을 target으로 끌어내림
     + 10.0 x thresh_penalty               <- threshold 음수 방지
     + 0.5 x gain_penalty                  <- gain이 초기값(3.0) 이상 커지는 것 방지
```

- `lambda_sparse`: 0에서 시작하여 10 epoch에 걸쳐 5.0까지 선형 증가 (curriculum)
- threshold는 매 step 후 `clamp_(min=0.01)` — 최솟값 보장
- gradient clipping = 1.0
- best model은 `train_acc + val_acc` 기준으로 저장

#### Step D: 최종 발화율 측정 및 저장

```python
# .pt 저장
torch.save(best_state, "models/sparse_{variant}_T{T}_fr{rate}.pt")

# .ptl export (TorchScript Lite for mobile)
export_sparse_ptl(model, T, ks, "sparse_{variant}_T{T}_fr{rate}.ptl")
```

#### Step E: 결과 기록

각 실험마다 수집: T, variant, kernel_size, target_rate, best_score, 초기/최종 발화율, threshold, gain

### 4.4 선행 작업

`train_sparse.py`를 실행하기 위해 필요한 선행 작업:

```
1) train_t3.py     -> T=3 에 대한 Teacher/Student 생성
2) train_t_sweep.py -> T=5,10,15 에 대한 Teacher/Student 생성
3) train_sparse.py  -> 위에서 만든 Student를 로드하여 발화율 희소화
```

5개 variant x 4개 T = **20개 Student 모델**이 미리 존재해야 전체 80개 실험이 수행됨.

### 4.5 GPU 관련

GPU 없이도 실행 가능 (자동 분기). 다만 80개 실험이므로:

| 환경 | 예상 소요 시간 |
|------|--------------|
| GPU (RTX A5000) | ~3시간 |
| CPU only | 수 시간 ~ 반나절 |

### 4.6 실행 결과

80/80 전체 실험 완료 (RTX A5000에서 약 3시간 소요).

---

## 5. 벤치마크 결과

### 5.1 모델 유형별 Best 구성 비교

| Type | Config | Params | Val Acc | Lat(ms) | E_actual(nJ) | E_poten(nJ) | vs CNN |
|------|--------|--------|---------|---------|-------------|-------------|--------|
| **CNN** | medium k=9 T=1 | 22,436 | **0.9980** | 0.075 | 8,156.6 | 8,156.6 | 1.00x |
| **QCNN** | largest k=9 T=1 | 47,088 | **0.9980** | 0.104 | 755.9 | 755.9 | **0.09x** |
| **Teacher** | large k=9 T=20 | 22,244 | 0.9901 | 3.792 | 163,132.9 | 163,132.9 | 20.00x |
| **Student** | large k=9 T=20 | 22,244 | 0.9901 | 2.329 | 163,132.9 | 30,940.2 | 20.00x |
| **Sparse** | large k=9 T=3 fr=30% | 22,247 | 0.9862 | 0.305 | 24,469.9 | 4,941.9 | 3.00x |

---

### 5.2 QCNN Best vs Sparse SNN Best

|  | QCNN Best | Sparse SNN Best |
|--|-----------|-----------------|
| **Config** | largest k=9 | large k=9 T=3 fr=30% |
| **Params** | 47,088 | 22,247 |
| **Size** | 183.9 KB | 92.9 KB |
| **Val Accuracy** | **0.9980** | 0.9862 |
| **Latency** | **0.104 ms** | 0.305 ms |
| **E_actual (nJ)** | **755.9** | 24,469.9 |
| **E_potential (nJ)** | **755.9** | 4,941.9 |
| **Firing Rate** | 100% (dense) | ~27% (sparse) |

비교 요약:
- **Accuracy**: QCNN이 1.18%p 우세 (99.80% vs 98.62%)
- **Latency**: QCNN이 2.9배 빠름 (0.1ms vs 0.3ms)
- **Energy (현재 구현)**: QCNN이 32.4배 효율적 — INT8 양자화의 압도적 우위
- **Energy (AC 연산 활용 시)**: 여전히 QCNN이 6.5배 효율적

현재 기준으로는 **QCNN이 모든 지표에서 우세**. Sparse SNN이 에너지 이점을 가지려면 뉴로모픽 전용 하드웨어(Loihi, SpiNNaker 등)에서 AC 연산을 네이티브로 지원해야 하며, 현재의 von Neumann 아키텍처(CPU/GPU)에서는 INT8 양자화가 더 실용적.

---

### 5.3 QCNN vs Sparse SNN — 모델 크기별 비교

#### 성능 비교표 (에너지 포함)

| Variant | QCNN Params | QCNN Acc | QCNN Lat(ms) | QCNN E_act(nJ) | QCNN E_pot(nJ) | Sparse Params | Sparse Acc | Sparse Lat(ms) | Sparse E_act(nJ) | Sparse E_pot(nJ) |
|---------|------------|----------|-------------|----------------|----------------|--------------|------------|---------------|------------------|------------------|
| smallest | 6,480 | **0.9862** | **0.086** | **102.7** | **102.7** | 7,655 | 0.7870 | 0.292 | 8,159.0 | 1,693.9 |
| small | 13,176 | **0.9961** | **0.090** | **210.0** | **210.0** | 12,519 | 0.9625 | 0.424 | 13,596.0 | 2,409.1 |
| medium | 22,176 | **0.9961** | **0.093** | **354.6** | **354.6** | 17,383 | 0.9822 | 0.312 | 19,033.0 | 3,360.2 |
| large | 33,480 | **0.9961** | **0.098** | **536.6** | **536.6** | 22,247 | 0.9862 | 0.305 | 24,469.9 | 4,941.9 |
| largest | 47,088 | **0.9980** | **0.104** | **755.9** | **755.9** | 27,111 | 0.9862 | 0.315 | 29,906.9 | 6,058.0 |

#### 상대값 요약 (QCNN 대비 Sparse)

| Variant | Acc 차이 | Latency | E_actual | E_potential | 모델 크기 |
|---------|---------|---------|----------|-------------|----------|
| smallest | -19.92%p | 3.4x 느림 | 79.5x 많음 | 16.5x 많음 | 1.42x 큼 |
| small | -3.36%p | 4.7x 느림 | 64.7x 많음 | 11.5x 많음 | 1.07x 비슷 |
| medium | -1.39%p | 3.4x 느림 | 53.7x 많음 | 9.5x 많음 | **0.85x 작음** |
| large | -0.99%p | 3.1x 느림 | 45.6x 많음 | 9.2x 많음 | **0.71x 작음** |
| largest | -1.18%p | 3.0x 느림 | 39.6x 많음 | 8.0x 많음 | **0.61x 작음** |

핵심 포인트:
- **정확도/레이턴시/에너지** 모두 QCNN이 우세 — 모든 크기에서 일관적
- **모델 크기(KB)**는 medium 이상부터 Sparse가 더 작음 (커널이 작고 양자화 오버헤드 없음)
- **smallest에서 Sparse 정확도 급락** (78.7%) — 작은 모델에서는 발화율 억제가 성능을 크게 떨어뜨림
- **large/largest에서 Sparse 정확도 98.6%** — 큰 모델은 희소화에도 견딤, QCNN과 격차 ~1%p

---

### 5.4 모든 모델 유형 x 크기별 종합 비교

- SG-SNN = Soft Gradient SNN (Teacher, Soft LIF)
- BS-SNN = Binary Spike SNN (Student, Hard LIF + KD)
- Sparse = Sparse SNN (BS-SNN + Firing Rate Regularization)

#### Accuracy (Val Acc)

| Variant | CNN | QCNN | SG-SNN | BS-SNN | Sparse |
|---------|-----|------|--------|--------|--------|
| smallest | 0.9862 | 0.9862 | 0.9684 | 0.7022 | 0.7870 |
| small | 0.9961 | 0.9961 | 0.9822 | 0.9763 | 0.9625 |
| medium | **0.9980** | 0.9961 | 0.9842 | 0.9862 | 0.9822 |
| large | 0.9941 | 0.9961 | **0.9901** | **0.9901** | 0.9862 |
| largest | 0.9961 | **0.9980** | **0.9901** | 0.9882 | 0.9862 |

#### Latency mean (ms)

| Variant | CNN | QCNN | SG-SNN | BS-SNN | Sparse |
|---------|-----|------|--------|--------|--------|
| smallest | **0.083** | 0.086 | 3.683 | 2.295 | 0.292 |
| small | **0.073** | 0.090 | 4.902 | 2.251 | 0.424 |
| medium | **0.075** | 0.093 | 0.592 | 0.363 | 0.312 |
| large | **0.080** | 0.098 | 3.792 | 2.329 | 0.305 |
| largest | **0.086** | 0.104 | 3.870 | 0.374 | 0.315 |

#### Energy Actual (nJ)

| Variant | CNN | QCNN | SG-SNN | BS-SNN | Sparse |
|---------|-----|------|--------|--------|--------|
| smallest | 2,361 | **103** | 54,393 | 54,393 | 8,159 |
| small | 4,830 | **210** | 90,640 | 90,640 | 13,596 |
| medium | 8,157 | **355** | 19,033 | 19,033 | 19,033 |
| large | 12,342 | **537** | 163,133 | 163,133 | 24,470 |
| largest | 17,386 | **756** | 199,380 | 29,907 | 29,907 |

#### Energy Potential (nJ) — AC 연산 활용 시

| Variant | CNN | QCNN | SG-SNN | BS-SNN | Sparse |
|---------|-----|------|--------|--------|--------|
| smallest | 2,361 | **103** | 54,393 | 11,445 | 1,694 |
| small | 4,830 | **210** | 90,640 | 17,448 | 2,409 |
| medium | 8,157 | **355** | 19,033 | 3,571 | 3,360 |
| large | 12,342 | **537** | 163,133 | 30,940 | 4,942 |
| largest | 17,386 | **756** | 199,380 | 5,908 | 6,058 |

#### 통합 테이블

| Variant | Type | T | k | FR% | Params | Acc | Lat(ms) | E_act(nJ) | E_pot(nJ) |
|---------|------|---|---|-----|--------|-----|---------|-----------|-----------|
| smallest | CNN | 1 | 9 | - | 6,612 | 0.9862 | 0.083 | 2,361.4 | 2,361.4 |
| | QCNN | 1 | 9 | - | 6,480 | 0.9862 | 0.086 | 102.7 | 102.7 |
| | SG-SNN | 20 | 3 | - | 7,652 | 0.9684 | 3.683 | 54,393.3 | 54,393.3 |
| | BS-SNN | 20 | 3 | - | 7,652 | 0.7022 | 2.295 | 54,393.3 | 11,445.4 |
| | Sparse | 3 | 3 | 30 | 7,655 | 0.7870 | 0.292 | 8,159.0 | 1,693.9 |
| small | CNN | 1 | 9 | - | 13,372 | 0.9961 | 0.073 | 4,829.8 | 4,829.8 |
| | QCNN | 1 | 9 | - | 13,176 | 0.9961 | 0.090 | 210.0 | 210.0 |
| | SG-SNN | 20 | 5 | - | 12,516 | 0.9822 | 4.902 | 90,639.9 | 90,639.9 |
| | BS-SNN | 20 | 5 | - | 12,516 | 0.9763 | 2.251 | 90,639.9 | 17,447.9 |
| | Sparse | 3 | 5 | 5 | 12,519 | 0.9625 | 0.424 | 13,596.0 | 2,409.1 |
| medium | CNN | 1 | 9 | - | 22,436 | 0.9980 | 0.075 | 8,156.6 | 8,156.6 |
| | QCNN | 1 | 9 | - | 22,176 | 0.9961 | 0.093 | 354.6 | 354.6 |
| | SG-SNN | 3 | 7 | - | 17,380 | 0.9842 | 0.592 | 19,033.0 | 19,033.0 |
| | BS-SNN | 3 | 7 | - | 17,380 | 0.9862 | 0.363 | 19,033.0 | 3,571.2 |
| | Sparse | 3 | 7 | 5 | 17,383 | 0.9822 | 0.312 | 19,033.0 | 3,360.2 |
| large | CNN | 1 | 9 | - | 33,804 | 0.9941 | 0.080 | 12,342.0 | 12,342.0 |
| | QCNN | 1 | 9 | - | 33,480 | 0.9961 | 0.098 | 536.6 | 536.6 |
| | SG-SNN | 20 | 9 | - | 22,244 | 0.9901 | 3.792 | 163,132.9 | 163,132.9 |
| | BS-SNN | 20 | 9 | - | 22,244 | 0.9901 | 2.329 | 163,132.9 | 30,940.2 |
| | Sparse | 3 | 9 | 30 | 22,247 | 0.9862 | 0.305 | 24,469.9 | 4,941.9 |
| largest | CNN | 1 | 9 | - | 47,476 | 0.9961 | 0.086 | 17,385.8 | 17,385.8 |
| | QCNN | 1 | 9 | - | 47,088 | 0.9980 | 0.104 | 755.9 | 755.9 |
| | SG-SNN | 20 | 11 | - | 27,108 | 0.9901 | 3.870 | 199,379.5 | 199,379.5 |
| | BS-SNN | 3 | 11 | - | 27,108 | 0.9882 | 0.374 | 29,906.9 | 5,907.9 |
| | Sparse | 3 | 11 | 30 | 27,111 | 0.9862 | 0.315 | 29,906.9 | 6,058.0 |

#### 패턴 요약

- **Accuracy**: CNN/QCNN > SG-SNN > BS-SNN/Sparse (smallest에서 SNN 급락)
- **Latency**: CNN ≈ QCNN >> Sparse > BS-SNN >> SG-SNN
- **Energy**: **QCNN 압도적 1위**, Sparse가 CNN보다 낮은 유일한 SNN (potential 기준, small 이하)
- **smallest SNN 문제**: BS-SNN 70%, Sparse 79% — 모델이 너무 작으면 Hard spike + 희소화가 정보를 파괴함

---

## 6. 입력 데이터 시간 길이 분석

### 6.1 전체 통계

전체 2,532개 sensor 파일 분석. 샘플링 간격: 10ms (100Hz).

| 항목 | 값 |
|------|-----|
| **샘플링 주기** | 10ms (100Hz) |
| **중간값** | **340ms (0.34초)** |
| **평균** | 572ms (0.57초) |
| **최소** | 70ms |
| **최대** | 22,690ms (이상치) |
| **90%가 이 이내** | 1,000ms (1초) |

### 6.2 시퀀스 길이 (행 수)

| 항목 | 값 |
|------|-----|
| min | 8 |
| max | 2,270 |
| mean | 58.2 |
| median | 35.0 |

### 6.3 시간 분포

| 시간 이하 | 비율 |
|-----------|------|
| 100ms (0.1s) | 0.9% |
| 200ms (0.2s) | 37.5% |
| 500ms (0.5s) | 53.2% |
| 800ms (0.8s) | 67.5% |
| 1,000ms (1.0s) | 90.4% |
| 1,500ms (1.5s) | 98.9% |
| 2,000ms (2.0s) | 99.0% |

### 6.4 제스처별 시간 길이

| 제스처 | 평균 | 중간값 | 범위 |
|--------|------|--------|------|
| **swipe_up** | 204ms (0.20s) | 190ms | 120~810ms |
| **swipe_down** | 181ms (0.18s) | 180ms | 80~440ms |
| **flick_up** | 749ms (0.75s) | 800ms | 70~1,530ms |
| **flick_down** | 1,155ms (1.15s) | 930ms | 80~22,690ms |

**요약**: swipe 계열은 약 **0.2초**, flick 계열은 약 **0.8~1.2초**의 짧은 제스처 데이터. 대부분의 샘플이 **1초 이내**(90.4%)이며, 시퀀스 길이로는 중간값 35행(= 350ms) 정도.
