# Section 4.3.1 On-Device 테스트 결과 분석

> Phase 1→2→3→4 동일 조건 비교 (T=10, k=9, FR=5%, Android On-Device CPU 8 threads)
> 테스트 기기: Samsung Galaxy S24+, Android 14, PyTorch 2.6.0 TorchScript Lite
> 63,300 forward passes per model, 2,532 gesture samples (4 classes)

---

## 1. 테스트 대상 모델 (20개 SNN + 5개 QCNN)

### Phase 1: Teacher (Soft LIF, T=10)
| Variant | 파일명 | Size |
|---------|--------|------|
| smallest | snn1d_teacher_smallest_T10.ptl | 47.8 KB |
| small | snn1d_teacher_small_T10.ptl | 73.9 KB |
| medium | snn1d_teacher_medium_T10.ptl | 109.2 KB |
| large | snn1d_teacher_large_T10.ptl | 153.3 KB |
| largest | snn1d_teacher_largest_T10.ptl | 206.6 KB |

### Phase 2: Student KD (Hard LIF + STE, T=10)
| Variant | 파일명 | Size |
|---------|--------|------|
| smallest | snn1d_student_smallest_T10.ptl | 44.6 KB |
| small | snn1d_student_small_T10.ptl | 70.8 KB |
| medium | snn1d_student_medium_T10.ptl | 106.0 KB |
| large | snn1d_student_large_T10.ptl | 150.2 KB |
| largest | snn1d_student_largest_T10.ptl | 203.5 KB |

### Phase 3: Sparse Student (Learnable Threshold + FR Reg, T=10, FR=5%)
| Variant | 파일명 | Size |
|---------|--------|------|
| smallest | sparse_smallest_T10_fr05.ptl | 52.0 KB |
| small | sparse_small_T10_fr05.ptl | 77.9 KB |
| medium | sparse_medium_T10_fr05.ptl | 113.3 KB |
| large | sparse_large_T10_fr05.ptl | 157.4 KB |
| largest | sparse_largest_T10_fr05.ptl | 210.7 KB |

### Phase 4: QSparse (INT8 Static Quantization, T=10, FR=5%)
| Variant | 파일명 | Size |
|---------|--------|------|
| smallest | qsparse_smallest_T10_fr05.ptl | 40.2 KB |
| small | qsparse_small_T10_fr05.ptl | 46.7 KB |
| medium | qsparse_medium_T10_fr05.ptl | 55.9 KB |
| large | qsparse_large_T10_fr05.ptl | 67.1 KB |
| largest | qsparse_largest_T10_fr05.ptl | 80.9 KB |

### 비교 기준: QCNN (INT8 Static Quantization)
| Variant | 파일명 | Size |
|---------|--------|------|
| smallest | qcnn1d_smallest_static.ptl | 16.5 KB |
| small | qcnn1d_small_static.ptl | 22.9 KB |
| medium | qcnn1d_medium_static.ptl | 31.9 KB |
| large | qcnn1d_large_static.ptl | 42.9 KB |
| largest | qcnn1d_largest_static.ptl | 56.4 KB |

### 추가 참조: EE QSparse (Early Exit, T=2, FR=5%)
| Variant | 파일명 | Size |
|---------|--------|------|
| smallest | ee_qsparse_smallest_T2_fr05.ptl | 24.1 KB |
| small | ee_qsparse_small_T2_fr05.ptl | 30.7 KB |
| medium | ee_qsparse_medium_T2_fr05.ptl | 39.9 KB |
| large | ee_qsparse_large_T2_fr05.ptl | 51.2 KB |
| largest | ee_qsparse_largest_T2_fr05.ptl | 64.8 KB |

---

## 2. Accuracy (%)

| Variant | QCNN | Ph1 Teacher | Ph2 Student | Ph3 Sparse | Ph4 QSparse T10 | EE QSparse T2 |
|---------|------|-------------|-------------|------------|-----------------|---------------|
| smallest | 95.34 | **98.50** | 98.18 | 98.10 | 97.99 | 96.41 |
| small | 97.24 | 98.10 | **98.78** | 98.78 | 98.82 | 95.77 |
| medium | 98.62 | 98.42 | **98.74** | 98.78 | 98.66 | 96.96 |
| large | 98.66 | 98.70 | 98.78 | 98.66 | **98.82** | 97.16 |
| largest | 98.89 | **99.05** | 98.54 | 98.78 | 98.82 | 97.55 |

### 제스처별 상세 Accuracy

#### smallest variant
| Model | Swipe Up | Swipe Down | Flick Up | Flick Down |
|-------|----------|------------|----------|------------|
| QCNN | 98.42 | 96.84 | 87.36 | 98.74 |
| Teacher | 99.21 | 98.89 | 97.63 | 98.26 |
| Student | 97.47 | 99.21 | 97.63 | 98.42 |
| Sparse | 97.00 | 98.74 | 98.10 | 98.58 |
| QSparse T10 | 97.79 | 98.42 | 97.47 | 98.26 |
| EE QSparse T2 | 96.21 | 94.31 | 97.63 | 97.47 |

#### largest variant
| Model | Swipe Up | Swipe Down | Flick Up | Flick Down |
|-------|----------|------------|----------|------------|
| QCNN | 99.53 | 98.89 | 97.79 | 99.37 |
| Teacher | 99.84 | 99.05 | 98.26 | 99.05 |
| Student | 99.84 | 98.10 | 98.42 | 97.79 |
| Sparse | 99.53 | 99.37 | 98.10 | 98.10 |
| QSparse T10 | 99.68 | 99.21 | 98.10 | 98.26 |
| EE QSparse T2 | 98.26 | 97.16 | 97.31 | 97.47 |

### Accuracy 분석

- **Phase 1→2→3→4(T=10) 정확도 변화가 0.5%p 이내**: Teacher(98.10~99.05%) → Student(98.18~98.78%) → Sparse(98.10~98.78%) → QSparse(97.99~98.82%)로, 동일 T=10 조건에서 sparsification과 INT8 양자화가 정확도를 실질적으로 훼손하지 않음을 On-Device에서 확인
- **모든 SNN Phase(T=10)가 QCNN보다 높거나 동등**: smallest에서 SNN 전 Phase가 97.99~98.50%로 QCNN(95.34%)을 2.6~3.2%p 초과
- **EE QSparse(T=2)만 1~2%p 하락**: T=10→T=2 timestep 축소로 인한 정보 손실이 원인이며, 그럼에도 95.77~97.55%로 실용적 수준 유지
- **QCNN smallest(95.34%)의 flick_up 87.36%가 가장 약한 지점**: INT8 양자화가 CNN에 더 큰 영향을 미치며, SNN은 flick_up에서도 96~98%로 안정적

---

## 3. Latency (avg_forward_ms)

| Variant | QCNN | Ph1 Teacher | Ph2 Student | Ph3 Sparse | Ph4 QSparse T10 | EE QSparse T2 |
|---------|------|-------------|-------------|------------|-----------------|---------------|
| smallest | 0.424 | 5.318 | 3.846 | 5.816 | 5.480 | **0.795** |
| small | 0.542 | 8.411 | 6.061 | 7.401 | 9.352 | **1.470** |
| medium | 0.693 | 11.933 | 9.378 | 10.468 | 12.184 | **1.396** |
| large | 0.867 | 13.423 | 11.597 | 12.083 | 14.928 | **1.574** |
| largest | 1.091 | 17.010 | 14.051 | 15.192 | 17.727 | **1.909** |

### Phase별 Latency 변화 (largest 기준)
```
Phase 1 Teacher:     17.010 ms  (baseline)
Phase 2 Student:     14.051 ms  (-17.4%)
Phase 3 Sparse:      15.192 ms  (-10.7%)  ← Teacher 대비
Phase 4 QSparse T10: 17.727 ms  (+4.2%)   ← 오히려 증가
EE QSparse T2:        1.909 ms  (-88.8%)  ← 극적 감소
QCNN:                 1.091 ms  (참고)
```

### Latency 분석

- **표준 CPU에서 Sparse(Phase 3)는 latency 개선이 미미하거나 오히려 증가**: spike=0일 때 연산을 건너뛰는 최적화가 표준 CPU에는 없으므로, Learnable Threshold의 추가 연산이 오버헤드로 작용
- **QSparse(Phase 4, T=10)도 Teacher보다 느림**: INT8 양자화의 qnnpack 백엔드가 소규모 Conv1d에서는 FP32 대비 유리하지 않으며, QuantStub/DeQuantStub 변환 오버헤드가 추가됨
- **Student(Phase 2)가 가장 빠른 T=10 모델**: Hard LIF의 단순 비교 연산이 Soft LIF의 sigmoid 연산보다 효율적
- **EE QSparse(T=2)만이 실질적 latency 개선 달성**: timestep 5배 감소(T=10→2) + Early Exit으로 QCNN의 1.5~1.9x 수준까지 접근

### QCNN 대비 Latency 배율
| Variant | Teacher | Student | Sparse | QSparse T10 | EE QSparse T2 |
|---------|---------|---------|--------|-------------|---------------|
| smallest | 12.5x | 9.1x | 13.7x | 12.9x | **1.87x** |
| small | 15.5x | 11.2x | 13.7x | 17.3x | **2.71x** |
| medium | 17.2x | 13.5x | 15.1x | 17.6x | **2.01x** |
| large | 15.5x | 13.4x | 13.9x | 17.2x | **1.82x** |
| largest | 15.6x | 12.9x | 13.9x | 16.2x | **1.75x** |

---

## 4. Battery / Energy (energy_per_inference_uAs)

| Variant | QCNN | Ph1 Teacher | Ph2 Student | Ph3 Sparse | Ph4 QSparse T10 | EE QSparse T2 |
|---------|------|-------------|-------------|------------|-----------------|---------------|
| smallest | 0.075 | 1.351 | 0.962 | 0.554 | 1.262 | **0.234** |
| small | 0.174 | 1.358 | 0.812 | 0.974 | 0.946 | **0.162** |
| medium | 0.215 | 1.091 | 0.760 | 0.879 | 1.027 | **0.258** |
| large | 0.180 | 1.471 | 1.052 | 1.434 | 1.512 | **0.336** |
| largest | 0.279 | 1.954 | 1.490 | 1.521 | 2.090 | **0.390** |

### Phase별 Energy 변화 (largest 기준)
```
Phase 1 Teacher:     1.954 uAs  (baseline)
Phase 2 Student:     1.490 uAs  (-23.7%)
Phase 3 Sparse:      1.521 uAs  (-22.2%)
Phase 4 QSparse T10: 2.090 uAs  (+7.0%)   ← 오히려 증가
EE QSparse T2:       0.390 uAs  (-80.0%)  ← 극적 감소
QCNN:                0.279 uAs  (참고)
```

### Energy 분석

- **Student(Phase 2)가 Teacher 대비 에너지 절감**: Hard LIF의 단순 연산이 Soft LIF의 sigmoid보다 에너지 효율적 (smallest: -28.8%, largest: -23.7%)
- **Sparse(Phase 3)는 Student와 유사한 수준**: 표준 CPU에서 spike=0 스킵이 불가하므로, 발화율 감소가 직접적 에너지 절감으로 연결되지 않음. 단, smallest에서는 0.554 uAs로 Teacher 대비 59% 감소 (모델 크기 효과)
- **QSparse T10(Phase 4)은 오히려 에너지 증가**: INT8 양자화 + QuantStub/DeQuantStub 오버헤드가 소형 모델에서 FP32 대비 불리
- **EE QSparse T2만이 실질적 에너지 절감**: T=2로 timestep을 줄여 80% 에너지 감소 달성

### QCNN 대비 Energy 배율
| Variant | Teacher | Student | Sparse | QSparse T10 | EE QSparse T2 |
|---------|---------|---------|--------|-------------|---------------|
| smallest | 18.0x | 12.8x | 7.4x | 16.8x | **3.12x** |
| small | 7.8x | 4.7x | 5.6x | 5.4x | **0.93x** |
| medium | 5.1x | 3.5x | 4.1x | 4.8x | **1.20x** |
| large | 8.2x | 5.8x | 8.0x | 8.4x | **1.87x** |
| largest | 7.0x | 5.3x | 5.5x | 7.5x | **1.40x** |

> EE QSparse small(0.162 uAs)이 QCNN small(0.174 uAs)보다 **에너지 효율이 더 좋음** (0.93x)

---

## 5. 모델 크기 비교

| Variant | QCNN | Teacher | Student | Sparse | QSparse T10 | EE QSparse T2 |
|---------|------|---------|---------|--------|-------------|---------------|
| smallest | 16.5 KB | 47.8 KB | 44.6 KB | 52.0 KB | 40.2 KB | 24.1 KB |
| small | 22.9 KB | 73.9 KB | 70.8 KB | 77.9 KB | 46.7 KB | 30.7 KB |
| medium | 31.9 KB | 109.2 KB | 106.0 KB | 113.3 KB | 55.9 KB | 39.9 KB |
| large | 42.9 KB | 153.3 KB | 150.2 KB | 157.4 KB | 67.1 KB | 51.2 KB |
| largest | 56.4 KB | 206.6 KB | 203.5 KB | 210.7 KB | 80.9 KB | 64.8 KB |

### 크기 변화 (largest 기준)
```
Phase 1 Teacher:     206.6 KB  (baseline)
Phase 2 Student:     203.5 KB  (-1.5%)    ← 동일 아키텍처
Phase 3 Sparse:      210.7 KB  (+2.0%)    ← Learnable params 추가
Phase 4 QSparse T10:  80.9 KB  (-60.8%)   ← INT8 양자화 효과
EE QSparse T2:        64.8 KB  (-68.6%)   ← T=2 + INT8
QCNN:                 56.4 KB  (참고)
```

---

## 6. 종합 분석 및 Section 4.3.1 시사점

### 6.1 Sparsification의 실질적 효과 (표준 CPU 기준)

Phase 3 Sparse는 발화율을 42%→7%로 낮춰 이론적으로 90~93%의 연산을 스킵할 수 있지만, **표준 모바일 CPU(ARM Cortex)에서는 이 스킵이 자동으로 반영되지 않는다.** PyTorch의 dense 텐서 연산은 spike=0인 위치도 동일하게 곱셈을 수행하기 때문이다.

그러나 Sparse의 의의는 다른 곳에 있다:
1. **정확도 보존**: Phase 1→3에서 정확도 변화 0.5%p 이내 (On-Device 실증)
2. **Phase 4(INT8 양자화)의 안정적 기반**: 발화율이 낮으면 activation 분포가 좁아져 INT8 양자화 시 정밀도 손실이 줄어듦
3. **뉴로모픽 하드웨어 준비**: spike=0 스킵이 하드웨어 레벨에서 지원되는 환경에서는 90~93% 연산 절감이 직접 실현됨

### 6.2 실질적 성능 개선 경로

On-Device 결과가 보여주는 실질적 최적화 경로:

```
Phase 1 Teacher (T=10)
  Acc: 98.5~99.1%  |  Lat: 5.3~17.0ms  |  Energy: 1.1~2.0 uAs  |  Size: 47.8~206.6KB
                    ↓
Phase 2 Student (T=10)  — Hard LIF 전환
  Acc: 98.2~98.8%  |  Lat: 3.8~14.1ms (-17%)  |  Energy: 0.8~1.5 uAs (-24%)
                    ↓
Phase 3 Sparse (T=10, FR=5%)  — FR 제어 (정확도 보존, 뉴로모픽 준비)
  Acc: 98.1~98.8%  |  Lat: 5.8~15.2ms (CPU에서 변화 없음)  |  Energy: 유사
                    ↓
Phase 4 QSparse (T=10, FR=5%)  — INT8 양자화 (모델 크기 60% 감소)
  Acc: 98.0~98.8%  |  Lat: 5.5~17.7ms (CPU에서 변화 없음)  |  Size: 40~81KB
                    ↓
EE QSparse (T=2, FR=5%)  — Early Exit + Timestep 축소 (실질적 속도/에너지 개선)
  Acc: 95.8~97.6%  |  Lat: 0.8~1.9ms (-89%)  |  Energy: 0.2~0.4 uAs (-80%)
```

### 6.3 핵심 발견

1. **SNN(T=10) 전 Phase가 98%+ 정확도 유지**: 동일 On-Device 조건에서 Phase 1→2→3→4 정확도 차이가 0.5%p 이내로, 4-Phase 파이프라인이 정확도를 훼손하지 않음을 실증
2. **표준 CPU에서의 latency/energy 개선은 주로 timestep 축소(T→2)와 Early Exit에서 발생**: Sparsification과 INT8 양자화 자체는 표준 CPU에서 직접적 속도 개선을 제공하지 않음
3. **INT8 양자화의 주 이점은 모델 크기 감소(60~69%)**: 메모리 제한적 always-on 시나리오에서 핵심
4. **EE QSparse small이 QCNN small보다 에너지 효율적**: 0.162 vs 0.174 uAs — SNN이 CNN을 에너지에서 역전하는 사례
5. **QCNN smallest의 flick_up 87.36%가 가장 약한 지점**: SNN은 전 제스처에서 96%+ 유지하여, 양자화 강건성에서 SNN이 우위

### 6.4 Section 4.3.1 재작성 방향 수정

기존 계획에서 "Skippable Ops = 90~93%"를 강조했으나, On-Device 결과를 반영하면:

- **Skippable Ops는 "이론적 상한"으로 표현**: "can potentially be skipped on spike-aware hardware"
- **표준 CPU에서의 실질적 이점**으로 초점 전환: (1) 정확도 보존, (2) INT8 양자화 안정성 기반, (3) 뉴로모픽 하드웨어 대비
- **Phase 2 Student의 latency/energy 개선(-17~24%)이 표준 CPU에서의 실질적 이득**임을 강조

---

## 7. 데이터 소스

- `Test_Result/section4.3.1_accuracy_0.csv`
- `Test_Result/section4.3.1__latency_0.csv`
- `Test_Result/section4.3.1__battery_0.csv`
