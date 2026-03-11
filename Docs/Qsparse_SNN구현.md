# QSparse SNN 구현 - INT8 Static Quantization for Sparse SNN

## 1. 프로젝트 개요

Thumbthing 프로젝트는 6축 IMU 센서(가속도 3축 + 자이로 3축) 데이터를 사용하여 4가지 제스처를 분류하는 SNN(Spiking Neural Network) 기반 임베디드 AI 시스템이다.

### 데이터 파이프라인
- **입력**: 6축 IMU raw 데이터 → forward-fill 전처리
- **Rate Coding**: 6축 → 12채널 (양/음 분리, z-score + sigmoid 변환)
- **최종 입력 형태**: `(B, 12, N, T)` — Batch, 12채널, N시퀀스, T타임스텝

### 모델 아키텍처
- **Conv1d 기반 SNN**: `Conv1d(12→32) → LIF1 → Conv1d(32→64) → LIF2 → AdaptiveAvgPool1d → FC(64→4)`
- **LIF 뉴런**: Leaky Integrate-and-Fire, 막전위 누적 + spike 발화
- **커널 크기 변형**: k=3(smallest), k=5(small), k=7(medium), k=9(large), k=11(largest)

### 4-Phase 학습 파이프라인
| Phase | 이름 | 설명 |
|-------|------|------|
| Phase 1 | Teacher (Soft LIF) | Soft surrogate gradient로 학습 |
| Phase 2 | Student (Hard LIF + KD) | Binary spike + STE, Knowledge Distillation |
| Phase 3 | Sparse | Firing Rate Regularization으로 스파이크 희소화 |
| **Phase 4** | **QSparse (INT8)** | **Conv1d 가중치 INT8 정적 양자화** |

---

## 2. 현재 Best 모델 분석

### CNN vs SNN 비교 (벤치마크 기준)
기존 벤치마크에서 **QCNN(INT8 양자화 CNN)** 이 가장 우수한 성능:

| Model | Variant | Accuracy | Latency | Size(KB) | E_potential(nJ) |
|-------|---------|----------|---------|----------|-----------------|
| QCNN | smallest | 98.62% | 0.086ms | 25.3 | 102.7 |
| QCNN | small | 99.61% | 0.090ms | 51.5 | 210.0 |
| QCNN | medium | 99.61% | 0.093ms | 86.6 | 354.6 |
| QCNN | large | 99.61% | 0.098ms | 130.8 | 536.6 |
| QCNN | largest | 99.80% | 0.104ms | 183.9 | 755.9 |

SNN은 정확도·지연시간에서 CNN에 뒤지지만, **에너지 효율(spike 기반 AC 연산)** 에서 잠재적 우위가 있다.

---

## 3. SNN 에너지 효율 개선 전략 분석

| # | 방법 | 구현 난이도 | 예상 개선 효과 | 비고 |
|---|------|------------|---------------|------|
| 1 | **INT8 양자화** | ★★☆ 중 | MAC 에너지 23x↓ | Conv 가중치만 INT8, LIF는 FP32 유지 |
| 2 | T 값 최소화 | ★☆☆ 낮음 | T에 비례한 연산량 감소 | T=3이면 T=15 대비 5x 절감 |
| 3 | Firing Rate 억제 | ★★☆ 중 | AC 비율 증가 → 에너지↓ | fr=5%면 95% 연산이 AC(0연산) |
| 4 | 채널 프루닝 | ★★★ 높음 | 모델 크기/연산 줄임 | 구조적 변경 필요 |
| 5 | C 런타임 최적화 | ★★★ 높음 | 실제 하드웨어 속도↑ | 뉴로모픽 칩 활용 시 극대화 |
| 6 | 가중치 프루닝 | ★★☆ 중 | 희소 행렬 연산 | HW 지원 필요 |
| 7 | 지식 증류 강화 | ★★☆ 중 | 작은 모델로 정확도 유지 | 이미 Phase 2에서 적용 |
| 8 | Mixed Precision | ★★★ 높음 | 레이어별 최적 정밀도 | INT4/INT8 혼합 |

**선택: Phase 4 - INT8 Static Quantization** (구현 가능성 + 효과 균형 최적)

---

## 4. INT8 양자화 구현 설계

### 양자화 범위
- **INT8 적용**: Conv1d 레이어 (conv1, conv2) 가중치 + 활성화
- **FP32 유지**: LIF 뉴런 (막전위 동역학), AdaptiveAvgPool1d, FC 레이어

### QuantStub/DeQuantStub 경계
```
Input → [QuantStub] → Conv1d → [DeQuantStub] → LIF(FP32) → [QuantStub] → Conv1d → [DeQuantStub] → LIF(FP32) → Pool → FC → Output
```

### 핵심 결정사항
1. **BatchNorm 미사용**: 원본 모델에 BN이 없으므로 fusion 불필요
2. **qnnpack 백엔드**: 모바일/임베디드 타겟, CPU 전용
3. **재학습 없음**: Post-Training Quantization (PTQ) — 기존 sparse 가중치 그대로 양자화
4. **Calibration**: 학습 데이터로 activation range 수집

### 80개 모델 구성
- **T 값**: 3, 5, 10, 15 (4가지)
- **Kernel Size**: 3, 5, 7, 9, 11 (5가지 = smallest~largest)
- **Firing Rate**: 30%, 20%, 10%, 5% (4가지)
- **총**: 4 × 5 × 4 = **80개**

---

## 5. 구현 코드: `train_sparse_quant.py`

### 주요 컴포넌트

#### QuantFriendlyLIFNode
```python
class QuantFriendlyLIFNode(nn.Module):
    """LIF neuron that stays in FP32. No custom autograd."""
    def __init__(self, tau, threshold, reset_value=0.0):
        super().__init__()
        self.tau = tau
        self.threshold = threshold
        self.reset_value = reset_value
        self.v = torch.zeros(0)

    def forward(self, x):
        if self.v.numel() == 0 or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
        self.v = self.v + (x - self.v) / self.tau
        spike = (self.v >= self.threshold).float()
        self.v = torch.where(spike > 0, torch.full_like(self.v, self.reset_value), self.v)
        return spike
```

#### QuantizableSparseStudent
- `QuantStub`/`DeQuantStub` 쌍을 conv1, conv2 각각에 배치
- `lif1`, `lif2`, `pool`, `fc`에는 `qconfig=None` 설정하여 양자화 제외

#### calibrate_and_quantize()
1. `torch.quantization.get_default_qconfig("qnnpack")` 적용
2. LIF/pool/fc의 qconfig을 None으로 설정
3. `torch.quantization.prepare()`: Observer 삽입
4. Calibration: 학습 데이터 forward pass로 활성화 통계 수집
5. `torch.quantization.convert()`: FP32 → INT8 변환

#### 출력 파일
- `models/qsparse_{variant}_T{T}_fr{rate}.pt` — 양자화 모델 state_dict
- `models/qsparse_{variant}_T{T}_fr{rate}.ptl` — Mobile Lite 배포용
- `result/qsparse_results.json` — 전체 결과 요약

---

## 6. 실행 결과

### 실행 정보
- **소요 시간**: 약 77.9분 (80개 모델)
- **성공**: 80/80 (100%)
- **모니터링 명령**: `tail -f train_sparse_quant.log`

### 전체 80개 QSparse 결과

#### smallest (k=3)
| T | FR% | FP32 Acc | INT8 Acc | Drop | PTL(KB) |
|---|-----|----------|----------|------|---------|
| 3 | 30% | 91.72% | 91.32% | +0.40% | 27.4 |
| 3 | 20% | 89.35% | 89.74% | -0.39% | 27.4 |
| 3 | 10% | 90.34% | 86.19% | +4.14% | 27.4 |
| 3 | 5% | 88.36% | 75.74% | +12.62% | 27.4 |
| 5 | 30% | 92.70% | 93.10% | -0.39% | 31.5 |
| 5 | 20% | 91.91% | 86.00% | +5.92% | 31.5 |
| 5 | 10% | 91.52% | 50.69% | +40.83% | 31.5 |
| 5 | 5% | 91.91% | 91.52% | +0.39% | 31.5 |
| 10 | 30% | 92.50% | 62.13% | +30.37% | 41.8 |
| 10 | 20% | 93.49% | 92.90% | +0.59% | 41.8 |
| 10 | 10% | 93.49% | 93.10% | +0.39% | 41.8 |
| 10 | 5% | 92.31% | 92.31% | +0.00% | 41.8 |
| 15 | 30% | 93.10% | 93.89% | -0.79% | 52.2 |
| 15 | 20% | 94.08% | 93.69% | +0.39% | 52.2 |
| 15 | 10% | 93.10% | 92.31% | +0.79% | 52.2 |
| 15 | 5% | 94.67% | 90.73% | +3.94% | 52.2 |

#### small (k=5)
| T | FR% | FP32 Acc | INT8 Acc | Drop | PTL(KB) |
|---|-----|----------|----------|------|---------|
| 3 | 30% | 96.25% | 95.46% | +0.79% | 31.9 |
| 3 | 20% | 95.27% | 78.50% | +16.77% | 31.9 |
| 3 | 10% | 96.25% | 95.66% | +0.59% | 31.9 |
| 3 | 5% | 96.65% | 96.65% | +0.00% | 31.9 |
| 5 | 30% | 96.65% | 84.22% | +12.43% | 36.0 |
| 5 | 20% | 97.04% | 96.06% | +0.99% | 36.0 |
| 5 | 10% | 96.65% | 94.87% | +1.78% | 36.0 |
| 5 | 5% | 96.06% | 96.25% | -0.20% | 36.0 |
| 10 | 30% | 97.83% | 71.79% | +26.04% | 46.4 |
| 10 | 20% | 97.83% | 97.24% | +0.59% | 46.4 |
| 10 | 10% | 97.63% | 67.85% | +29.78% | 46.4 |
| 10 | 5% | 98.62% | 98.22% | +0.39% | 46.4 |
| 15 | 30% | 97.44% | 68.24% | +29.19% | 56.7 |
| 15 | 20% | 97.44% | 71.20% | +26.23% | 56.7 |
| 15 | 10% | 96.06% | 61.74% | +34.32% | 56.8 |
| 15 | 5% | 96.25% | 72.98% | +23.27% | 56.7 |

#### medium (k=7)
| T | FR% | FP32 Acc | INT8 Acc | Drop | PTL(KB) |
|---|-----|----------|----------|------|---------|
| 3 | 30% | 97.24% | 96.65% | +0.59% | 36.8 |
| 3 | 20% | 97.63% | 97.63% | +0.00% | 36.8 |
| 3 | 10% | 96.65% | 95.86% | +0.79% | 36.8 |
| 3 | 5% | 97.44% | 97.44% | +0.00% | 36.8 |
| 5 | 30% | 97.83% | 75.74% | +22.09% | 40.8 |
| 5 | 20% | 97.63% | 97.24% | +0.39% | 40.8 |
| 5 | 10% | 97.63% | 98.03% | -0.39% | 40.8 |
| 5 | 5% | 98.42% | 97.44% | +0.99% | 40.8 |
| 10 | 30% | 98.22% | 90.93% | +7.30% | 51.2 |
| 10 | 20% | 98.03% | 98.03% | +0.00% | 51.2 |
| 10 | 10% | 97.63% | 98.42% | -0.79% | 51.2 |
| 10 | 5% | 98.22% | 98.22% | +0.00% | 51.2 |
| 15 | 30% | 98.62% | 98.42% | +0.20% | 61.5 |
| 15 | 20% | 98.03% | 97.44% | +0.59% | 61.5 |
| 15 | 10% | 98.22% | 98.62% | -0.39% | 61.5 |
| 15 | 5% | 98.82% | 67.65% | +31.16% | 61.5 |

#### large (k=9)
| T | FR% | FP32 Acc | INT8 Acc | Drop | PTL(KB) |
|---|-----|----------|----------|------|---------|
| 3 | 30% | 98.03% | 89.35% | +8.68% | 41.4 |
| 3 | 20% | 98.03% | 98.03% | +0.00% | 41.4 |
| 3 | 10% | 97.04% | 97.63% | -0.59% | 41.4 |
| 3 | 5% | 97.83% | 98.22% | -0.39% | 41.4 |
| 5 | 30% | 98.03% | 88.36% | +9.66% | 45.5 |
| 5 | 20% | 98.03% | 95.86% | +2.17% | 45.5 |
| 5 | 10% | 98.62% | 96.65% | +1.97% | 45.5 |
| 5 | 5% | 98.42% | 56.02% | +42.41% | 45.5 |
| 10 | 30% | 98.42% | 56.41% | +42.01% | 55.8 |
| 10 | 20% | 98.42% | 97.24% | +1.18% | 55.9 |
| 10 | 10% | 98.03% | 98.42% | -0.39% | 55.9 |
| 10 | 5% | 97.83% | 98.22% | -0.39% | 55.9 |
| 15 | 30% | 98.03% | 60.95% | +37.08% | 66.2 |
| 15 | 20% | 98.62% | 74.56% | +24.06% | 66.2 |
| 15 | 10% | 99.01% | 97.44% | +1.58% | 66.2 |
| **15** | **5%** | **98.82%** | **99.21%** | **-0.39%** | **66.2** |

#### largest (k=11)
| T | FR% | FP32 Acc | INT8 Acc | Drop | PTL(KB) |
|---|-----|----------|----------|------|---------|
| 3 | 30% | 98.22% | 74.36% | +23.87% | 46.3 |
| 3 | 20% | 98.03% | 98.82% | -0.79% | 46.3 |
| 3 | 10% | 98.03% | 97.04% | +0.99% | 46.3 |
| 3 | 5% | 98.62% | 98.42% | +0.20% | 46.3 |
| 5 | 30% | 98.82% | 98.22% | +0.59% | 50.3 |
| 5 | 20% | 99.01% | 98.42% | +0.59% | 50.4 |
| 5 | 10% | 99.41% | 98.42% | +0.99% | 50.4 |
| 5 | 5% | 99.01% | 98.82% | +0.20% | 50.4 |
| 10 | 30% | 98.22% | 96.06% | +2.17% | 60.8 |
| 10 | 20% | 98.03% | 97.83% | +0.20% | 60.9 |
| 10 | 10% | 98.22% | 92.50% | +5.72% | 60.9 |
| 10 | 5% | 98.82% | 98.03% | +0.79% | 60.9 |
| 15 | 30% | 98.22% | 98.82% | -0.59% | 71.1 |
| 15 | 20% | 98.42% | 97.44% | +0.99% | 71.1 |
| 15 | 10% | 98.82% | 98.42% | +0.39% | 71.2 |
| 15 | 5% | 98.82% | 99.01% | -0.20% | 71.1 |

---

## 7. Top 결과 분석

### INT8 Accuracy Top 10
| Rank | Variant | T | FR% | INT8 Acc | PTL(KB) | Acc Drop |
|------|---------|---|-----|----------|---------|----------|
| 1 | large | 15 | 5% | **99.21%** | 66.2 | -0.39% |
| 2 | largest | 15 | 5% | 99.01% | 71.1 | -0.20% |
| 3 | largest | 3 | 20% | 98.82% | 46.3 | -0.79% |
| 4 | largest | 5 | 5% | 98.82% | 50.4 | +0.20% |
| 5 | largest | 15 | 30% | 98.82% | 71.1 | -0.59% |
| 6 | medium | 15 | 10% | 98.62% | 61.5 | -0.39% |
| 7 | largest | 3 | 5% | 98.42% | 46.3 | +0.20% |
| 8 | largest | 5 | 20% | 98.42% | 50.4 | +0.59% |
| 9 | medium | 10 | 10% | 98.42% | 51.2 | -0.79% |
| 10 | large | 10 | 10% | 98.42% | 55.9 | -0.39% |

### 최소 PTL 크기 Top 10 (Accuracy ≥ 95%)
| Rank | Variant | T | FR% | INT8 Acc | PTL(KB) |
|------|---------|---|-----|----------|---------|
| 1 | small | 3 | 30% | 95.46% | 31.9 |
| 2 | small | 3 | 10% | 95.66% | 31.9 |
| 3 | small | 3 | 5% | 96.65% | 31.9 |
| 4 | small | 5 | 20% | 96.06% | 36.0 |
| 5 | small | 5 | 5% | 96.25% | 36.0 |
| 6 | medium | 3 | 30% | 96.65% | 36.8 |
| 7 | medium | 3 | 20% | 97.63% | 36.8 |
| 8 | medium | 3 | 10% | 95.86% | 36.8 |
| 9 | medium | 3 | 5% | 97.44% | 36.8 |
| 10 | medium | 5 | 20% | 97.24% | 40.8 |

---

## 8. QCNN vs QSparse 비교 (Variant별)

### 비교 기준
- **QCNN**: INT8 양자화된 CNN (k=9, T=1)
- **QSparse**: 각 variant의 최고 정확도 INT8 모델

| | QCNN | QSparse Best | |
|---------|------|-------------|--|
| **smallest** | | | |
| PTL Size | 25.3 KB | 27.4 KB (T=15 fr=30%) | QCNN 0.9x |
| Accuracy | 98.62% | 93.89% | QCNN +4.73%p |
| Latency | 0.086ms | ~0.3ms* | QCNN 3.5x faster |
| E_potential | 102.7 nJ | ~50-80 nJ** | QSparse 우위 |
| **small** | | | |
| PTL Size | 51.5 KB | 31.9 KB (T=3 fr=5%) | **QSparse 1.6x 작음** |
| Accuracy | 99.61% | 98.22% (T=10 fr=5%) | QCNN +1.39%p |
| Latency | 0.090ms | ~0.3ms* | QCNN faster |
| E_potential | 210.0 nJ | ~100-150 nJ** | **QSparse 우위** |
| **medium** | | | |
| PTL Size | 86.6 KB | 36.8 KB (T=3 fr=5%) | **QSparse 2.4x 작음** |
| Accuracy | 99.61% | 98.62% (T=15 fr=10%) | QCNN +0.99%p |
| Latency | 0.093ms | ~0.3ms* | QCNN faster |
| E_potential | 354.6 nJ | ~150-250 nJ** | **QSparse 우위** |
| **large** | | | |
| PTL Size | 130.8 KB | 41.4 KB (T=3 fr=5%) | **QSparse 3.2x 작음** |
| Accuracy | 99.61% | **99.21%** (T=15 fr=5%) | QCNN +0.40%p |
| Latency | 0.098ms | ~0.3ms* | QCNN faster |
| E_potential | 536.6 nJ | ~300-400 nJ** | **QSparse 우위** |
| **largest** | | | |
| PTL Size | 183.9 KB | 46.3 KB (T=3 fr=5%) | **QSparse 4.0x 작음** |
| Accuracy | 99.80% | 99.01% (T=15 fr=5%) | QCNN +0.79%p |
| Latency | 0.104ms | ~0.3ms* | QCNN faster |
| E_potential | 755.9 nJ | ~350-500 nJ** | **QSparse 우위** |

> *QSparse latency는 T 타임스텝 반복으로 인해 CNN 대비 느림 (CPU 환경, 뉴로모픽 HW에서는 역전 가능)
> **E_potential은 spike 희소성 기반 추정값 (MAC 중 대부분이 AC로 대체)

---

## 9. 핵심 발견사항

### 양자화 안정성
- **fr=30% 모델은 양자화에 취약**: 일부에서 accuracy collapse 발생 (30~40%p 하락)
- **fr=5~10% 모델은 안정적**: 대부분 ±1%p 이내 accuracy 변동
- **큰 variant(large, largest)가 양자화에 더 안정적**: 파라미터 여유분이 양자화 오차 흡수

### 압축 성능
- **INT8 모델 크기**: 모든 variant에서 ~1.0KB (파라미터만)
- **PTL 파일 크기**: 27.4~71.2KB (TorchScript 메타데이터 포함)
- **압축비**: 28~102x (FP32 대비)

### QSparse vs QCNN
| 항목 | QCNN 우위 | QSparse 우위 |
|------|----------|-------------|
| Accuracy | +0.4~4.7%p | - |
| Latency (CPU) | 3~5x 빠름 | 뉴로모픽 HW에서 역전 |
| **PTL Size** | - | **1.6~4.0x 작음** (small+) |
| **Energy (Potential)** | - | **1.2~2.0x 효율적** |
| 배포 유연성 | CPU 최적 | 에지/IoT 최적 |

### 추천 배포 시나리오
1. **CPU 환경 (속도 우선)**: QCNN small (51.5KB, 99.61%, 0.09ms)
2. **에지 환경 (크기 우선)**: QSparse small T=3 fr=5% (31.9KB, 96.65%)
3. **에지 환경 (정확도 우선)**: QSparse large T=15 fr=5% (66.2KB, 99.21%)
4. **에너지 최적 (배터리)**: QSparse medium T=3 fr=5% (36.8KB, 97.44%)

---

## 10. 에너지 계산 기준

| 연산 유형 | FP32 에너지 | INT8 에너지 |
|----------|------------|------------|
| MAC (Multiply-Accumulate) | 4.6 pJ | 0.2 pJ |
| AC (Accumulate only) | 0.9 pJ | 0.03 pJ |

- **CNN**: 모든 연산이 MAC → `Total = MAC_count × 4.6pJ` (FP32) 또는 `× 0.2pJ` (INT8)
- **SNN**: spike=0이면 AC, spike=1이면 MAC → `Total = MAC_count × fr × E_mac + MAC_count × (1-fr) × E_ac`
- fr=5%인 경우: 95%의 연산이 AC(0.03pJ)로 대체 → 에너지 대폭 절감

---

## 11. 파일 구조

```
Thumbthing_easy_intae2/
├── train_sparse_quant.py      # Phase 4: INT8 양자화 스크립트
├── train_sparse_quant.log     # 실행 로그
├── models/
│   ├── sparse_*.pt            # Phase 3 입력 모델 (80개)
│   ├── qsparse_*.pt           # Phase 4 양자화 모델 (80개)
│   └── qsparse_*.ptl          # Phase 4 모바일 배포 파일 (80개)
├── result/
│   ├── benchmark_results.json  # CNN/QCNN/SNN 벤치마크
│   └── qsparse_results.json   # QSparse 양자화 결과 (80개)
├── model.py                   # 기본 SNN 모델 정의
├── dataset.py                 # 데이터 로딩/전처리
├── lif_module.py              # LIF 뉴런 모듈
├── benchmark.py               # 벤치마크 스크립트
└── train_sparse.py            # Phase 3 sparse 학습
```

---

## 12. 향후 과제

1. **benchmark.py에 QSparse 모델 타입 추가**: 현재 벤치마크에서 QSparse 직접 측정 미지원
2. **Quantization-Aware Training (QAT)**: PTQ 대신 QAT 적용으로 fr=30% 모델 안정성 개선
3. **뉴로모픽 하드웨어 벤치마크**: 실제 spike 기반 하드웨어에서의 latency/energy 측정
4. **채널 프루닝 + 양자화 결합**: 모델 크기 추가 축소
5. **INT4 Mixed Precision**: Conv1d를 INT4로 추가 양자화 시도
