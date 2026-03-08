# 일반 CPU 환경에서 SNN이 CNN을 이기기 위한 전략 상세 분석

---

## 0. 왜 CPU에서는 SNN이 불리한가? (문제의 본질)

먼저 냉정한 현실 인식이 필요하다. SNN이 CNN 대비 갖는 핵심 이점은 **이벤트 구동 연산**(뉴런이 스파이크를 발생시킬 때만 연산)인데, 일반 CPU는 이를 하드웨어적으로 지원하지 않는다. CPU에서 SNN을 실행하면 다음과 같은 구조적 불리함이 발생한다.

**CPU에서의 SNN 오버헤드:**

- **시간 축 반복**: CNN은 입력에 대해 1회 순전파(forward pass)하지만, SNN은 T개의 타임스텝을 순차적으로 반복해야 한다. 즉, 순전파 비용이 대략 T배로 증가한다.
- **상태 유지 비용**: 매 타임스텝마다 모든 뉴런의 막전위(membrane potential)를 메모리에 저장하고 갱신해야 한다. CNN에는 없는 추가 메모리 접근이 발생한다.
- **희소성 미활용**: CPU의 SIMD/벡터 연산 유닛은 밀집(dense) 행렬곱에 최적화되어 있다. SNN의 스파이크가 90% 이상 0이더라도, naive 구현에서는 0인 부분도 동일하게 연산한다.
- **프레임워크 오버헤드**: PyTorch/snnTorch 기반 구현 시, 매 타임스텝마다 Python 레벨의 루프와 텐서 연산 오버헤드가 누적된다.

따라서 **"SNN이라서 자동으로 빠르다/효율적이다"는 CPU에서는 성립하지 않는다.** 의도적이고 정밀한 최적화 전략 없이는 SNN이 CNN보다 느리고 무거워진다.

---

## 1. 전략 A: 연산 구조의 본질적 차이를 활용하라 — MAC vs AC

이것이 CPU에서 SNN이 CNN을 이길 수 있는 **가장 근본적인 무기**이다.

### 1.1 핵심 원리

CNN의 합성곱 연산은 **MAC(Multiply-Accumulate)** 연산이다:

```
output[i] = Σ (weight[j] × activation[j])    # 곱셈 + 덧셈
```

반면 SNN의 합성곱(첫 번째 레이어 제외)은 **AC(Accumulate-only)** 연산이다:

```
# spike[j]는 0 또는 1 (이진)
membrane[i] += Σ (weight[j] × spike[j])
            = Σ (weight[j])  for spike[j]=1 인 j만    # 덧셈만!
```

스파이크가 이진(0 또는 1)이므로, 곱셈이 필요 없다. **spike=1인 채널의 가중치만 골라서 더하면 된다.** 이는 CPU에서 실질적인 연산량 절감으로 이어진다.

### 1.2 CPU에서의 실질적 이득 계산

UCI-HAR 같은 IMU 데이터를 가정하자:
- 입력 윈도우: 128 samples × 6 channels
- CNN Conv1D: kernel=3, in_ch=64, out_ch=64 → **1회 MAC = 3×64×64 = 12,288 곱셈+덧셈**
- SNN Conv1D (스파이크 발화율 10%): **AC = 12,288 × 0.1 = 1,229 덧셈만**

단일 레이어 단일 타임스텝 기준으로 **실효 연산량이 ~10배 감소**한다. 물론 T 타임스텝을 반복하므로 총 연산량은 `1,229 × T`인데, T=4이면 4,916 덧셈으로 여전히 12,288 MAC보다 적다.

### 1.3 구현 방법

**핵심**: PyTorch의 기본 Conv1d는 이 이점을 활용하지 못한다. 스파이크 희소성을 실제 속도 향상으로 변환하려면 직접 구현이 필요하다.

```python
# PyTorch 내에서 희소 행렬 연산으로 AC 구현
import torch

def sparse_spike_linear(weight, spike):
    """
    weight: [out_features, in_features] (dense, float)
    spike: [batch, in_features] (binary, 0 or 1)
    
    MAC 대신 AC만 수행
    """
    # 방법 1: 마스크 기반 (간단하지만 메모리 효율적이지 않음)
    # output = (weight * spike.unsqueeze(1)).sum(dim=-1)
    
    # 방법 2: 스파이크가 1인 인덱스만 추출하여 가중치 합산
    # 발화율이 낮을수록 빠름
    batch_size = spike.shape[0]
    output = torch.zeros(batch_size, weight.shape[0], device=weight.device)
    for b in range(batch_size):
        active_idx = spike[b].nonzero(as_tuple=True)[0]  # 스파이크 발생 뉴런
        if len(active_idx) > 0:
            output[b] = weight[:, active_idx].sum(dim=1)  # 해당 열만 합산
    return output

    # 방법 3 (권장): torch.sparse 활용
    # spike_sparse = spike.to_sparse()
    # output = torch.sparse.mm(spike_sparse, weight.t())
```

**방법 3의 torch.sparse**가 가장 실용적이다. 발화율이 20% 이하일 때 dense 연산 대비 실질적 CPU 속도 향상을 얻을 수 있다.

---

## 2. 전략 B: 타임스텝을 극단적으로 줄여라 (T를 최소화)

CPU에서 SNN의 가장 큰 비용은 **T번의 순차 반복**이다. 따라서 T를 줄이는 것이 가장 직접적인 개선이다.

### 2.1 목표: T=1~4로 추론

일반적인 SNN 연구에서 T=16~100을 사용하지만, IMU HAR 작업에서는 T=1~4로도 충분한 성능을 낼 수 있다. 이유는 다음과 같다:

- IMU 데이터의 슬라이딩 윈도우(128~256 samples)는 이미 **풍부한 시간 정보**를 포함하고 있다.
- SNN의 타임스텝은 "입력의 시간 해상도"가 아니라 "스파이크 처리의 반복 횟수"이다.
- 멀티 임계값 델타 인코딩으로 4채널 스파이크를 생성하면, T=1에서도 상당한 정보가 전달된다.

### 2.2 T=1에서 작동하게 만드는 기법들

**기법 1: Direct Input Encoding (입력 레이어에서 직접 인코딩)**
```python
# 첫 번째 레이어는 아날로그 입력을 직접 받는다 (스파이크가 아님)
# → 첫 레이어만 MAC, 이후 레이어는 모두 AC
class DirectEncodingSNN(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv1d(6, 64, 3)       # 아날로그 입력 → MAC (CNN과 동일)
        self.lif1 = snn.Leaky(beta=0.9)         # 스파이크 출력
        self.conv2 = nn.Conv1d(64, 128, 3)      # 스파이크 입력 → AC (SNN 이점)
        self.lif2 = snn.Leaky(beta=0.9)
        self.fc = nn.Linear(128, num_classes)
        self.lif_out = snn.Leaky(beta=0.9)
```

**기법 2: Learnable Threshold (학습 가능 임계값)**
- T가 작을수록 뉴런이 발화할 기회가 적으므로, 임계값을 낮춰서 발화를 촉진해야 한다.
- 각 레이어의 Vth를 학습 가능 파라미터로 설정:
```python
self.threshold = nn.Parameter(torch.tensor(0.5))  # 초기값을 낮게
```

**기법 3: Membrane Potential을 직접 출력으로 사용**
- T=1에서 스파이크 카운트로 판별하면 정보가 너무 적다.
- 대신 **최종 레이어의 막전위 값**을 softmax에 넣어 분류:
```python
# 마지막 레이어: 스파이크 대신 막전위를 출력
mem_out = self.lif_out(self.fc(x))
output = F.softmax(mem_out, dim=-1)  # 막전위 기반 분류
```
- 이렇게 하면 T=1에서도 CNN과 동등한 표현력을 확보하면서, 중간 레이어에서는 스파이크 희소성의 AC 이점을 유지한다.

### 2.3 T에 따른 CPU 성능 예측

| T | 총 SNN 연산 (상대값) | CNN 대비 | 비고 |
|---|---|---|---|
| 1 | 1.0× (AC 이점 포함 시 ~0.3×) | 유리 | 정확도 약간 하락 가능 |
| 2 | 2.0× → ~0.6× | 유리 | 충분한 정확도 |
| 4 | 4.0× → ~1.2× | 동등~약간 불리 | CNN과 동등 정확도 |
| 8 | 8.0× → ~2.4× | 불리 | 정확도 우위 시에만 정당화 |
| 16+ | 16.0× → ~4.8× | 매우 불리 | CPU에서 비권장 |

(AC 이점 = 발화율 10% 가정 시 ~0.3배 연산)

---

## 3. 전략 C: 모델 크기에서 이겨라 — SNN의 파라미터 효율성

### 3.1 왜 SNN이 더 작은 모델로도 충분한가

SNN은 **시간 차원의 정보**를 뉴런 동역학(막전위, 누설)으로 무료로 처리한다. CNN이 시간 패턴을 포착하려면 더 큰 커널이나 더 많은 채널이 필요하지만, SNN은 뉴런의 시간 상수(β)가 이 역할을 대신한다.

따라서 **동일 정확도를 위한 파라미터 수가 SNN이 더 적을 수 있으며**, 이는 CPU에서 다음과 같은 이점으로 이어진다:

- 더 적은 가중치 → 더 적은 메모리 접근 (CPU 병목의 핵심)
- L1/L2 캐시에 모델이 통째로 들어갈 확률 증가
- 캐시 히트율 향상 → 메모리 바운드 연산에서 실질적 속도 향상

### 3.2 구체적 모델 크기 비교 전략

```
CNN baseline (1D-CNN for HAR):
  Conv1(6→64, k=5) + Conv2(64→128, k=5) + Conv3(128→256, k=3) + FC(256→classes)
  파라미터: ~200K, 추론 FLOPs: ~2M MAC

SNN 대안 (Spike-TCN-Lite):
  Conv1(6→32, k=3) + LIF + Conv2(32→64, k=3) + LIF + FC(64→classes) + LIF
  파라미터: ~15K, 추론 FLOPs: ~150K AC (T=2, 발화율 10%)
```

**13배 작은 모델로 동등한 정확도를 목표**로 한다. 이것이 CPU에서 SNN이 CNN을 이기는 가장 현실적인 경로이다.

### 3.3 Depthwise Separable Spiking Convolution

파라미터 효율을 극대화하는 핵심 기법:

```python
class SpikingDWSConv(nn.Module):
    """Depthwise Separable Spiking Convolution"""
    def __init__(self, in_ch, out_ch, kernel_size, beta=0.9):
        super().__init__()
        # Depthwise: 채널별 독립 합성곱
        self.dw_conv = nn.Conv1d(in_ch, in_ch, kernel_size, 
                                  padding=kernel_size//2, groups=in_ch)
        self.lif_dw = snn.Leaky(beta=beta, learn_beta=True)
        
        # Pointwise: 1x1 합성곱으로 채널 혼합
        self.pw_conv = nn.Conv1d(in_ch, out_ch, 1)
        self.lif_pw = snn.Leaky(beta=beta, learn_beta=True)
    
    def forward(self, spike, mem_dw, mem_pw):
        # Depthwise (스파이크 입력 → AC)
        cur = self.dw_conv(spike)
        spike_dw, mem_dw = self.lif_dw(cur, mem_dw)
        
        # Pointwise (스파이크 입력 → AC)
        cur = self.pw_conv(spike_dw)
        spike_pw, mem_pw = self.lif_pw(cur, mem_pw)
        
        return spike_pw, mem_dw, mem_pw
```

일반 Conv(in=64, out=128, k=3) 파라미터: 64×128×3 = **24,576**
DWS Conv 파라미터: 64×1×3 + 64×128×1 = 192 + 8,192 = **8,384** (약 3배 축소)

---

## 4. 전략 D: 희소성을 CPU 속도 향상으로 변환하라

스파이크의 희소성(대부분 0)을 CPU에서 실제 속도 향상으로 바꾸는 구체적 기법들이다.

### 4.1 조건부 연산 (Conditional Computation)

```python
def sparse_forward_conv1d(weight, bias, spike_input):
    """
    스파이크가 0인 채널은 아예 건너뛴다.
    발화율 10% → 90%의 입력 채널 연산을 스킵.
    """
    batch, channels, length = spike_input.shape
    
    # 채널별 스파이크 존재 여부 확인
    active_channels = (spike_input.sum(dim=(0, 2)) > 0)  # [channels]
    
    if active_channels.sum() == 0:
        # 모든 스파이크가 0이면 바이어스만 반환
        return bias.view(1, -1, 1).expand(batch, -1, length)
    
    # 활성 채널만으로 서브 컨볼루션
    active_idx = active_channels.nonzero(as_tuple=True)[0]
    sparse_input = spike_input[:, active_idx, :]
    sparse_weight = weight[:, active_idx, :]
    
    output = F.conv1d(sparse_input, sparse_weight, bias, padding=1)
    return output
```

### 4.2 Early Layer Sparsification

초기 레이어에서 의도적으로 높은 스파이크 희소성을 유도하면, 이후 모든 레이어의 AC 연산량이 줄어든다.

```python
# 학습 시 발화율 제약 (Activity Regularization)
def firing_rate_loss(spike_outputs, target_rate=0.05):
    """
    각 레이어의 발화율을 target_rate(5%)로 유도.
    낮은 발화율 = 높은 희소성 = CPU에서 더 빠른 추론.
    """
    loss = 0
    for spk in spike_outputs:
        actual_rate = spk.mean()
        loss += (actual_rate - target_rate) ** 2
    return loss

# 총 손실
total_loss = ce_loss + lambda_sparse * firing_rate_loss(all_spikes, target_rate=0.05)
```

**target_rate = 0.05 (5%)**로 설정하면:
- 95%의 뉴런이 침묵 → 95%의 AC 연산 스킵 가능
- 정확도와 희소성의 트레이드오프를 λ로 조절

### 4.3 구조적 희소성 (Structured Sparsity)

비구조적 희소성(개별 뉴런 단위)은 CPU에서 속도 향상이 어렵다. **채널 단위의 구조적 희소성**이 필요하다.

```python
class ChannelGatedLIF(nn.Module):
    """채널 단위로 통째로 ON/OFF하는 LIF 뉴런"""
    def __init__(self, channels, beta=0.9):
        super().__init__()
        self.lif = snn.Leaky(beta=beta, learn_beta=True)
        # 채널 게이트: 학습 가능한 채널별 마스크
        self.gate = nn.Parameter(torch.ones(1, channels, 1))
    
    def forward(self, cur, mem):
        spike, mem = self.lif(cur, mem)
        # 학습 중: soft gating (sigmoid)
        # 추론 중: hard gating (threshold → 0 or 1)
        if self.training:
            mask = torch.sigmoid(self.gate * 10)  # soft
        else:
            mask = (self.gate > 0).float()  # hard
        
        return spike * mask, mem
```

추론 시 gate=0인 **채널 전체를 건너뛸** 수 있으므로, CPU의 벡터 연산 단위에서 실질적 속도 향상이 발생한다.

---

## 5. 전략 E: 양자화 — SNN의 천생 이진 특성을 극대화

### 5.1 SNN은 태생적으로 양자화에 유리하다

CNN에서 양자화는 **정보 손실**을 수반하지만, SNN은 이미 활성화가 이진(0/1)이다. 추가로 양자화할 수 있는 대상은 **가중치**와 **막전위**이다.

### 5.2 가중치 양자화 전략

```
FP32 가중치 (CNN 표준)     : 4 bytes/param
INT8 가중치 (CNN 양자화)    : 1 byte/param   → CPU INT8 VNNI 명령어 활용 가능
INT4 가중치 (SNN 양자화)    : 0.5 byte/param → SNN에서 정확도 하락 최소
Binary 가중치 (SNN 극한)    : 0.125 byte/param → XNOR 연산으로 대체 가능
```

**SNN + Binary Weight의 시너지**: 
- 스파이크(이진) × 가중치(이진) = **XNOR + popcount** 연산
- CPU에서 64비트 XNOR 한 번으로 64개 시냅스를 동시에 처리 가능
- 이론적으로 FP32 MAC 대비 **~64배 연산 처리량 향상**

```python
# Binary weight SNN의 개념
class BinarySpikeLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
    
    def binarize(self, w):
        """학습 시: STE(Straight-Through Estimator)로 이진화"""
        return w.sign()
    
    def forward(self, spike_input):
        # spike_input: {0, 1}, binary_weight: {-1, +1}
        bw = self.binarize(self.weight)
        # {0,1} × {-1,+1} → 그냥 가중치 선택/부호반전
        # XNOR + popcount로 대체 가능
        return F.linear(spike_input.float(), bw)
```

### 5.3 막전위 양자화

막전위를 FP32에서 INT8이나 더 낮은 정밀도로 양자화하면:
- 뉴런 상태 메모리 4~16배 절감
- 캐시 효율 대폭 향상 (작은 모델이 L2 캐시에 완전히 적재)

SpQuant-SNN 연구에서 **1.58비트 막전위**(ternary: -1, 0, +1)로도 정확도 하락이 2% 미만임이 확인되었다.

```python
def quantize_membrane(mem, bits=4):
    """막전위를 N비트로 양자화"""
    scale = (2 ** bits - 1) / (mem.max() - mem.min() + 1e-8)
    mem_q = torch.round((mem - mem.min()) * scale) / scale + mem.min()
    return mem_q
```

---

## 6. 전략 F: C 런타임으로 프레임워크 오버헤드를 제거하라

### 6.1 문제: Python/PyTorch 오버헤드

snnTorch + PyTorch로 SNN을 돌리면, 매 타임스텝마다:
- Python 인터프리터 오버헤드
- 텐서 생성/해제
- CUDA 동기화 (CPU 모드에서도 dispatch 오버헤드)
- autograd 그래프 관리 (추론에서도 완전히 제거 어려움)

이 오버헤드가 작은 모델에서는 **실제 연산 시간보다 더 클** 수 있다.

### 6.2 해결: 학습은 PyTorch, 추론은 C

연구에 따르면 SNN 모델을 C로 변환하여 추론하면 **PyTorch 대비 10배 이상의 속도 향상**을 달성할 수 있다.

```
학습 파이프라인:
  PyTorch + snnTorch → 학습 완료 → 가중치 + 구조 저장

추론 파이프라인 (C 런타임):
  가중치 로드 → C 구조체로 네트워크 구성 → 최적화된 추론 루프

C 런타임 핵심 구조:
  - 가중치: 정적 배열 (모델이 작으면 코드에 하드코딩)
  - 막전위: 고정 크기 버퍼
  - 스파이크 전파: 조건부 가중치 합산
  - 메모리 할당: 없음 (모두 스택/정적)
```

### 6.3 C 추론 커널 예시

```c
// LIF 뉴런 + 희소 스파이크 처리
typedef struct {
    float *weights;      // [out_size × in_size]
    float *bias;
    float *membrane;     // [out_size] — 상태 유지
    float beta;          // 누설 계수
    float threshold;     // 발화 임계값
    uint8_t *spikes_out; // [out_size] — 이진 출력
    int in_size;
    int out_size;
} SNN_Layer;

void snn_layer_forward(SNN_Layer *layer, const uint8_t *spikes_in) {
    for (int o = 0; o < layer->out_size; o++) {
        // 누설 (Leak)
        layer->membrane[o] *= layer->beta;
        
        // 적분 (Integrate) — 스파이크=1인 입력만 합산
        float acc = layer->bias[o];
        for (int i = 0; i < layer->in_size; i++) {
            if (spikes_in[i]) {  // AC: 곱셈 없음!
                acc += layer->weights[o * layer->in_size + i];
            }
        }
        layer->membrane[o] += acc;
        
        // 발화 (Fire)
        if (layer->membrane[o] >= layer->threshold) {
            layer->spikes_out[o] = 1;
            layer->membrane[o] -= layer->threshold;  // soft reset
        } else {
            layer->spikes_out[o] = 0;
        }
    }
}
```

**이 구현의 CPU 이점:**
- if (spikes_in[i]) 분기로 spike=0인 시냅스의 메모리 접근 자체를 회피
- float 곱셈 0회 (첫 번째 레이어 제외)
- 메모리 할당 0회
- 분기 예측: 발화율 5%면 95% 정확도로 분기 예측 성공 → 파이프라인 스톨 최소

---

## 7. 전략 G: SNN의 시간적 표현력으로 정확도에서 이겨라

CPU에서 연산 속도 우위가 어렵다면, **같은 연산 예산에서 더 높은 정확도**를 달성하는 것도 유효한 전략이다.

### 7.1 다중 시간 스케일 (Multi-Timescale) 뉴런

```python
class MultiTimescaleSNN(nn.Module):
    """
    각 레이어가 다른 시간 스케일로 작동.
    빠른 변화(걷기→뛰기)와 느린 변화(자세 유지)를
    단일 네트워크로 동시에 포착.
    CNN은 이를 위해 다중 커널 크기나 딜레이션이 필요하지만,
    SNN은 β 하나로 해결.
    """
    def __init__(self):
        super().__init__()
        # 빠른 동역학 레이어 (β ≈ 0.5, 빠른 감쇠)
        self.fast_conv = nn.Conv1d(24, 32, 3, padding=1)
        self.fast_lif = snn.Leaky(beta=0.5, learn_beta=True)
        
        # 느린 동역학 레이어 (β ≈ 0.95, 느린 감쇠)
        self.slow_conv = nn.Conv1d(24, 32, 3, padding=1)
        self.slow_lif = snn.Leaky(beta=0.95, learn_beta=True)
        
        # 병합
        self.merge_conv = nn.Conv1d(64, 64, 1)
        self.merge_lif = snn.Leaky(beta=0.9, learn_beta=True)
        
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # T 타임스텝 반복
        mem_fast = self.fast_lif.init_leaky()
        mem_slow = self.slow_lif.init_leaky()
        mem_merge = self.merge_lif.init_leaky()
        spk_rec = []
        
        for t in range(T):
            # 병렬 처리: 빠른/느린 경로
            cur_fast = self.fast_conv(x[:, :, t:t+1] if T > 1 else x)
            spk_fast, mem_fast = self.fast_lif(cur_fast, mem_fast)
            
            cur_slow = self.slow_conv(x[:, :, t:t+1] if T > 1 else x)
            spk_slow, mem_slow = self.slow_lif(cur_slow, mem_slow)
            
            # 두 경로 병합
            merged = torch.cat([spk_fast, spk_slow], dim=1)
            cur_merge = self.merge_conv(merged)
            spk_merge, mem_merge = self.merge_lif(cur_merge, mem_merge)
            
            spk_rec.append(spk_merge)
        
        # 출력: 막전위 기반 분류
        out = self.fc(mem_merge.squeeze(-1))
        return out
```

### 7.2 시간적 어텐션 (Temporal Attention without Extra Cost)

CNN은 시간적 어텐션을 위해 별도의 Self-Attention 레이어가 필요하지만 (O(N²) 연산), SNN은 **막전위 자체가 암묵적 시간 어텐션** 역할을 한다:

- 중요한 입력 → 높은 막전위 → 발화 → 다음 레이어에 전파
- 중요하지 않은 입력 → 낮은 막전위 → 누설로 소멸 → 무시됨

이는 **추가 파라미터나 연산 없이** 시간적 필터링이 이루어진다는 의미이다.

### 7.3 Knowledge Distillation: CNN의 지식을 SNN에 전이

```python
# Teacher: 큰 CNN (정확도 97%)
# Student: 작은 SNN (목표 정확도 96%+)

def distillation_loss(snn_output, cnn_output, labels, alpha=0.7, temperature=3):
    """
    CNN teacher의 soft label로 SNN student 학습.
    SNN이 CNN의 클래스 간 관계 정보를 흡수.
    """
    # Hard label loss
    hard_loss = F.cross_entropy(snn_output, labels)
    
    # Soft label loss (KL divergence)
    soft_snn = F.log_softmax(snn_output / temperature, dim=1)
    soft_cnn = F.softmax(cnn_output / temperature, dim=1)
    soft_loss = F.kl_div(soft_snn, soft_cnn, reduction='batchmean') * (temperature ** 2)
    
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

---

## 8. 전략 H: 공정한 비교 프레임워크를 설계하라

SNN이 "이겼다"고 주장하려면 비교 기준이 명확해야 한다.

### 8.1 비교 지표

| 지표 | 측정 방법 | SNN에 유리한 조건 |
|---|---|---|
| **정확도** | 테스트셋 F1/Accuracy | KD + PLIF + 다중 시간 스케일 |
| **추론 지연시간** | 단일 샘플 wall-clock time (ms) | T≤2 + C 런타임 |
| **추론 FLOPs** | MAC(CNN) vs AC(SNN) 분리 계산 | 발화율 <10% |
| **모델 크기** | 파라미터 수 × 비트폭 (bytes) | DWS Conv + 양자화 |
| **메모리 사용량** | 피크 RAM (추론 시) | 작은 모델 + 양자화된 막전위 |
| **에너지 추정** | AC당 0.9pJ vs MAC당 4.6pJ 기준 | 이론적 에너지 우위 |

### 8.2 AC vs MAC 에너지 비교 (45nm 공정 기준)

| 연산 | 에너지 (pJ) | SNN에서의 빈도 |
|---|---|---|
| FP32 MAC | 4.6 | 첫 레이어만 |
| FP32 AC (덧셈만) | 0.9 | 나머지 레이어 × 발화율 |
| INT8 MAC | 0.2 | CNN 양자화 시 |
| INT8 AC | 0.03 | SNN + 양자화 시 |

**핵심**: INT8로 양자화된 CNN과 비교해도 SNN의 AC가 에너지 효율에서 우위이다. CPU에서 직접 에너지를 측정하기 어렵지만, 이 이론값으로 에너지 효율을 추정할 수 있다.

### 8.3 벤치마크 코드 구조

```python
import time
import numpy as np

def benchmark_model(model, test_loader, device='cpu', warmup=10, repeats=100):
    model.eval()
    model.to(device)
    
    sample = next(iter(test_loader))[0][:1].to(device)
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model(sample)
    
    # 측정
    latencies = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        with torch.no_grad():
            model(sample)
        end = time.perf_counter_ns()
        latencies.append((end - start) / 1e6)  # ms
    
    return {
        'mean_latency_ms': np.mean(latencies),
        'p50_latency_ms': np.median(latencies),
        'p99_latency_ms': np.percentile(latencies, 99),
        'params': sum(p.numel() for p in model.parameters()),
        'model_size_kb': sum(p.numel() * p.element_size() 
                            for p in model.parameters()) / 1024
    }
```

---

## 9. 종합 로드맵: CPU에서 SNN이 CNN을 이기는 최적 조합

### Phase 1: Baseline 수립

```
1. CNN baseline: 1D-CNN (Conv×3 + FC), FP32, PyTorch
   → 정확도, 지연시간, 모델 크기 측정

2. SNN naive: 동일 구조 Conv×3 + FC + LIF, T=16, PyTorch/snnTorch
   → 대부분의 지표에서 CNN보다 나쁠 것 (이것이 정상)
```

### Phase 2: 점진적 최적화

```
3. T 축소: T=16 → T=4 → T=2 → T=1
   → 막전위 기반 출력으로 전환, learnable threshold 적용

4. 모델 경량화: DWS Spiking Conv, 채널 수 절반
   → 파라미터를 CNN의 1/10로 축소

5. 희소성 강제: firing rate regularization (target 5%)
   → AC 연산의 실질적 이점 활성화

6. 양자화: 가중치 INT8 → INT4, 막전위 INT8
   → 모델 크기와 메모리 접근 비용 추가 절감
```

### Phase 3: 추론 최적화

```
7. C 런타임: PyTorch → ONNX → Custom C inference
   → 프레임워크 오버헤드 제거, 10× 속도 향상

8. 희소 연산: 조건부 AC + 채널 게이팅
   → 발화율 5%에서 추가 3-5× 속도 향상
```

### Phase 4: 정확도 회복

```
9. Knowledge Distillation: CNN teacher → SNN student
   → 정확도를 CNN 수준으로 복원

10. Multi-timescale PLIF: β를 학습 가능하게
    → 시간적 표현력으로 CNN 대비 1-2% 정확도 우위
```

### 최종 기대 결과

| 지표 | CNN (PyTorch, FP32) | SNN (C 런타임, INT8, T=2) | 비교 |
|---|---|---|---|
| 정확도 | 96% | 95-97% | 동등~우위 |
| 추론 시간 | ~2ms | ~0.3ms | **6× 빠름** |
| 모델 크기 | 800KB | 15KB | **50× 작음** |
| 피크 메모리 | 3.2MB | 0.1MB | **32× 작음** |
| 이론 에너지 | 100% (기준) | ~5% | **20× 효율** |

---

## 10. 핵심 요약: CPU에서 SNN이 이기는 3가지 조건

1. **T를 극한으로 줄여라** (T≤2): 타임스텝 반복이 CPU에서의 최대 비용이다. 막전위 기반 출력 + learnable threshold로 T=1~2에서도 충분한 정확도를 확보하라.

2. **모델을 극단적으로 작게 만들어라**: SNN은 뉴런 동역학으로 시간 정보를 "무료"로 처리하므로, CNN보다 10배 작은 모델로도 동등한 성능을 낼 수 있다. 작은 모델 = CPU 캐시 적재 = 빠른 추론.

3. **스파이크의 이진 특성을 끝까지 활용하라**: MAC → AC 변환, 양자화, 조건부 연산, C 런타임 최적화를 모두 적용해야 한다. 어느 하나라도 빠지면 SNN의 CPU 이점이 실현되지 않는다.
