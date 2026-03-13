# Section 4.3.1 Rewrite — Sparsity and Firing-Rate Control

> Replace the existing 4.3.1 paragraph and Figure 12 with the following Table + text.

---

## Table X: Effect of Phase-3 sparsity-aware fine-tuning on firing rate, accuracy, and operation reduction (T=10, k=9, target firing rate = 5%).

| Variant  | Ch Config | Phase-2 Avg FR (%) | Phase-3 Avg FR (%) | Phase-2 Score | Phase-3 Score | Skippable Ops (%) |
|----------|-----------|-------------------|-------------------|--------------|--------------|------------------|
| smallest | (16, 32)  | 42.3              | 10.1              | 1.950        | 1.957        | 89.9             |
| small    | (24, 48)  | 42.0              | 8.3               | 1.971        | 1.977        | 91.7             |
| medium   | (32, 64)  | 34.6              | 6.6               | 1.968        | 1.976        | 93.4             |
| large    | (40, 80)  | 38.4              | 7.1               | 1.974        | 1.983        | 92.9             |
| largest  | (48, 96)  | 35.3              | 7.3               | 1.974        | 1.981        | 92.7             |

> Avg FR = mean of Layer 1 and Layer 2 firing rates. Score = train accuracy + validation accuracy. Skippable Ops = 1 − Phase-3 Avg FR, representing the fraction of synaptic multiply-accumulate operations that can be skipped due to zero-valued binary spikes.

---

## Text (replaces existing 4.3.1 paragraph)

**4.3.1 Sparsity and Firing-Rate Control.** Table X quantifies the effect of Phase-3 sparsity-aware fine-tuning across all five channel-capacity variants (T = 10). Before fine-tuning, the Phase-2 binary-spike student exhibits natural firing rates of 35–42% averaged over both spiking layers. After applying learnable thresholds with curriculum-scheduled firing-rate regularization (target rate 5%, λ ramped from 0 to 5.0 over the first 10 epochs), the average firing rate drops to 6.6–10.1%, enabling 89.9–93.4% of synaptic multiply-accumulate operations to be skipped at inference. Notably, this aggressive sparsification does not degrade accuracy: the combined train + validation score either matches or slightly *improves* over the Phase-2 baseline across all variants, suggesting that firing-rate regularization also acts as an implicit capacity regularizer that mitigates overfitting. Larger models (medium through largest) consistently achieve firing rates below 8%, approaching the 5% target more closely than the smallest variant.

---

## Data Source

All numbers are derived from `result/channel_variant_sparse_results.json` (Phase-3) and `result/channel_variant_teacher_student_results.json` (Phase-2), using T=10, target_rate=0.05 entries.

### Detailed Breakdown (T=10, FR target=5%)

**smallest (16, 32):**
- Phase-2 FR: L1=33.9%, L2=50.7% → Avg=42.3%
- Phase-3 FR: L1=10.4%, L2=9.7% → Avg=10.1%
- Phase-2 student best_score: 1.9497
- Phase-3 sparse best_score: 1.9566

**small (24, 48):**
- Phase-2 FR: L1=30.1%, L2=53.8% → Avg=42.0%
- Phase-3 FR: L1=7.7%, L2=8.9% → Avg=8.3%
- Phase-2 student best_score: 1.9709
- Phase-3 sparse best_score: 1.9768

**medium (32, 64):**
- Phase-2 FR: L1=22.5%, L2=46.6% → Avg=34.6%
- Phase-3 FR: L1=5.3%, L2=7.9% → Avg=6.6%
- Phase-2 student best_score: 1.9684
- Phase-3 sparse best_score: 1.9758

**large (40, 80):**
- Phase-2 FR: L1=26.9%, L2=49.8% → Avg=38.4%
- Phase-3 FR: L1=6.6%, L2=7.5% → Avg=7.1%
- Phase-2 student best_score: 1.9738
- Phase-3 sparse best_score: 1.9832

**largest (48, 96):**
- Phase-2 FR: L1=22.3%, L2=48.2% → Avg=35.3%
- Phase-3 FR: L1=6.1%, L2=8.5% → Avg=7.3%
- Phase-2 student best_score: 1.9743
- Phase-3 sparse best_score: 1.9812
