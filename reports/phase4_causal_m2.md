# Phase 4 — M2 causal validation

**Features under test:** [3223]
**Variants:** ['C1', 'C2', 'C3']  
**Truncated D1 samples:** 201  
**Random-k controls:** 5 independent draws  
**Clamp-up value:** 10.0 (uniform; refine in iteration if needed)

## Mean P(pivot) by condition

| Condition | mean | median | std | n |
|---|---:|---:|---:|---:|
| baseline | 0.3324 | 0.3138 | 0.1936 | 201 |
| candidate_ablate | 0.2483 | 0.2357 | 0.1462 | 201 |
| candidate_clamp_up | 0.3063 | 0.3020 | 0.1547 | 201 |
| random_ablate_0 | 0.3324 | 0.3138 | 0.1936 | 201 |
| random_ablate_1 | 0.3324 | 0.3138 | 0.1936 | 201 |
| random_ablate_2 | 0.3324 | 0.3138 | 0.1936 | 201 |
| random_ablate_3 | 0.3325 | 0.3138 | 0.1936 | 201 |
| random_ablate_4 | 0.3324 | 0.3138 | 0.1936 | 201 |
| random_clamp_up_0 | 0.3353 | 0.3161 | 0.1941 | 201 |
| random_clamp_up_1 | 0.3313 | 0.3113 | 0.1933 | 201 |
| random_clamp_up_2 | 0.3319 | 0.3124 | 0.1933 | 201 |
| random_clamp_up_3 | 0.3312 | 0.3119 | 0.1936 | 201 |
| random_clamp_up_4 | 0.3323 | 0.3134 | 0.1947 | 201 |

## Bidirectional + control check

- **Ablate:** candidate drop = +0.0841; random-k null mean = −0.0000005 ± 0.0000114 (n=5); candidate exceeded **5/5** random draws.
- **Clamp-up:** candidate rise = −0.0262; random-k null mean = −0.0000045 ± 0.0017 (n=5); candidate exceeded 0/5 random draws (effect is in the wrong direction).

Reported this way because the random-k null distribution at the truncated pre-pivot position is **near-degenerate**: random single-feature ablation changes P(pivot) by essentially zero, with essentially zero variance. Quoting a σ multiple here would mislead — the candidate is qualitatively separated from a *flat* null, not a tail event in a smooth one. (An earlier draft of this script reported "+7397σ" on the ablate side; that number is an artifact of dividing a real effect by an almost-zero denominator and has been retired.)

## Kill check

**PARTIAL — necessity yes, sufficiency no.** Ablation separates qualitatively from the random-k null (5/5 draws below the candidate's 25% relative drop, with the null itself degenerate at ≈0). Clamp-up at value=10 moves P(pivot) the *wrong* direction relative to a similarly flat null. A separate run at clamp value=3 (see `phase4_causal_m2_3223_clamp3.md`) produces the same qualitative asymmetry — necessity holds, sufficiency does not. The construction's commit is consistent with a multi-feature coordination in which feat 3223 is one indispensable component, and pegging it high alone pushes the model into an OOD state where pivot probability falls.