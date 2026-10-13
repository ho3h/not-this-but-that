# Phase 4 — M2 causal validation

**Features under test:** [9841]
**Variants:** ['C1', 'C2', 'C3']  
**Truncated D1 samples:** 201  
**Random-k controls:** 5 independent draws  
**Clamp-up value:** 10.0 (uniform; refine in iteration if needed)

## Mean P(pivot) by condition

| Condition | mean | median | std | n |
|---|---:|---:|---:|---:|
| baseline | 0.3324 | 0.3138 | 0.1936 | 201 |
| candidate_ablate | 0.3325 | 0.3138 | 0.1936 | 201 |
| candidate_clamp_up | 0.3266 | 0.3063 | 0.1912 | 201 |
| random_ablate_0 | 0.3324 | 0.3138 | 0.1936 | 201 |
| random_ablate_1 | 0.3324 | 0.3138 | 0.1936 | 201 |
| random_ablate_2 | 0.3324 | 0.3138 | 0.1936 | 201 |
| random_ablate_3 | 0.3324 | 0.3138 | 0.1936 | 201 |
| random_ablate_4 | 0.3324 | 0.3138 | 0.1936 | 201 |
| random_clamp_up_0 | 0.3353 | 0.3161 | 0.1941 | 201 |
| random_clamp_up_1 | 0.3313 | 0.3113 | 0.1933 | 201 |
| random_clamp_up_2 | 0.3319 | 0.3124 | 0.1933 | 201 |
| random_clamp_up_3 | 0.3342 | 0.3145 | 0.1938 | 201 |
| random_clamp_up_4 | 0.3320 | 0.3130 | 0.1934 | 201 |

## Bidirectional + control-beating check

- **Candidate ablate drop:** -0.0000
- **Random ablate drop:** +0.0000 ± 0.0000
- **Candidate clamp-up rise:** -0.0058
- **Random clamp-up rise:** +0.0005 ± 0.0017

- **Drop z vs random:** -9.78σ
- **Rise z vs random:** -3.70σ

## Kill check

**FAIL** — neither direction beats controls. The construction is not localized to this feature/supernode. Pivot to a larger supernode or stop. Honest negative result is still worth writing up (PRD §0 — that was the lesson).