# Phase 4 — M2 causal validation

**Features under test:** [5371, 9841, 13611, 3877, 6137]
**Variants:** ['C1', 'C2', 'C3']  
**Truncated D1 samples:** 201  
**Random-k controls:** 5 independent draws  
**Clamp-up value:** 10.0 (uniform; refine in iteration if needed)

## Mean P(pivot) by condition

| Condition | mean | median | std | n |
|---|---:|---:|---:|---:|
| baseline | 0.3324 | 0.3138 | 0.1936 | 201 |
| candidate_ablate | 0.3325 | 0.3138 | 0.1936 | 201 |
| candidate_clamp_up | 0.3111 | 0.2846 | 0.1838 | 201 |
| random_ablate_0 | 0.3325 | 0.3138 | 0.1937 | 201 |
| random_ablate_1 | 0.3325 | 0.3138 | 0.1937 | 201 |
| random_ablate_2 | 0.3324 | 0.3138 | 0.1936 | 201 |
| random_ablate_3 | 0.3326 | 0.3138 | 0.1938 | 201 |
| random_ablate_4 | 0.3324 | 0.3138 | 0.1936 | 201 |
| random_clamp_up_0 | 0.3306 | 0.3083 | 0.1942 | 201 |
| random_clamp_up_1 | 0.3301 | 0.3109 | 0.1947 | 201 |
| random_clamp_up_2 | 0.3296 | 0.3072 | 0.1938 | 201 |
| random_clamp_up_3 | 0.3291 | 0.3114 | 0.1925 | 201 |
| random_clamp_up_4 | 0.3286 | 0.3100 | 0.1920 | 201 |

## Bidirectional + control-beating check

- **Candidate ablate drop:** -0.0000
- **Random ablate drop:** -0.0001 ± 0.0001
- **Candidate clamp-up rise:** -0.0213
- **Random clamp-up rise:** -0.0028 ± 0.0008

- **Drop z vs random:** +0.75σ
- **Rise z vs random:** -23.73σ

## Kill check

**FAIL** — neither direction beats controls. The construction is not localized to this feature/supernode. Pivot to a larger supernode or stop. Honest negative result is still worth writing up (PRD §0 — that was the lesson).