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
| candidate_clamp_up | 0.2650 | 0.2582 | 0.1491 | 201 |
| random_ablate_0 | 0.3324 | 0.3138 | 0.1936 | 201 |
| random_ablate_1 | 0.3324 | 0.3138 | 0.1936 | 201 |
| random_ablate_2 | 0.3324 | 0.3138 | 0.1936 | 201 |
| random_ablate_3 | 0.3325 | 0.3138 | 0.1936 | 201 |
| random_ablate_4 | 0.3324 | 0.3138 | 0.1936 | 201 |
| random_clamp_up_0 | 0.3333 | 0.3145 | 0.1938 | 201 |
| random_clamp_up_1 | 0.3321 | 0.3138 | 0.1935 | 201 |
| random_clamp_up_2 | 0.3323 | 0.3134 | 0.1935 | 201 |
| random_clamp_up_3 | 0.3321 | 0.3133 | 0.1936 | 201 |
| random_clamp_up_4 | 0.3324 | 0.3137 | 0.1940 | 201 |

## Bidirectional + control-beating check

- **Candidate ablate drop:** +0.0841
- **Random ablate drop:** -0.0000 ± 0.0000
- **Candidate clamp-up rise:** -0.0674
- **Random clamp-up rise:** -0.0000 ± 0.0005

- **Drop z vs random:** +7397.48σ
- **Rise z vs random:** -132.73σ

## Kill check

**PARTIAL** — only one direction beats controls. The feature(s) may be necessary-but-not-sufficient (ablate-only passes) or sufficient-but-not-necessary (clamp-only passes). Worth reporting as a partial causal claim; investigate the asymmetry.