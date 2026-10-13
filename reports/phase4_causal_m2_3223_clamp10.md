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

## Bidirectional + control-beating check

- **Candidate ablate drop:** +0.0841
- **Random ablate drop:** -0.0000 ± 0.0000
- **Candidate clamp-up rise:** -0.0262
- **Random clamp-up rise:** -0.0000 ± 0.0017

- **Drop z vs random:** +7397.48σ
- **Rise z vs random:** -15.53σ

## Kill check

**PARTIAL** — only one direction beats controls. The feature(s) may be necessary-but-not-sufficient (ablate-only passes) or sufficient-but-not-necessary (clamp-only passes). Worth reporting as a partial causal claim; investigate the asymmetry.