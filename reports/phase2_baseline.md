# Phase 2 — Behavioral baseline (M1)

Per-model construction rates on D2 neutral prompts. M1 is computed **per sentence**, not per generation (see HANDOVER §5 P2 gotcha — a 300-token generation contains multiple sentences and the resampling unit is the sentence). CIs are 95% bootstrap with 2 000 resamples.

## Setup

- 102 D2 prompts × 5 seeds = 510 generations per model
- Sampling: temperature=0.8, top_p=0.95, max_new_tokens=150
- Classifier: strict mode (regex + spaCy dependency filter)

## Construction rates

| Model | n_sent | C1 | C2 | C3 | C4 | any_core (C1∪C2∪C3) |
|---|---:|---|---|---|---|---|
| pythia_70m | 3229 | 0.015 (0.011–0.020) | 0.003 (0.001–0.005) | 0.001 (0.000–0.002) | 0.000 (0.000–0.000) | 0.019 (0.015–0.024) |
| gpt2 | 2682 | 0.013 (0.008–0.017) | 0.002 (0.000–0.004) | 0.002 (0.000–0.003) | 0.000 (0.000–0.000) | 0.016 (0.012–0.021) |
| gemma_2b | 1433 | 0.003 (0.001–0.006) | 0.000 (0.000–0.000) | 0.001 (0.000–0.003) | 0.000 (0.000–0.000) | 0.004 (0.001–0.008) |
| gemma_2b_it | 1078 | 0.001 (0.000–0.003) | 0.000 (0.000–0.000) | 0.017 (0.009–0.024) | 0.000 (0.000–0.000) | 0.018 (0.010–0.026) |

## Base vs instruct (Gemma 2 2B)

The Phase 6 genealogy hypothesis (PRD §8 P6) predicts the construction is *dormant in base, amplified by instruct*. Phase 2 surfaces the gap.

- **C1**: base = 0.003 [0.001,0.006], instruct = 0.001 [0.000,0.003], ratio = 0.33×
- **C2**: base = 0.000 [0.000,0.000], instruct = 0.000 [0.000,0.000], ratio = 1.00×
- **C3**: base = 0.001 [0.000,0.003], instruct = 0.017 [0.009,0.024], ratio = 11.96×
- **any_core**: base = 0.004 [0.001,0.008], instruct = 0.018 [0.010,0.026], ratio = 4.21×

**any_core gap (instruct − base): +0.013.**
CIs do not overlap — the gap is clean. Phase 6 has a live story to chase.
