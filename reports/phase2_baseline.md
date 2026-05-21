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
| gemma_2b | 0 | — | — | — | — | — |
| gemma_2b_it | 0 | — | — | — | — | — |

