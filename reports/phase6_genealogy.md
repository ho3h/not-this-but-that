# Phase 6 — Genealogy (base vs instruct)

**Features under test:** [3223]

## SAE reconstruction quality

Gemma Scope is trained on the base model. PRD §8 P6 says we must verify reconstruction on the instruct model before trusting any instruct-side number.


| Model | var explained (mean) | var explained (median) | L2 err (mean) |
|---|---:|---:|---:|
| gemma-2-2b (base) | -5.860 | -5.885 | 248.486 |
| gemma-2-2b-it (instruct) | -7.147 | -7.158 | 304.053 |

**Instruct VE = -7.147** — SAE transfer is poor. Numbers below are reported but should NOT be quoted without this caveat.

## Feature activation on D2 (mean of last-token activations, n=50)

| Feature | base mean | instruct mean | ratio | base %active | instruct %active |
|---:|---:|---:|---:|---:|---:|
| 3223 | 0.148 | 0.000 | 0.00× | 2.00% | 0.00% |

## Ablation effect on truncated D1 with-prompts

| Model | baseline P(pivot) | ablate P(pivot) | drop (mean ± std) | n |
|---|---:|---:|---:|---:|
| gemma-2-2b (base) | 0.3324 | 0.2483 | +0.0841 ± 0.0690 | 201 |
| gemma-2-2b-it (instruct) | 0.4775 | 0.3255 | +0.1520 ± 0.0862 | 201 |

**Drop ratio (instruct / base): 1.81×**

## Genealogy verdict

**PARTIAL** — feature activation is similar between base (0.00× ratio) but ablation drops 1.81× more in instruct. The feature is more *causally load-bearing* in instruct without firing more on average.