# Phase 5 — Quality preservation (M3)

**Features under test:** [3223]

## Fluency (D3 perplexity)

| Condition | geom-mean PPL | mean | median | n chunks |
|---|---:|---:|---:|---:|
| baseline | 24.628 | 25.080 | 24.529 | 6 |
| ablate | 24.633 | 25.058 | 24.355 | 6 |
| clamp_up | 24.785 | 25.228 | 24.548 | 6 |

- Ablate / baseline = **1.000×**
- Clamp-up / baseline = **1.006×**

**Fluency PASS** — perplexity ratio under 1.2× baseline.

## Meaning (D1 with vs without baseline cosine)

- mean cosine = 0.773, median = 0.821, p10 = 0.567, p90 = 0.937 (n = 226)

*Baseline reference only*: this is how much meaning the D1 paraphraser preserved across the construction's removal. Phase 7 compares baseline generation vs intervened generation under the same metric — the actual product claim is the Phase 7 number.

## Coherence — LLM-judge

Phase 5 v2 wires Anthropic Claude as the LLM judge on a sample of Phase 4 M1 ablate generations (baseline vs intervened, blind to condition), 1-5 coherence rating. Skipped here.