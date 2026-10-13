# Phase 5 — Quality preservation (M3)

**Features under test:** [3223, 9909, 12898, 9816, 6759, 4197, 4516, 1250, 2137, 11864, 2282, 12524, 8530, 7361, 8820, 2184, 12923, 12506, 14401, 9606, 1608, 6336, 347, 10822, 12767]

## Fluency (D3 perplexity)

| Condition | geom-mean PPL | mean | median | n chunks |
|---|---:|---:|---:|---:|
| baseline | 24.628 | 25.080 | 24.529 | 6 |
| ablate | 26.581 | 27.112 | 26.263 | 6 |
| clamp_up | 27.659 | 28.268 | 27.419 | 6 |

- Ablate / baseline = **1.079×**
- Clamp-up / baseline = **1.123×**

**Fluency PASS** — perplexity ratio under 1.2× baseline.

## Meaning (D1 with vs without baseline cosine)

- mean cosine = 0.773, median = 0.821, p10 = 0.567, p90 = 0.937 (n = 226)

*Baseline reference only*: this is how much meaning the D1 paraphraser preserved across the construction's removal. Phase 7 compares baseline generation vs intervened generation under the same metric — the actual product claim is the Phase 7 number.

## Coherence — LLM-judge

Phase 5 v2 wires Anthropic Claude as the LLM judge on a sample of Phase 4 M1 ablate generations (baseline vs intervened, blind to condition), 1-5 coherence rating. Skipped here.