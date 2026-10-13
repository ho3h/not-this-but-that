# Dose response: switching off more and more (exploratory)

40 meaning/value prompts from D4 × 5 samples per dose, gemma-2-2b-it in chat format, error-term-preserving
ablation at every position except BOS, common random numbers across doses (`scripts/dose_response.py`).
Habit = detector v3 family or cousin, plus "goes beyond". Perplexity is on held-out human prose
(`data/D3_fluency.txt`) relative to normal. Quiz = 20 short general-knowledge questions, greedy.
Blind ratings: 20 answers per dose, shuffled, rated 1–5 by an annotator who never saw the dose
(`data/dose_quality_*`).

| dose | features off | habit (family or cousin) | + goes beyond | "but" | degenerate | perplexity | quiz | fluency | quality |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 0 | 29% | 34% | 8% | 0% | 1.00× | 90% | 5.0 | 4.0 |
| L20 top 25 | 25 | 25% | 32% | 0% | 0% | 1.04× | 90% | 4.8 | 4.0 |
| L20 top 100 | 100 | 26% | 36% | 0% | 0% | 1.17× | 90% | 4.8 | 4.1 |
| L20 all 500 | 500 | 22% | 32% | 1% | 0% | 1.31× | 85% | 3.8 | 3.9 |
| 3 layers top 100 | 300 | 22% | 30% | 0% | 2% | 1.63× | 95% | 3.8 | 3.9 |
| 3 layers all | 1894 | 9% | 10% | 0% | 2% | 2.49× | 75% | 2.1 | 2.8 |

Exploratory; doses reuse D4 prompts already used for confirmation v2. 200 answers per dose, so
differences of a few points between the lower doses are within noise.
