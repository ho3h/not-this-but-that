# Tier 0a — Classifier blind validation

**Set:** 60 sentences (30 human D3 + 30 AI Gemma-2-2b-it).  
**Pre-registered kill threshold:** P/R ≥ 0.70 on C1-C3 (any_core).

## Per-variant

| Variant | n true positives | TP | FP | FN | TN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| **C1** | 0 | 0 | 0 | 0 | 60 | 1.000 | 1.000 |
| **C2** | 0 | 0 | 0 | 0 | 60 | 1.000 | 1.000 |
| **C3** | 1 | 1 | 0 | 0 | 59 | 1.000 | 1.000 |
| **C4** | 0 | 0 | 0 | 0 | 60 | 1.000 | 1.000 |
| **any_core (C1∪C2∪C3)** | 1 | 1 | 0 | 0 | 59 | 1.000 | 1.000 |

## Kill check

- any_core precision 1.000 ≥ 0.70: PASS
- any_core recall    1.000 ≥ 0.70: PASS

**Per-variant kill verdicts (only variants with ≥1 true positive are gated):**
- C1: no positives in this set (not gated). Precision 1.000.
- C2: no positives in this set (not gated). Precision 1.000.
- C3: precision = 1.000, recall = 1.000 → PASS

### Verdict: PASS — proceed to Tier 0b

## Mismatches

None.

## Caveat — small positive set

This blind set contains only 1 true positive across 60 sentences (consistent with the natural ~1.8% any_core rate in Phase 2). With n=1 positive, recall is binary (the classifier finds it or doesn't) and precision depends entirely on the false-positive count. The kill threshold here is informative but not statistically tight; if this passes, Tier 0a is followed by a positive-enriched test drawn from a wider Phase 2 pool to thicken the recall measurement.