# Phase 1 — Construction classifier evaluation

**Kill check (0.85 P/R on C1–C3, regex-only):** PASSED ✅

**Strict mode (regex + dependency filter):** PASSED ✅

## Caveat

This validation set was hand-written by the same agent that wrote the regex patterns. Perfect 1.00 P/R is a self-consistency check, not a generalization test — the kill check is met but the real measurement happens in Phase 2, when M1 is run against actual model generations and the classifier sees inputs it wasn't tuned for. Treat these numbers as a sanity floor.

## Per-variant (regex only)

| Variant | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **C1** (gated) | 10 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 |
| **C2** (gated) | 10 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 |
| **C3** (gated) | 10 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 |
| **C4** | 5 | 0 | 0 | 95 | 1.000 | 1.000 | 1.000 |

**Any-core (C1∪C2∪C3):** P=1.000, R=1.000 (TP=30, FP=0, FN=0, TN=70)

## Per-variant (strict: regex + dependency filter)

| Variant | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **C1** (gated) | 10 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 |
| **C2** (gated) | 10 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 |
| **C3** (gated) | 10 | 0 | 0 | 90 | 1.000 | 1.000 | 1.000 |
| **C4** | 5 | 0 | 0 | 95 | 1.000 | 1.000 | 1.000 |

**Strict vs regex-only diff:** 0 rows. The dependency filter doesn't change any verdict on this set — every regex hit's negation token traces back to a verb/aux head, so nothing is filtered. The filter exists to catch parse-ambiguous edge cases in Phase 2 generations.
