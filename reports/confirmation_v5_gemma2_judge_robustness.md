# Confirmation v5 (Gemma 2): second blind judge (non-gating)

The registered verdicts used one blind annotator per check (preference: two instances of the same model, items 1–60
and 61–120). A second, independent annotator on a different model judged every pair again with the same frozen
instructions (`data/confirmation_v5_gemma2_pairs_labels_j2.tsv`, `data/confirmation_v5_gemma2_never_pairs_labels_j2.tsv`).

| check | judge | switched off wins | ties | normal wins | share | gate |
|---|---|---:|---:|---:|---:|---:|
| preference (H3) | registered | 47 | 4 | 69 | 41% | ≥ 40% |
| preference (H3) | second | 59 | 5 | 56 | 51% | |
| warning clarity (H4) | registered | 19 | 9 | 32 | 39% | ≥ 40% |
| warning clarity (H4) | second | 24 | 8 | 28 | 47% | |

Agreement between judges: 82 of 120 preference pairs, 42 of 60 warning pairs. Both registered results sit within a
few points of their gates, and the second judge places both closer to parity. The registered verdicts (H3 PASS,
H4 FAIL) are unchanged; the spread between judges is the honest error bar on them.
