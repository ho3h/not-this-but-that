# Confirmation v4: second blind judge (non-gating)

The registered preference gate used one blind annotator (two instances of the same model, items 1–60 and
61–120, `data/confirmation_v4_pairs_labels.tsv`). A second, independent annotator on a different model
judged all 120 pairs with the same frozen instruction (`data/confirmation_v4_pairs_labels_j2.tsv`).

| judge | chat 25 wins | ties | normal wins | share for chat 25 |
|---|---:|---:|---:|---:|
| registered | 54 | 4 | 62 | 47% |
| second | 52 | 6 | 62 | 46% |

The two agree on 91 of 120 pairs. Neither preference differs from an even split by more than chance
(54 vs 62 and 52 vs 62, two-sided sign tests p ≈ 0.5 and 0.4).
