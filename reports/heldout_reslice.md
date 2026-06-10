# Held-out re-slices (selection/evaluation independence)

The deployed coalition was selected on the first 40 D1 prefixes
(`pivot_attribution_n40.json`, promote-ranked top-25). These slices
of the committed eval JSONs are out-of-sample w.r.t. that selection.
Union = v2 detector post-2026-06-09 fix; strict = v1 non-strict.

| slice | n | union baseline → ablated | rel drop (95% CI) | mid-p | strict |
|---|---:|---|---|---:|---|
| Q5c neutral, 26 confirmation-split prompts (78 pairs) | 78 | 7 → 5 | 28.6% [+0.0%, +66.7%] | 0.25 | 5 → 0 |
| Q5b primed, prefixes 40-99 — outside the n=40 selection set (180 pairs) | 180 | 18 → 9 | 50.0% [-10.0%, +78.8%] | 0.06391 | 11 → 0 |
| Q5b primed, prefixes 80-99 — outside every attribution scan (60 pairs) | 60 | 1 → 0 | 100.0% [+0.0%, +100.0%] | 0.5 | 1 → 0 |

Prompt-level sign test, full Q5c (seeds collapsed within prompt): 8 prompts went construction→clean, 1 went clean→construction, 9 hit in both; mid-p = 0.02148 over 102 prompts.

Q5d (the two-feature eval) has no held-out slice: all 40 of its
prefixes are inside the selection set. It should be read as an
in-sample demo number, not a confirmation.
