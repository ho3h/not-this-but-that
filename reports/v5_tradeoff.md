# Two switches on the same fresh prompts (Gemma 2)

Confirmation v5 registered the specific switch. `scripts/v5_original_arm.py` (exploratory) ran the original 25 on
the same 140 D7 and 30 D6 prompts with the same seeds; its normal answers are word-for-word identical to v5's
(600 of 600), so the comparison is like for like. Warning clarity was judged blind by the registered-style
annotator and by a second annotator on a different model, with the registered H4 instruction.

| switch | habit (600 everyday answers) | drop | ordinary negation vs normal | warning clarity share (judge 1 / judge 2) |
|---|---:|---:|---:|---:|
| normal | 92 | | 1.00 | |
| original 25 (v3/v4 set) | 8 | 91% | 0.60 | 23% / 27% |
| specific 25 (v5, registered) | 26 | 72% | 0.95 | 39% / 47% |

Warning clarity counts: original 25 off 12 wins / 4 ties / 44 losses (judge 1), 9 / 14 / 37 (judge 2); specific 19 / 9 / 32
and 24 / 8 / 28. Files: `reports/v5_original_arm.json`, `data/v5_original_never_pairs_*`,
`reports/confirmation_v5_gemma2_run.md`, `reports/confirmation_v5_gemma2_judge_robustness.md`.

Reading: there is a trade-off. Swapping two double-duty features gives back a fifth of the habit reduction and
recovers most of the lost negation and most of the warning clarity, though not quite all of it.
