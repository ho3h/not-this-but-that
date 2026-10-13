# A more specific switch (exploratory)

The decision-point features also carry ordinary negation (`negation_check.md`). `scripts/specific_selection.py` scores
every strong habit candidate (top 100 family, top 60 cousin) on a second set of decision points: the token before an
ordinary negation ("you don't need", "it can't", "never leave") in the same normal selection answers, outside any
construction. A feature is *specific* if switching it off alone lowers the probability of that negation by less than
5% of baseline. The specific set takes the top specific features by the same ranking as before.

## Which features carry ordinary negation

Gemma 2 (layer 20): of the 25, two are unspecific.

- 15646: alone lowers ordinary negation by 12% · phrases indicating conditions or exceptions in narratives
- 10655: alone lowers ordinary negation by 42% · terms related to legal processes and court proceedings

Gemma 3 (layer 22, set B): three are unspecific.

- 298: alone lowers ordinary negation by 10% · [no label; promotes: are, were, aren, jsou, have, são]
- 1138: alone lowers ordinary negation by 60% · [no label; promotes: nicht, ikke, não, không, tidak, не]
- 6524: alone lowers ordinary negation by 6% · starts questions

## Held-out evaluation (D2 prompts × 2, 200 tokens; 30 'no' questions)

| model / set | features | habit | negation words per 1,000 | don't-type contractions | 'no' questions right |
|---|---:|---:|---:|---:|---:|
| Gemma 2 · normal | 0 | 41/204 (—) | 2.23 | 15 | 28/30 |
| Gemma 2 · current | 25 | 1/204 (−98%) | 0.94 | 9 | 28/30 |
| Gemma 2 · specific | 25 | 3/204 (−93%) | 2.27 | 18 | 28/30 |
| Gemma 3 · normal | 0 | 94/204 (—) | 7.36 | 53 | 28/30 |
| Gemma 3 · current | 24 | 1/204 (−99%) | 2.01 | 26 | 28/30 |
| Gemma 3 · specific | 25 | 73/204 (−22%) | 8.34 | 104 | 28/30 |

Blind warning clarity on the 30 'never' questions (60 pairs each, vs normal): Gemma 2 current 16 wins / 10 ties / 34
losses (35%); Gemma 2 specific 23 / 15 / 22 (51%), with warnings that read as advice in 12 specific answers against 15
normal (`data/specific_never_pairs_gemma2_*`).

## Reading

In Gemma 2 the habit and ordinary negation can be pulled apart: two of the 25 do double duty, and swapping them out
keeps nearly all of the habit drop while leaving negation and warnings as they were. In Gemma 3 they cannot, at this
layer: its strongest habit feature (1138, which promotes "not" in many languages) is the general negation feature, and
without it most of the habit survives. Both are tested under pre-registration in confirmation v5
(`confirmation_v5_gemma2_prereg.md`, `confirmation_v5_gemma3_prereg.md`).
