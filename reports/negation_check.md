# Can the model still say no? (exploratory)

See `scripts/negation_check.py`. Questions: `data/negation_check_questions.json`.

## gemma2

| condition | 'no' questions right | 'yes' questions right | P(right), 'no' qs | 'never' answers with a prohibition | negations per 1k words |
|---|---:|---:|---:|---:|---:|
| normal | 28/30 | 30/30 | 0.94 | 98% | 26.3 |
| 25 off | 28/30 | 30/30 | 0.93 | 100% | 12.3 |
| matched 25 | 28/30 | 30/30 | 0.94 | 98% | 22.4 |

## gemma3

| condition | 'no' questions right | 'yes' questions right | P(right), 'no' qs | 'never' answers with a prohibition | negations per 1k words |
|---|---:|---:|---:|---:|---:|
| normal | 28/30 | 28/30 | 0.93 | 98% | 29.2 |
| A | 28/30 | 28/30 | 0.93 | 100% | 11.3 |
| B | 28/30 | 28/30 | 0.93 | 98% | 12.6 |
| C | 28/30 | 28/30 | 0.93 | 100% | 13.8 |
| matched 25 | 28/30 | 28/30 | 0.93 | 97% | 27.0 |

## Blind warning-clarity check

For each model, the 60 answers to the 30 "what should you never do" questions were paired (normal vs switched off),
order randomised, and judged blind: which answer does a better job of telling the person what to avoid, with clear,
correct warnings that a skimming reader could not mistake for advice (`data/negation_pairs_*`).

| model | switched off wins | ties | normal wins | share for switched off | answers with a warning that reads as advice (switched off / normal) |
|---|---:|---:|---:|---:|---:|
| Gemma 2, the 25 | 16 | 10 | 34 | 35% | 15 / 12 |
| Gemma 3, set B | 11 | 20 | 29 | 35% | 17 / 3 |

Typical difference: under a heading such as "things you should never do", the normal answer writes "**Never connect the
positive (red) cable to the negative (black) battery terminal.**"; the switched-off answer writes "**Connect the positive and negative cables to
the wrong terminals.**" Correct in context, but less clear on its own.

Negation words in everyday answers (confirmation v4, Gemma 2): 6.7 → 3.1 per 1,000 words with the 25 off (matched
control 6.4); "don't / doesn't / can't / won't / didn't" 278 → 172.
