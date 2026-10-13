# Self-priming: what happens after one "not X but Y"? (exploratory)

Gemma 2 2B-it. 40 held-out prompts, the answer started with one of three openers carrying the same content; 8 samples each, 200 new tokens. Only the continuation is scored (detector v3, family or cousin).

| prompts | opener | continuations using not X but Y | rate | in the next sentence | median position of first use (characters) |
|---|---|---:|---:|---:|---:|
| all 40 | construction | 18 / 320 | 5.6% | 2 | 471 |
| all 40 | plain | 67 / 320 | 20.9% | 60 | 3 |
| all 40 | both | 45 / 320 | 14.1% | 34 | 3 |
| 20 process | construction | 1 / 160 | 0.6% | 0 | 332 |
| 20 process | plain | 6 / 160 | 3.8% | 5 | 12 |
| 20 process | both | 9 / 160 | 5.6% | 5 | 146 |
| 20 meaning | construction | 17 / 160 | 10.6% | 2 | 518 |
| 20 meaning | plain | 61 / 160 | 38.1% | 55 | 3 |
| 20 meaning | both | 36 / 160 | 22.5% | 29 | 3 |

Rate ratios (95% prompt-clustered bootstrap CI):

- all 40, construction/plain: **0.27×** (0.12× to 0.49×)
- all 40, construction/both: **0.40×** (0.19× to 0.69×)
- all 40, both/plain: **0.67×** (0.48× to 0.90×)
- 20 process, construction/plain: **0.17×** (0.00× to 1.00×)
- 20 process, construction/both: **0.11×** (0.00× to 0.60×)
- 20 process, both/plain: **1.50×** (0.71× to 5.50×)
- 20 meaning, construction/plain: **0.28×** (0.12× to 0.53×)
- 20 meaning, construction/both: **0.47×** (0.22× to 0.84×)
- 20 meaning, both/plain: **0.59×** (0.39× to 0.80×)

## Reading

The prediction (one use primes more) was wrong. After the construction opener the model rarely uses it again, and when it does, it is far down the answer. After a plain grand claim ("X is Y"), it usually supplies the missing foil in the very next sentence ("It's not just about books anymore, it's about community"). The construction behaves less like a contagion and more like a slot: an answer about meaning gets roughly one, and a grand claim without a foil invites it. For scale, the same 40 prompts with no opener (turn-up run, 2 samples each) used it in 11 of 40 whole answers to meaning prompts and 0 of 40 to process prompts.
