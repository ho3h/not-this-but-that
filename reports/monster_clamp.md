# Clamping the features on (exploratory)

10 mundane prompts, 1 sample, 140 tokens.

| condition | answers using not X but Y | degenerate |
|---|---:|---:|
| normal | 0 / 10 | 0 |
| all 25 x1 | 0 / 10 | 10 |
| all 25 x2 | 0 / 10 | 10 |
| all 25 x4 | 0 / 10 | 1 |
| top 5 x1 | 0 / 10 | 10 |
| top 5 x2 | 0 / 10 | 10 |
| top 5 x4 | 0 / 10 | 10 |
| 13565 x1 | 1 / 10 | 5 |
| 13565 x2 | 0 / 10 | 10 |
| 13565 x4 | 0 / 10 | 10 |
| 15646 x1 | 1 / 10 | 0 |
| 15646 x2 | 6 / 10 | 1 |
| 15646 x4 | 3 / 10 | 10 |
