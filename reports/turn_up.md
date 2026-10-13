# Turning the 25 up as well as off (exploratory)

D2 + D2b prompts (none used to choose the features), 2 samples each, common random numbers across
conditions, chat template, 200 tokens. Habit = detector v3 family or cousin. x2 / x3 multiply the
features' layer-20 contribution; matched 25 = confirmation v4's frozen firing-rate-matched control.
CIs are 95% prompt-clustered bootstrap.

## All prompts

| condition | answers | with the habit | 95% CI | "but" | degenerate | words |
|---|---:|---:|---:|---:|---:|---:|
| 25 off | 304 | 1 (0.3%) | 0.0–1.0% | 14% | 0% | 146 |
| normal | 304 | 35 (11.5%) | 7.9–15.8% | 22% | 0% | 147 |
| 25 x2 | 304 | 112 (36.8%) | 30.3–43.4% | 27% | 0% | 144 |
| 25 x3 | 304 | 165 (54.3%) | 47.4–61.2% | 31% | 0% | 143 |
| matched 25 x3 | 304 | 45 (14.8%) | 10.5–19.1% | 20% | 1% | 146 |

## Meaning/value prompts

| condition | answers | with the habit | 95% CI | "but" | degenerate | words |
|---|---:|---:|---:|---:|---:|---:|
| 25 off | 186 | 1 (0.5%) | 0.0–1.6% | 10% | 1% | 147 |
| normal | 186 | 29 (15.6%) | 10.2–21.5% | 12% | 0% | 147 |
| 25 x2 | 186 | 93 (50.0%) | 41.9–58.6% | 21% | 0% | 145 |
| 25 x3 | 186 | 129 (69.4%) | 61.8–76.9% | 31% | 1% | 144 |
| matched 25 x3 | 186 | 37 (19.9%) | 14.0–26.3% | 12% | 1% | 147 |

## Process/scene prompts

| condition | answers | with the habit | 95% CI | "but" | degenerate | words |
|---|---:|---:|---:|---:|---:|---:|
| 25 off | 118 | 0 (0.0%) | 0.0–0.0% | 21% | 0% | 144 |
| normal | 118 | 6 (5.1%) | 1.7–9.3% | 38% | 0% | 146 |
| 25 x2 | 118 | 19 (16.1%) | 9.3–23.7% | 37% | 0% | 142 |
| 25 x3 | 118 | 36 (30.5%) | 21.2–40.7% | 31% | 0% | 141 |
| matched 25 x3 | 118 | 8 (6.8%) | 2.5–11.0% | 32% | 0% | 145 |

## Blind checks

**Are the extra hits real?** 60 detector hits (40 from 25 x3, 20 from normal), shown as excerpts in random order
to an annotator who never saw the condition and was asked whether each dismisses one description in favour of
a grander one (`data/turn_up_precision_*`). Confirmed: 25 x3 32 of 40 (80%), normal 19 of 20 (95%). Most of the
rejected x3 hits are additive "not only X but also Y", which the detector counts as part of the family and this
stricter rubric does not. Scaling each arm by its confirmed share, the reframing rate still rises about fourfold
(about 11% → 43%).

**Is the writing still good?** 30 answers per arm (normal, 25 x3, matched 25 x3), shuffled and rated 1–5 by a
blind annotator (`data/turn_up_quality_*`): fluency 4.77 / 4.60 / 4.63, usefulness 3.93 / 3.80 / 3.87. No x3
answer was rated below 4 for fluency.
