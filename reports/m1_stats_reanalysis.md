# M1 — bootstrap CIs + paired McNemar

Re-analysis of the Q5b/Q5c/Q5d generations (saved JSON) with proper
inferential statistics. Scoring uses the v2 union detector
(strict + permissive — see `src/classifier/detect_v2.py`).

| probe | n | baseline rate | ablated rate | absolute drop (95% CI) | rel drop (95% CI) | McNemar mid-p |
|---|---:|---:|---:|---:|---:|---:|
| Q5b primed | 300 | 46/300 = 15.33% | 8/300 = 2.67% | +0.1267 [+0.0800, +0.1767] | +82.31% [+68.29%, +93.33%] | 1.832e-08 |
| Q5c neutral | 306 | 36/306 = 11.76% | 7/306 = 2.29% | +0.0948 [+0.0621, +0.1307] | +80.44% [+67.33%, +92.31%] | 1.537e-08 |
| Q5d minimal (n=16) | 16 | 6/16 = 37.50% | 1/16 = 6.25% | +0.3116 [+0.1250, +0.4375] | +85.22% [+60.00%, +100.00%] | 0.03125 |
| Q5d minimal n=120 | 120 | 23/120 = 19.17% | 5/120 = 4.17% | +0.1498 [+0.0750, +0.2250] | +77.87% [+57.14%, +94.44%] | 6.604e-05 |

Per-probe paired tables (b = baseline-hit & ablated-clean; c = baseline-clean & ablated-hit):

- **Q5b primed**: both yes 2, baseline-only 44, ablated-only 6, both no 248; discordant 50.
- **Q5c neutral**: both yes 6, baseline-only 30, ablated-only 1, both no 269; discordant 31.
- **Q5d minimal (n=16)**: both yes 1, baseline-only 5, ablated-only 0, both no 10; discordant 5.
- **Q5d minimal n=120**: both yes 3, baseline-only 20, ablated-only 2, both no 95; discordant 22.