# M1 — bootstrap CIs + paired McNemar

Re-analysis of the Q5b/Q5c/Q5d generations (saved JSON) with proper
inferential statistics. Scoring uses the v2 union detector
(strict + permissive — see `src/classifier/detect_v2.py`).

| probe | n | baseline rate | ablated rate | absolute drop (95% CI) | rel drop (95% CI) | McNemar mid-p |
|---|---:|---:|---:|---:|---:|---:|
| Q5b primed | 300 | 37/300 = 12.33% | 18/300 = 6.00% | +0.0634 [+0.0200, +0.1100] | +50.37% [+20.59%, +72.50%] | 0.004534 |
| Q5c neutral | 306 | 18/306 = 5.88% | 10/306 = 3.27% | +0.0261 [+0.0065, +0.0490] | +43.65% [+14.29%, +70.00%] | 0.01172 |
| Q5d minimal (n=16) | 16 | 5/16 = 31.25% | 2/16 = 12.50% | +0.1873 [+0.0625, +0.3750] | +60.65% [+20.00%, +100.00%] | 0.125 |
| Q5d minimal n=120 | 120 | 19/120 = 15.83% | 9/120 = 7.50% | +0.0835 [+0.0250, +0.1500] | +51.84% [+18.18%, +78.57%] | 0.01273 |

Per-probe paired tables (b = baseline-hit & ablated-clean; c = baseline-clean & ablated-hit):

- **Q5b primed**: both yes 5, baseline-only 32, ablated-only 13, both no 250; discordant 45.
- **Q5c neutral**: both yes 9, baseline-only 9, ablated-only 1, both no 287; discordant 10.
- **Q5d minimal (n=16)**: both yes 2, baseline-only 3, ablated-only 0, both no 11; discordant 3.
- **Q5d minimal n=120**: both yes 6, baseline-only 13, ablated-only 3, both no 98; discordant 16.