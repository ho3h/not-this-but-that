# Pre-registered confirmation run — verdict

Analyzed once at n = 300 pairs per `confirmation_prereg.md` (registered 2026-06-10).

- Union: baseline 3/300 (1.00%) → ablated 1/300 (0.33%)
- Relative drop: **66.7%** (95% prompt-clustered bootstrap CI [-100.0%, +100.0%])
- McNemar mid-p (two-sided): **0.375** (kills 3, leaks 1)
- Strict detector (secondary, non-gating): 2 → 0
- "More than just" cousin (secondary, non-gating): 2 → 3
- Context the gates don't capture: the D2b baseline rate (1.0%) is a fifth of the exploratory D2 rate (5.9%) — the predicted ~18 baseline events did not materialize, so the test had no power. The direction is consistent with the exploratory result; the magnitude is unverifiable at this event count.

## Verdict against the pre-registered gates: **KILL**

(PASS requires rel drop ≥ 30% and mid-p < 0.05; KILL if rel drop < 15% or mid-p ≥ 0.05.)
