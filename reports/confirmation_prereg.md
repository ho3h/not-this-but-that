# Pre-registered confirmation run — coalition ablation on fresh neutral prompts

**Registered: 2026-06-10, before any generation on the D2b prompt set.**
Commit this file (and `confirmation_prereg.json` + `data/D2b_confirmation_prompts.json`)
before running `scripts/confirmation_run.py`. The runner reads the frozen
parameters from the JSON twin of this file and refuses to run if they're absent.

## What is frozen

- **Coalition (25 features, layer-20 Gemma Scope 16k):**
  347, 1250, 1608, 2137, 2184, 2282, 3223, 4197, 4516, 6336, 6759, 7361,
  8530, 8820, 9606, 9816, 9909, 10822, 11864, 12506, 12524, 12767, 12898,
  12923, 14401
  (= the promote-ranked top-25 of `reports/pivot_attribution.json`, the n=40
  selection artifact every prior eval used. Not re-derived at run time.)
- **Detector:** the v2 union (strict v1 non-strict ∪ permissive) exactly as
  committed on 2026-06-10 — `src/classifier/detect_v2.py` post
  negation-mandatory revision, FP audit at `reports/permissive_fix_audit.md`.
  No detector changes between registration and analysis.
- **Prompts:** the 50 D2b prompts (`data/D2b_confirmation_prompts.json`),
  written after coalition + detector were frozen, never used anywhere before.
- **Generation config:** `gemma-2-2b-it`, SAE spliced in for BOTH conditions
  (baseline = empty hooks), 50 new tokens, temperature 0.8, top_p 0.95,
  seeds 0–5. n = 50 prompts × 6 seeds = **300 pairs**.

## Pre-registered prediction

Based on the exploratory Q5c result (negated family 5.9% → 3.3%, −44%
relative, CI 14–70%, mid-p 0.012), we predict the family construction rate
on D2b drops by **≥ 30% relative** with McNemar mid-p < 0.05.

## Decision rule (single analysis, no interim looks)

- **PASS** if family relative drop ≥ 30% AND McNemar mid-p (two-sided) < 0.05.
- **KILL** if family relative drop < 15% OR mid-p ≥ 0.05.
- Between 15% and 30% with p < 0.05: report as "replicated direction, smaller
  magnitude" — neither pass nor kill; the post must quote the D2b number.
- Reported alongside, non-gating: the strict-detector drop (exploratory
  expectation ≥ 70%) and the "more than just" cousin count (exploratory
  expectation: flat or rising — the reroute).
- One analysis, run once at n = 300. The runner checkpoints for crash
  recovery but does not print running rates, and no decisions are taken on
  partial data. No detector edits, no prompt edits, no n extension after
  the first generation.

## Power note

At the exploratory baseline rate (5.9%) we expect ~18 baseline family events
in 300 pairs; at a true relative drop of 44% the expected McNemar mid-p is
~10⁻². At a true drop of 30% the test is marginal — that is intentional:
the gate is set at the smallest effect we claimed, not the one we hope for.

## Reporting commitment

The outcome — pass, kill, or in-between — gets reported in the Medium post
and README verbatim, including the kill. The phrase committed for a kill:
"the pre-registered confirmation failed; the coalition's effect on fresh
prompts is smaller than the exploratory data suggested, and the headline
should be read accordingly."
