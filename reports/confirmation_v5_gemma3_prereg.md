# Pre-registered confirmation v5 (Gemma 3 4B-it): replication and the negation trade-off

**Registered 2026-10-03, before any generation on the D7 or D6 prompts.** Frozen by SHA-256
in `confirmation_v5_gemma3_prereg.json`; the runner refuses to generate or analyse if any frozen file differs.

## Why

Replication of the decision-point procedure in Gemma 3 4B-it with Gemma Scope 2 instruct SAEs
(16k, l0_medium). Exploratory work (`reports/g3_*`): layer 22 chosen on held-out decision points
(family top 25: −99%); because Gemma 3 uses the "more than just" cousin as often as the family, the
pilot (`reports/g3_pilot.md`) compared sets and chose **set B**: the top 20 of the family ranking
plus the top 5 of the cousin ranking (24 unique features; outputs label the arm "25 off"). On D2,
set B cut the habit 106 → 4 of 204. Unlike Gemma 2, a set that spares ordinary negation did not work
here: dropping the three negation-carrying features (chief among them 1138, which promotes "not" in
many languages) left most of the habit (94 → 73; `reports/specific_eval_gemma3.json`). In Gemma 3 at
this layer the habit and ordinary negation look entangled. This test registers that trade-off.

- **Feature set:** set B (`features` in the JSON, 24 features at layer 22).
- **Control:** 24 features matched one-for-one by firing rate (`scripts/matched_control.py`, seed 7).

## What is frozen

- **Prompts:** 140 fresh D7 prompts (`data/D7_prompts.json`), written blind to all results: advice,
  meaning, message, copy, encouragement and essay (20 each) plus 20 explicit requests for the pattern;
  30 fresh D6 "what should you never do" prompts (`data/D6_never_prompts.json`).
- **Generation:** google/gemma-3-4b-it via MLX, chat template, error-term-preserving switch-off at layer 22
  (every position except BOS); normal, switched-off and control rows decoded together with common random
  numbers; 5 samples per D7 prompt and 2 per D6 prompt, up to 300 new tokens, temperature 0.8,
  top-p 0.95, seeds `20261200 + prompt id` (D6: `20261200 + 1000 + index`).
- **Detector:** `detect_v3.py`; habit = family or cousin.
- **Capability checks:** ARC-Easy (300), WikiText-2 perplexity (first 120,000 characters), "a" before
  a vowel, and the 30 "no" questions of `data/negation_check_questions.json` (prediction = Yes vs No
  next-token probability).
- **Blind preference (H3):** 120 pairs (normal vs switched off) sampled from the everyday D7 prompts
  with `random.Random(2026)`, A/B order randomised, judged by an annotator who sees only the question
  and the two answers, with this instruction: *"For each pair, decide which answer is better: better written and more useful for the person who asked. Answer A, B, or T for a genuine tie. Answers were cut off at about 300 tokens; do not penalise either for ending mid-sentence."*
- **Blind warning clarity (H4):** all 60 D6 answer pairs (normal vs switched off), shuffled with the
  same generator, A/B randomised, judged blind with: *"For each pair, decide which answer does a better job of telling the person what to avoid: the warnings are clear and correct, and a reader skimming the answer could not mistake a warning for a recommendation (for example, a bullet that reads as an instruction to do the dangerous thing). Answer A, B, or T for a genuine tie. Answers were cut off at about 300 tokens; do not penalise either for ending mid-sentence."*
- **Ordinary negation (H5):** occurrences of not, n't, no, never per 1,000 words in the everyday D7
  answers, excluding any that fall inside a detected family, cousin, reversed or "goes beyond" span.

## Hypotheses and gates (analysed once)

**H1 (primary).** On the 600 everyday triples, the habit falls with the set switched off. PASS if the
relative drop ≥ 50% and McNemar mid-p < 0.05; KILL if < 25% or mid-p ≥ 0.05; otherwise PARTIAL.

**H2 (specificity).** The drop exceeds the matched control's drop by at least 40 percentage points.

**H3 (cost).** All of: ARC-Easy within 3 points of normal; WikiText-2 perplexity ≤ 1.10× normal;
"a" before a vowel in ≤ 2% of answers; blind preference share (wins plus half of ties) ≥ 40%; at most
2 of the 30 "no" questions lost.

**H4 (warnings stay clear).** Blind warning-clarity share for the switched-off answers ≥ 40%.

**H5 (ordinary negation kept).** Ordinary negation rate with the set off ≥ 0.80 × normal.

**Reported, non-gating:** per-genre drops; tiers for both arms; explicit-request prompts; "but"
share; mean words; all-negation rate.

## Prediction (registered before generation)

H1, H2 and H3 PASS; H4 and H5 FAIL. That is: the habit can be switched off in Gemma 3 too, but only
by turning down ordinary negation and making warnings less clear. Any other pattern will be reported
as a departure from this prediction.

## Reporting commitment

- If the prediction holds: *"In Gemma 3 4B, switching off 24 decision-point features cut the habit
  by X% on 140 fresh prompts, but also cut ordinary negation and made warnings less clear: in this
  model the habit and negation share machinery."*
- Otherwise: report each gate's verdict and numbers, in the post and README.
