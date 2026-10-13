# Pre-registered confirmation v5 (Gemma 2): the specific switch

**Registered 2026-10-03, before any generation on the D7 or D6 prompts.** Frozen by SHA-256
in `confirmation_v5_gemma2_prereg.json`; the runner refuses to generate or analyse if any frozen file differs.

## Why

Confirmations v3 and v4 passed, but a later check (`reports/negation_check.md`) found that the
25 also carry ordinary negation: negation words roughly halve, and warnings get vaguer (blind share 35%).
Scoring every strong candidate on ordinary-negation decision points (`scripts/specific_selection.py`,
exploratory) showed that two of the 25 do double duty (10655: −42% of ordinary negation alone; 15646:
−12%). The **specific set** drops them and takes the next two habit features that leave ordinary
negation alone. On held-out D2 prompts (exploratory, `reports/specific_eval_gemma2.json`) it cut the
habit 41 → 3, kept negation (2.27 vs 2.23 per 1,000), and the blind warning share was 51%. This test
asks whether that holds on fresh prompts, with the negation checks as gates.

- **Feature set:** the specific 25 (`features` in the JSON): 23 of the v3/v4 set plus 5619 and 3315.
- **Control:** 25 features matched one-for-one by firing rate (`scripts/matched_control.py`, seed 7).

## What is frozen

- **Prompts:** 140 fresh D7 prompts (`data/D7_prompts.json`), written blind to all results: advice,
  meaning, message, copy, encouragement and essay (20 each) plus 20 explicit requests for the pattern;
  30 fresh D6 "what should you never do" prompts (`data/D6_never_prompts.json`).
- **Generation:** google/gemma-2-2b-it via MLX, chat template, error-term-preserving switch-off at layer 20
  (every position except BOS); normal, switched-off and control rows decoded together with common random
  numbers; 5 samples per D7 prompt and 2 per D6 prompt, up to 300 new tokens, temperature 0.8,
  top-p 0.95, seeds `20261100 + prompt id` (D6: `20261100 + 1000 + index`).
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

## Prediction

All five gates PASS.

## Reporting commitment

- All PASS: *"On 140 fresh prompts, switching off 25 features chosen to spare ordinary negation cut
  the habit by X%, with a matched control unchanged, no measurable cost, ordinary negation kept and
  warnings as clear as normal."*
- Any FAIL or KILL: report which gate failed, with its numbers, in the post and README.
