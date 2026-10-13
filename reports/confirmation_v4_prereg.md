# Pre-registered confirmation v4: the chat-chosen 25 on fresh, varied prompts

**Registered 2026-10-03, before any generation on the D5 prompts.** Frozen by SHA-256
in `confirmation_v4_prereg.json`; the runner refuses to generate or analyse if any
frozen file differs.

## Why

Confirmation v3 passed on 102 everyday "describe / explain" prompts with 150-token
answers. This test asks whether the result holds on the kinds of things people
actually ask chatbots, with longer answers, against a matched control, and with
stronger cost checks.

## What is frozen

- **Feature set:** the same chat-chosen 25 as v3 (`chat_features` in the JSON).
- **Control:** 25 features matched one-for-one to the chat 25 by firing rate
  (draw 0 of `decision_point_ladder.py`, listed in the JSON).
- **Prompts:** 140 fresh D5 prompts (`data/D5_confirmation_v4_prompts.json`),
  written blind: advice, meaning, message, copy, encouragement and essay (20 each),
  plus 20 explicit requests for the pattern.
- **Generation:** `google/gemma-2-2b-it` via MLX, chat template, error-term-preserving
  switch-off at layer 20 (every position except BOS); normal, chat-25 and control rows
  decoded together with common random numbers; 5 samples per prompt, up to 300 new
  tokens, temperature 0.8, top-p 0.95, seeds `20261004 + prompt id`.
- **Detector:** `detect_v3.py`; habit = family or cousin.
- **Capability checks:** ARC-Easy (300 test questions, `data/arc_easy_300.json`),
  WikiText-2 test perplexity (first 120,000 characters,
  `data/wikitext2_test_excerpt.txt`), "a" before a vowel.
- **Blind preference:** 120 pairs (normal vs chat 25) sampled from the everyday
  prompts with `random.Random(2026)`, A/B order randomised; judged by an annotator who
  sees only the question and the two answers, with this instruction: *"For each pair,
  decide which answer is better: better written and more useful for the person who
  asked. Answer A, B, or T for a genuine tie. Answers were cut off at about 300
  tokens; do not penalise either for ending mid-sentence."*

## Hypotheses and gates (analysed once)

**H1 (primary).** On the 600 everyday triples, the habit falls with the chat 25 off.
PASS if the relative drop ≥ 50% and McNemar mid-p < 0.05; KILL if < 25% or mid-p ≥ 0.05;
otherwise PARTIAL.

**H2 (specificity).** The chat-25 drop exceeds the matched control's drop by at least
40 percentage points.

**H3 (cost).** All of: ARC-Easy accuracy within 3 points of normal; WikiText-2
perplexity ≤ 1.10× normal; "a" before a vowel in ≤ 2% of answers; blind preference
share for the chat 25 (wins plus half of ties) ≥ 40%.

**Reported, non-gating:** per-genre drops; family, cousin, "goes beyond" and reversed
tiers for both arms; explicit-request prompts (can the model still produce the pattern
when asked?); "but" share; mean words.

## Reporting commitment

- All PASS: *"On 140 fresh prompts across six everyday genres, switching off the 25
  features cut the habit by X%, with a matched control unchanged and no measurable cost."*
- Any FAIL or KILL: report which gate failed, with its numbers, in the post and README.
