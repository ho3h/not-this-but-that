# Pre-registered confirmation v3: squash the habit with 25 chat-chosen features

**Registered 2026-10-03, before any generation on the held-out prompts.** Frozen by
SHA-256 in `confirmation_v3_prereg.json`; the runner refuses to generate or analyse
if any frozen file differs.

## Why

Confirmation v2 showed that switching off the original 25 features (chosen on the
base model, at the comma before the turn) leaves the habit intact: the model
reroutes. An exploratory follow-up (`reroute_attribution.json`, `dose_targeted.md`)
chose features in the chat model itself, at the token where it commits to the
negation ("…duck pond | isn't just…"). Switching off the top 25 of those cut the
habit from 29% to 2% of answers on D4 meaning prompts, with perplexity on human
prose at 1.05× and no quiz loss. Those features were chosen and tested on D4
prompts, so this test uses prompts that played no part in choosing them.

## What is frozen

- **Chat features (primary arm):** the top 25 of the family ranking in
  `reroute_attribution.json` with a positive mean drop, listed in the JSON.
- **Original coalition (secondary arm):** the 25 features of every earlier test.
- **Prompts:** the 102 D2 prompts (`data/D2_neutral_prompts.json`), never used to
  choose the chat features, in Gemma's chat template.
- **Model and engine:** `google/gemma-2-2b-it` via MLX (`v2_mlx.py`, `v2_mlx_multi.py`);
  error-term-preserving switch-off at layer 20, every position except BOS; normal,
  original-25 and chat-25 rows decoded together with common random numbers.
- **Sampling:** 5 samples per prompt (510 triples), temperature 0.8, top-p 0.95,
  up to 150 new tokens, seeds `20261003 + prompt index`.
- **Detector:** `src/classifier/detect_v3.py`. Habit = family or cousin.

## Hypotheses and gates

**H1 (primary).** With the chat 25 off, the habit falls.
PASS if the relative drop ≥ 50% and McNemar mid-p < 0.05. KILL if the drop < 25%
or mid-p ≥ 0.05. Otherwise PARTIAL.

**H2 (co-primary, cost).** With the chat 25 off: perplexity on held-out human
prose (`data/D3_fluency.txt`) ≤ 1.15× normal; accuracy on the 20-question quiz
within one question of normal; answers with "a" before a vowel ≤ 2%.
PASS only if all three hold.

**Reported, non-gating:** family, cousin, "goes beyond" and reversed tiers for both
arms; the original 25 on the same prompts; share of answers containing "but";
mean words.

## Reporting commitment

- H1 PASS and H2 PASS: *"In a pre-registered test on 102 prompts not used to choose
  them, switching off 25 features chosen in the chat model cut the habit by X%,
  with no measurable cost."*
- H1 PASS, H2 FAIL: *"…cut the habit by X%, at a measurable cost: [failed check]."*
- PARTIAL: *"…cut the habit by X%, short of the 50% registered."*
- KILL: *"The chat-chosen features did not reliably reduce the habit on new prompts;
  the exploratory squash did not replicate."*
