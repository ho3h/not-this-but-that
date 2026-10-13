# Pre-registered confirmation v2: chat format, error term preserved

**Registered 2026-10-02, before any generation on the D4 prompt set.** This file,
the prompts, the engine, the detectors and the runner are frozen by SHA-256
hashes recorded in the JSON twin (`confirmation_v2_prereg.json`); the runner
refuses to generate or analyse if any hash differs. (Frozen by hash because the
author hadn't yet approved a commit; commit the registration before
`--analyze` for a public timestamp.)

## Why a second confirmation

The first pre-registered confirmation (`confirmation_prereg.md`) failed its gates
because its fresh prompts barely elicited the construction (1.0% baseline), so it
had no power. Three method problems in the exploratory runs also needed fixing
before the claim could be tested properly:

1. **No chat template.** The exploratory generation runs fed prompts to
   `gemma-2-2b-it` as plain text. Here every prompt is wrapped in Gemma's chat
   template and generation stops at `<end_of_turn>`.
2. **No error term.** The exploratory runs replaced the layer-20 residual with
   the SAE reconstruction in both conditions (including BOS), so the "normal"
   condition was a modified model. Here the normal condition is the unmodified
   model, and the intervention subtracts only the 25 coalition features' decoder
   contributions, keeping everything the SAE does not explain.
3. **SAE fidelity.** The Tier 0b fidelity gate failed at VE −1.33. Re-measured
   excluding BOS, which Gemma Scope was not trained on, VE is 0.78 (base) and
   0.73 (it) with 99% of loss recovered on base (`sae_fidelity.md`), above the
   pre-registered 0.60.

## What is frozen

- **Coalition:** the same 25 layer-20 features as every earlier eval (promote-ranked
  top 25 of `pivot_attribution.json`, selected on the base model).
- **Model and engine:** `google/gemma-2-2b-it`, published bf16 weights, MLX
  (`scripts/v2_mlx.py`, MLX 0.31.2, mlx-lm 0.31.3). The engine patches an MLX
  RoPE bug that corrupts batched decoding (`tests/test_mlx_rope_batch.py`);
  its next-token distributions match Hugging Face transformers on the same
  prompt (KL 1e-5), and coalition-feature firing agrees on 99.8% of positions.
- **Intervention:** `h → h − JumpReLU(h·W_enc[:,C] + b_enc[C])·W_dec[C]` at the
  layer-20 residual, every position except BOS, every generation step.
- **Prompts:** 120 fresh D4 prompts (`data/D4_confirmation_v2_prompts.json`),
  80 M (asks what something means or why it matters) and 40 P (asks for a
  process, a scene or facts), opening verbs balanced within type, written by an
  annotator who saw no generations or results.
- **Sampling:** temperature 0.8, top-p 0.95, up to 150 new tokens. Each call
  draws 10 normal and 10 coalition-off answers for one prompt; sample *k* of
  each condition shares its random numbers (common random numbers), seeded
  `20261002 + 10·id + stage`.
- **Detectors:** exactly as committed: family = `detect_construction_v2`
  (v1 regex ∪ permissive layer), cousin = `detect_more_than_just`, textbook =
  `detect_construction(strict=False)`. Curly quotes are mapped to ASCII first.

## Design: event-driven, at most two stages

Stage 1 draws 10 answer pairs per prompt (1,200 pairs). If the normal answers to
M prompts contain at least **40** family hits, the analysis runs on stage 1.
Otherwise stage 2 draws 10 more pairs per prompt and the analysis runs once on
both stages. The stage decision reads normal answers only. No other interim
looks; the runner prints progress, never rates.

## Hypotheses and decision rules

**H1 (primary).** Switching off the coalition reduces the family construction in
answers to M prompts. Paired by (prompt, sample).

- **PASS** if the relative drop ≥ 30% and McNemar mid-p (two-sided) < 0.05.
- **KILL** if the relative drop < 15% or mid-p ≥ 0.05.
- Between 15% and 30% with mid-p < 0.05: **replicated direction, smaller
  magnitude**; the post must quote the number.

**H2 (secondary).** In normal answers, the family-or-cousin rate on M prompts is
at least twice the rate on P prompts, with a prompt-clustered 95% bootstrap
interval for the ratio whose lower bound exceeds 1. Otherwise **not confirmed**.

**Reported, non-gating:** the textbook-form drop, the cousin count (the
exploratory runs saw it hold or rise), family plus cousin, the family drop on
all prompts and on P prompts, the share of answers containing "but", and mean
words per answer.

## Reporting commitment

Whatever the outcome, it is reported in the post and README. Committed wording:

- PASS: *"In a pre-registered test on 120 fresh prompts, in chat format and with
  the model otherwise unmodified, switching off the 25 features cut the
  construction by X% (McNemar mid-p Y)."*
- In between: *"The second pre-registered confirmation replicated the direction
  at a smaller magnitude (X%)."*
- KILL: *"The second pre-registered confirmation failed: switching off the
  coalition did not reliably reduce the construction in chat-formatted answers,
  and the generation results above should be read as exploratory."*
- H2 not confirmed: *"The prompt-type pattern did not replicate in a
  pre-registered test."*
