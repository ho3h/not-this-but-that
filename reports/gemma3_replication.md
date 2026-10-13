# Gemma 3 4B-it: replication of the decision-point procedure

Model `google/gemma-3-4b-it` (MLX, bf16), Gemma Scope 2 instruct residual SAEs (`gemma-scope-2-4b-it-res`,
16k, l0_medium). Engine `scripts/g3_mlx.py`; exploratory stages `scripts/g3_pipeline.py`; pre-registered test
`confirmation_v5_gemma3_prereg.md` → `confirmation_v5_gemma3_run.md`.

## Engine and SAE checks

- The MLX batched-decode RoPE bug is present in Gemma 3 too (rows 1+ drift by up to 44 logits); folding the batch
  into the head axis (`FoldedRope`) makes every row in a batch identical. Batch size 1 vs 4 still differs by up to
  1.25 logits in bf16, so every comparison keeps all conditions in one batch.
- SAE fidelity excluding BOS (`g3_smoke.json`): variance explained 0.92 / 0.86 / 0.83 at layers 17 / 22 / 29,
  loss recovered 99.9% / 99.1% / 99.4%.

## How often it uses the habit (normal answers, `g3_baseline.json`)

| prompts | answers with the habit |
|---|---:|
| D4 meaning/value | 294 / 400 (74%) |
| D4 process/scene | 26 / 200 (13%) |
| D2 everyday | 268 / 510 (53%) |

Gemma 2 2B-it on the same D4 meaning prompts: 234 / 800 (29%).

## The decision point (`g3_attribution_L*.json`, `g3_ladder_L*.json`)

Held-out decision points from D2 answers; drop in the probability of committing to the turn:

| layer | family, top 25 | cousin, top 25 | random / matched 25 |
|---|---:|---:|---:|
| 17 | 81% | 72% | ≈ 0 |
| 22 | 99% (top 2: 81%) | 29% | ≈ 0 |
| 29 | 99.9% (top 2) | −13% | ≈ 0 |

The strongest layer-22 features are output-side plans: 1138 promotes "not" in many languages (nicht, ikke, não, не…),
5020 promotes "Not / 并非", 1273 promotes "beyond / besides", 14946 "much more". Because the cousin is as common as the
family in Gemma 3, the pilot (`g3_pilot.md`) chose set B: family top 20 + cousin top 5 (24 features).

## Pre-registered result (`confirmation_v5_gemma3_run.md`)

Prediction registered before generation: H1–H3 PASS, H4 and H5 FAIL. Outcome: exactly that.

- H1: habit 156 → 27 of 600 everyday answers (−83%, CI 73–91%). PASS.
- H2: matched control 156 → 143. PASS.
- H3: ARC-Easy 88.7% → 88.3%; WikiText 1.003×; "a" + vowel 1.3%; "no" questions 28/28; blind preference 45 / 6 / 69
  (share 40.0%, gate ≥ 40%). PASS, at the gate; a second blind judge gives 35 / 31 / 54 (42%), so the normal answers are
  modestly preferred (`confirmation_v5_gemma3_judge_robustness.md`).
- H4: warning clarity 10 / 18 / 32 (32%). FAIL.
- H5: ordinary negation 0.39× normal. FAIL.
- Not gated: with the set off, 30 of 700 answers contain a doubled "it it" (normal 0, control 0).
- Explicit requests still produce the pattern (100% → 100%).

## Reading

The procedure transfers: in a newer, larger model about two dozen features at the decision point control the habit,
and a matched set does nothing. What changes is the cost. In Gemma 3 the habit's main feature is the model's general
plan to say "not"; a set that spares it leaves most of the habit (−22%, `specific_switch.md`). Switching the habit off
there means switching down negation.
