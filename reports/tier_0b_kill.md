# Tier 0b — KILL

**Pre-registered threshold:** `base_var_explained_min: 0.60` (with `instruct_var_explained_min: 0.50` as the secondary).

**Measured:** best VE I could obtain on the base model at Gemma Scope L20 / width-16k was **−1.33**. Far below the threshold. The pre-registration also says `kill_if_any_fails: true` for Tier 0 calibration as a whole, so this kills the entire Tier 0 gate. Proceeding past this would violate the adversarial protocol.

## What I did

1. Captured `blocks.20.hook_resid_post` from `gemma-2-2b` on a sample of D3 text. Variance ≈ 130–375 per chunk, L2-per-token / √d_model ≈ 8–10 (sensible scale).
2. Ran `SAE.from_pretrained(release='gemma-scope-2b-pt-res-canonical', sae_id='layer_20/width_16k/canonical')` and computed `sae(resid)`. Direct VE = **−12.65**.
3. Verified via `run_with_cache_with_saes` that the cache's `hook_sae_input` == raw residual (max |Δ| = 0). So the input the SAE sees is the same residual I captured directly. The `hook_sae_recons` produced by the installed SAE has variance ≈ 6 167, vs input variance 375 — the cached reconstruction is itself ~16× the input scale.
4. Inspected the SAE: `cfg.normalize_activations: 'none'`, `cfg.apply_b_dec_to_input: False`, `W_dec` columns unit-normed (max norm = 1.000), encoder norms reasonable, JumpReLU thresholds in 4.5–30 range.
5. Discovered `sae.fold_activation_norm_scaling_factor(scaling_factor)` — a method whose docstring/signature implies Gemma Scope was trained with an activation-norm scaling factor that needs to be folded in. Swept scaling factors from 0.001 to 100; best VE observed = **−1.33 at scaling_factor ≈ 0.10**. Still far below threshold; the curve doesn't suggest a magic value that recovers sensible recon.

## The internal contradiction

The model's *forward-pass behaviour* under SAE installation is sensible:

- Baseline P(pivot) on truncated D1 prompts is 0.33 (Phase 4) — far above noise-level for a 256k-token vocab.
- Ablating feature 3223 produces a coherent 25% drop in P(pivot), consistent across 201 prompts.
- The model continues to generate fluent English under SAE installation (Phase 7).

If the SAE were truly producing reconstructions with VE = −12, the post-SAE residual stream would be garbage and the model's downstream layers would emit noise. They don't. The reconciliation is most likely one of:

- **(a)** Gemma Scope uses an error-term / splice-in pattern where the *forward-pass* residual is `resid` (unchanged), and `hook_sae_recons` is only a *diagnostic* of what the SAE encoder→decoder would have reconstructed. The mismatch then says the diagnostic is broken (or expects normalization I'm not applying), but the actual SAE-mediated *intervention* in the forward pass (ablating a feature in the encoded space and propagating the delta) is fine.
- **(b)** My measurement code is wrong (calling `sae(x)` in a way the API doesn't support standalone), and the real SAE forward path applies normalization I'm not replicating.
- **(c)** Gemma Scope's published weights aren't usable through sae_lens 6.43.0 without an additional step that isn't documented in the API I'm using.

In any of (a)/(b)/(c), the science isn't necessarily broken — but my pre-registered verification step is broken, and per the protocol I cannot wave that away. The threshold was 0.60. The measurement is −1.33. That's a kill.

## What this kills, what it doesn't

**Kills:**
- The Phase 6 genealogy claim's prerequisite. The 1.81× ablation-drop ratio between base and instruct cannot be cited until VE measurement works.
- All of Tier 0 (per the `kill_if_any_fails: true` setting).

**Does NOT kill (these stand independently of SAE reconstruction quality):**
- **Phase 2 behavioural finding.** Sentence-level construction rates and the variant-composition shift (instruct 94% C3 vs other models C1-dominant) come from the classifier on raw model outputs. No SAE involved.
- **Tier 0a classifier validation.** Passed blindly on independent text (P=0.80, R=1.00). No SAE involved.

**Conditionally killed (claim depends on this VE measurement working):**
- **Phase 4 necessity claim** (feat 3223 ablation drops P(pivot) by 25%). The ablation IS a real causal intervention — it changes the model's forward output. But interpreting *which feature in feature-space* it corresponds to depends on the SAE's encoder/decoder being meaningful, which is exactly what VE was meant to certify. If interpretation (a) above is right, the intervention is real but the labelling of "the feature you ablated is feature 3223" is shakier than the writeup claims.

## What I am NOT doing (prime directive)

I am not going to:
- Tune the threshold downward to make this pass.
- Argue that VE = −1.33 is "close enough" because the model still works.
- Skip ahead to Tier 1 in the hope that specificity will compensate.
- Invent a "VE_v2" metric that measures something different and passes.

I AM stopping here, writing this report, and surfacing it to the human.

## Three honest paths forward (your call)

1. **Debug the VE measurement properly.** Possibilities to try:
   - Verify with a different SAE library or a known-good Gemma Scope tutorial that uses different load mechanics.
   - Open an issue / read recent sae_lens changelogs for breaking changes to the Gemma Scope loader path in 6.x.
   - Try `gemma-scope-2b-pt-res` (non-canonical id) or a different layer's SAE to see if the issue is canonical-specific.
   - Ask a credentialed mech-interp person who has worked with Gemma Scope what the expected VE measurement pattern is.

2. **Accept that VE measurement is broken and downgrade the writeup accordingly.** The genealogy claim becomes "the ablation drop ratio is 1.81×, but the prerequisite reconstruction-quality check could not be performed in this codebase; treat as exploratory until verified." The Phase 4 necessity claim becomes "the ablation produces a real causal change in P(pivot), but feature-level interpretability is conditional on the SAE encoder/decoder being meaningful, which we cannot independently verify."

3. **Stop here.** The writeup as it stands — with the kill reported honestly — is itself a reportable result: *we attempted to validate the SAE's prerequisite and could not, so the mechanism-level claims are softer than we initially wrote.* That's an honest exploration outcome.

Whichever you pick, I do not pick it for you. The adversarial methodology says stop on kill.
