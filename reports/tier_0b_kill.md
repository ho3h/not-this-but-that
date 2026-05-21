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

---

## Debugging session result (post-kill, bounded by the prime-directive fork)

**Took Path 1 (debug) as a bounded session, validated against external ground truth — not against my 0.60 threshold.**

### Found

1. **`use_error_term = False`** on this SAE. So the SAE does *replace* the residual stream in the forward pass; it isn't splicing in with an error term. But the model's top-5 next-token predictions under SAE installation are essentially identical to the pure forward (top 5 in the same order, probabilities within 1–2%). That's only possible if the SAE is operating roughly correctly inside the model.

2. **The canonical sae_lens evaluation path (`get_recons_loss`, `get_sparsity_and_variance_metrics`) uses the same `encode → decode` pattern I was using, with an `ActivationScaler` that scales by `sqrt(d_in) / mean_norm` ≈ 0.13.** I replicated this. VE was still −1.40.

3. **Direct calibration against an external ground truth: L0.** The canonical SAE I loaded is internally `layer_20/width_16k/average_l0_71` — meaning the canonical SAE has L0 = 71 features active per token. Sweeping runtime-scale s and measuring L0:

   | s | L0 | nonzero_frac | cosine sim | VE |
   |---:|---:|---:|---:|---:|
   | 0.1 | 12.4 | 0.075% | +0.37 | −1.87 |
   | 0.2 | 23.4 | 0.143% | +0.54 | −1.82 |
   | **0.5** | **74.1** | **0.452%** | **+0.83** | −4.30 |
   | 1.0 | 152.9 | 0.933% | +0.92 | −5.68 |
   | 2.0 | 435.0 | 2.66% | +0.91 | −6.58 |

   **At s=0.5, L0 lands at 74.1 against the canonical 71.** Cosine similarity reaches 0.83. The SAE *is* operating correctly when fed properly-scaled inputs — the feature counts and reconstruction direction match what Gemma Scope advertises.

4. **But VE-as-a-number stays negative across every formula I tried** — variance-based, MSE-based, per-position sum-of-squares, and the legacy / new sae_lens formulas. Most likely: there's a mean-offset / `b_dec` issue (b_dec has norm 206) where the directional reconstruction is right but the magnitude offset from the data mean is wrong enough to make `1 − ||x − recon||² / ||x − mean(x)||²` go very negative. This is a *measurement* issue, not a *science* issue, but I can't isolate the exact line without more sae_lens-internal debugging than this bounded session allows.

### What this means for the kill

The kill stands by the letter of the pre-registration (`base VE ≥ 0.60` is the threshold, and I couldn't reach it). It is softened by the calibration evidence: the SAE *is* the SAE Gemma Scope advertises (L0 within 4 % of canonical, cosine 0.83), and the model's forward output under SAE installation is sensible. The forward-pass causal interventions in Phase 4 are therefore likely real — they operate on a functioning SAE — even though my *standalone diagnostic* of that SAE's reconstruction is broken.

### Fork outcome

Per the bounded session protocol I committed to: *"If even the tutorial path fails on your stack after that session, you stop, take Path 2 (downgrade the writeup)"*. The tutorial path failed — my `encode → decode` reproduces the same negative VE as the canonical `get_recons_loss` arithmetic. **Taking Path 2.** The writeup needs to be downgraded to mark Phase 6 (genealogy) and the feature-labelling part of Phase 4 explicitly conditional on a verification this codebase cannot perform. The L0 + cosine calibration evidence can be cited as a *proxy* that the SAE is functional, but proxy is not prerequisite.
