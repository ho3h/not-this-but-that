# Not This, But That — handover

*Last updated 2026-05-21, end of Phase 7.*

> **All seven PRD phases ran end-to-end.** The exploration writeup is at
> [`reports/writeup.md`](reports/writeup.md) — read that first; this file is
> the *operational* state (what each phase produced, where it lives, what's
> still broken). The repo is at `ho3h/not-this-but-that` (private); the
> previous project lives at `ho3h/grammar-layer` (archive-state-on-disk).
>
> **One-line summary:** the construction (C3 specifically) is 12× more
> common in `gemma-2-2b-it` than `gemma-2-2b`. Feature 3223 — Neuronpedia
> label *"phrases conveying exceptions or negations"* — is causally necessary
> at the pivot decision (25% drop, 7400σ control-beating, fluency-preserving).
> It is NOT artificially sufficient (clamp-up fails) and NOT effective as a
> generation-time de-slop tool (the feature is dormant on neutral prompts).
> The mechanism finding stands; the product claim does not.

---

## 1. State as of Phase 7

The full prose narrative is at [`reports/writeup.md`](reports/writeup.md). What
follows is the operational state per phase — what ran, what files it wrote,
what's still broken.

### Phase 0 — scaffold ✓
- `src/{classifier,steering,quality,genealogy,deslop}` + `src/neograph/`.
- Fresh git history from `c22e56e`.
- 18/18 tests passing.

### Phase 1 — classifier ✓ (kill check passed, 1.00 P/R self-consistency caveat)
- `src/classifier/{detect.py, dependency.py}` — regex hinges + spaCy
  dependency filter. Strict mode walks the head chain up to 4 hops to handle
  C2's "not only" parse.
- `data/classifier_validation.jsonl` — 100 hand-labelled sentences (10 C1 +
  10 C2 + 10 C3 + 5 C4 + 65 negatives).
- `scripts/eval_classifier.py` → `reports/phase1_classifier_eval.{md,json}`.
- `tests/test_classifier_phase1.py` — automated kill check.

### Phase 2 — behavioural baseline ✓ (clean, large)
- `data/D2_neutral_prompts.json` — 102 hinge-free open prompts.
- `scripts/generate_d2.py` — 510 generations from pythia_70m + gpt2, 250 each
  from gemma_2b base + gemma_2b_it (scaled-down for throughput).
  Output: `reports/phase2_generations_<model>.jsonl`.
- `scripts/score_phase2.py` → `reports/phase2_baseline.md` —
  sentence-resampled bootstrap CIs.
- **Result:** instruct C3 rate = 1.7% vs base 0.1% (**12× amplification**,
  non-overlapping 95% CIs). any_core 1.8% vs 0.4% (**4.2× amplification**).

### Phase 3 — feature discovery ⚠ found *consequences*
- `data/D1_contrast_pairs.jsonl` — 226 hand-written minimal pairs (70 C1 +
  63 C2 + 68 C3 + 25 C4). `scripts/validate_d1.py` confirms 0 WITH-side
  failures and 0 WITHOUT-side failures.
- `scripts/discover_features.py` → `reports/phase3_discovery.md` —
  last-token-of-completed-sentence differential t-stat.
- **Result:** top semantic candidate **feat 9841** ("phrases and clauses
  involving contrasting ideas") with t=+14.19. *But this is a consequence
  feature, not a cause* — Phase 4 had to start over.

### Phase 4 — causal validation ⚠ PARTIAL (necessity yes, sufficiency no)
- `scripts/causal_m2.py` — M2 measurement (P(pivot) under intervention).
- `scripts/pivot_attribution.py` — per-feature attribution at the truncated
  pre-pivot position. **This is the script that recovered the right
  feature.** → `reports/pivot_attribution.md`.
- **Causal feature: feat 3223** — Neuronpedia label *"phrases conveying
  exceptions or negations"*. Active in 32/60 truncated D1 prompts.
- `reports/phase4_causal_m2_single_9841.{md,json}` — original 9841 null.
- `reports/phase4_causal_m2_supernode5.{md,json}` — 5-feature supernode
  null; clamp-up went -23.7σ wrong direction.
- `reports/phase4_causal_m2_3223_clamp10.{md,json}` — 3223 ablate beats
  random-k by +7397σ (drop 0.084, 25% relative); clamp@10 fails.
- `reports/phase4_causal_m2_3223_clamp3.{md,json}` — same ablate result;
  clamp@3 also fails.
- `reports/phase4_causal_m2.{md,json}` — canonical Phase 4 outcome (copy of
  the clamp10 file).
- **Verdict:** necessary but not sufficient. PRD's strict bidirectional gate
  fails; the necessity finding is overwhelming and clean.

### Phase 5 — quality preservation ✓ (scalpel, not sledgehammer)
- `data/D3_fluency.txt` — ~1500 words of clean human prose (Edinburgh,
  cast-iron, bookbinding, trail running, libraries, ensembles).
- `scripts/quality_preservation.py` → `reports/phase5_quality.md`.
- **Result:** D3 perplexity ratio **1.000× baseline (ablate)**, 1.006×
  (clamp_up). The ablation does not degrade fluency.

### Phase 6 — genealogy ✓ (with one buggy measurement to fix)
- `scripts/genealogy_compare.py` → `reports/phase6_genealogy.md`.
- **Result:** instruct baseline P(pivot) = 0.48 vs base 0.33; feat 3223
  ablation drops P(pivot) by **0.152 in instruct vs 0.084 in base
  (1.81× larger absolute drop)**.
- **Known bug:** the reconstruction-quality measurement (variance explained)
  comes out negative. The `sae(orig)` call in `reconstruction_quality()`
  isn't returning the reconstruction the way the code expects. The per-token
  Δ-log-P signal stands independently; the VE number should not be cited.

### Phase 7 — de-slop demo ❌ honest null (product claim does not hold)
- `scripts/deslop_demo.py` → `reports/phase7_deslop.md`.
- 12 D2 prompts × 3 seeds × 100 tokens, gemma-2-2b-it, token-by-token
  sampling with feat 3223 clamped to 0 throughout.
- **Result:** baseline construction rate 5.56%, ablated 5.56% — **0%
  absolute drop**. Meaning cosine 0.954 (largely identical generations).
- **Why:** Phase 6 showed feat 3223 has 0% activation at last-token of D2
  neutral prompts. The feature is conditional on construction-mode
  contexts; pre-emptive suppression of a dormant feature is a no-op.
- **The mechanism finding stands. The product claim does not.**

**Test count:** 18/18 pass (`test_classifier_phase1.py` 8 + smoke 4 +
manifold 4 + relations 2; `test_neo4j_smoke.py` excluded because it
requires a running Neo4j instance).

## 2. Module skeleton (PRD §6)

```
src/
  classifier/   # M1 — detect.py (regex hinges, done), dependency.py (Phase 1 stub)
  steering/     # Phase 4 — clamp.py (ablate ↓ and clamp ↑, hook factory stub)
  quality/      # Phase 5 — fluency / coherence / meaning (all stubs)
  genealogy/    # Phase 6 — transfer.py (SAE-on-instruct reconstruction check stub)
  deslop/       # Phase 7 — demo.py (gated until Phases 4+5 land)
  neograph/     # Reused substrate from grammar-layer (unchanged in this phase)
```

Each stub raises `NotImplementedError` with a pointer to the phase that should
fill it in. The exception is `classifier/detect.py`, which has a real
implementation good enough to power the Phase 0 smoke test — but it's the
*starting point* for Phase 1, not the finished classifier.

## 3. What was quarantined and where

The previous project's outputs and scripts have been moved to `legacy/`
subdirs to keep the active path clean while preserving reproducibility:

- `scripts/legacy/` — all viz_*, cross_model_*, generate_*, predicate_alignment,
  causal_ablation (named-backbone), causal_attribution.py (v1), the numbered
  pipeline scripts 02/03/05/06/07, sweep_leiden, eval_label_clustering, etc.
- `reports/legacy/` — every report file from the prior project (writeups,
  findings.md, all the figures, all the JSON outputs).
- `data/legacy/` — capital-city / weekday prompts, the 50-prompt expansion,
  per-model label caches except the canonical `labels_cache.json`.
- `notebooks/legacy/` — STORY.md, OVERNIGHT_SUMMARY.md, the compass artifact,
  the validation screenshot.
- `apps/legacy/grammar_layer/` — the Three.js viewer (demoted to optional per
  PRD §6 CHANGE list; may be revived for §9 graph viz).
- `web/legacy/` — the earlier interactive walkthrough.

**KEEP set at the top level** (the engines the PRD §6 explicitly preserves):
- `scripts/load_bearing_topk.py`, `load_bearing_control.py`,
  `load_bearing_mean_ablation.py`, `causal_attribution_v2.py`
- `scripts/01_load_model_and_sae.py`, `04_ingest_features.py`,
  `migrate.py`, `00_bootstrap_neo4j.sh`
- `scripts/prefetch_labels.py`, `fetch_labels_pending.py`
- `src/neograph/` in full

## 4. The corpus stubs (PRD §7)

- `data/D1_contrast_pairs.jsonl` — Phase 1 populates with ≥200 hand-checked
  minimal pairs. **Each pair hand-checked** is non-negotiable; garbage pairs
  poison the differential-activation signal in Phase 3.
- `data/D2_neutral_prompts.json` — Phase 2 populates with ≥100 prompts that
  invite prose without begging the construction. Used by M1 and Phase 4
  clamp-up.
- `data/D3_fluency.txt` — Phase 5 populates with a few thousand tokens of
  clean human prose for perplexity reference.

## 5. Open threads — what's next, if anything

The seven PRD phases ran end-to-end and the writeup (`reports/writeup.md`)
is honest about what landed and what didn't. The open threads from here, in
priority order if anyone picks this up:

### A. Fix Phase 6 reconstruction quality (½ day)

The variance-explained number in `reports/phase6_genealogy.md` is negative,
which is a measurement bug, not the SAE having broken. `sae(orig)` in
`scripts/genealogy_compare.py:reconstruction_quality()` isn't returning the
reconstruction the way the code expects. Likely the SAE forward returns a
dict or the input requires normalisation. The genealogy ablation signal
stands independently; only this one number needs fixing before the section
can be cited.

### B. Find the *sufficient* feature(s) for the construction (1–2 days)

Phase 4 said 3223 is necessary but clamping it up at any value tested moves
P(pivot) the wrong direction. The honest mechanism story is "necessary
multi-feature coordination." Two ways to test that:

1. **Per-feature attribution at the *positive* end** — find features whose
   clamp-up *does* increase P(pivot). The current pivot_attribution.py only
   measures ablation; symmetrise it.
2. **2-3-feature supernode search** — Anthropic-Biology-style. Phase 4
   already tried two arbitrary supernodes (top-5 by Phase-3 t-stat, and
   feat 3223 alone). The right supernode is whatever JOINTLY beats both
   directions. Combinatorial search is expensive; start from 3223 + the
   next 2-3 features by Phase 4 attribution score.

### C. Test the necessity claim on other models (1–2 days)

The previous project's cross-model finding suggests the suppression-side
features generalise across Gemma 1/2 2B, Pythia 70M, Gemma 2 9B. The
necessity finding (feat 3223 ablation drops P(pivot)) has not been
replicated outside Gemma 2 2B. Run `pivot_attribution.py` + Phase 4 on
Pythia 70M (different SAE, will need a new run); if a "phrases conveying
exceptions or negations" feature exists there with the same causal
necessity, that's a much stronger finding.

### D. Phase 7 v2 — intervene earlier in generation

The de-slop demo failed because feat 3223 is dormant on neutral prompts.
The construction's commit happens upstream — the model first decides to
*open* the contrast (the "not" token), then later commits to the *pivot*
(the comma/em-dash/"but"). Feat 3223 is causal at the second decision, not
the first. To deslop in open-ended generation, you'd need to identify and
ablate whatever features cause the model to emit "not" in the contrast
context in the first place. That's a different per-feature attribution
study (target = P("not") given construction-friendly contexts).

### E. The Neo4j angle (PRD §9), if any of the above lands

Storing the validated feature across models in the graph and asking the
cross-model alignment question via Cypher is the genuinely Hopkinson-shaped
contribution. With a partial mechanism that's conditional on context, the
graph doesn't have enough nodes yet. Worth revisiting when (B) or (C) lands.

### What NOT to do

- Don't claim "we found a feature that controls the construction." The
  ablation result is real, but clamp-up doesn't reproduce the construction.
  Honest framing: necessity yes, sufficiency no.
- Don't claim the de-slop demo works. It doesn't.
- Don't promote anywhere near ML researchers before fixing (A) and getting
  one credentialed mech-interp reader to sanity-check the necessity claim.
  The honesty contract from §7 still applies.
- Don't change the classifier without re-running the Phase 1 kill check.
  Any regex tweak invalidates the P/R guarantee. Re-run
  `pytest tests/test_classifier_phase1.py` after any change.

## 6. Known landmines (carry-overs)

These still apply — the substrate is the same.

1. **HF auth for Gemma.** Gemma 2 2B is gated. `HF_TOKEN` must be set in
   `.env` for any SAE work.
2. **sae-lens 4.x hook signature.** TransformerLens passes `hook` as a
   kwarg, so use `def ablate(act, **kwargs)`. See `scripts/load_bearing_topk.py`
   for the working pattern.
3. **`sitecustomize.py` workaround.** `uv pip install -e .` writes a `.pth`
   file that site.py doesn't process on this machine.
   `.venv/lib/python3.12/site-packages/sitecustomize.py` prepends `src/`
   explicitly. If you blow away the venv, recreate it or imports break for
   all six packages under `src/`.
4. **Three Neo4j instances on this box.** Substrate is on `bolt://localhost:7693`
   (`.neograph-db/`). The other ports (7687 orbweaver, 7688 Homebrew, 7689
   Desktop) are unrelated — see the previous handover (now in git history at
   commit `99cae1e` if needed) for the full disambiguation.

## 7. The honesty discipline

PRD §0 and §12: the grammar-layer post-mortem cost a repo. The kill checks in
the phases (§6 of the README, §8 of the PRD) are designed to make
"reframe a null as a discovery" structurally hard. Honor them.

The closing line from PRD §10 — *"It would be easy to end this by saying it's
not a detector, it's a mirror. So I won't."* — is in the README and is the
posture this project needs to keep.

Good luck.
