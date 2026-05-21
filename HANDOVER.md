# Not This, But That — handover

*Last updated 2026-05-20, end of Phase 1.*

> **Phase 0 (scaffold) and Phase 1 (construction classifier) are done.** The
> repo is fresh — it's not the `graphgeometry` branch; it's its own GitHub
> repo at `ho3h/not-this-but-that` (private). The previous project lives at
> `ho3h/grammar-layer` archived at `19f2426`. Read the README first — it
> states scope, prior work (§2), the differentiation against Kuznetsov
> 2503.03601 (§3), the anti-scope (§4), the metrics (§5), and the build
> phases with kill checks (§6). Then this file.

---

## 1. State as of Phase 1

**Phase 0 — scaffold.**
- Module skeleton at `src/{classifier,steering,quality,genealogy,deslop}` +
  `src/neograph/` carried over. All Phase 0 stubs that haven't been touched
  by Phase 1 still raise `NotImplementedError` with phase pointers.
- README, HANDOVER, anti-scope, kill checks all in place. The `legacy/`
  carry-over from `graphgeometry` did *not* come into this repo — fresh git
  history from `c22e56e initial: not-this-but-that, Phase 0 scaffold`.

**Phase 1 — classifier (this section).**
- **Kill check passed.** `tests/test_classifier_phase1.py::test_regex_only_kill_check`
  is the automated gate: precision/recall **≥ 0.85** on C1, C2, C3 against the
  100-sentence hand-labelled validation set. Current: **1.00 P/R across all
  four variants**, both regex-only and strict (regex + spaCy dep filter).
- **Validation set:** `data/classifier_validation.jsonl` — 10 C1 + 10 C2 +
  10 C3 + 5 C4 + 65 negatives (15 contracted-negation paraphrases of
  positives, 10 "paraphrase_of_C4", 15 neutrals-with-`not` to test FP risk,
  15 clean neutrals). The composition matches HANDOVER's Phase 1 spec.
- **The 1.00 P/R caveat:** the validation set was hand-written by the same
  agent that tuned the regex. This is a self-consistency check, not a
  generalization test. The real measurement is Phase 2 M1 — when the
  classifier sees actual model generations. Don't quote 1.00 P/R as a
  finding; quote it as "the kill check is met and the classifier is ready
  for Phase 2."
- **`classifier/detect.py`:** Phase 1 broadened the regex to accept
  contracted-negation openers (`isn't / aren't / wasn't / weren't`), the
  full subject-copula contraction family in pivot position (`he's / she's
  / we're / they're / this is / that's`), and fixed the C1/C3 overlap
  predicate (half-open interval intersection — Phase 0's bug let "It's not
  just an update — it's a rethink." count for both C1 and C3, inflating
  C1's FP).
- **`classifier/dependency.py`:** Phase 0 was a stub returning `True`.
  Phase 1 added a spaCy `en_core_web_sm` parse: locate the negation token
  inside the regex hit's span, walk up the head chain, reject only if no
  ancestor within 4 hops is a `VERB`/`AUX`. The head-chain walk is
  necessary because C2's "not only" parses with "not" as `advmod` of
  "only" (not directly of the verb). Strict mode adds ~8 ms/sentence; the
  filter doesn't change any verdict on the validation set (0 strict-vs-regex
  diffs) — it exists for Phase 2 real-generation edge cases.
- **`scripts/eval_classifier.py`:** the gate. Run via
  `uv run python scripts/eval_classifier.py`. Writes
  `reports/phase1_classifier_eval.{md,json}` with both modes side-by-side.

**Test count:** 19/19 pass (`tests/test_classifier_phase1.py` adds 8 to the
Phase 0 baseline of 11).

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

## 5. Priority queue — start of Phase 2

Phase 1 cleared the kill check. The next agent's first task is Phase 2:
behavioral baseline — measure M1 (spontaneous construction rate) across the
model set on the D2 neutral prompts.

### P2 — Behavioral baseline (PRD §8 Phase 2, 1–2 days)

**The motivating expectation:** instruct construction rate ≫ base. If this
gap doesn't exist on the four-model set, the Phase 6 genealogy story is
already in trouble — flag it then, continue, because the within-model
mechanism (Phases 3–5) can still hold.

**Concrete steps:**

1. **Populate `data/D2_neutral_prompts.json`.** ≥100 open prompts that
   invite prose without begging the construction. Examples in PRD §7: *"Write
   two sentences about why a city invested in public transit."* Avoid
   prompts containing `not`, `but`, `however`, or any of the C1–C4 hinges in
   the prompt itself — those bias generation.

2. **Generate continuations.** For each (model, prompt, seed):
   - `gemma-2-2b` (base), `gemma-2-2b-it` (instruct), `gpt2`, `EleutherAI/pythia-70m-deduped`.
   - ≥5 seeds × ≥100 prompts × 4 models = ≥2000 generations. ~150 tokens each.
   - Reuse `scripts/01_load_model_and_sae.py` for model loading; the SAE
     isn't needed for M1 (the classifier is model-agnostic). Write a new
     `scripts/generate_d2.py` that just does HF model + tokenizer +
     `model.generate()` with deterministic seeds.

3. **Score with the classifier.** Use `classifier.rate(texts, strict=True)`.
   Strict mode is right here — Phase 2 is where the dep check earns its
   keep on real generations. Report per-model: M1 = construction rate per
   variant + any_core, plus bootstrap CIs.

4. **Write `reports/phase2_baseline.md`.** Per-model M1 table, with
   bootstrap CIs. State the base-vs-instruct gap explicitly; flag if it's
   smaller than expected.

**Exit criterion (informational, not a kill check):** instruct ≫ base by a
meaningful margin (rule of thumb: ≥2× rate, non-overlapping CIs). If not,
proceed but note the genealogy risk; the within-model mechanism work
(Phases 3–5) is still worth doing.

### Important Phase 2 gotchas

- **Sampling strategy matters.** `temperature=0.0` will collapse to mode and
  give a misleading M1. Use `temperature=0.7-1.0` with top-p sampling — the
  construction rate is a property of the sampling distribution, not the
  argmax. Record exact `temperature`, `top_p`, `top_k` in the report.
- **`gemma-2-2b-it` chat template.** Apply it (via `tokenizer.apply_chat_template`)
  before generation. Without it the instruct model behaves erratically; with
  it the construction rate jumps. This *is* the genealogy effect; record both.
- **Bootstrap with sentence-level resampling, not generation-level.** A
  300-token generation might contain 2–3 sentences; if M1 is computed
  per-sentence, the resampling unit is the sentence. Document the choice.

### What NOT to do in Phase 2

- Don't start feature discovery (Phase 3) before Phase 2 lands.
  Phase 3's D1 differential-activation hunt is expensive; ground the
  motivation first.
- Don't try Gemma 2 9B or Mistral 7B yet. Phase 2 is about establishing M1
  across the small/medium open models — bigger models come in later if the
  scale curve is interesting.
- Don't change the classifier without rerunning the Phase 1 kill check.
  Any regex tweak invalidates the precision/recall guarantee. Re-run
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
