# Not This, But That

> Find the feature inside a small open LLM that produces the **"not X, but Y"** writing tell, prove you can switch it off without wrecking the prose, and tell the story in the voice of the tell itself.

This isn't a detector. It's the same engine that lived in the previous repo (`grammar-layer`), repointed at a better question.

The previous project rediscovered a known phenomenon — SAE "suppression features" — and over-claimed novelty. The post-mortem was clear: the framework was sound, the framing was not, and the finding was someone else's from 2024. The machinery, though, was good. *Not* this — but *that*.

---

## Status

**Phase 0 — scaffold complete.** Module skeleton present, smoke test passes, prior work and anti-scope are stated up front (this README). No mechanism claim has been validated yet; nothing here has been peer-read by a credentialed mech-interp researcher; the result, if any, will pertain to **one model (Gemma 2 2B)** and **one stylistic construction**. Do not read past this line as if any of the downstream phases have landed.

---

## 1. The thing we are studying

Wikipedia's *Signs of AI writing* page (shortcut **WP:AIPARALLEL**) describes the target precisely: LLMs over-produce parallel constructions built on *not / but / however*. We study four variants:

| # | Variant | Example | Lexical hinge |
|---|---------|---------|---------------|
| **C1** | Contrastive correction | "It's *not* a tool, it's a revolution." | `not` … `it's` / `,` pivot |
| **C2** | Additive escalation | "*Not only* does it scale, *but* it adapts." | `not only` … `but` |
| **C3** | Minimize-then-elevate | "It's *not just* an update — it's a rethink." | `not just` … `but` / `—` |
| **C4** | Triadic negation | "No mandate. No approval. Just power." | `no` … `no` … `just` |

**C1–C3 are the core. C4 is a rhythmic cousin** (added to the WP discussion in 2026); kept in the corpus, kept out of the primary causal claim until C1–C3 are nailed.

Why this construction and not "rule of three" or "delve": it has a **lexical hinge** — a specific tokenizable pivot (`but`, `it's`, `just`, `—`). That converts "did the model enter the construction" into a next-token probability, which is exactly what the existing ablation harness measures. The rule of three is rhythmic, unanchored, SAE-illocalizable. **Out of scope for v1.**

---

## 2. Prior work — read this section first

The single thing that sank `grammar-layer` was a missing related-work section. We fix that *before* writing analysis.

**Closest prior art** (the work this project must differentiate from, row by row in §3):
- **"Feature-Level Insights into Artificial Text Detection with Sparse Autoencoders"** — Kuznetsov et al., [arXiv:2503.03601](https://arxiv.org/abs/2503.03601), March 2025. SAE features from Gemma-2-2b's residual stream + a steering approach + the finding that LLMs have a distinct writing style.

**Genealogy of the tell:**
- "Understanding the Effects of RLHF on the Quality and Detectability of LLM-Generated Texts" — [arXiv:2503.17965](https://arxiv.org/abs/2503.17965), March 2025.
- "The Last Fingerprint: How Markdown Training Shapes LLM Prose" — arXiv:2603.27006 — RLHF amplifies stylistic tells because evaluators reward structured, emphatic prose.

**Methodological lineage:**
- Bloom & Lin, ["Understanding SAE Features with the Logit Lens"](https://www.lesswrong.com/posts/qykrYY6rXXM7EEs8Q/understanding-sae-features-with-the-logit-lens), LessWrong, March 2024 — the original suppression-vs-prediction features framing.
- Marks, Rager, Michaud, Belinkov, Bau & Mueller, ["Sparse Feature Circuits"](https://arxiv.org/abs/2403.19647), ICLR 2025 — signed indirect-effect attribution.
- Anthropic, ["Circuit Tracing"](https://transformer-circuits.pub/2025/attribution-graphs/methodology.html) + ["On the Biology of a Large Language Model"](https://transformer-circuits.pub/2025/attribution-graphs/biology.html), Transformer Circuits, March 2025.
- Lieberum et al., ["Gemma Scope"](https://arxiv.org/abs/2408.05147), August 2024 — the SAEs we use.
- Zhang, Wang & Su, "Mechanistic Knobs: Retrieving and Steering High-Order Semantic Features via SAEs" — arXiv:2601.02978 — precedent for steering behavior-level linguistic features.

**Cultural anchor:**
- Wikipedia, [*Signs of AI writing*](https://en.wikipedia.org/wiki/Signs_of_AI_writing) (WP:AIPARALLEL). The reason anyone cares.

---

## 3. What this is, and what 2503.03601 was

| | Kuznetsov et al. 2503.03601 (prior art) | Not This, But That |
|---|---|---|
| Unit of analysis | Aggregate "AI-ness" of a text | One named construction (C1–C3) |
| Feature claim | Features *correlate* with AI text | A specific feature with *bidirectional causal* effect |
| Validation | Steering / detection statistics | Ablate → rate drops **and** clamp → rate rises, vs random-k / bottom-k controls |
| Genealogy | — | Base vs instruct: is the feature *dormant in base, amplified by tuning*? |
| Output | Detection insight | A working "off switch" + quality-preservation evidence |

If, after Phase 4, no row holds up better than the prior art — the contribution is dead and we say so. See the kill criteria in `docs/phases.md` (created from PRD §8 when Phase 1 starts).

---

## 4. What this is NOT

- **Not a detector.** AI-text detection (DetectGPT, GPTZero, watermarking, Binoculars) is crowded, adversarially doomed, and gets gamed the moment it ships. We study *mechanism and control*, not classification.
- **Not a claim about "AI writing" in general.** We study *one model's* tell. Gemma 2 2B is small and is **not** the model whose prose anyone actually complains about (that's GPT-4o / Claude / Gemini — all closed, no public SAEs). The honest claim is *"here is the mechanism of this tell in this model,"* not *"here is why AI writes like this."*
- **Not a paper, yet.** It is an exploration until a credentialed mech-interp reader says otherwise. No self-citation BibTeX. No "Reviewer: Closed" theater. No genre-cosplay of a Transformer Circuits report.

---

## 5. The metrics

`grammar-layer` measured effect on a **single factual token**. The construction is **multi-token**, so the metric changes. Three measurements:

- **M1 — Construction rate (behavioral ground truth).** Generate N continuations (N ≥ 50, ≥ 5 seeds) from neutral prompts; a hinge-detector classifier (`src/classifier/`) labels each for C1–C4 presence. Rate = fraction containing the construction. Model-agnostic. Needs no SAE.
- **M2 — Pivot probability (the causal lever).** Given a context that has opened the negation ("It's not just an update"), measure `P(pivot token completes the construction)` — `P(—) + P(it's) + P(but) | …`. Clean next-token probability — the existing ablation harness applies directly.
- **M3 — Quality preservation (the thing that makes it useful, not a trick).** After any intervention: fluency (perplexity on held-out), coherence (LLM-judge 1–5), meaning (embedding cosine vs original). The *product* claim lives or dies on M3. Removing the tell and wrecking the prose is a failure, not a finding.

---

## 6. Build phases (each has a kill check — honor it)

| Phase | What | Kill check |
|---|---|---|
| **0** | Scaffold the refactor — modules, README, smoke test | repo runs end-to-end; README leads with prior work ✅ |
| **1** | Build & validate the construction classifier | precision/recall ≥ 0.85 on C1–C3 against hand-labelled set |
| **2** | Behavioral baseline (M1) across `gemma-2-2b`, `gemma-2-2b-it`, GPT-2 small, Pythia 70M | instruct rate ≫ base rate (if not, Phase 6 already in trouble) |
| **3** | Feature discovery — D1 contrast pairs, differential SAE activation at the pivot | a feature or 2–3-feature supernode shows clean, label-interpretable separation |
| **4** | **Causal validation** — bidirectional ablate + clamp on D2, vs random-k / bottom-k controls | both directions beat controls; if not, report null honestly, don't reframe |
| **5** | Quality preservation (M3) on the intervention | fluency/coherence/meaning all preserved → product claim survives |
| **6** | Genealogy — base vs instruct on the validated feature; SAE-transfer caveat verified empirically | reconstruction quality on instruct is acceptable; instruct-side numbers are trusted only conditionally |
| **7** | De-slop demo + writeup — inference-time steering vector, before/after | small judge eval: less AI-sounding while staying fluent |

The kill checks are the entire epistemic upgrade over the previous project. **Do not reframe a null as a discovery.** That was the lesson; it cost a repo to learn.

---

## 7. Repo layout

```
src/
  classifier/      # M1 — regex hinges + dependency-based FP filter
  steering/        # SAE-feature clamp hooks (ablate ↓ and clamp ↑)
  quality/         # M3 — fluency / coherence / meaning
  genealogy/       # Phase 6 — base vs instruct, with SAE-transfer check
  deslop/          # Phase 7 demo — inference-time steering vector
  neograph/        # Reused substrate from grammar-layer (Neo4j feature graph,
                   # SAELens loaders, Neuronpedia label cache, manifold fits)

scripts/
  load_bearing_topk.py            # KEEP — per-prompt top-K ablation harness
  load_bearing_control.py         # KEEP — random-k / bottom-k targeting controls
  load_bearing_mean_ablation.py   # KEEP — mean-ablation OOD robustness
  causal_attribution_v2.py        # KEEP — per-feature attribution
  01_load_model_and_sae.py        # KEEP — model + SAE smoke
  04_ingest_features.py           # KEEP — feature ingestion
  prefetch_labels.py              # KEEP — Neuronpedia label warm
  fetch_labels_pending.py         # KEEP — per-model label cache
  migrate.py / 00_bootstrap_neo4j.sh  # KEEP — substrate setup
  legacy/                         # grammar-layer-era scripts, preserved for
                                  # reproducibility of the prior writeups

data/
  D1_contrast_pairs.jsonl   # Phase 1 — ≥200 hand-checked minimal pairs
  D2_neutral_prompts.json   # Phase 2 — ≥100 prompts that don't beg the construction
  D3_fluency.txt            # Phase 5 — held-out human prose for perplexity
  labels_cache.json         # Gemma 2 2B Neuronpedia autointerp cache
  legacy/                   # Capital-city / weekday / arithmetic prompts (prior project)

reports/                    # New outputs land here; reports/legacy/ has the prior writeups
notebooks/legacy/           # STORY.md, OVERNIGHT_SUMMARY.md and other prior-project artifacts
apps/legacy/                # Three.js viewer from the previous project (may be revived in §9)
web/legacy/                 # Earlier interactive walkthrough
```

The **Neo4j substrate (`src/neograph/`)** is the genuinely Hopkinson-shaped piece worth preserving: a multi-relation feature graph indexed for vector search and Cypher-queryable across models. Its v2-era role here is to store validated construction features and their cross-model alignments — the graph-native question from PRD §9: *does the "not X, but Y" feature in Gemma align (by decoder cosine) with the corresponding feature in Pythia, and is the alignment stronger among instruct-tuned models?*

---

## 8. Reproduce — Phase 0 only

```bash
# 1. Install. Requires uv (https://docs.astral.sh/uv/) and Python 3.12.
uv sync

# 2. Set HF_TOKEN in .env (Gemma 2 is gated on Hugging Face).
echo "HF_TOKEN=hf_..." > .env

# 3. Smoke test — imports + classifier hits canonical C1–C4 examples.
.venv/bin/python -m pytest -W ignore tests/test_refactor_smoke.py -v
```

Phases 1–7 are not yet runnable. Their scripts will land under `scripts/` (engines reused) and new modules under `src/<name>/` as each phase ships.

---

## 9. Honesty contract

Carried from the last project, non-negotiable:

1. **Cite prior work in the README before writing a line of analysis.** ✅ §2 above.
2. **State the Gemma-2-2b external-validity ceiling in every writeup's opening.** This is *one model's* tell, not "why AI writes like this."
3. **Report quality and faithfulness, not just the headline effect.** M3 is a hard gate on the product claim.
4. **Honor the kill checks.** A null result reported honestly is worth more than a grand result that doesn't hold.
5. **Before promoting it anywhere near ML researchers or the Neo4j AI Ethics committee:** get one credentialed mech-interp reader to sanity-check the causal claim. Posting first, checking later, is exactly the move that hurt last time.

---

## 10. License

MIT — see [LICENSE](LICENSE).

---

*It would be easy to end this by saying it's not a detector, it's a mirror. So I won't.*
