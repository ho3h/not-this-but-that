# Not This, But That

> Find the feature inside a small open LLM that produces the **"not X, but Y"** writing tell, prove you can switch it off without wrecking the prose, and tell the story in the voice of the tell itself.

This isn't a detector. It's the same engine that lived in the previous repo (`grammar-layer`), repointed at a better question.

The previous project rediscovered a known phenomenon — SAE "suppression features" — and over-claimed novelty. The post-mortem was clear: the framework was sound, the framing was not, and the finding was someone else's from 2024. The machinery, though, was good. *Not* this — but *that*.

---

## Status

This project has run two parallel investigations through the same adversarial methodology. **Read them in this order:**

1. **[`reports/operating_protocol.md`](reports/operating_protocol.md)** — Discovery/Confirmation firewall, BH-FDR multiplicity correction, pre-registered Tiers 0–6, prime directive. The methodology is the load-bearing intellectual contribution; the empirical findings are the worked example of it working.
2. **[`reports/the_explicit_decision.md`](reports/the_explicit_decision.md)** — Path A (mechanism) vs Path B (register) split. Two separate writeups, deliberately not merged.
3. **[`reports/path_b_register_writeup.md`](reports/path_b_register_writeup.md)** — the behavioural finding, narrowed by cross-family replication, with the honest cross-family null reported.
4. **[`reports/writeup.md`](reports/writeup.md)** — the Path A exploration writeup with its caveats. Conditional on the Tier 0b prerequisite.

### Path B — narrowed empirical finding

**Within Gemma 2 2B-it (Tier 0a + Discovery → Confirmation → BH-FDR passed):**
- Construction rate is **4.2× the base model's** (Gemma 2 2B-it: 1.8% / Gemma 2 2B base: 0.4%); non-overlapping bootstrap CIs.
- **94% of construction usage is C3** ("It's not just X — it's Y", minimize-then-elevate).
- Constructions cluster at the **beginning of generations** (median relative position 0.10 vs 0.50 for non-construction sentences; Mann-Whitney p_BH = 0.043 on the held-out Confirmation split).

**Cross-family replication on Qwen 2.5 7B Instruct: NULL.** C3 share of any_core = 0%; H17 direction does not replicate (n=3 positives, no power either way). The 94%-C3 / opener-position signature is Gemma-2-2b-it-specific, not a cross-family instruct register. The surface register (opening summary + bullets + bold headers) appears shared cross-family by eyeball but was not measured. Gemma 2 9B-it would have tested the within-Gemma-family scale question but the HF token doesn't have access.

**Retired:** "Instruct-tuning installs the C3 register across families." Also retired: H19 (was an apostrophe-in-contraction artifact, not a real quote-mark effect).

### Path A — mechanism story (conditional, unresolved)

- **Discovery candidate**: SAE feature #3223 ("phrases conveying exceptions or negations") at L20 of Gemma Scope width-16k. Ablation at the pre-pivot decision point drops P(pivot) by 25% relative; 5/5 random-k controls produced no measurable change.
- **Necessity yes, sufficiency no**: clamping the feature *up* doesn't reproduce the construction.
- **De-slop product claim retracted**: ablating the feature during open-ended generation gives 0% drop in construction rate, because the feature is dormant on neutral prompts. It gates the *commit* to the pivot, not the *entry* into construction-mode.
- **Foundation crack**: Tier 0b (Gemma Scope reconstruction-quality verification) returns negative VE through every measurement path, which is definitionally an instrument bug. Proxy evidence the SAE is functional (L0 = 74 vs canonical 71; cosine 0.83) does not substitute for the prerequisite the pre-registration demanded. Until a credentialed mech-interp reader unblocks the VE-reproduction recipe, the mechanism story stays as Discovery candidates, not findings.

### Foundation cracks
1. **CLOSED** — classifier blind-validation. [Tier 0a](reports/tier_0a_classifier_blind_eval.md): P = 0.80, R = 1.00 on 90 independently-sourced sentences. Above the pre-registered ≥ 0.70 gate. **Caveat (H09 lesson):** the classifier's C3 regex partly leans on the "just/merely/simply" lexicon, so the 94%-C3 share partly reflects what the classifier can see. Worth a sentence of honesty in any onward use.
2. **OPEN** — Tier 0b VE reconstruction. See [`reports/tier_0b_kill.md`](reports/tier_0b_kill.md).

Nothing here has been peer-read by a credentialed mech-interp researcher. Path B is shippable as a small modest finding; Path A is conditional. The honesty contract (§9 below) still applies.

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
- Goodfire / Bhalla et al., ["Do Sparse Autoencoders Find True Features?"](https://arxiv.org/abs/2604.28119), arXiv:2604.28119, April 2026 — the key warning that meaningful concepts may be tiled across low-dimensional manifolds of SAE atoms rather than represented by isolated feature directions.
- Zhang, Wang & Su, "Mechanistic Knobs: Retrieving and Steering High-Order Semantic Features via SAEs" — arXiv:2601.02978 — precedent for steering behavior-level linguistic features.

**How this differs from the Goodfire manifold work:** we do not claim that
the global 2D UMAP is the manifold, nor that single SAE atoms are complete
concepts. The graph is used as a substrate for hypotheses and interventions:
activation-linked sets, behavior coalitions, communities, and prompt-retrieved
slices. The visualization is currently an atlas over atoms; the research claim
is the causal effect of a validated behavior coalition.

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

## 6. Build phases — what each kill check actually returned

| Phase | What | Kill check | Outcome |
|---|---|---|---|
| **0** | Scaffold the refactor | repo runs end-to-end, README leads with prior work | ✅ |
| **1** | Build & validate the construction classifier | precision/recall ≥ 0.85 on C1–C3 | ✅ 1.00 P/R (self-consistency caveat) |
| **2** | Behavioral baseline (M1) across 4 models | instruct rate ≫ base | ✅ instruct **12× C3**, **4.2× any_core**, non-overlapping CIs |
| **3** | Feature discovery — differential SAE activation | clean label-interpretable feature | ⚠ found consequence features, not causes; recovered via per-feature pre-pivot attribution → feat **3223** "phrases conveying exceptions or negations" |
| **4** | **Causal validation** — bidirectional vs random-k | both directions beat controls | ⚠ **PARTIAL** — ablate drops P(pivot) 25%, separated qualitatively from a near-degenerate random-k null (5/5 controls produced no change); clamp-up fails (necessity yes, sufficiency no) |
| **5** | Quality preservation (M3) | fluency / coherence / meaning preserved | ✅ D3 perplexity ratio **1.000×** baseline (scalpel) |
| **6** | Genealogy — base vs instruct | reconstruction quality acceptable; gap signal | ✅ instruct ablation drop **1.81× larger** than base; VE recon measurement has a bug to fix |
| **7** | De-slop demo + writeup | ablation during generation reduces M1 | ❌ **0% drop** — feature is conditional on construction-mode contexts; the mechanism stands, the product claim does not |

The full prose narrative — what worked, what didn't, what we can and cannot claim — is at **[`reports/writeup.md`](reports/writeup.md)**. The honesty contract was honored: where a null landed, the writeup reports it as a null.

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
