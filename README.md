# Not This, But That

*"This isn't a setback, it's a springboard."* Every chatbot writes that sentence. I went looking for the single internal feature in Gemma 2 2B that produces it — the way refusal famously lives in a single direction — and found that there isn't one. There's a **coalition of twenty-five SAE features**, two of which do a third of the work each. Silence all twenty-five and the textbook form of the tic nearly vanishes (**−93%** under the strict, blind-validated detector); the whole negated family drops **44%** on out-of-sample neutral prompts (18/306 → 10/306 paired generations, McNemar mid-p 0.012), with the prose staying fluent. And then the model reroutes: its favourite escape — the affirmative cousin *"it's more than just X. It's Y"* — actually **rises** under the kill, putting the widest honest count at **−25%**. The catch that became the finding: what those features actually run is the model's **contrast machinery** — silence them and the share of generations containing "but" falls from 13.4% to 1.3% while words-per-generation stays identical at 32 — and the "not X, but Y" construction is just that machinery's loudest output.

This repo is the code, data, and receipts behind the writeup. Every number the post quotes maps to an artifact in the [receipts table](#receipts) below.

## The finding in five bullets

- **It's a coalition, not a switch.** Twenty-five layer-20 SAE features, selected by per-feature causal attribution to the pivot decision. Leave-one-out says two are indispensable — **3223** (*"phrases conveying exceptions or negations"*, cost-when-removed 0.073) and **9909** (*"references to digital technology"*, 0.074) — one is secondary (12898, 0.021), and the other twenty-two are redundant supporters at under 0.013 each. Ablating the two cores alone buys most of what twenty-five buy.
- **Single-feature and prompt-level attacks fail.** A gauntlet of seven — ask nicely (−38%), logit bans ("works" by making seven common words unsayable), in-context exemplars ("work" by replacing the model's voice), directional orthogonalization (−12%), CAA steering (kills only where the prose collapses), scoped and global single-feature ablation — failed outright or worked only via collateral. The model reroutes around any one feature: the Hydra Effect, applied to rhetoric.
- **The selection had to be causal, not structural.** Feature 3223's decoder neighbours (−20%), co-activation partners (−18%, i.e. nothing beyond 3223 alone), and Leiden community-mates (−0.1%) all failed to find the coalition; the causal-attribution top-10 dropped P(pivot) 61%. Structural similarity in this SAE does not predict coalition membership.
- **It's layer-20-local.** The same pipeline at L12 buys −30%, at L25 −52%, at L20 −72% — and joint ablation across all three layers (75 features) floors at 0.079, indistinguishable from L20 alone at 0.077. Heads at several layers, one heart.
- **Necessity, not sufficiency.** Ablating the coalition removes the construction; clamping it up does not reproduce it. That asymmetry is the deepest unresolved thing in the project.

## What this is not

- **Not a detector.** AI-text detection is crowded and adversarially doomed. This is about mechanism and control. (The slop-o-meter is a demo of what the project's measurement instrument sees, not a product.)
- **Not a claim about "AI writing" in general.** One construction, one model. Gemma 2 2B was chosen for its public SAEs rather than for the infamy of its prose. The honest claim is *here is the machinery of this tell in this model*, not *here is why AI writes like this*.
- **Not peer-reviewed.** Nothing here has been read by a credentialed mech-interp researcher. The receipts exist so you can check the work, not so you can skip checking it.

## See it

**[Live demo](https://not-this-but-that.vercel.app/demo/)** — the hosted story page: side-by-side baseline-vs-ablated playbacks you can step through token by token, with the construction highlighted as it forms — or doesn't.

The full **playground** — the 16,384-dot feature atlas, concept search, lasso ablation, surgical de-slop, the behaviour mixer, the audit trail — needs the live model and runs locally. Honest requirements: a Hugging Face account with access to the gated Gemma 2 weights, ~10 GB of downloads on first run, and enough memory to hold both Gemma 2 2B variants plus the SAE (~12 GB; developed on Apple Silicon/MPS, falls back to CPU). The graph-backed features additionally want a local Neo4j (below).

## Quickstart

```bash
# 1. Install. Requires uv (https://docs.astral.sh/uv/); Python 3.12 is pinned.
uv sync
.venv/bin/python -m spacy download en_core_web_sm   # the strict detector's dependency check

# 2. Credentials. Gemma 2 is gated on Hugging Face — accept the licence at
#    huggingface.co/google/gemma-2-2b-it, then put a read token in .env:
cp .env.example .env    # edit: HF_TOKEN=hf_...

# 3. Start the probe daemon (holds model + SAE in memory, serves the demo).
scripts/probe_run.sh start

# 4. Open the playground.
open http://127.0.0.1:8765/demo/
```

The map, community names, and baked examples are committed under `web/demo/`, so the atlas renders and the story page works without any graph. Generation, ablation, and P(pivot) probes run against the daemon.

### Graph features (optional)

Surgical de-slop, the behaviour mixer, the audit trail, and the live neighbour overlays are Cypher under the hood and need a local **Neo4j 5 with GDS and APOC** at `bolt://localhost:7693` (credentials in `.env.example`).

`scripts/00_bootstrap_neo4j.sh` builds one, but it is honestly macOS/Homebrew-pinned: it clones a Homebrew-installed Neo4j 2026.03.1 into `.neograph-db/` and downloads GDS. Any Neo4j 5+ instance with GDS works — point `NEO4J_URI` at it. Then the ingest chain:

1. `scripts/01_load_model_and_sae.py` — smoke test: Neo4j reachable (GDS + APOC), HF auth, model + SAE load and produce (1, seq, 16384) activations.
2. `scripts/migrate.py` — applies schema constraints + vector indexes, idempotently.
3. `scripts/04_ingest_features.py` — writes the 16,384 `:SAEFeature` nodes with decoder/encoder vectors and pulls Neuronpedia auto-interp labels for each.
4. `scripts/seed_behaviours.py` — seeds the `:Behaviour` nodes and weighted `INCLUDES` edges (the `ai-ism` coalition, straight from `reports/pivot_attribution.json` + the leave-one-out costs).
5. `scripts/build_label_embeddings.py` — embeds all 16,384 feature labels and writes the vector index that concept search and surgical de-slop retrieve against (~2 min on CPU).

`reports/label_embeddings.npy` (and its index JSON) are gitignored — run step 5 locally or surgical de-slop has nothing to retrieve with. One more honesty note: the `DECODER_SIMILAR` / `CO_ACTIVATES_WITH` edges (524k / 272k) and the 18 Leiden communities were built with GDS via `src/neograph/relations.py` rather than a numbered script; the demo degrades gracefully without them — only the live graph-neighbour expansions need the edges.

With the graph up, the playground's three demos are each one Cypher query:

- **Surgical de-slop** — embed your prompt, vector-retrieve matching feature labels, intersect with the `ai-ism` `:Behaviour`'s `INCLUDES` edges, silence only the overlap. For some prompts the intersection is empty and the graph says *leave it alone* — which is the point.
- **Mix your own chatbot** — four `:Behaviour` subgraphs (`ai-ism`, `bullets`, `hedging`, `formal_register`) with a slider each; intensities compose into one weighted Cypher UNION of features to silence.
- **Audit trail** — every generation writes an `:Intervention` node with `(:Intervention)-[:USED_SOURCE]->(:Source)-[:SELECTED]->(:SAEFeature)` paths, so "why did this run silence feature 3223?" is a path query, weeks later. Each demo has a `show Cypher` button.

## Receipts

Every claim, with the artifact that backs it. All paths under `reports/`.

| Claim | Receipt |
|---|---|
| Neutral prompts, out-of-sample, n=306 pairs: family 18 → 10 (**−44%**, CI 14–70%, mid-p 0.012; 8 of 9 changed prompts toward clean); strict 14 → 1 (**−93%**); family+cousin 28 → 21 (**−25%**, the cousin itself 10 → 11) | [`m1_stats_reanalysis.md`](reports/m1_stats_reanalysis.md), [`m1_rescore_union.json`](reports/m1_rescore_union.json) |
| Prompt-level: 10 prompts changed status, all 10 toward clean (sign-test p ≈ 0.001) | [`m1_stats_reanalysis.md`](reports/m1_stats_reanalysis.md) |
| The detector fixes that revised the 80% draft headline, twice — every dropped/added hit hand-inspectable | [`permissive_fix_audit.md`](reports/permissive_fix_audit.md) |
| Primed prompts, n=300 (40% in-sample): prefix-inclusive family 267 → 120 (−55%), strict 172 → 31 (−82%); completion-only −51% (p=4.5×10⁻³), held-out half −50% | [`m1_stats_reanalysis.md`](reports/m1_stats_reanalysis.md), [`heldout_reslice.md`](reports/heldout_reslice.md) |
| Two-feature core 3223+9909, n=120 (in-sample): completion-only −53%; prefix-inclusive only −15% vs the full coalition's −55% — two features carry the decision, twenty-five carry the paragraph | [`m1_stats_reanalysis.md`](reports/m1_stats_reanalysis.md) |
| Contrast machinery, beyond the tic: "but"-share 13.4% → 1.3% (neutral) and 75% → 1.7% (primed); commas 1.52 → 0.58/gen; words/gen 32 → 32 | [`collateral_syntax.md`](reports/collateral_syntax.md) |
| Ablation ladder at the pivot decision (n=80): 1/2/5/10/25/50/100 features → −18/−37/−51/−61/−72/−75/−79%; top-25 at n=200 = −76% | [`asymptote_ladder.json`](reports/asymptote_ladder.json) |
| Controls: 100 random features < 0.001 effect; coalition beats all 20 matched-activation null draws (drop +0.200 vs best null +0.001) | [`matched_activation_null.json`](reports/matched_activation_null.json) |
| Leave-one-out: two indispensable cores (3223: 0.073, 9909: 0.074), secondary 12898 (0.021), rest < 0.013 | [`q3_leave_one_out.json`](reports/q3_leave_one_out.json) |
| Layer-local: L12 −30%, L20 −72%, L25 −52%; joint L12+L20+L25 floor 0.079 ≈ L20-alone 0.077 | [`q7c_cross_layer_joint.json`](reports/q7c_cross_layer_joint.json) |
| Fluency: perplexity on held-out human prose 1.079× (coalition) / 1.000× (single feature) | [`phase5_quality_coalition_top25.md`](reports/phase5_quality_coalition_top25.md) |
| Strict classifier blind-validated at P = 0.80, R = 1.00 on 90 independently-sourced sentences | [`tier_0a_classifier_blind_eval.md`](reports/tier_0a_classifier_blind_eval.md) |
| The seven-attack gauntlet, re-scored per-generation with the union detector | [`gauntlet_union_rescore.json`](reports/gauntlet_union_rescore.json) |
| Blinded LLM judge over all 1,452 generations: 88–92% agreement with the family tier; primed kill confirmed −48% (p ≈ 5×10⁻⁴); widest semantic count on neutral prompts statistically flat (the reroute, fully visible) | [`llm_judge_rescore.md`](reports/llm_judge_rescore.md) |
| Pre-registered confirmation on 50 fresh prompts: **KILL by its frozen gates** — direction consistent (3 → 1) but the fresh concrete-register prompts barely elicit the tic (1.0% baseline vs 5.9% exploratory). The headline rates are a property of discussion-register prompts; the tic is topic-conditional | [`confirmation_prereg.md`](reports/confirmation_prereg.md), [`confirmation_run.md`](reports/confirmation_run.md) |
| The canonical coalition (n=40 selection run; the n=100 rerun overlaps 19/25 with identical top-3) | [`pivot_attribution.json`](reports/pivot_attribution.json), [`pivot_attribution_n100.json`](reports/pivot_attribution_n100.json) |

Detector provenance, stated plainly: the strict classifier is the only blind-validated one; the permissive layer that completes the family is FP-audited but not blind-validated; the affirmative "more than just" cousin is counted separately (it rises under ablation); the gauntlet's original referee was a third detector (P = R = 0.857 on its own holdout); a blinded frontier-model judge agrees with the family tier on 88–92% of all generations ([`llm_judge_rescore.md`](reports/llm_judge_rescore.md)). The statistics regenerate from the committed generation JSONs without a GPU — `m1_stats_reanalysis.py`, `rescore_union.py`, `heldout_reslice.py`, `collateral_syntax.py`, `audit_permissive_fix.py` in `scripts/`. The Discovery/Confirmation protocol the project ran under is at [`reports/operating_protocol.md`](reports/operating_protocol.md).

## Where this could be wrong

- **It's post-hoc.** The pre-registration committed to single-feature attacks; the coalition is what came out of those failing. On the order of fifty analyses ran before the headline number existed. What protects the result is effect size, the out-of-sample neutral eval, and the matched-activation null — not pre-registration. A properly pre-registered confirmation run on fresh prompts is the most important experiment not yet done.
- **Several supporting numbers are in-sample.** The ladder, leave-one-out, matched null, and cross-layer tests all use the 80 prompts the coalition was selected on. The neutral-prompt eval (−93% strict / −44% family / −25% incl. cousin) is the out-of-sample one. The primed eval is 40% in-sample, but its held-out half shows the same drop (−50% vs −51%); re-selecting the coalition at n=100 keeps 19/25 members and the same top three.
- **Sufficiency is unresolved.** Clamping the coalition up doesn't reproduce the construction. The pre-registered interchange-patching retrial hasn't been run; the asymmetry may be a real fact about coalitions or an experiment away from dissolving.
- **One model, one SAE, one width.** Gemma 2 2B, Gemma Scope L20/16k canonical. No cross-model replication yet; this is an existence proof, not a law.
- **The fluency defence uses the wrong instrument.** 1.079× perplexity on held-out *human* prose says the model still predicts clean text, not that its own de-slopped generations stayed good. Blinded preference judging hasn't been run; the circumstantial evidence is the collateral table (identical length, identical sentence counts).

## Repo map

```
src/
  classifier/   the detectors — strict (regex + spaCy dependency check) and the v2 union
  steering/     SAE hook factories: ablate / clamp
  quality/      fluency, coherence, meaning preservation (M3)
  genealogy/    base vs instruct comparisons
  deslop/       generation-time coalition ablation
  gauntlet/     the seven single-feature / prompt-level attacks
  firewall/     Discovery/Confirmation data-split enforcement
  neograph/     Neo4j substrate: schema, ingest, relations, manifold fits
scripts/        the pipeline + the probe daemon (probe_run.sh / probe_daemon.py / PROBE_README.md)
data/           D1 contrast pairs (226), D2 neutral prompts (102), D3 fluency prose, committed splits
reports/        every artifact cited above, plus the full writeups
web/demo/       story page + playground (served by the daemon at /demo/)
cypher/         graph schema + saved queries
bloom/          Neo4j Bloom perspective for the feature graph
tests/          pytest suite
```

## Lineage

The methods this project leans on, and where they came from: the SAEs are Gemma Scope ([Lieberum et al. 2024](https://arxiv.org/abs/2408.05147)); the directional-ablation attack that failed here is the one that *worked* on refusal ([Arditi et al., NeurIPS 2024](https://arxiv.org/abs/2406.11717)); the steering vectors are CAA ([Rimsky et al., ACL 2024](https://arxiv.org/abs/2312.06681)); the rerouting behaviour has a name, the Hydra Effect ([McGrath et al. 2023](https://arxiv.org/abs/2307.15771)); and the closest prior art on SAE features of AI-text style is [Kuznetsov et al. 2025](https://arxiv.org/abs/2503.03601). The construction itself is antithetic parallelism, named by Robert Lowth in 1753 — the model didn't invent it, it just can't stop.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q --ignore=tests/test_neo4j_smoke.py
# 24 passed
```

`test_neo4j_smoke.py` needs the live graph on port 7693; everything else runs cold.

## License

MIT — see [LICENSE](LICENSE).

---

*It would be easy to end this by saying it's not a detector, it's a mirror. So I won't.*
