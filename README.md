# Not this but that

Chatbots keep writing "It isn't just X, it's Y". This repository looks for the machinery behind that habit inside Gemma 2 2B (and then Gemma 3 4B), using Google DeepMind's Gemma Scope sparse autoencoders (SAEs). It tests what happens when you ablate that machinery, when you amplify it, and where the habit came from in the first place.

**Read it:** [the post on Medium](https://medium.com/p/00fbd8453c74) · [interactive version](https://not-this-but-that.vercel.app/demo/)

## In short

- **25 features decide it.** Ablating them in Gemma 2 2B cut the habit from 17% of answers to 0.6% (88 → 3 of 510 answers to 102 questions it had never seen; pre-registered). On 140 fresh questions of six kinds: −93%. Ablating 25 decoy features changed nothing, and quiz scores and blind quality ratings held.
- **The price is some negation.** Gemma used about half as many "not"s overall. It still got 28 of 30 yes/no questions right, but its warnings got vaguer: a blind judge preferred the normal warnings 34 times to 16.
- **It works in reverse.** Amplifying the same 25 raises the habit from 12% of answers to 37% (×2) and 54% (×3). Decoys at ×3: 15%.
- **It replicates in Gemma 3 4B** (−83% with 24 features, pre-registered). There the habit and ordinary negation share machinery, as predicted in advance.
- **The question predicts it.** 234 of 800 answers to "why does this matter" questions used it, against 2 of 400 answers to "how does this work" questions.
- **Why chatbots do it (exploratory).** The habit is old in print: "is not just" was about 15 times as common in 2019 as in 1920. Instruction tuning makes answers longer; preference tuning raises the habit per word on meaning questions (1.7×). People and AI judges are indifferent to it at equal length, while an open reward model prefers it over the same idea without the contrast 77% of the time.

Every number links to a receipt below, including the pre-registered tests that failed.

## Findings in detail

- **Twenty-five features in the chat model decide it (pre-registered, held-out).** Features chosen in `gemma-2-2b-it` itself, at the token where it commits to the turn ("A city's coffee culture | isn't just…", "…it's | more than just…"), by switching off each active layer-20 feature one at a time (`scripts/reroute_attribution.py`, 40 decision points where the chat model was about to write "isn't just" in D4 answers; 40 "more than just" decision points ranked the cousin). Switched off on 102 prompts never used to choose them (confirmation v3, chat template, error-term-preserving ablation, 510 paired answers): the sentence or its "more than just" cousin falls from 88 answers to 3 (−97%, 95% prompt-clustered CI 92–100%, mid-p 6×10⁻²⁵). Cost gates passed: perplexity on human prose 1.05×, quiz 90% → 90%, no "a"/"an" errors; "but" kept (12.9% → 11.2%); blind fluency 4.88 vs 4.90, blind usefulness 4.75 vs 4.72.
- **At the decision point, on held-out answers, the effect is specific to these 25.** Pull towards the negation (38 decision points cut from confirmation v3's normal answers): top 1/2/5/10/25 → −46/−68/−85/−97/−98%. Twenty draws of 25 random features: mean −0.4% (max 0.7%); twenty draws matched on firing rate: mean −0.1% (max 1.6%). Alone, 15646 ("conditions or exceptions") gives −46%, 10252 −38%, 10655 −22%, 13565 −21%. The cousin ("more than just", 36 decision points) depends almost entirely on feature 13565, "comparison or enhancement": alone −91%, every other member ≤ 2%; leaving it on costs 91 of the 94 points ([`decision_point_ladder.md`](reports/decision_point_ladder.md), exploratory).
- **A second, harder pre-registered test passed every gate (confirmation v4).** Same 25 on 140 fresh D5 prompts written blind (advice, meaning, message, copy, encouragement, essay, plus 20 explicit requests), 300-token answers, 5 samples, normal / 25 off / 25 firing-rate-matched features off decoded together. H1: habit 94 → 7 (−93%, CI 85–99%, mid-p 2×10⁻²⁴). H2: matched control 94 → 103 (no drop). H3: ARC-Easy 85.7% → 85.3%, WikiText-2 perplexity 1.049×, "a" before a vowel 0.3%, blind pairwise preference 54 wins / 4 ties / 62 losses for the 25-off answers (share 47%, gate ≥ 40%); a second blind judge on a different model gives 52 / 6 / 62 and agrees on 91 of 120 pairs. When a prompt asks for the pattern explicitly, the model still produces it (96% vs 100%): the 25 remove the habit and leave the ability.
- **The catch: the 25 also carry ordinary negation (exploratory, found after both tests).** In v4's everyday answers, negation words fall from 6.7 to 3.1 per 1,000 (matched control 6.4), and "don't / doesn't / can't / won't / didn't" from 278 to 172. The model can still say no (28/30 "no" questions right, on and off; `negation_check.md`), but on 30 "what should you never do" questions a blind judge preferred the normal answers' warnings 34 to 16 (10 ties): switched off, warnings often lose "never"/"don't" and read as bare imperatives under a "should never" heading. The registered "no measurable cost" verdicts stand by their frozen gates; this cost was outside them.
- **A cleaner switch trades strength for side effects (pre-registered confirmation v5, Gemma 2).** Scoring candidates on ordinary-negation decision points showed two of the 25 doing double duty (10655 cuts ordinary negation 42% alone, 15646 12%). The specific 25 swap them for 5619 and 3315. On 140 fresh D7 prompts: habit 92 → 26 (−72%, CI 59–83%) **PASS**; matched control 92 → 95 **PASS**; ARC 85.7% → 85.7%, WikiText 1.048×, "no" questions 28/28, blind preference 41% **PASS**; warning clarity 39% (gate ≥ 40%) **FAIL**; ordinary negation 0.95× **PASS**. A second blind judge on a different model gives 51% preference and 47% warnings. The original 25 on the same prompts and seeds (exploratory, identical normal rows): −91%, ordinary negation 0.60×, warnings 23% / 27%.
- **Replicated in Gemma 3 4B-it, with the trade-off predicted in advance (pre-registered confirmation v5, Gemma 3).** Same procedure with Gemma Scope 2 instruct SAEs at layer 22; 24 features (family top 20 + cousin top 5, because Gemma 3 uses the cousin as often as the family). Gemma 3 uses the habit in 74% of answers to meaning prompts (Gemma 2: 29%). On D7: habit 156 → 27 (−83%, CI 73–91%) **PASS**; matched control 156 → 143 **PASS**; ARC 88.7% → 88.3%, WikiText 1.003×, "no" questions 28/28, preference 40.0% **PASS** (at the gate; second blind judge 42%, so a modest quality cost); warnings 32% **FAIL**; ordinary negation 0.39× **FAIL**, exactly as registered. Its strongest habit feature (1138) promotes "not" in many languages; a set without it leaves most of the habit (−22%). With the set off, 30 of 700 answers contain a doubled "it it" (0 normally).
- **The same 25 turn it up (exploratory).** Scaling their layer-20 contribution on 152 D2 + D2b prompts: 25 off 0.3%, normal 11.5%, ×2 36.8%, ×3 54.3% of answers; 25 matched features at ×3 14.8%. On process/scene prompts 5% → 31%. Blind checks: 32/40 ×3 hits confirmed as the reframing move (19/20 normal), blind fluency 4.60 vs 4.77.
- **What's left is looser wording.** A blind judge counting any reframing move sees 49/150 → 27/150 answers (mid-p 6×10⁻⁴); the remainder is "goes beyond", "transcends" and similar.
- **Brute force breaks the model first.** Ranking features by how much they promote the pivot and switching off more of them barely moves the habit until 1,894 features at layers 12, 20 and 25 are off (29% → 9% of answers), at perplexity 2.49×, quiz 75% and blind fluency 2.1/5. Adding the top "more than just" features to the 25 reaches 0% but breaks "a"/"an" agreement (exploratory).
- **No single switch.** The negation feature 3223 ("phrases conveying exceptions or negations") isn't in the top 500 of the chat-model attribution at the decision point. Seven attacks centred on it, or applied from outside the model, either left most of the habit in place or removed it by damaging the output: asking nicely 14 → 8 cases, banning seven common words 14 → 0, few-shot examples 14 → 0 (the model copies their style), ablating or orthogonalizing feature 3223 15 → 11–12, and a CAA steering vector 14 → 2 (the prose degenerates).
- **The prompt predicts the habit (pre-registered).** In confirmation v2's normal answers, 234 of 800 answers to meaning/value prompts contained the construction or its cousin (detector v3), against 2 of 400 for process/scene prompts. **H2 confirmed.**
- **Earlier route, superseded.** The project first selected a coalition of 25 in the base model, at the comma after the negation. It lowered the base model's pivot probability by 72% in-sample, but in chat answers the habit rerouted into "isn't just X. It's Y." and its pre-registered test (confirmation v2, H1) **failed its gates**: family 85 → 80 on meaning prompts. On confirmation v3's prompts the same coalition gives 88 → 75. The first pre-registered confirmation (50 concrete prompts) had a 1.0% baseline and no power. All of it is in the receipts below.

## Scope

- One construction, one 2024-era model, one SAE. Gemma 2 2B was chosen for its public SAEs and feature labels; the claims are about this habit in this model.
- This isn't an AI-text detector.
- Not peer-reviewed. The receipts are there so you can check the work.

## See it

The [interactive version](https://not-this-but-that.vercel.app/demo/) replays recorded answers with and without the 25, word by word, and shows the 16,384-feature map. The full playground (your own prompts, switching features off, concept search, lasso, the graph tools) needs the model, so it runs locally. Requirements: a Hugging Face account with access to the gated Gemma 2 weights, ~10 GB of downloads on first run, and about 22 GB of memory (both Gemma 2 2B variants load in fp32, plus the SAE). Developed on Apple Silicon (MPS); everything else, CUDA machines included, falls back to CPU because the device check only knows MPS. The graph-backed tools also need a local Neo4j (below).

## Quickstart

```bash
# 1. Install. Requires uv (https://docs.astral.sh/uv/); Python 3.12 is pinned.
uv sync
.venv/bin/python -m spacy download en_core_web_sm   # the strict detector's dependency check

# 2. Credentials. Gemma 2 is gated on Hugging Face: accept the licence at
#    huggingface.co/google/gemma-2-2b-it, then put a read token in .env:
cp .env.example .env    # edit: HF_TOKEN=hf_...

# 3. Start the probe daemon (holds model + SAE in memory, serves the demo).
scripts/probe_run.sh start

# 4. Open the playground.
open http://127.0.0.1:8765/demo/
```

The map, community names, and baked examples are committed under `web/demo/`, so the atlas renders and the story page works without any graph. Generation, ablation, and P(pivot) probes run against the daemon.

### Graph features (optional)

Smart silence, the behaviour mixer, the audit trail and the live neighbour overlays are Cypher under the hood and need a local **Neo4j 5 with GDS and APOC** at `bolt://localhost:7693` (credentials in `.env.example`).

`scripts/00_bootstrap_neo4j.sh` builds one, but it is macOS/Homebrew-pinned: it clones a Homebrew-installed Neo4j 2026.03.1 into `.neograph-db/` and downloads GDS. Any Neo4j 5+ instance with GDS works; point `NEO4J_URI` at it. Then the ingest chain:

1. `scripts/01_load_model_and_sae.py`: smoke test: Neo4j reachable (GDS + APOC), HF auth, model + SAE load and produce (1, seq, 16384) activations.
2. `scripts/migrate.py`: applies schema constraints + vector indexes, idempotently.
3. `scripts/04_ingest_features.py`: writes the 16,384 `:SAEFeature` nodes with decoder/encoder vectors and pulls Neuronpedia auto-interp labels for each.
4. `scripts/seed_behaviours.py`: seeds the `:Behaviour` nodes and weighted `INCLUDES` edges (the `ai-ism` behaviour is the 25 from confirmation v3, weighted by their decision-point attribution, with leave-one-out costs from `reports/decision_point_ladder.json`).
5. `scripts/build_label_embeddings.py`: embeds all 16,384 feature labels and writes the vector index that concept search and smart silence retrieve against (~2 min on CPU).

`reports/label_embeddings.npy` (and its index JSON) are gitignored, so run step 5 locally or smart silence has nothing to retrieve with. The `DECODER_SIMILAR` / `CO_ACTIVATES_WITH` edges (524k / 272k) and the 18 Leiden communities were built with GDS via `src/neograph/relations.py` (no numbered script), and the per-feature activation densities they and the matched null rely on came from a capture step (`03_capture_activations.py`) that isn't in the repo. The demo degrades gracefully without the edges; only the live graph-neighbour expansions need them.

With the graph up, the playground's three graph tools are each one Cypher query:

- **Smart silence**: embed your prompt, vector-retrieve matching feature labels, intersect with the `ai-ism` `:Behaviour`'s `INCLUDES` edges, and silence only the overlap. For some prompts the intersection is empty and nothing is silenced.
- **Behaviour mixer**: four `:Behaviour` subgraphs with a slider each; intensities compose into one weighted Cypher UNION of features to silence. Only `ai-ism` was found by causal testing. `bullets`, `hedging` and `formal_register` are the top 25 label-embedding matches for a seed phrase (`scripts/seed_behaviours.py`) and have not been tested causally.
- **Audit trail**: every generation writes an `:Intervention` node with `(:Intervention)-[:USED_SOURCE]->(:Source)-[:SELECTED]->(:SAEFeature)` paths, so "why did this run silence feature 3223?" is a path query, weeks later. Each demo has a `show Cypher` button.

## Receipts

Every claim, with the artifact that backs it. All paths under `reports/`.

### The 25

| Claim | Receipt |
|---|---|
| **Pre-registered confirmation v3**: chat-chosen 25 on 102 held-out D2 prompts × 5: habit 88 → 3 (−97%, CI 92–100%), **PASS**; cost gates **PASS** (perplexity 1.052×, quiz 90 → 90%, "a"+vowel 0.0%); original 25 on the same prompts 88 → 75 | [`confirmation_v3_prereg.md`](reports/confirmation_v3_prereg.md), [`confirmation_v3_run.md`](reports/confirmation_v3_run.md) |
| Blind checks on confirmation v3: any-reframing judge 49/150 → 27/150; blind fluency/quality normal 4.90/4.72, chat 25 4.88/4.75, original 25 4.72/4.38 | [`confirmation_v3_blind_checks.json`](reports/confirmation_v3_blind_checks.json) |
| Decision-point ladder on held-out decision points (exploratory): family top 1/2/5/10/25 → −46/−68/−85/−97/−98%; 20 random 25s mean −0.4%, 20 firing-rate-matched 25s mean −0.1%; singles and leave-one-out; cousin carried by 13565 (alone −91%) | [`decision_point_ladder.md`](reports/decision_point_ladder.md), `scripts/decision_point_ladder.py` |
| **Pre-registered confirmation v4**: 140 fresh D5 prompts × 5, 300 tokens: H1 habit 94 → 7 (−93%, CI 85–99%) **PASS**; H2 matched control 94 → 103 **PASS**; H3 ARC-Easy 85.7 → 85.3%, WikiText 1.049×, "a"+vowel 0.3%, blind preference share 47% **PASS**; explicit requests 100% → 96% | [`confirmation_v4_prereg.md`](reports/confirmation_v4_prereg.md), [`confirmation_v4_run.md`](reports/confirmation_v4_run.md), [`confirmation_v4_judge_robustness.md`](reports/confirmation_v4_judge_robustness.md) |
| Negation side effect (exploratory): negation words 6.7 → 3.1 per 1,000 in v4; "no" questions 28/30 on and off; blind warning clarity 16 wins / 10 ties / 34 losses for the 25 off (Gemma 3 set B: 11 / 20 / 29) | [`negation_check.md`](reports/negation_check.md), `scripts/negation_check.py`, `data/negation_*` |
| **Pre-registered confirmation v5, Gemma 2** (specific 25, D7 + D6): H1 92 → 26 (−72%) PASS, H2 PASS, H3 PASS (preference 41%), H4 warnings 39% FAIL, H5 ordinary negation 0.95× PASS; second judge 51% / 47% | [`confirmation_v5_gemma2_prereg.md`](reports/confirmation_v5_gemma2_prereg.md), [`confirmation_v5_gemma2_run.md`](reports/confirmation_v5_gemma2_run.md), [`confirmation_v5_gemma2_judge_robustness.md`](reports/confirmation_v5_gemma2_judge_robustness.md) |
| Original vs specific 25 on the same fresh prompts (exploratory): −91% / −72% habit, 0.60× / 0.95× ordinary negation, 23% / 39% warnings | [`v5_tradeoff.md`](reports/v5_tradeoff.md), `scripts/v5_original_arm.py` |
| Specific selection (exploratory): which features carry ordinary negation, in both models | [`specific_switch.md`](reports/specific_switch.md), `scripts/specific_selection.py`, `scripts/specific_eval.py` |
| **Pre-registered confirmation v5, Gemma 3 4B-it** (set B, 24 features, layer 22): H1 156 → 27 (−83%) PASS, H2 PASS, H3 PASS (preference 40.0%; second judge 42%), H4 32% FAIL, H5 0.39× FAIL, as predicted | [`confirmation_v5_gemma3_prereg.md`](reports/confirmation_v5_gemma3_prereg.md), [`confirmation_v5_gemma3_run.md`](reports/confirmation_v5_gemma3_run.md), [`confirmation_v5_gemma3_judge_robustness.md`](reports/confirmation_v5_gemma3_judge_robustness.md), [`gemma3_replication.md`](reports/gemma3_replication.md) |
| Gemma 3 engine, SAE fidelity, baseline, attribution, ladders and set pilot (exploratory) | `scripts/g3_mlx.py`, `scripts/g3_pipeline.py`, `scripts/g3_pilot.py`, `reports/g3_*` |
| Turning the 25 up (exploratory): 0.3% / 11.5% / 36.8% / 54.3% of answers at off / normal / ×2 / ×3; matched ×3 14.8%; blind precision 32/40 at ×3; blind fluency 4.60 vs 4.77 | [`turn_up.md`](reports/turn_up.md), `scripts/turn_up.py`, `data/turn_up_*` |
| Choosing features in the chat model at the commitment token (exploratory) | [`reroute_attribution.json`](reports/reroute_attribution.json), `scripts/reroute_attribution.py` |
| Dose response and targeted doses with costs (exploratory) | [`dose_response.md`](reports/dose_response.md), [`dose_targeted.md`](reports/dose_targeted.md) |
| **Detector v3** (sensitivity re-score; registered verdicts unchanged): fixes contracted, degree, modal and fronted cousin forms, mid-sentence "not just X, they Y", "less about X, more about Y", curly apostrophes; drops v1 hedge false positives. Confirmation v2 meaning prompts: family 86 → 84, family + cousin 234 → 244, "goes beyond" 63 → 82; prompt type 234/800 vs 2/400. Blind precision check: 40/40 new cousin hits, 4/4 new family hits (after removing two weak patterns) | [`detector_v3.md`](reports/detector_v3.md), [`src/classifier/detect_v3.py`](src/classifier/detect_v3.py) |
| Chat vs base model, detector v3 rescore of phase 2 (exploratory): 52/250 vs 11/250 answers, 4.7× (prompt-clustered CI 2.4–12×) | [`base_vs_chat_v3.md`](reports/base_vs_chat_v3.md), `scripts/base_vs_chat_v3.py` |
| SAE fidelity, excluding BOS: VE 0.78 (base) / 0.73 (it), 99% loss recovered; including BOS reproduces the Tier 0b failure | [`sae_fidelity.md`](reports/sae_fidelity.md) |
| **Pre-registered confirmation v2** (120 fresh prompts, chat template, unmodified baseline, 1,200 pairs): H1 family on meaning prompts 85 → 80, −6%, mid-p 0.63, **KILL**; textbook 71 → 36 (−49%, mid-p 9×10⁻⁵); cousin 113 → 127; "but" 52 → 5. H2 prompt type 23.4% vs 0.25%, **CONFIRMED** | [`confirmation_v2_prereg.md`](reports/confirmation_v2_prereg.md), [`confirmation_v2_run.md`](reports/confirmation_v2_run.md) |

### Why it happens (exploratory)

| Claim | Receipt |
|---|---|
| News: "isn't just" per 1,000 English-language GDELT articles containing "according to", 1.20 (2019–22) to 3.08 (2025–26), peaking in July 2025 at 3.5× and falling by more than half since; the "not only" control stays flat | [`trend.md`](reports/trend.md), `scripts/trend_gdelt.py`, `scripts/trend_analysis.py`, `data/trend/` |
| Books: "is not just" 0.32 (1920) to 5.08 (2019) per million words, Google Books Ngram | [`trend.md`](reports/trend.md), `data/trend/ngram/` |
| Training stages, OLMo 2 1B base / SFT / DPO / Instruct on the same 152 questions: SFT makes about 5× as many answers use it, in answers about 6× longer (flat per word); on meaning questions the per-word rate rises 1.72× from SFT to Instruct (95% CI 1.09–2.92) | [`why_stages.md`](reports/why_stages.md), `scripts/why_stages.py` |
| People: 4,917 one-sided LMArena battles, 55.9% raw, odds ratio 1.01 once length is accounted for | [`why_arena.md`](reports/why_arena.md), `scripts/why_arena.py` |
| AI judges in preference data (UltraFeedback, OLMo 2 preference mix): 51% at equal length | [`why_prefdata.md`](reports/why_prefdata.md), `scripts/why_prefdata.py` |
| Reward model (Skywork-Reward-V2-Qwen3-1.7B) on 600 answers with one sentence rewritten: prefers the original 76.7% of the time over the same idea without the contrast, 53.2% over the bare claim | [`why_rewardmodel.md`](reports/why_rewardmodel.md), `scripts/why_rewardmodel.py`, [`why_rewardmodel_items.jsonl`](reports/why_rewardmodel_items.jsonl) |
| Opening sentences (40 questions, 3 openers, 8 answers each) | [`priming.md`](reports/priming.md), `scripts/priming.py` |
| Clamping "conditions or exceptions" at twice its usual peak: 6 of 10 dull answers take the construction | [`monster_clamp.md`](reports/monster_clamp.md), `scripts/monster_clamp.py` |
| The six 2026 company quotes in the post, verified word for word on each company's own site (no claim about who wrote them) | `scripts/gallery_brands.py`, [`d5_brands.png`](reports/figures/d5_brands.png) |
| Summary and related work | [`why.md`](reports/why.md), [`why_literature.md`](reports/why_literature.md) |

### Earlier route: the base-model coalition (superseded)

| Claim | Receipt |
|---|---|
| Neutral prompts, out-of-sample, n=306 pairs: family 18 → 10 (**−44%**, CI 14–70%, mid-p 0.012; 8 of 9 changed prompts toward clean); strict 14 → 1 (**−93%**); family+cousin 28 → 21 (**−25%**, the cousin itself 10 → 11) | [`m1_stats_reanalysis.md`](reports/m1_stats_reanalysis.md), [`m1_rescore_union.json`](reports/m1_rescore_union.json) |
| Prompt-level, seeds collapsed: 9 prompts changed status, 8 toward clean (sign-test mid-p 0.021) | [`heldout_reslice.md`](reports/heldout_reslice.md) |
| The detector fixes that revised the 80% draft headline, twice, with every dropped or added hit open to inspection | [`permissive_fix_audit.md`](reports/permissive_fix_audit.md) |
| Primed prompts, n=300 (40% in-sample): prefix-inclusive family 267 → 120 (−55%), strict 172 → 31 (−82%); completion-only −51% (p=4.5×10⁻³), held-out half −50% | [`m1_stats_reanalysis.md`](reports/m1_stats_reanalysis.md), [`heldout_reslice.md`](reports/heldout_reslice.md) |
| Two-feature core 3223+9909, n=120 (in-sample): completion-only −53%; prefix-inclusive only −15% vs the full coalition's −55%. Two features carry the decision; twenty-five carry the paragraph | [`m1_stats_reanalysis.md`](reports/m1_stats_reanalysis.md) |
| Contrast machinery, beyond the tic: "but"-share 13.4% → 1.3% (neutral) and 75% → 1.7% (primed); commas 1.52 → 0.58/gen; words/gen 32 → 32 | [`collateral_syntax.md`](reports/collateral_syntax.md) |
| Ablation ladder at the pivot decision (n=80, base model): 1/2/5/10/25/50/100 features → −18/−37/−51/−61/−72/−75/−79%; top-25 at n=200 = −76% | [`joint_ablation.json`](reports/joint_ablation.json), [`asymptote_ladder.json`](reports/asymptote_ladder.json) (a raw daemon response; `scripts/q7full_ladder.py --layer 20` regenerates it) |
| Controls: 100 random features < 0.001 effect; coalition beats all 20 matched-activation null draws (drop +0.200 vs best null +0.001) | [`matched_activation_null.json`](reports/matched_activation_null.json) |
| Leave-one-out: two indispensable cores (3223: 0.073, 9909: 0.074), secondary 12898 (0.021), rest < 0.013 | [`q3_leave_one_out.json`](reports/q3_leave_one_out.json) |
| Layer-local: L12 −30%, L20 −72%, L25 −52%; joint L12+L20+L25 floor 0.079 ≈ L20-alone 0.077 | [`q7c_cross_layer_joint.json`](reports/q7c_cross_layer_joint.json) |
| Fluency: perplexity on held-out human prose 1.079× (coalition) / 1.000× (single feature) | [`phase5_quality_coalition_top25.md`](reports/phase5_quality_coalition_top25.md) |
| Strict classifier blind-validated at P = 0.80, R = 1.00 on 90 independently-sourced sentences | [`tier_0a_classifier_blind_eval.md`](reports/tier_0a_classifier_blind_eval.md) |
| The seven-attack gauntlet, n=90 per arm, re-scored per generation with the family detector: ask nicely 14 → 8, word ban 14 → 0, exemplars 14 → 0, scoped and global 3223 ablation 15 → 11, orthogonalization 15 → 12, CAA 14 → 2 | [`gauntlet_union_rescore.json`](reports/gauntlet_union_rescore.json) |
| Blinded LLM judge over all 1,452 generations: 88–92% agreement with the family tier; primed kill confirmed −48% (p ≈ 5×10⁻⁴); widest semantic count on neutral prompts statistically flat (the reroute, fully visible) | [`llm_judge_rescore.md`](reports/llm_judge_rescore.md) |
| Pre-registered confirmation on 50 fresh prompts: **KILL by its frozen gates**. Direction consistent (3 → 1), but the fresh, mostly concrete prompts barely elicit the construction (1.0% baseline against 5.9% exploratory), so the test had no power | [`confirmation_prereg.md`](reports/confirmation_prereg.md), [`confirmation_run.md`](reports/confirmation_run.md) |
| Which forms absorbed the reroute (exploratory, post hoc): two-sentence "isn't just X. It's Y." 15 → 41; same-sentence 61 → 31 | [`confirmation_v2_forms.md`](reports/confirmation_v2_forms.md) |
| Prompt type (exploratory, post hoc): with 152 prompts labelled blind to outcomes, the construction or its cousin appears in 9.4% of answers to meaning/value prompts against 1.3% for process/scene prompts; 7.0× pooled (95% prompt-clustered CI 2.8–30×), 3.0× stratified by prompt set | [`prompt_type_split.md`](reports/prompt_type_split.md), [`data/prompt_type_labels.tsv`](data/prompt_type_labels.tsv) |
| The canonical coalition (n=40 selection run; the n=100 rerun overlaps 19/25 with identical top-3) | [`pivot_attribution.json`](reports/pivot_attribution.json), [`pivot_attribution_n100.json`](reports/pivot_attribution_n100.json) |

Detector provenance: the main results count the habit with `detect_v3` (family or cousin), whose added patterns were blind-checked for precision (40/40 cousin, 4/4 family); the registered v3 and v4 verdicts were scored with it from the start. For the earlier route: the strict classifier is the only blind-validated one (P = 0.80, R = 1.00); the permissive layer that completes the family is FP-audited but not blind-validated; the gauntlet's original referee was a third detector (P = R = 0.857 on its own holdout); a blinded frontier-model judge agrees with the family tier on 88–92% of those generations ([`llm_judge_rescore.md`](reports/llm_judge_rescore.md)). The demo's JavaScript highlighter mirrors detector v3 (family and cousin), and `tests/test_js_detector_parity.py` checks that it flags exactly what the Python detector flags on every committed generation. The generation-based statistics regenerate from the committed JSONL files without a GPU (`--analyze` on the confirmation runners, `rescore_v3.py`, `base_vs_chat_v3.py`, `prompt_type_split.py` and the earlier-route scripts in `scripts/`); the decision-point numbers need the model. The Discovery/Confirmation protocol the project ran under is at [`reports/operating_protocol.md`](reports/operating_protocol.md).

## Where this could be wrong

- **Two models from one family.** The procedure has been confirmed under pre-registration in Gemma 2 2B-it (three tests) and Gemma 3 4B-it (one test). Both are Google models with Google SAEs; other families and larger models are untested. Earlier analyses on the base-model coalition ran about fifty tests; that coalition's pre-registered test failed, which is why the selection moved to the chat model's decision point.
- **The decision-point ladder is exploratory.** It is measured on held-out decision points, but its conditions were not pre-registered. The generation tests are the confirmatory evidence.
- **Detectors draw lines.** The habit is counted by regex tiers (`detect_v3`), blind-checked for precision on the patterns it added. A model that escapes into wording no detector covers would look like a success; the blind any-reframing judge (33% → 18%) is the check on that, and it says some of the urge survives in looser wording.
- **Cost gates are only as good as their coverage.** Both registered tests passed every cost gate, and a later check found a cost none of them measured: weaker negation, including vaguer warnings. Other side effects may exist that nothing here measures.
- **AI raters only.** Fluency, usefulness and the reframing judge are blind LLM annotators that never saw the condition. No human ratings.
- **SAE fidelity.** Gemma Scope's layer-20 SAE explains about three-quarters of the variance on this model once BOS is excluded (VE 0.78 base / 0.73 chat, 99% of loss recovered, [`sae_fidelity.md`](reports/sae_fidelity.md)). The interventions subtract only the chosen features' contributions and keep the reconstruction error, and never touch BOS, so an imperfect SAE limits how much of the model's activity the features account for; whatever it misses passes through unchanged. The pre-registered Tier 0b fidelity check failed at VE −1.33 because it included BOS ([`tier_0b_kill.md`](reports/tier_0b_kill.md)).
- **Turning up is exploratory.** Scaling the 25 up makes the habit several times more common, against a matched control, but that run was not pre-registered, and scaling existing activations is weaker evidence than writing the features into contexts where they are silent.
- **One SAE width and one layer per model.** 16k features at layer 20 (Gemma 2) and layer 22 (Gemma 3). Behavioural runs on Qwen 2.5 7B Instruct and gpt-oss-20b ([`path_b_register_writeup.md`](reports/path_b_register_writeup.md)) found the "isn't X, it's Y" form far rarer there, measured with the strict detector only.
- **Provenance gaps (earlier route).** `asymptote_ladder.json`, `q3_leave_one_out.json` and `q7c_cross_layer_joint.json` are raw daemon responses without a committed producer script. The Q5c JSON header records `max_new_tokens=100`, but `scripts/q5c_runner.py` uses 50.

## Repo map

```
src/
  classifier/   the detectors: strict (regex + spaCy dependency check) and the v2 union
  steering/ quality/ genealogy/ deslop/
                planned modules, still stubs: the working hooks, generation and
                perplexity code live in scripts/probe_daemon.py and scripts/quality_preservation.py
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

Comments in older scripts cite sections of the original project plan ("PRD §…"), which isn't included here.

## Lineage

The methods this project leans on, and where they came from: the SAEs are Gemma Scope ([Lieberum et al. 2024](https://arxiv.org/abs/2408.05147)); the directional-ablation attack that failed here is the one that *worked* on refusal ([Arditi et al., NeurIPS 2024](https://arxiv.org/abs/2406.11717)); the steering vectors are CAA ([Rimsky et al., ACL 2024](https://arxiv.org/abs/2312.06681)); the rerouting behaviour has a name, the Hydra Effect ([McGrath et al. 2023](https://arxiv.org/abs/2307.15771)); and the closest prior art on SAE features of AI-text style is [Kuznetsov et al. 2025](https://arxiv.org/abs/2503.03601). The construction itself is antithetic parallelism, which Robert Lowth named in 1753. The model inherited it and overuses it.

## Data and licences

The code is MIT. The data comes from several places:

- **Model outputs.** Gemma 2 and Gemma 3 generations are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). OLMo 2 generations (`reports/why_stages_generations.jsonl`) come from the Allen Institute for AI's [OLMo 2 1B](https://huggingface.co/allenai/OLMo-2-0425-1B) checkpoints (Apache 2.0).
- **Feature labels.** Neuronpedia's auto-interpretation labels for Gemma Scope ([neuronpedia.org](https://www.neuronpedia.org/)), cached in `data/labels_cache.json`, `data/g3_labels_cache.json` and `web/demo/layout.json`.
- **Evaluation sets.** `data/arc_easy_300.json` is 300 questions from AI2's [ARC](https://allenai.org/data/arc) (CC BY-SA 4.0). `data/wikitext2_test_excerpt.txt` is an excerpt of [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext) (CC BY-SA 3.0).
- **Preference data.** `reports/why_rewardmodel_items.jsonl` holds 600 answers drawn from [arena-human-preference-55k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k), [UltraFeedback (cleaned)](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned) and the [OLMo 2 preference mix](https://huggingface.co/datasets/allenai/olmo-2-1124-7b-preference-mix), each with one sentence rewritten. See each dataset card for its licence.
- **Trend data.** `data/trend/gdelt/` caches responses from the [GDELT Project](https://www.gdeltproject.org/)'s DOC 2.0 API; `data/trend/ngram/` comes from the [Google Books Ngram Viewer](https://books.google.com/ngrams).
- **Everything else** (prompts, labels, blind ratings, reports) was written for this project.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q --ignore=tests/test_neo4j_smoke.py
# 32 passed
```

`test_neo4j_smoke.py` needs the live graph on port 7693; everything else runs cold (`test_js_detector_parity.py` skips without Node, `test_mlx_rope_batch.py` without MLX).

## License

MIT. See [LICENSE](LICENSE).

