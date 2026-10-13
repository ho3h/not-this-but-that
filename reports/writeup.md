# Not This, But That — an exploration

> **⚠️ This is the historical Phase 0-7 audit trail (2026-05-21). For the
> current finding — a 25-feature coalition that kills the construction at
> 80% relative on n=306 — read [`medium_post_draft.md`](medium_post_draft.md).**
> The "pre-emptive removal doesn't work" line below is about *single-feature*
> attacks, which the coalition finding superseded.

*One model, one construction. The commit-gate is necessary, conditional, and instruct-amplified. Pre-emptive removal doesn't work — and that's the explanation, not the apology.*

**Status:** This is an exploration, not a paper. It has not been peer-read by a credentialed mech-interp researcher. The result, if any, pertains to **Gemma 2 2B / Gemma 2 2B-it** specifically — not "AI writing" in general. The models people actually complain about (GPT-4o, Claude, Gemini) are closed and have no public SAEs; nothing here transfers to them. Cite this work, if at all, with that ceiling stated up front.

**Two foundation cracks. One closed, one not.**

- **M1 classifier blind-validation:** **CLOSED.** [Tier 0a](tier_0a_classifier_blind_eval.md) — the classifier was re-tested on 90 sentences (30 human D3 + 30 AI Gemma-2-2b-it + 30 enriched with "not"-containing sentences from four models) that the regex was never tuned on. Hand labels were committed before the classifier ran. Result: precision 0.80, recall 1.00 on any_core (above the pre-registered ≥0.70 gate). The single false positive was a litotes case (`It's not without challenges, but the rewards…`) that's borderline in either direction. M1 generalises beyond the original self-consistency set.

- **SAE reconstruction-quality verification:** **OPEN.** [Tier 0b](tier_0b_kill.md) — the pre-registered prerequisite check (variance-explained ≥ 0.60 on base) cannot be satisfied with this codebase. Every measurement path (raw `sae(x)`, `fold_activation_norm_scaling_factor`, runtime-scaling, the canonical `sae_lens.evals` arithmetic) returns a negative VE. **Proxy evidence the SAE itself is functional:** L0 at runtime-scale s=0.5 lands at **74.1** against the canonical advertised **71** (within 4 %), and reconstruction cosine similarity is **0.83**. The model's top-5 next-token predictions under SAE installation are essentially identical to the pure forward pass. The SAE is operating; the *variance-explained number* is the broken instrument. Per the pre-registered protocol the kill stands until VE can be reproduced against the field's published number for this SAE.

The downstream consequence: **the Phase 6 genealogy claim and the feature-*labelling* part of Phase 4 are explicitly conditional on a verification this codebase cannot perform.** The Phase 4 *intervention* is real — perturbing the SAE encoder at the candidate index produces a clean 25 % drop in P(pivot), and that drop is the model's response to a real perturbation. But the chain *"the thing you ablated is the SAE feature Neuronpedia labels 3223 specifically"* requires the SAE's encoder/decoder to be the meaningful object the published VE certifies it to be. We have proxy evidence for that (L0, cosine, forward-pass invariance), not the prerequisite measurement.

---

## Thesis

Inside Gemma 2 2B's residual stream at layer 20, there is a single sparse-autoencoder feature — Neuronpedia-labelled *"phrases conveying exceptions or negations"*, index **#3223** — that is **causally necessary** for the model's commit to the pivot of the "not X, but Y" construction. Ablating it at the pre-pivot decision point cuts the pivot's next-token probability by **25% relative**; no random single-feature ablation at the same position moves the probability at all. The ablation is fluency-preserving on held-out human prose. The same feature is **measurably more load-bearing in the instruct-tuned model than in the base model on identical contexts**.

The feature is **not sufficient on its own**. Forcing it to a high activation does not reproduce the construction; it pushes the model into an out-of-distribution state where pivot probability *drops*. And ablating it throughout open-ended generation does not de-slop the prose: the feature is dormant on neutral prompts, so suppressing it pre-emptively is a no-op. These two nulls are not failures of the mechanism finding — they are evidence of its shape. The commit-gate is conditional on prior context, and the *decision to enter* construction-mode lives upstream of feature 3223.

That's the story. The rest of this document is the audit trail.

---

## The question, and what the prior project missed

Wikipedia's *Signs of AI writing* (WP:AIPARALLEL) lists the **"not X, but Y"** construction — *"It's not a tool, it's a revolution"* — among the most-cited stylistic tells of LLM prose. We study one model's tell across four variants:

| | Variant | Example |
|---|---|---|
| C1 | Contrastive correction | "It's *not* a tool, it's a revolution." |
| C2 | Additive escalation | "*Not only* does it scale, *but* it adapts." |
| C3 | Minimize-then-elevate | "It's *not just* an update — it's a rethink." |
| C4 | Triadic negation | "No mandate. No approval. Just power." |

The previous repo (`grammar-layer`, archived) rediscovered a known phenomenon — SAE *suppression features* — and over-framed it. Two of the suppression features it surfaced, #15596 ("forms of to be") and #10142 ("the word is"), are precisely the features the model avoids reaching for when it uses the construction *instead of* a plain copular sentence. The grammar-layer project saw the avoidance side; this project sees the recruitment side.

Closest prior art: [Kuznetsov et al., arXiv:2503.03601](https://arxiv.org/abs/2503.03601) — SAE features from Gemma-2-2b, a steering approach, "LLMs have a distinct writing style." Our differentiation, row by row:

| | Kuznetsov 2503.03601 | This project |
|---|---|---|
| Unit of analysis | Aggregate "AI-ness" of a text | One named construction (C1–C3) |
| Feature claim | Correlation | **Causal necessity** at the pivot decision |
| Validation | Steering / detection statistics | Ablate + clamp-up vs random-k controls |
| Genealogy | — | Base vs instruct on identical contexts |
| Output | Detection insight | A necessary feature, a quality-preserved scalpel, and an honest null on the product side |

---

## Behavioural baseline (Phase 2): the gap is in the *variant*, not the rate

We generated open-ended continuations on 102 / 50 neutral prompts × 5 seeds from Pythia 70M, GPT-2 small, Gemma 2 2B, and Gemma 2 2B-it, then scored every sentence for C1–C4 presence with the strict classifier (regex hinges + spaCy dependency filter). Sentence-resampled bootstrap 95% CIs:

| Model | C1 | C2 | C3 | any_core | C3 share of any_core |
|---|---|---|---|---|---|
| pythia_70m | 1.5% | 0.3% | 0.1% | **1.9%** (1.5–2.4) | 5% |
| gpt2 | 1.3% | 0.2% | 0.2% | **1.6%** (1.2–2.1) | 13% |
| **gemma_2b (base)** | 0.3% | 0.0% | 0.1% | **0.4%** (0.1–0.8) | 25% |
| **gemma_2b_it** | 0.1% | 0.0% | 1.7% | **1.8%** (1.0–2.6) | **94%** |

The naive headline — *"instruct uses the construction 4.2× more than base"* — is true but understates the result and invites an obvious objection: if instruct-tuning installs the construction, why do the base Pythia and GPT-2 models already produce it at 1.9% and 1.6%, *higher than* Gemma instruct's 1.8%?

The honest reading is two facts:

1. **Gemma's base is the outlier-low.** At 0.4% any_core, it sits a factor of four below the other base models. Instruct-tuning brings it to 1.8%, roughly where Pythia and GPT-2 already are.
2. **The variant composition is qualitatively different.** Instruct Gemma is **94% C3** — almost everything it produces is the "not just X — Y" minimize-then-elevate variant. Every other model is C1-dominant; C3 is 5–25% of the construction usage in those models. That shift, not the rate change, is the signal.

Instruct-tuning didn't manufacture the construction. It moved Gemma into a specific stylistic register that the other base models also occupy, and concentrated the variant.

The Phase 4 causal feature we identify next is the gate for *committing to the pivot once the construction has been opened*. Phase 6 will show it is more load-bearing in the model that uses the C3 variant 94% of the time. That's the mechanistic correlate of the variant-composition shift.

---

## Mechanism (Phases 3–4): finding the wrong feature, then the right one

**Phase 3 — last-token-of-completed-sentence differential.** On 226 hand-written D1 contrast pairs (`{with construction, without — meaning-matched paraphrase}`), we ranked SAE features at L20 by t-statistic on `act_with − act_without` at the last token of each sentence. The top semantically interpretable hit:

> feat **9841** — *"phrases and clauses involving contrasting ideas or situations"* — t = **+14.19** across all variants.

The label looks right. It is not the cause.

**Phase 4a — intervention at the truncated pre-pivot position.** We truncated each D1 with-sentence at the comma/em-dash that commits the construction and measured P(pivot token) under intervention vs baseline. Feat 9841 produced **zero change** under ablation (it is not active at this position), and clamping it to a high value pushed P(pivot) *down*. A 5-feature supernode of the top Phase-3 hits behaved the same way — clamp-up moved P(pivot) the wrong direction by orders of magnitude beyond any random baseline. These are **consequence features**: they fire on sentences that have *already* used the construction. Phase 3's last-token-of-completed-sentence convention surfaces them, but they are not the causal lever.

**Phase 4b — per-feature attribution at the right position.** We then ablated *every active feature, one at a time*, at the truncated pre-pivot position across 60 D1 with-prompts, and ranked by mean attribution to P(pivot). The top result:

> feat **3223** — *"phrases conveying exceptions or negations"* — mean attribution drop = **+0.076 absolute**, active in 32/60 truncated prompts, score 1.5× the next interpretable feature.

The label is right and the effect is at the right position.

**Phase 4c — bidirectional intervention on feat 3223.**

| Condition | mean P(pivot) | Δ from baseline | relative |
|---|---:|---:|---:|
| baseline | 0.3324 | — | — |
| ablate(3223) | 0.2483 | **−0.0841** | **−25%** |
| clamp_up(3223, value=10) | 0.3063 | −0.0262 | wrong direction |
| clamp_up(3223, value=3) | 0.2650 | −0.0674 | wrong direction (worse) |

**Random-k controls (n=5, single-feature ablation at the same position):**

| Statistic | random-k null | candidate |
|---|---|---|
| mean drop | −0.0000005 | +0.0841 |
| std of drops | 0.00001 | — |
| max drop in 5 draws | 0.0000 | +0.0841 |
| draws below candidate | 5/5 | — |

A note on the statistics. The script originally reported these as σ multiples. **It shouldn't have, and the writeup no longer does.** The random-k null distribution at this position is *near-degenerate* — ablating an arbitrary single feature at the truncated pre-pivot position changes P(pivot) by essentially zero, with essentially zero variance. Dividing the candidate's drop by that variance gives a sigma count in the thousands, which is meaningless: it isn't a tail event in a smooth distribution, it's a qualitative separation from a flat one. The truthful report is the table above — five out of five random single-feature ablations produced no change; the candidate produced a 25% relative drop. Reporting that as a sigma multiple would be the same genre of impressive-sounding-but-meaningless number that sank the last project.

The bidirectional reading is **necessity yes, sufficiency no**. Ablating feat 3223 removes a quarter of the model's pivot probability at the decision point; pegging it high — at either of two tested values — does not reproduce the construction, it pushes the model somewhere else. The construction's commit is a *coordinated* multi-feature event, of which 3223 is one indispensable component. By the PRD's strict bidirectional kill check, Phase 4 fails. By the honest necessity-only reading, it produces a real and clean result that the next two phases support.

**A caveat the foundation cracks add here:** the *intervention* — ablating the SAE's encoder output at index 3223 — produces a real change in the model's forward output (the 25 % drop is a direct measurement on the model's logits, not a function of any SAE reconstruction quality). What's conditional on the unverified VE prerequisite is the *interpretation* that "what we ablated is *the* feature corresponding to Neuronpedia's 'phrases conveying exceptions or negations' label." Proxy evidence the encoder is doing what we think it's doing: at the runtime scaling that matches the canonical L0 (s = 0.5, L0 = 74 vs 71), the candidate's feature activation pattern on D1 with-prompts is what we'd expect for a contrast-related feature. But proxy is not prerequisite.

---

## Quality (Phase 5): the ablation is a scalpel

| Condition | held-out PPL (geo-mean over 6 D3 chunks) | ratio vs baseline |
|---|---:|---:|
| baseline | 24.628 | 1.000× |
| ablate(3223) | 24.633 | **1.000×** |
| clamp_up(3223) | 24.785 | 1.006× |

Ablating feat 3223 does not measurably degrade the model's fluency on held-out clean human prose covering Edinburgh, cast-iron cookware, bookbinding, trail running, libraries, and chamber music. The lever — where it works at all — is a scalpel.

(Coherence — an LLM-judge rating on intervened generations — is the third M3 leg per PRD §5. It's stubbed; folding in Anthropic Claude as judge on a small sample is the next quality check.)

---

## Genealogy (Phase 6, EXPLICITLY CONDITIONAL): the same intervention is more load-bearing in instruct

> **This section is conditional on the open foundation crack.** The PRD's pre-registered prerequisite for citing genealogy numbers — Gemma Scope reconstruction-quality verification on the instruct model — could not be performed in this codebase. The numbers below are reported, but every claim in this section needs to be read with that flag attached.

We ran the Phase 4 ablation on identical truncated D1 with-prompts in both base and instruct, applying the *same* SAE.

| Model | baseline P(pivot) | ablate P(pivot) | absolute drop | relative drop |
|---|---:|---:|---:|---:|
| gemma-2-2b (base) | 0.3324 | 0.2483 | 0.0841 | 25.3 % |
| gemma-2-2b-it (instruct) | **0.4775** | 0.3255 | **0.1520** | **31.8 %** |

Two facts move together. The instruct model is **more committed to the pivot at the pre-decision point** (P(pivot) = 0.48 vs 0.33), and ablating at SAE feature index 3223 produces a **1.81× larger absolute drop** in instruct. *Within the conditional caveat*, the same causal-intervention site is **more load-bearing in the instruct model on identical contexts** — the within-mechanism correlate of the Phase 2 variant-composition shift toward C3.

**Why this is currently unverifiable in this codebase.** Gemma Scope is trained on the *base* model. Applying its encoder to the instruct model is an empirical assumption that PRD §8 P6 requires us to verify by measuring reconstruction quality on instruct, and the resulting variance-explained number on base + instruct is what would license the cross-model comparison. Our VE measurement — through every path tried, including the canonical `sae_lens.evals` arithmetic — returns negative values that are inconsistent with the SAE being functional at all. Proxy evidence the SAE *is* functional (L0 = 74 vs canonical 71; cosine sim = 0.83; the model's top-5 next-token predictions under SAE installation are essentially identical to the pure forward) does not substitute for the published-VE check the protocol demanded. See [`tier_0b_kill.md`](tier_0b_kill.md).

The per-token Δ-log-P numbers above are a direct measurement on the instruct model — they describe what happens to instruct-Gemma's pivot probability when we perturb its residual stream at a specific encoded-feature index. That fact stands. What *cannot* yet be said without the VE prerequisite is that **the encoded feature being perturbed is the same semantic object across the two models.** If the SAE doesn't transfer well to instruct, the "instruct feature 3223" might be a different direction than the "base feature 3223," and the 1.81× ratio would be measuring a partly-incommensurable comparison. The current evidence is consistent with the genealogy claim but does not yet license it.

---

## Why pre-emptive de-slop fails (Phase 7), and why that is the result

We ran a small generation-time intervention on Gemma 2 2B-it: 12 D2 prompts × 3 seeds × 100 tokens of token-by-token sampling, comparing baseline against a run with feat 3223 clamped to 0 throughout.

| Condition | construction rate (any_core) | meaning preservation (MiniLM cosine) |
|---|---:|---:|
| baseline | 5.6% (2/36 generations) | — |
| ablated | 5.6% (2/36 generations) | 0.954 |

**Zero absolute drop in construction rate.** Meaning is preserved (cosine 0.95) because the generations are very nearly identical — same prompts, same seeds, same sampling, and ablating a feature that wasn't going to fire is a no-op.

This is internally consistent with everything above. Phase 6 measured feat 3223's activation at the last token of D2 neutral prompts: **0% in instruct**, 2% in base. The feature isn't a *propensity* signal — it doesn't fire on prompts that *might* lead to the construction. It fires on contexts that have *already opened* the construction's contrast, and gates the commit to the pivot. Suppressing it pre-emptively on a neutral prompt is operating on something the model wasn't going to use anyway.

What this tells us about the shape of the mechanism. The construction has at least two decision points:

1. **Entry**: an upstream decision to *open* the contrast, emitting the "not" token in construction context. We have not identified what gates this.
2. **Commit**: the decision to complete the pivot once the contrast has been opened. This is what feat 3223 gates, and where ablating it cuts pivot probability by a quarter.

A de-slop tool that removes the construction from open-ended generation would need to operate on (1), not (2). Phase 7's null is the experimental confirmation that 3223 lives at the second gate, not the first. That isn't a failed product claim buried at the end; it's the cleanest available evidence for where in the residual stream the construction's commit-gate actually lives.

---

## What we can and cannot say

We **can** say:

1. **The variant composition shift is real and large, and the M1 classifier that measures it generalises beyond its own training data.** Gemma 2 2B-it is **94 % C3** in its construction usage on D2 neutral prompts; the other models (base Pythia 70M, base GPT-2 small, base Gemma 2 2B) are C1-dominant. Sentence-resampled 95 % CIs on any_core do not overlap between Gemma base (0.4 %) and Gemma instruct (1.8 %). Classifier validation on 90 blind, independently-sourced sentences (Tier 0a) returned P = 0.80, R = 1.00 on any_core — above the pre-registered ≥ 0.70 gate.

2. **Perturbing a specific SAE-encoded feature index at the pre-pivot position produces a clean, control-beating drop in P(pivot).** At index 3223 (Gemma Scope L20, width-16k, Neuronpedia-labelled "phrases conveying exceptions or negations"), ablation drops P(pivot) by **25 % relative** on the truncated D1 with-prompts. All five random-k single-feature controls at the same position produced *no measurable change* — the candidate is qualitatively separated from a degenerate null. This claim is **partially conditional**: the *intervention* is a direct measurement on the model's forward output; the *identification* of the perturbed direction as the SAE feature Neuronpedia labels depends on the SAE's encoder/decoder being functional in the way the published VE certifies, which we could not independently verify (see foundation crack).

3. **The ablation is fluency-preserving on held-out human prose** (perplexity ratio = 1.000 × baseline). The lever is a scalpel by the perplexity metric; coherence-by-LLM-judge is still stubbed.

We **cannot yet** say (independent of the foundation crack):

1. **That clamping the candidate feature *up* causes the model to use the construction.** It doesn't. The construction's commit is a multi-feature coordination, and pegging one feature high pushes the model into an OOD state where pivot probability drops.

2. **That ablating the candidate feature during open-ended generation removes the construction.** It doesn't, because the feature is dormant on neutral prompts. The de-slop product claim does not hold.

3. **That this finding generalises beyond Gemma 2 2B.** We did not replicate it on Pythia 70M, GPT-2 small, or any other open SAE-equipped model.

We **cannot say at all without fixing the foundation crack**:

4. **That the same SAE feature is more load-bearing in instruct than in base.** The 1.81 × ratio in Phase 6 is a real measurement on the instruct model's forward output, but its *interpretation as a cross-model genealogy signal* requires the SAE to transfer cleanly from base to instruct, which is exactly what the reconstruction-quality check would have shown. Until that check works on this codebase the genealogy claim is unverified.

5. **That what we ablated is "feature 3223" as Neuronpedia describes it.** The proxy evidence — L0 = 74 against canonical 71, cosine sim = 0.83 — is consistent with the SAE encoder being functional, but the pre-registered prerequisite was a variance-explained number we couldn't reproduce. Until that reproduces, the *feature-level* interpretation is conditional on the *direction-level* intervention being correctly indexed.

---

## What this is not

- **Not a detector.** AI-text detection is adversarially doomed; we don't compete there.
- **Not a claim about "AI writing" in general.** One model. One construction. One mechanism. Gemma 2 2B isn't the model anyone complains about; that's GPT-4o / Claude / Gemini, all closed, no public SAEs.
- **Not a paper.** It is an exploration. Before this circulates near ML researchers, the open foundation cracks need fixing and one credentialed mech-interp reader needs to sanity-check the necessity claim. That gate from last time still stands.

---

## What's still to fix

In priority order:

1. **CLOSED — Validate the classifier against blind, independently-sourced text.** Done. P = 0.80, R = 1.00 on n = 90 sentences the regex was never tuned on. See [`tier_0a_classifier_blind_eval.md`](tier_0a_classifier_blind_eval.md).

2. **OPEN — Fix the Phase 6 reconstruction-quality measurement.** Every measurement path through this codebase returns negative VE for the canonical Gemma Scope L20 width-16k SAE — including the canonical `sae_lens.evals` arithmetic. Negative VE is definitionally an instrument bug; this SAE is widely used and reconstructs at published-VE levels elsewhere. Proxy evidence the SAE is functional (L0 = 74 vs canonical 71; cosine sim = 0.83; forward-pass invariance under installation) is in [`tier_0b_kill.md`](tier_0b_kill.md). What this fix requires is more than a bounded debug session — probably the right move is to file an issue with sae_lens 6.43.0 covering the Gemma Scope canonical loader path, or to ask a credentialed mech-interp person who has worked with Gemma Scope what the expected VE-reproduction recipe is. Until this lands, the genealogy claim and the feature-labelling part of the necessity claim stay conditional.

3. **OPEN — Get one credentialed mech-interp reader on the causal claim.** The necessity-without-sufficiency result is the kind of thing that can be either real and reportable or an artifact of intervention design. An outside read is the cheapest way to tell which. Specifically: (a) does the ablation-vs-clamp-up asymmetry reflect a genuine multi-feature coordination, or could it be an artifact of the clamp values (10.0 and 3.0) being OOD? (b) does the VE-measurement issue I hit have a known recipe in the Gemma Scope community? An experienced reader will know whether either distinction matters here.

After those, **and only after**: the optional next question is what gates *entry* into construction-mode — the upstream "decide to emit 'not' in contrast context" decision that Phase 7 told us lives somewhere besides feat 3223. That's a fresh experiment, not a redo, and worth chasing only because it's interesting in its own right.

**Adversarial pre-registration.** The seven-tier adversarial methodology Theo specified (with the prime directive at §0) is committed at [`pre_registration.yaml`](../pre_registration.yaml). Tier 0a passed, Tier 0b killed. Per protocol the next tiers (1–6: specificity attack, matched-activation null, interchange-patching sufficiency retrial, generalisation to Gemma 9B, genealogy causal with fixed VE, red-team-the-method) are *blocked* on fixing the foundation crack at item 2 above. They are not "next steps" in this writeup's sense — they are the apparatus that will either confirm or kill the partial finding here, and they should run only after the prerequisite the protocol named has actually been met.

---

## What worked, what didn't

| Phase | Designed | Outcome |
|---|---|---|
| 0 | Scaffold | ✓ |
| 1 | Classifier ≥ 0.85 P/R | ✓ on self-consistent set; **untested on independent text** |
| 2 | Behavioural M1, base vs instruct | ✓ variant composition shift is the signal, not the rate |
| 3 | Feature discovery | ⚠ found consequence features; recovered via per-feature pre-pivot attribution |
| 4 | Bidirectional causal lever | ⚠ necessity overwhelming, sufficiency fails — multi-feature coordination |
| 5 | Quality preservation | ✓ scalpel (perplexity); coherence still stubbed |
| 6 | Genealogy | ✓ 1.81× larger ablation effect in instruct; **VE recon measurement buggy** |
| 7 | De-slop demo | ✓ as **evidence**: feat 3223 gates commit, not entry; pre-emptive removal is a no-op |

The Neo4j cross-model alignment angle (PRD §9) has not been built. It needs more than one validated mechanism to align across; revisit when (1)–(3) above land.

---

It would be easy to end this by saying it's not a detector, it's a mirror. So I won't.
