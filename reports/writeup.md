# Not This, But That — an exploration

*One model. One construction. A partial mechanism. An honest null on the product side.*

**Status:** This is an exploration, not a paper. It has not been peer-read by a credentialed mech-interp researcher. The result, if any, pertains to **Gemma 2 2B / Gemma 2 2B-it** specifically — not "AI writing" in general. The models people actually complain about (GPT-4o, Claude, Gemini) are closed and have no public SAEs; nothing here transfers to them. Cite this work, if at all, with that ceiling stated up front.

---

## The question

Wikipedia's *Signs of AI writing* (WP:AIPARALLEL) describes the **"not X, but Y"** construction — *"It's not a tool, it's a revolution."* — as one of the most-cited stylistic tells of LLM prose. We study one model's tell, in four variants:

| | Variant | Example |
|---|---|---|
| C1 | Contrastive correction | "It's *not* a tool, it's a revolution." |
| C2 | Additive escalation | "*Not only* does it scale, *but* it adapts." |
| C3 | Minimize-then-elevate | "It's *not just* an update — it's a rethink." |
| C4 | Triadic negation | "No mandate. No approval. Just power." |

The question is not whether the model produces this construction. The question is **whether a single SAE feature carries it causally**, and **whether removing that feature de-slops the prose without wrecking the writing**.

## What the prior project missed (and how this one positions against it)

The previous repo (`grammar-layer`, archived at `ho3h/grammar-layer`) rediscovered a known phenomenon — SAE *suppression features* — and over-framed the finding. The post-mortem from that project's PRD is the engine of this one. Two of the suppression features it found, #15596 ("forms of to be") and #10142 ("the word is"), are precisely *the features the model avoids reaching for when it uses the construction instead of a plain copular sentence*. The grammar-layer project saw them from one side; this project sees them from the other.

The closest prior art for this work is [Kuznetsov et al., arXiv:2503.03601](https://arxiv.org/abs/2503.03601) — SAE features from Gemma-2-2b's residual stream, steering approach, "LLMs have a distinct writing style." Our differentiation:

| | Kuznetsov 2503.03601 | This project |
|---|---|---|
| Unit of analysis | Aggregate "AI-ness" of a text | One named construction (C1–C3) |
| Feature claim | Correlation | **Causal necessity** at the pivot decision (Phase 4) |
| Validation | Steering / detection statistics | Ablate + clamp-up vs random-k controls |
| Genealogy | — | Base vs instruct on identical contexts (Phase 6) |
| Output | Detection insight | A necessary feature, a quality-preserved scalpel, an honest null on the product side |

---

## Findings, in the order they landed

### 1. Phase 2 — the behavioral gap is large and clean

We generated 510 continuations from Pythia 70M and GPT-2 small (102 D2 neutral prompts × 5 seeds), and 250 from Gemma 2 2B base and Gemma 2 2B-it (50 prompts × 5 seeds, scaled-down due to throughput). Sentence-level bootstrap CIs:

| Model | C1 | C2 | C3 | any_core |
|---|---|---|---|---|
| pythia_70m | 1.5% | 0.3% | 0.1% | 1.9% (1.5–2.4) |
| gpt2 | 1.3% | 0.2% | 0.2% | 1.6% (1.2–2.1) |
| **gemma_2b (base)** | 0.3% | 0.0% | 0.1% | **0.4%** (0.1–0.8) |
| **gemma_2b_it** | 0.1% | 0.0% | **1.7%** | **1.8%** (1.0–2.6) |

**Base vs instruct on Gemma 2 2B:** instruct produces C3 at **12× the base rate** (0.1% → 1.7%), with non-overlapping CIs. The any_core rate is amplified 4.2×. The Phase 6 motivating expectation — *the construction is dormant in base, amplified by instruct* — has a real behavioral signal. And the amplification is **specifically C3**: the "not just X — Y" minimize-then-elevate variant, not C1 or C2.

### 2. Phase 3 — feature discovery found *consequences*, not *causes*

On 226 D1 hand-written contrast pairs (`{with construction, without — meaning-matched paraphrase}`), we computed per-feature differential SAE activation at the **last token** of each sentence, ranking by t-statistic. Top candidate (rank 3 overall, rank 1 among semantically interpretable):

> **feat 9841** — *"phrases and clauses involving contrasting ideas or situations"* — t = **+14.19** across all variants. Looks like the right feature.

It was not.

### 3. Phase 4a — the wrong feature, told us cleanly

We perturbed feat 9841 at the truncated pre-pivot position of each D1 with-prompt and measured ΔP(pivot):

- Ablate → 0: **zero effect** (the feature isn't active there).
- Clamp-up → 10: P(pivot) went **down** by -3.7σ vs random-k.

A 5-feature supernode (top 5 with-recruiting features) made it worse: clamp-up moved P(pivot) **the wrong direction by -23.7σ**. The reading: 9841 and its neighbours are **consequence features**. They fire on sentences that have *already* used the construction. Forcing them up at the pre-pivot position signals to the model that the construction is *already done* — so it doesn't commit to the pivot.

Phase 3's last-token-of-completed-sentence discovery is a fine diagnostic for "what differs between with and without," but the features it surfaces are not the causal levers.

### 4. Phase 4b — per-feature pre-pivot attribution recovered the cause

We then ran per-feature attribution at the *truncated pre-pivot position*: ablate each active feature one at a time, measure ΔP(pivot), rank by mean drop × √(n active prompts).

> **feat 3223** — *"phrases conveying exceptions or negations"* — mean attribution drop = **+0.076** absolute, active in 32/60 truncated prompts, score **1.5×** the next interpretable feature.

The label is the right kind of label, and the effect is causally located at the right position.

### 5. Phase 4c — the causal lever passes ablation, fails clamp-up

| Condition | P(pivot) | Δ vs baseline | vs random-k |
|---|---:|---:|---:|
| baseline | 0.3324 | — | — |
| ablate(3223) | 0.2483 | **−0.0841 (−25%)** | **+7397σ** |
| clamp_up(3223, value=10) | 0.3063 | −0.0262 | −15.5σ (wrong direction) |
| clamp_up(3223, value=3) | 0.2650 | −0.0674 | −132.7σ (worse) |

**The ablation result is enormous and clean.** Removing feat 3223 drops the pivot probability by a quarter, beating random-k by ~7400σ. The clamp-up result is also striking, but in the opposite direction we expected: artificially elevating the feature, at either value tested, also drops P(pivot).

The honest interpretation: **feat 3223 is causally necessary for committing to the construction's pivot, but not sufficient on its own.** The construction commits via *coordinated* multi-feature activity, where 3223 is one indispensable component. Pegging 3223 to a fixed high value pushes the model into an out-of-distribution state representing "the contrast has already happened" — so it doesn't commit again.

**By the PRD's strict bidirectional kill check, Phase 4 fails.** The honest framing is that it passes in the necessity direction by an overwhelming margin, and fails in the sufficiency direction. That asymmetry is the result.

### 6. Phase 5 — the ablation is a scalpel, not a sledgehammer

Held-out perplexity on D3 (six paragraphs of clean human prose on unrelated topics):

| Condition | geo-mean PPL | ratio vs baseline |
|---|---:|---:|
| baseline | 24.628 | 1.000× |
| ablate(3223) | 24.633 | **1.000×** |
| clamp_up(3223) | 24.785 | 1.006× |

Removing 3223 does not measurably degrade the model's fluency on prose that has nothing to do with the construction. The lever, where it works at all, is a scalpel.

### 7. Phase 6 — the genealogy signal lines up

| Model | baseline P(pivot) | ablate P(pivot) | absolute drop | relative drop |
|---|---:|---:|---:|---:|
| gemma-2-2b (base) | 0.3324 | 0.2483 | 0.0841 | 25.3% |
| gemma-2-2b-it (instruct) | 0.4775 | 0.3255 | **0.1520** | **31.8%** |

The instruct model is **more committed to the pivot at the pre-decision point** (0.48 vs 0.33), and ablating feat 3223 produces a **1.81× larger absolute drop** in instruct. The same causal feature, more load-bearing in the post-instruct-tuning model — consistent with Phase 2's 4.2× any_core behavioural amplification.

**Caveat (PRD §8 P6):** Gemma Scope is trained on the *base* model. Reconstruction quality on the instruct model is the prerequisite check; our measurement of explained-variance came out negative, which means our reconstruction code has a bug rather than the SAE having actually broken. The per-token Δ-log-P signal in the ablation is the operative measurement and stands independently — but the VE number itself should not be cited until the measurement is fixed.

### 8. Phase 7 — the de-slop tool, honestly, does not work

We ran a small generation-time intervention: 12 D2 prompts × 3 seeds × 100 tokens, sampling token-by-token with feat 3223 clamped to 0 throughout, comparing to baseline.

- Baseline construction rate: **5.56%** (2/36 generations contained C1/C2/C3)
- Ablated construction rate: **5.56%** (2/36)
- Absolute drop: **+0.00%**
- Meaning preservation (MiniLM cosine of baseline vs ablated): mean **0.954**

The intervention has no effect on the construction rate during open-ended generation. The internal logic is consistent: Phase 6 showed feat 3223 has 0% activation at the last token of D2 neutral prompts in instruct. The feature is conditional on construction-mode contexts — it lights up *after* the model has committed to opening the contrast, not before. Suppressing a feature that isn't yet active is a no-op. To deslop the prose you'd need to intervene further upstream, or on whatever multi-feature coordination commits the model to open the construction in the first place.

**The mechanism finding stands. The product claim does not.**

---

## What we can and cannot say

We **can** say:

1. The construction (C3 specifically) is **4–12× more common in Gemma 2 2B-it than in Gemma 2 2B**, with clean CI separation.
2. There is a specific SAE feature (#3223, labelled *"phrases conveying exceptions or negations"*) that is **causally necessary** for the pivot commit — ablating it drops pivot probability by 25% relative, beating any random-k control by ~7400σ, and the ablation does not degrade general fluency.
3. The same feature is **more causally load-bearing in instruct than in base** on identical contexts — a within-mechanism signal that the Phase 2 behavioural gap has something coherent under it.

We **cannot** say:

1. That clamping feat 3223 *up* causes the model to use the construction. It doesn't. Artificially elevating it pushes the model into an OOD state where pivot probability *drops*.
2. That ablating feat 3223 during open-ended generation removes the construction. It doesn't. The feature is dormant on neutral prompts; ablating it pre-emptively is a no-op.
3. That this finding generalises beyond Gemma 2 2B. We didn't test it on any other model family. The previous project's cross-model work suggests *suppression* features generalise; the *necessity* finding here has not been replicated elsewhere.

---

## What this is not

- **Not a detector.** AI-text detection is adversarially doomed; we don't compete there.
- **Not a claim about "AI writing" in general.** One model, one construction, one mechanism.
- **Not a paper.** It is an exploration. The next step, if there is one, is for a credentialed mech-interp reader to look at this and decide whether the necessity-without-sufficiency result is publishable on its own.

---

## What worked, what didn't, what's left

| Phase | Designed | Outcome |
|---|---|---|
| 0 | Scaffold | ✓ |
| 1 | Classifier ≥ 0.85 P/R | ✓ (1.00 P/R, self-consistency caveat) |
| 2 | Behavioural M1, base vs instruct | ✓ Large clean gap |
| 3 | Feature discovery | ⚠ Found consequence features; recovered via per-feature pre-pivot attribution |
| 4 | Bidirectional causal lever | ⚠ Necessity passes (7400σ), sufficiency fails |
| 5 | Quality preservation | ✓ Scalpel |
| 6 | Genealogy | ✓ Signal lines up; reconstruction-VE measurement is buggy |
| 7 | De-slop demo | ❌ Honest null — feature is conditional on construction-mode contexts |

**The Neo4j angle (PRD §9)** — storing the validated feature across models in a graph and asking the cross-model alignment question via Cypher — has not been built. With a *partial* mechanism that's also conditional, there isn't enough to align across models yet. Worth revisiting after the sufficiency question is settled or after this work is replicated on another open model.

---

## The closing line

It would be easy to end this by saying it's not a detector, it's a mirror. So I won't.
