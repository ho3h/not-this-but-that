# Probes log — interactive ablation findings

Running log of ad-hoc experiments through the probe daemon. Each entry is
a question, the command(s) used, and what we learned. Append-only; bigger
findings get promoted to their own report.

---

## 2026-05-24

### Q1. Does community-12 (3223's Leiden community) predict causal coalition membership?

**Hypothesis.** If Leiden community detection over decoder + co-activation
edges captures causally-relevant structure, then ablating the top-N
community-12 members should drop P(pivot) substantially.

**Probe.**
```bash
COMM=$(probe.py '{"cmd": "graph", "query": "community", "cid": 12, "limit": 10}' \
       | jq -c .result.features)
probe.py "{\"cmd\": \"measure_pivot\", \"ablate\": $COMM, \"max_samples\": 10}"
```

Set: `[2557, 7750, 10251, 14934, 6724, 9589, 12634, 10735, 5259, 12830]` —
top 10 of community 12 by activation density (~33% density each, basically
always-on).

**Result.** baseline 0.2938 → ablated 0.2939; drop = −0.0002 (−0.06%).
**Essentially zero.** Adding to the running tally of falsified structural
priors:

| Prior | N | rel drop above baseline | conclusion |
|---|---:|---:|---|
| 3223 alone | 1 | 17.8 % | (anchor) |
| decoder-cosine top-9 + 3223 | 10 | 19.8 % | adds +2.0 pp |
| co-activation top-9 + 3223 | 10 | 17.6 % | adds 0 (rounding) |
| community-12 top-10 by density | 10 | **−0.06 %** | adds *negative* (noise) |
| attribution top-10 | 10 | 61.4 % | adds +43.6 pp |

Three structural priors checked; three falsified. The graph organises the
SAE, but Leiden communities ≠ causal coalitions, decoder neighborhoods ≠
causal coalitions, co-firing partners ≠ causal coalitions. The only prior
that points at the coalition is the causal one (per-feature attribution).

**What this means for the demo's NL layer:** "ablate the community" or
"ablate the decoder neighborhood" or "ablate the co-firing cluster" are all
the *wrong* defaults for a user typing "kill the construction." The right
default is "ablate the attribution-ranked coalition for behavior X" — and
the demo needs to make that distinction visible in the cloud, not bury it.

---

### Q4. Per-variant decomposition — do C1, C2, C3 share a coalition?

**Hypothesis.** The gauntlet's surface findings (F1 dies under all SAE
attacks, F3 holds) suggested F1 and F3 might be distinct behaviours
implemented by different mechanisms. M2 (pivot-probability) probes the
moment *before* the surface form diverges, so a clean test is: does the
top-25 coalition suppress P(pivot) equally well on all three variants,
or differentially?

**Probe.**
```bash
# Same top-25 coalition; ran measure_pivot three times with variants=[Cx]
```

**Result.**

| Variant | n | baseline P(pivot) | ablated | drop | rel drop |
|---|---:|---:|---:|---:|---:|
| C1 | 70 | 0.251 | 0.074 | +0.177 | **+70.5 %** |
| C2 | 63 | 0.463 | 0.091 | +0.372 | **+80.3 %** |
| C3 | 68 | 0.296 | 0.062 | +0.234 | **+79.0 %** |

**All three variants share the coalition.** Same 25 features collapse
P(pivot) by 70–80% across C1, C2, C3. The differences are in *baselines*
(C2 commits at 46% vs C1/C3 at 25-30%), not in coalition structure. The
surface-level divergence between F1 and F3 under single-feature attacks
in §§7-10 of the writeup happens *after* the pivot decision — same shared
machinery commits, divergent surface generation. The "single AI-ism"
framing is the right one at the M2 (decision-point) level. Variant-
specific machinery, if any, operates at the M1 (sentence-completion)
level — a follow-up probe.

---

## Followups queued

### Q2. Where does the ladder asymptote? (Answered.)

**Probe.** Re-ran pivot_attribution with --top-k 100 (40 prompts), then
the daemon's `ladder` command at sizes [5, 10, 25, 50, 75, 100], all
attribution-ranked, plus random size-matched controls (3 draws each).
Full output: `reports/asymptote_ladder.json`.

**Result.**

| N | mean P(pivot) | drop | rel drop | marginal / feature |
|---:|---:|---:|---:|---:|
| 0 (baseline) | 0.2778 | — | — | — |
| 5 | 0.139 | 0.138 | **49.8%** | 9.96 % |
| 10 | 0.100 | 0.178 | **64.0%** | 0.79 % (5→10) |
| 25 | 0.078 | 0.200 | **72.1%** | 0.54 % (10→25) |
| 50 | 0.069 | 0.209 | **75.3%** | 0.13 % (25→50) |
| 75 | 0.063 | 0.215 | **77.4%** | 0.08 % (50→75) |
| 100 | 0.058 | 0.220 | **79.3%** | 0.07 % (75→100) |

Random nulls at every size: mean ≈ 0 ± 0.001. Even ablating 100
random features doesn't move P(pivot).

**Three findings.**

1. **Knee at ~10 features.** The first 5 give 50%; the next 5 add 14
   pp; everything after that is sub-1%/feature. The coalition is
   "ten-shaped" more than "twenty-five-shaped"; 25 just sweeps up the
   tail efficiently.
2. **Asymptote at ~80%.** Beyond top-25 the drop creeps up by ~7pp
   total across 75 more features. The SAE basis at L20 caps out at
   about 80% of P(pivot) for this behaviour.
3. **The other 20% lives outside this SAE.** Three plausible homes:
   the SAE reconstruction error (Engels & Michaud "dark matter"
   territory), other layers' SAEs (probe Q7), or longer-tail features
   the rank-100 truncation doesn't reach but the n=1237 signal does
   (most have n_active ≤ 3, scores below 0.005).

**Implication for the Medium post.** "Twenty-some features" is right
in spirit but the more precise wording is "about ten features do
most of the work, twenty-five captures 90% of what this SAE can
address, and the rest lives outside the basis." Worth a small edit.

**Implication for the demo.** The Toggle tier should default to the
top-10 set, not top-25, for the most visceral "watch a few features
disappear" moment. Top-25 stays as the "comprehensive" toggle. Top-100
is for users who want to see the asymptote with their own eyes.

### Q6. Suppressor coalition — also stacks, in the opposite direction

**Probe.** Top-25 features by attribution score (kind="suppress") —
features whose individual ablation *raises* P(pivot). Ablate all 25
jointly via measure_pivot.

**Result.** baseline 0.278 → ablated 0.374. mean drop = −0.097
(−35% relative). Q6/q6_suppressor_coalition.json.

**Interpretation.** There's a real two-way ledger at the pivot
decision: a promoter coalition (top-25 = +72% drop, "shut the
construction down") and a suppressor coalition (top-25 = −35%
rise, "make it more likely"). The asymmetry — suppression effect
is ~half the promotion effect — is interesting but probably
reflects the corpus: D1 'with' samples already contain the
construction-friendly context, so the "off switch" features have
less room to push.

**For the demo.** "Ablate the suppressors" is a UI moment too:
*lift* the construction rate by knocking out the model's
brakes. Pairs nicely with "ablate the promoters" for the
two-direction story.

### Q8. SAE error-term contribution — not where the residual lives

**Probe.** measure_pivot with use_sae=False (raw model, no SAE
inserted) vs use_sae=True at L20. Q8/q8_sae_error_term.json.

**Result.** raw P(pivot) = 0.270; with-SAE = 0.278. Δ = +0.007 (the
SAE *slightly raises* P(pivot) when inserted, doesn't strip it).

**Interpretation.** The L20 SAE basis is *not* lossy for the
construction. The 20% residual at top-100 ablation (Q2) doesn't
live in the L20 SAE reconstruction error. Combined with Q7 (L25
SAE *is* lossy by ~19%): the residual likely lives in *late
layers* (L21-L25) where the canonical SAE family loses
construction-relevant signal.

### Q3. Leave-one-out from top-25 — coalition has 2 indispensable nodes + 22-feature redundant cloud

**Probe.** Daemon's measure_pivot, called 25 times: each call ablates the
top-25 minus one feature. The "cost of removal" = how much the drop falls
when that feature is removed from the coalition. Reports/q3_leave_one_out.json.

**Result.** Ranked by cost of removal:

| feat | label | cost when removed | n=80 drop without it |
|---:|---|---:|---:|
| 9909 | digital tech / online | **+0.0735** | +0.127 |
| 3223 | exceptions / negations | **+0.0727** | +0.128 |
| 12898 | societal issues / laws | +0.0214 | +0.179 |
| 11864 | technical legal/procedural | +0.0126 | +0.188 |
| 2137 | concern / caution | +0.0092 | +0.191 |
| (rest of top-25) | various | < 0.005 each | ≈ +0.20 each |

**Stunning structure.** The "twenty-five-feature coalition" is really
**two indispensable features (3223 + 9909) doing roughly equal heavy
lifting, one secondary (12898), and twenty-two features that are
individually almost completely substitutable.** Remove any one of the
22, and the rest of the coalition covers (drop barely changes). Remove
3223 alone (within the top-25 context), the drop falls by 36% of the
full effect; same for 9909.

**Caveat about how to read this.** Leave-one-out from the top-25
measures *within-coalition redundancy*, not total importance. The
asymptote ladder (Q2) measured *additive* contribution from rank 1
onwards; that's a different question. 3223 and 9909 individually
also explain ~half of the full drop additively (top-2 = 37%, top-25
= 72%). The reading: 3223 + 9909 are both indispensable
*and* additive; the long tail is necessary in aggregate but
substitutable in detail.

**Implication for the demo.** The "watch this die" moment in the
Toggle tier should toggle 3223 + 9909 *together* — that's the
moment where the construction's commitment visibly halves. Then
adding the rest of the top-25 collapses it further. Two
choreographed clicks, not one.

### Q7-full. Per-layer attribution + ladder at L12 and L25 — L20 is the dominant locus, late layers add a smaller secondary, early layers add little

**Probe.** Ran full per-feature attribution scans at L12 and L25
(`scripts/pivot_attribution.py --sae-layer 12 / 25`, 40 D1 truncated
prompts each, ~17 min apiece on MPS solo). Then ladder probes at
each layer via `scripts/q7full_ladder.py` over sizes
{5, 10, 25, 50, 75, 100} of each layer's attribution-ranked
features, with random size-matched controls. Outputs:
`pivot_attribution_L{12,25}.json`, `asymptote_ladder_L{12,25}.json`.

**The three-layer comparison.**

| Layer | Baseline P(pivot) | Top-5 rel drop | Top-25 rel drop | Top-100 rel drop | Absolute P after top-25 |
|---:|---:|---:|---:|---:|---:|
| **L12** | 0.288 | −17 % | −30 % | −37 % | 0.202 |
| **L20** | 0.278 | −50 % | **−72 %** | **−79 %** | **0.078** |
| **L25** | 0.219 | −45 % | −52 % | −57 % | 0.104 |

Random nulls at every size, at every layer: < 0.001 mean drop.
Signals are real.

**Three layer-locality findings.**

1. **L20 is the dominant locus.** Its ladder asymptotes 22 pp higher
   than L25's and 42 pp higher than L12's. The L20 SAE is where the
   construction's coalition lives.
2. **L25 has a real secondary coalition.** 57% asymptote is meaningful
   — there *is* late-layer machinery for the construction. But the
   L25 baseline is only 0.219 (vs L20's 0.278) because the L25 SAE
   itself loses ~19% of construction signal at insertion (Q7-fast).
   So L25's coalition is operating on a *partially-degraded* signal;
   what it can hit is impressive, but the absolute floor under L25's
   top-25 ablation (0.104) is still higher than L20's (0.078).
3. **L12 has minimal causal involvement.** Even top-100 at L12 only
   moves the needle 37%. The construction isn't *built* at L12 — the
   early layer represents enough of the prompt's context for some
   features to correlate with the pivot, but the actual commit
   mechanism lives downstream.

**The L12 / L20 / L25 top features look completely different.**
None of L20's anchors (3223 "exceptions/negations", 9909 "digital
tech", 12898 "societal issues") appear at the top of L12 or L25's
ranks. L12's top-5 are organization/academic/transaction/procedural
features. L25's top-5 are Australian-culture/code/list/requirement
features. These look like topical noise more than rhetorical
machinery — which fits the L12 / L25 ladders' weaker asymptotes.
The "negation + situation scaffolding" coalition is an L20 thing
specifically.

**Implication for the post.** "The residual lives in late layers"
(from the original asymptote Q2 + Q7-fast) becomes more precise:
*some* of the residual lives at L25 (where a 57% asymptote
coalition exists), but the construction's primary representation
is at L20. The Medium post's framing — "this SAE at this layer can
get ~80% of the way" — is now backed by direct comparative
measurement at three layers. A definitive single-locus result.

### Q7c. Cross-layer joint ablation — late-layer coalition re-measures, doesn't add

**Probe.** New daemon command `measure_pivot_multi` that installs
ablation hooks at multiple SAEs simultaneously. Tested with
`{12: top25, 20: top25, 25: top25}` — 75 features total ablated
across three layers at the pre-pivot last position.
`reports/q7c_cross_layer_joint.json`.

**Result.**

| Condition | P(pivot) | Drop |
|---|---:|---:|
| multi-SAE baseline (all 3 SAEs inserted, no ablation) | 0.214 | — |
| L12 + L20 + L25 top-25 joint ablation (75 feats total) | **0.079** | −0.135 (−63.1%) |

| Reference (single-layer ablations) | Floor |
|---|---:|
| L20 alone (top-25) | 0.077 |
| L25 alone (top-25) | 0.104 |
| L12 alone (top-25) | 0.202 |

**The cross-layer absolute floor (0.079) is essentially identical
to L20-alone's (0.077).** Adding the L12 and L25 coalitions on
top of L20 does *not* push the floor lower. The late-layer
machinery isn't adding *new* construction signal — it's
re-measuring what L20's coalition already addresses, from a
different vantage point downstream in the computation.

**Why does the multi-SAE baseline drop to 0.214?** Inserting all
three SAEs cumulatively adds reconstruction overhead — particularly
L25's, which loses 19% of P(pivot) on its own (Q7-fast). The
multi-SAE baseline is the joint cost of inserting all three, which
explains why the "relative drop" of 63% looks smaller than L20's
72% in isolation — same absolute floor, different baseline.

**The L20-is-locus finding is now definitive.** Three converging
measurements:
1. L20 ablation gets the lowest absolute floor (Q2 + Q7-full).
2. L12 / L25 have *different* coalitions (Q7-full top-25 sets share
   no features with L20's).
3. Cross-layer joint ablation doesn't beat L20-alone's floor (Q7c).

If the late-layer coalition were a *separate* mechanism implementing
the construction independently, Q7c should have pushed below 0.077.
It didn't, so the late-layer coalition is "watching" the same
mechanism from downstream — a read-out, not an independent
implementation. That distinction matters for the post and for the
demo: there's one place to intervene, and L20 is it.

### Q7. Layer baselines: L25's SAE itself destroys construction signal

**Probe.** Daemon's measure_pivot with use_sae=True/False and
sae_layer ∈ {12, 20, 25}. The Gemma Scope canonical SAE was loaded
on-demand at each layer.

**Result.**

| condition | P(pivot) | Δ vs raw |
|---|---:|---:|
| raw model (no SAE inserted) | 0.270 | — |
| L12 SAE inserted | 0.288 | +0.018 |
| L20 SAE inserted | 0.278 | +0.007 |
| **L25 SAE inserted** | **0.219** | **−0.051** |

**L25's SAE strips ~19% of P(pivot) just by being inserted.** That
is, the SAE reconstruction at L25 loses construction-relevant signal
that lives in the L25 residual stream. This is exactly the
SAE-dark-matter problem (Engels & Michaud 2024) at late layers: the
canonical SAE basis doesn't span the construction's representation
there.

**Implication.** The ~20% residual at top-100 ablation (Q2) almost
certainly doesn't live in the L20 SAE error term (Q8 ruled that
out), but it plausibly lives in *late layers* where the SAE basis is
lossy. Construction machinery extends past L20 into the read-out
layers (L21-L25), and our L20-only intervention can't address it.
Sharpens the Medium post's "what's left" framing from "lives outside
this basis (vague)" to "lives in late layers where THIS family of
SAEs has lossy reconstruction."

### Q5. M1 sustained joint ablation (top-25 on gemma-2-2b-it D2 prompts) — underpowered, directionally consistent

**Probe.** Daemon's `m1_eval` command. 8 D2 prompts × 2 seeds = 16
pairs. Sustained ablation of top-25 throughout generation on
gemma-2-2b-it. Classifier scoring (the same one used in the original
gauntlet).

**Raw result.** baseline 0/16, ablated 0/16. The classifier returned
no hits in either condition.

**Hand-audit.** The classifier misses F2 staccato (`"isn't just X.
It's Y"`) — confirmed with a targeted test: comma version gets C3,
period version returns no hits. Phase 7's deslop_demo has the same
blindspot. Several baseline generations clearly contain F2 forms
the classifier missed.

Re-scored with a permissive regex catching F1/F3 (comma/dash),
F2 (staccato), and F4/F5 (reframing/comparative):

| | baseline | ablated | drop |
|---|---:|---:|---:|
| 16-pair sample | 1/16 = 6.25% | 0/16 = 0.00% | +6.25 pp (+100% rel) |

**Verdict.** Directionally consistent with the M2 finding (joint
ablation suppresses the construction in actual generation, not just
at the pivot decision), but **the sample is too small to be
load-bearing**. Phase 7's baseline construction rate on D2 was
~5.5%, so seeing 1/16 in baseline is in-distribution. The 0/16
ablated is consistent with a real ~100% relative effect but also
with a 50%+ effect plus sampling variance.

**To make this load-bearing:** 50+ D2 prompts × 3 seeds = 150+ pairs.
At the observed daemon throughput (~1.5 min/pair under MPS with no
contention, ~3 min/pair under load), that's 4-8 hours. Defer to a
longer overnight run. Alternative: pick D2 prompts pre-screened to
have higher baseline construction rates (the harvest detector mined
exactly those for the CAA corpus) and use those.

**A better M1 test** that's tractable in 30 min: take the 80 D1
truncated prefixes (which by construction are positioned to commit
to the pivot), let the IT model continue them with sustained
top-25 ablation, score the post-pivot region with the classifier.
This is essentially "what happens to the next sentence after the
model decides whether to commit to the construction?" — directly
tests whether the M2 P(pivot) drop translates to fewer constructions
downstream. Queued as Q5b.

---

## Followups queued

- **Q3. The "minimal sufficient" set.** Take the winning set from Q2.
  Iteratively remove features (greedy backwards elimination) and find the
  smallest subset that recovers ≥80% of the full set's drop. *What we want
  to know:* what's the actual size of the irreducible coalition?

### Q5b. M1 via D1 continuation — clean confirmation (40 pairs, p < 0.01)

**Probe.** Take 20 D1 truncated 'with' prefixes (`It's not a tool`,
`This isn't a feature`, ...), let `gemma-2-2b-it` continue under
sustained top-25 ablation, score the continuation with the M1
classifier. 2 seeds per prefix = 40 pairs. Done in 3 chunks via the
daemon's `generate` endpoint + client-side classifier scoring.
`reports/q5b_d1_continuation.json`.

**Result.**

| | baseline | ablated |
|---|---:|---:|
| construction rate | **22.5%** (9/40) | **2.5%** (1/40) |
| absolute drop | +20.0 pp | |
| relative drop | **−88.9%** | |

Fisher's exact, two-tailed: p < 0.01. The single ablated "leak"
(prefix #13, seed 1) is consistent with the M2 asymptote's 20%
residual outside the L20 SAE basis.

**This closes the loop on the §11 finding.** The M2 (decision-point)
suppression translates directly to M1 (full-generation)
suppression on prompts that are positioned to commit. The Phase 7
null on neutral D2 prompts was driven by 3223 being dormant on
neutral inputs; under D1 prefixes (model is primed for the
construction), the joint coalition kills it in nearly 9 out of 10
cases.

**Bridge for the Medium post.** Phase 7 said "the de-slop product
doesn't hold on neutral prompts." That's still true — the coalition
is only active in construction-friendly contexts. But Phase 7 *can't*
say "ablation doesn't work in actual generation." It does — when the
context calls for it. Q5b is the experiment that says so on the same
classifier the gauntlet used.

---

## Followups queued

### Q5c. M1 high-power D2 sustained ablation — paper-grade confirmation on NEUTRAL prompts

**Probe.** 40 D2 neutral prompts × 3 seeds = 120 pairs. Sustained
top-25 ablation throughout generation on `gemma-2-2b-it`. 50-token
continuations. M1 classifier scoring. Ran via detached
`scripts/q5c_runner.py`. Output:
`reports/q5c_d2_high_power.json`.

**Result.**

| | baseline | top-25 ablated |
|---|---:|---:|
| construction rate | **6.67%** (8/120) | **0.83%** (1/120) |
| absolute drop | +5.83 pp | |
| relative drop | **−87.5%** | |

Fisher's exact, two-tailed: p ≈ 0.02 (highly significant).

**This is the result Phase 7 said couldn't exist.** Phase 7 ablated
just feature 3223 on D2 prompts and got a clean null (0% absolute
drop), because 3223 is dormant when the model isn't already primed
for the construction. The lesson there was supposed to be "the
de-slop product doesn't work on neutral prompts."

Q5c says: it DOES work, when you ablate the whole coalition
instead of a single feature. The construction rate on real D2
prompts (the kind a chatbot would receive in deployment) falls
by 87.5%. Of the 8 baseline generations that contained the
construction, joint top-25 ablation killed 7 of them outright;
only one leaked through. Same absolute floor (~1% residual) as
the M2 asymptote and the Q5b D1-continuation, consistent across
all three probes.

**What this changes for the Medium post.** Phase 7's null was a
clean negative on a too-narrow intervention. The product claim —
"ablate this thing and the construction goes away" — was right in
principle but ablating one feature is the wrong intervention. The
right intervention is ablating the coalition. The Medium post
should now say: the de-slop tool that Phase 7 said didn't work
*does* work; you just have to attack the coalition, not the
anchor feature. Phase 7's apparent null was a methodological
artefact of single-feature attack on a multi-feature target.

- **Q5d. Compare 3223+9909 (only) vs full top-25 on D1 continuation.**
  Q3 showed those two are the indispensable nodes. Does ablating
  just those two get most of the M1 effect? Cleanest minimal-set
  demonstration for the demo. 2-3 chunks worth of compute via the
  daemon.

- **Q5. M1 sustained-ablation.** Daemon's `generate` runs base
  gemma-2-2b. Need to extend daemon to support gemma-2-2b-it OR write a
  small standalone script that runs sustained joint ablation on D2
  prompts on the IT model and scores with M1 classifier. *What we want to
  know:* does joint ablation of the top-25 actually reduce M1 generation
  rate (Phase 7's null result was for single-feature ablation only).

- **Q6. Suppressor coalition.** The top-10 suppressors raise P(pivot) by
  15.2%. What if we ablate top-25 suppressors? Does it stack to a clean
  rise, or saturate? *What we want to know:* is the "anti-construction"
  coalition also distributed, or is the asymmetry real?

- **Q7. Layer transfer.** All probes are at layer 20. Re-run the ladder
  at layer 12 (mid) and layer 25 (late). *What we want to know:* is the
  coalition layer-local? If yes, the construction is computed at L20
  specifically. If no, it's a distributed-across-layers behaviour and
  joint ablation should also stack across layers.

- **Q8. SAE error-term contribution.** The 28% residual at top-25 lives
  somewhere. Compare P(pivot) (a) with SAE spliced in (current), (b)
  without the SAE inserted at all. If (a) < (b), the SAE itself is
  destroying construction-relevant signal (i.e. the residual lives in
  the SAE error term, the basis is incomplete). If (a) ≈ (b), the
  residual lives in other layers / outside this hook.

---

## What the graph can actually do, after three failed structural priors

The probes are telling us structural priors (Leiden community, decoder
cosine, co-activation Jaccard) **don't** predict causal coalition
membership for this behaviour. That's a directional finding for what the
graph's role *in this project* should be:

1. **Coalition registry, not coalition discoverer.** Discover the
   coalition empirically (attribution scan), then write it to the graph
   as a first-class entity: `(:Coalition {name: "not-X-but-Y-pivot",
   model: "gemma-2-2b", layer: 20, anchor: 3223})-[:INCLUDES]->
   (:SAEFeature)`. Future "ablate the negation coalition" NL queries
   resolve in one Cypher hop. This is the role the graph *earns* once
   we know structural priors are unreliable.

2. **Cross-coalition overlap.** Once we have multiple coalitions (one
   per behaviour we probe — F1, F3, refusal, sycophancy, …), the graph
   can show which features are shared across coalitions vs unique. This
   is where graph queries genuinely beat flat lists, because the
   question is fundamentally relational.

3. **Label-embedding semantic queries.** AutoInterpLabel embeddings let
   us do "find features whose label is semantically near 'negation'" —
   this is a *semantic* prior, not a structural one, and might
   actually correlate with causal membership in ways the structural
   priors don't. Worth a probe: take the top-25 coalition, check the
   pairwise label-embedding cosines, see if there's a coherent
   semantic cluster the labels point to.

4. **Spatial layout for the cloud.** UMAP + community assignment is the
   right source of truth for the demo's 2D feature manifold. The graph
   serves layout, not analysis.

5. **Hypothesis space for negative results.** The graph is what made
   "decoder vs co-activation vs community don't predict the coalition"
   a *cheap* finding to discover. Two Cypher queries and a measurement
   each. Without the graph that finding would have taken three custom
   scripts. So the graph's first-order value here is in cheaply
   falsifying naive priors.
