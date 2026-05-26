# I Found the Neuron That Makes AI Write Like AI. Then I Tried to Kill It.

*Spoiler: it isn't one neuron. It's twenty-five. And they don't really want to die.*

---

There's a sentence I want you to read out loud.

> "This isn't a setback, it's a springboard."

You know that sentence. You've read it five hundred times this year. You couldn't pick the chatbot that wrote it out of a lineup of chatbots, because every chatbot writes that sentence. Different topic, same shape. *It isn't X — it's Y.* I'll call it the **AI-ism** for the rest of this piece, because that's what it is. The bot's tic. The deep tell.

I wanted to know if I could turn it off.

Not from the prompt side — that's been tried. You can ask the model not to do it; the model nods, then does it three sentences later. You can ban the word "just" from its vocabulary; the model finds a comma, finds an em-dash, finds another verb. The construction is too useful to the model. It is, on the inside of the model's head, doing something.

What I wanted to know is what that *something* was, and whether you could reach inside the model — open it up like a clock — and switch the something off. Most of the work in mechanistic interpretability looks for the *one feature* that runs a behaviour. Refusal, for instance, turned out to live in one direction in the residual stream of every model anyone's checked. Erase that direction, the model stops refusing to write your stalker fanfic. The literature kept publishing variants on this: *a single feature mediates X*, *one direction is the locus of Y*.

I bet the AI-ism would be the same. I spent a weekend being wrong about that.

---

## A digression into 1753

The construction isn't a chatbot invention. It's a 2,800-year-old rhetorical move.

In 1741 the Oxford Professor of Poetry, a 31-year-old clergyman named Robert Lowth, started a series of Latin lectures on the literary structure of biblical Hebrew. Published in 1753 as *Praelectiones de sacra poesi Hebraeorum*. He gave the move a name: **antithetic parallelism**. Two clauses, structurally mirrored, one denied and one affirmed. *"A wise son brings joy to his father, but a foolish son grief to his mother."* The book of Proverbs is composed almost entirely of it. Hebrew poetry uses it as a beat.

Modern chatbots front-load the negation — *"it's not X, it's Y"* instead of *"the wise A, but the foolish B"* — but it's the same engine. Lowth was looking at it three millennia after it was invented. The chatbot has only had it for two years.

Which means: the move itself is fine. It's an old beat. The thing that's wrong is the *over-application* — every chatbot reaches for it on every prompt, regardless of whether the antithesis actually clarifies anything. Half of LinkedIn caption advice in 2026 is just "stop doing this." So: can we reach into the chatbot and remove its eagerness to do the antithetic parallelism move, while leaving the rest of the chatbot intact?

That's a question with a real answer, if you have the right tools. I went looking.

---

## The setup, in plain English

I'll move fast because the punchline is more interesting than the apparatus.

**The model.** Google's Gemma 2 2B — an open instruction-tuned model, two billion parameters, runs on a laptop. Small enough to probe directly, big enough to produce fluent prose. It produces the AI-ism, as a sentence rate of about 7% across a set of neutral open-ended prompts, which sounds low until you realise every other paragraph contains one.

**The microscope.** The way you look inside a model these days is a **sparse autoencoder** — an SAE. It's a little learned dictionary. You feed it the model's internal state at a given layer and it decomposes that state into a sum of 16,384 *features*, of which only a handful are firing at any given moment. The clever part: each feature gets an automatic English-language label. Feature 3223 in this model is labelled *"phrases conveying exceptions or negations."* Feature 9909 is *"references to digital technology and online interactions."* Feature 10678 (a different layer) is *"phrases related to careful planning."* That kind of thing. The labels come from another model that reads the example sentences each feature fires on and writes a summary. They're noisy but mostly truthful.

**The substrate.** I keep all 16,384 features in a Neo4j graph as nodes. Edges between them encode three structural relationships: which features write similar things to the model's scratchpad (decoder cosine, 1.3 million edges), which features fire together (co-activation, 349k edges), and which automatically-discovered cluster they belong to (Leiden community detection, 18 communities). I'll come back to this — it's the navigational instrument that turns a flat 16k-feature list into a queryable map of meanings. Without it you can't reason about features by hand.

**The probe.** Take a sentence the model is *about to* commit to the AI-ism with — something like *"It's not a tool"*. Truncate it right before the model picks the next word. Look at the probability the model assigns to any of the comma-and-pivot words that complete the construction (*", but"*, *", it's"*, *"—"*). Call that probability **P(pivot)**. Across 80 such truncated prompts, P(pivot) at baseline is about 0.28. That's the thing I'm going to try to push down.

If I can find the feature, or features, that drives P(pivot), and disable them, the construction should stop. That's the bet.

---

## The naive attack

I did what anyone would do. I ranked all 16,384 features by how much zeroing them out drops P(pivot). The winner, by a wide margin, was **feature 3223**. Label: *phrases conveying exceptions or negations.* You can imagine my satisfaction. The literal negation feature. Cinematic.

I zeroed it. P(pivot) dropped by 18%. Real drop, beat every random baseline.

So I went for the kill. I took feature 3223's direction in the model's internal vector space and *projected it out of every layer* — the Arditi et al. 2024 method, the one that erased refusal behaviour from a dozen chat models. It was supposed to make the construction structurally impossible. Cannot be represented. The model literally cannot write notes along that axis.

I ran the test. The first generation, on a prompt about an Antarctic research station, started:

> *"Imagine a world, **not of green meadows and warm breezes, but of endless white**, the sun a distant memory…"*

The intervention designed to make the construction *impossible* opened with the construction. Inside a model whose residual stream had been continuously stripped of feature 3223's direction. Forty-three generations later, the construction rate had moved by about negative two percentage points.

I tried six more single-feature attacks. Different mechanisms — ban the pivot words at the logit layer, ablate the feature only when it fires, ablate it always, build a learned steering vector and subtract it from the residual stream. They all did the same partial thing: the most textbook form of the construction (*"it's not X, it's Y"*) went away, and a different form (*"it's not just X, it's Y"*) often went up. The model wasn't being killed. The model was *rerouting*.

There's a name for this. It's called the **Hydra Effect** — Anthropic and DeepMind have been writing about it for two years. Knock out one path that produces a behaviour, another path compensates. The Hydra grows another head. Most of the published cases are about factual recall: ablate the neuron that remembers Paris is the capital of France, the model finds another way to say Paris. I'd never seen it called out for a rhetorical behaviour, but here it was. The model was being a Hydra about contrastive correction.

Which means feature 3223 wasn't really *the* feature. It was *a* feature. There were others. I just hadn't found them.

---

## The realization

Here's the structural lesson I should have started with.

The published refusal-direction work showed that one direction in the residual stream mediates refusal. That's a beautiful clean result. It made everyone in interp assume *most* behaviours work like that. *Necessary and sufficient.* One thing.

But there's no reason that has to be true. A behaviour can be *necessary* in one feature and *not sufficient* — the model needs the feature, *and* the model has other ways of getting the same effect when the feature isn't available. The cleanest evidence for this is when you turn off the feature and the behaviour drops a little but not to zero. That's what 3223 looks like.

The model is, in some sense, hedging. The construction is implemented redundantly. A coalition of features cooperates to produce it, and the coalition is robust to losing any one member.

If that's the case, the right attack isn't to find a smarter way to disable one feature. The right attack is to find *the rest of the coalition* and disable all of them at once.

This is where the graph stopped being decorative.

---

## The eighth attack

I needed to find the coalition. The honest version of that question is: which features, in addition to 3223, have a large causal contribution to P(pivot)?

I ran the same per-feature attribution scan but this time kept the top hundred, not just the top one. The top three: feature 3223 (negation), feature 9909 (digital technology), feature 12898 (societal issues / laws / marginalised groups). The next twenty-two: a heterogeneous cloud of features about concern, accountability, scientific measurement, urgency, ethics, professional roles. Not what you'd guess. Not all about negation. Topical scaffolding.

Then I ablated them in joint-set ladders — top-2, top-5, top-10, top-25, top-50, top-75, top-100 — and measured P(pivot) at each rung.

| set size | P(pivot) | drop from baseline | rel drop |
|---:|---:|---:|---:|
| 1 (just 3223) | 0.228 | −0.050 | **−18%** |
| 2 | 0.174 | −0.104 | **−37%** |
| 5 | 0.135 | −0.143 | **−51%** |
| 10 | 0.107 | −0.171 | **−61%** |
| 25 | 0.078 | −0.200 | **−72%** |
| 50 | 0.069 | −0.209 | −75% |
| 100 | 0.058 | −0.220 | −79% |

There's the coalition. The first five features get you to 51%. The first ten get you to 61%. By twenty-five you've taken away 72% of the model's commitment to the construction at the decision point. By a hundred you asymptote at about 79%. After that the SAE basis at this layer has nothing more to give you — the remaining 21% lives at later layers or in the SAE's reconstruction error (I'll get to that). I also ran this exact ladder at higher power (200 D1 prompts instead of 80) and the asymptote sharpened slightly — top-25 = 76% drop, top-100 = 82% drop. Same shape, tighter numbers. [Reproducible run.](./asymptote_ladder_n200.json)

The control: ablating a hundred *random* features moves P(pivot) by under 0.001. The 79% drop isn't an artefact of mass ablation. It's specific to this set.

The model wasn't being a Hydra against the coalition. It was being a Hydra against my attempts to find one node of the coalition at a time. Hit it from all twenty-five directions simultaneously and it loses three-quarters of its grip.

---

## In actual prose, on actual prompts

The number above is at one token position. The interesting question is whether the kill survives in actual generation — does the model, allowed to write a full paragraph with the coalition silenced at every token, produce the construction less often?

I tested it two ways, and there's a methodology footnote I have to make up-front. The construction has at least two surface forms the model uses interchangeably: the same-sentence one (*"isn't X, it's Y"*) and the across-sentence one (*"isn't X. It's Y."* — what I'll call the *period-form*). The regex-plus-spaCy classifier this project inherited from earlier work catches the same-sentence form but misses the period-form. So I re-scored every generation with both that classifier *and* a more permissive regex that catches the period-form, then took the union. Reporting both is the only honest thing to do; the union is the "everything I can see" number. (The full re-score, with strict, permissive, and union counts side by side, is at [`reports/m1_rescore_union.json`](./m1_rescore_union.json). A reader who wants to write an even more permissive detector and get a lower number is welcome to.)

**Primed prompts.** Take 100 of the construction-bearing prefixes (*"It's not a tool"*, *"This isn't a setback"*, *"He's not a teacher"*) and let the instruction-tuned model continue each, three seeds. Baseline construction rate, union of detectors: **15%, forty-six of three hundred**. Same prompts, same seeds, top-25 coalition silenced throughout: **3%, eight of three hundred**. **An 82% relative drop** (95% CI from prompt-clustered bootstrap: +68% to +93%; McNemar's mid-p exact paired test, two-sided: **p < 10⁻⁶**). The strict-only number reads 8.3% → 0.7% = 92%; the strict classifier under-counts the baseline because it misses period-form, and over-counts the kill because it doesn't see when the model reroutes into period-form.

**Neutral prompts.** A hundred-and-two open-ended prompts the model hasn't seen — *"describe a hospital cafeteria at 2 a.m.", "sketch a portrait of a postal worker", "discuss the role of mentorship"* — three seeds each, three hundred and six pairs total. This is the test that matters, because these are the prompts a real chatbot user actually sends. Baseline, union: **36 of 306, 12%**. Silenced: **7 of 306, 2%**. **An 80% relative drop** (95% bootstrap CI: +67% to +92%; McNemar's mid-p two-sided: **p < 10⁻⁶**). Strict-only: 4.6% → 0.3% = 92%, same caveat.

This is the result an earlier phase of this project said was impossible. We'd tested *single-feature* ablation on neutral prompts and got a clean null — the construction rate didn't move. That null was correct, given what we were testing. It was wrong about what it meant. The de-slop tool doesn't work when you target the wrong thing. It works *mostly* when you target the coalition. There's a long tail of period-form rerouting the coalition doesn't fully cover — about 4% of generations still produce the construction, in a form the model wouldn't have used as often without the intervention. That's worth being honest about, not because it weakens the finding but because it's the shape of the finding.

The clean-kill cases are the visceral ones. *"This isn't a setback"* primes the model to write *"…it's a springboard"* in baseline; under coalition ablation the same prompt produces *"…it's a — opportunity to make the best decision."* The construction doesn't form. *"He's not a teacher"* baseline reels off three contrastive sentences in a row (*"He's not a teacher, but I think he's helping us all learn. He's not a politician, but I think he's showing us what's right. He's not a musician, but I think he's inspiring us."*); under ablation the model degenerates, just repeats the prefix, never finds a continuation. The machinery the prompt would have used isn't available, and the model can't find a route around it.

There's also a nicer follow-up: I ablated *just the two indispensable features*, 3223 and 9909, on the primed prompts at full power (40 prefixes × 3 seeds = 120 pairs). Construction rate union: **19.2% → 4.2%, a 78% relative drop** (95% bootstrap CI: +57% to +94%; McNemar's mid-p: p ≈ 7 × 10⁻⁵). Compare that to the full top-25 coalition's 80% drop on the same setup. **The two-feature core gets nearly all of the kill — the other twenty-three are mostly cleanup.** That's the demo moment: two clicks instead of twenty-five.

---

## What the coalition actually looks like

I ran a leave-one-out from the top-25: for each feature, I ablated *the other twenty-four* and measured how much the drop fell. That tells you which features are individually indispensable (removing them costs a lot) versus individually substitutable (removing them costs almost nothing because the rest cover for them).

The result is the cleanest structural thing I found in this project. Two features stick out:

- **Feature 3223**, *phrases conveying exceptions or negations.* Cost when removed: 0.073.
- **Feature 9909**, *references to digital technology and online interactions.* Cost when removed: 0.074.

Then there's a secondary tier, mostly **feature 12898** (*references to societal issues, particularly laws and marginalised groups*) at 0.021. And then twenty-two features whose individual cost is under 0.01. Remove any one of them in isolation and the coalition barely notices.

So the coalition has a structure: **two indispensable core features doing about a third of the work each, and a long tail of twenty-two redundant supporters**. The reason the seven single-feature attacks all failed isn't that the construction is uncuttable — it's that they were all hitting the same one or two nodes of a coalition that has twenty-five.

I tested whether extending the analysis past rank 25 changes the picture — leave-one-out from the top *fifty* features instead of the top twenty-five. The result is even cleaner than the top-25: the same two indispensables (3223 + 9909 cost ~0.07 each), the same one secondary (12898 cost 0.020), three more weak-secondary features (cost 0.01-0.02), and then *all forty-three remaining features in the top-50 have cost-when-removed below 0.005 — and the twenty-five features ranked 26-50 all sit below 0.001*. There's no second indispensable tier hiding past rank 25. The coalition really is two-cored.

The labels are the second surprise. I expected the coalition to be a cluster of "negation" features. It isn't. 3223 is about negation. 9909 is about digital tech. 12898 is about laws and marginalised groups. The remaining features are about concern, accountability, ethics, scientific measurement, urgency, professional roles. *They look topical, not syntactic.*

What I think is going on: the construction commits when the negation feature plus the *situational scaffolding around it* is active. Contrastive correction lands when there's a thing to *care about* — a problem, a stake, a domain in which the contrast carries rhetorical weight. The negation feature alone is just negation; combine it with "this is a serious topic about which a corrective claim might matter" and you get *it's not X, it's Y*. The coalition isn't a syntactic engine; it's a *rhetorical context* detector.

I don't have a clean proof of that interpretation, but the labels point exactly there.

---

## The first surprise: the graph guessed wrong

I want to dwell on the graph for a second, because it's the part that earned its place by being wrong before it was right.

The hypothesis the graph naturally generates is *structural*: features that are like 3223 are probably in the coalition with it. There are three obvious structural priors. The graph holds all three as queryable edges:

- **Decoder neighbours**: features that write similar patterns to the residual stream (cosine of their decoder columns).
- **Co-activators**: features that fire together across the corpus (PMI, Jaccard overlap).
- **Community-mates**: features in the same Leiden cluster.

So I tried each. Three Cypher queries. Three measurements. Then ladder probes on each set.

| prior | size | P(pivot) drop |
|---|---:|---:|
| 3223 alone | 1 | −18% |
| 3223 + 9 decoder neighbours | 10 | −20% |
| 3223 + 9 co-activators | 10 | −18% |
| Top 10 of 3223's community | 10 | **−0.1%** |
| Causal attribution top-10 | 10 | **−61%** |

All three structural priors *failed*. The decoder neighbours added about two percentage points beyond what 3223 alone did. The co-activators added literally nothing. The community-mates — the top features in the Leiden cluster containing 3223 — moved P(pivot) by under one part per thousand. The features that *implement* the construction with 3223 aren't its decoder neighbours, aren't its frequent firing partners, and aren't its community-mates.

The only prior that worked was the direct causal one: *which feature, when removed, drops P(pivot) most?* And then take the top N of those.

This is the graph contributing a negative result, and that's a real contribution. The graph let me ask three structurally-motivated hypotheses in an afternoon, fail at each, and move on. Without the graph that finding would have taken three custom scripts, three measurement runs, and three writeups. With the graph it took half an afternoon.

The graph's productive role in this project, in retrospect, is:

1. **It made the negative result cheap.** I learned that structural similarity in this SAE doesn't predict causal coalition membership. That's useful to know — it generalises beyond this construction.
2. **It made the coalition legible.** Once I had the twenty-five feature indices, the graph turned them into a list of labelled, communitied, density-annotated, hover-able dots in a map. Without that, the coalition is twenty-five integers.
3. **It made the demo possible.** The playground I built lets you type a concept ("negation", "punctuation", "names") into a box and the graph instantly highlights every feature whose label matches. You can silence them with one click. The "type a meaning, kill it" gesture is the magic moment of the demo, and it's a Cypher query and a label-similarity match away from impossible.

A flat sixteen-thousand-feature list is not navigable. The graph turns it into a map. The model performs the interventions; the graph chooses the targets and gives them names. That's the role.

---

## The second surprise: it really is at one layer

Gemma 2 2B has twenty-six layers. The work above is all at layer 20. The natural follow-up is: is layer 20 special, or does the construction live everywhere?

I re-ran the whole pipeline — per-feature causal attribution, plus the ablation ladder — at layer 12 (early-middle) and layer 25 (late). Each layer with its own top-25 features. Different SAEs, different feature numberings, different coalitions.

| layer | baseline P(pivot) | after top-25 ablation | relative drop |
|---:|---:|---:|---:|
| L12 | 0.288 | 0.202 | −30% |
| **L20** | **0.278** | **0.077** | **−72%** |
| L25 | 0.219 | 0.104 | −52% |

L12 is *"the construction isn't built here yet"*. The early layer represents enough of the prompt's topical context that some features correlate with the pivot, but the machinery that *commits* to the construction hasn't run.

L25 is more interesting. It has a real coalition (52% drop), but its baseline is lower because inserting the L25 SAE itself loses 19% of P(pivot). The late-layer SAE basis is genuinely too narrow for the construction — what it reconstructs faithfully isn't quite the right thing. Some construction-relevant signal lives in the SAE's *error term* at L25.

And the kicker, the bit that nails the locality finding: I joint-ablated **L12 + L20 + L25 top-25 simultaneously** — seventy-five features across three layers — and the absolute floor was 0.079. Same as L20-alone's 0.077. The late-layer coalition isn't an independent implementation. It's the same mechanism, observed from downstream. There's one place to intervene on the construction, and it's layer 20.

The Hydra has heads at multiple layers, but it has one heart.

---

## What I think this means

I'll be honest about the scope. This is one construction in one open-source model, with one SAE family at one layer. I have not shown this generalises to Llama or Mistral; I haven't shown that *other* rhetorical behaviours in this model have similarly small coalitions. The work I've done here is an existence proof: at least *this* behaviour, in *this* model, is implementable as a small, identifiable, coalition that you can disable.

The published interpretability literature has spent two years finding behaviours that live in one direction. Refusal. Sycophancy. Honesty. Single directions. Single switches.

I expected the AI-ism to be the same. What I found is that even when one feature is genuinely the right anchor — feature 3223 is exactly what you'd want the negation feature to be — the behaviour the feature anchors can still be a coalition. The model can run the construction without using the feature, because the model has twenty-two redundant supporters. Single-feature attacks failed not because they were wrong about the feature but because they were wrong about the architecture: behaviours have coalition addresses, not switch addresses, and the prior you need to find the coalition is causal, not structural.

The good news is that twenty-five features is not a lot. The coalition is small enough to enumerate, identify by label, and silence. The bad news, for anyone who wants to *positively control* the model — make it produce the construction reliably by turning the coalition up — is that I tried clamping the coalition to high values and it doesn't reproduce the construction. *Necessity yes, sufficiency no.* You can take the construction away. You can't put it back the same way.

That asymmetry is the deepest thing in this writeup, and I don't fully understand it.

---

## Where I could be wrong

I want to be specific about this, because a friend with mech-interp credentials read a draft and pointed at four things, and I'd rather pre-empt the hostile read than get caught by it.

**This is post-hoc.** The project's pre-registration committed to single-feature attacks, which is what the original gauntlet ran. The coalition is the finding that came *out* of those attacks failing, which means several of the adversarial controls the pre-registration committed to need to be re-run *against the coalition*. The random-k null in [`reports/joint_ablation.md`](./joint_ablation.md) beat the coalition by orders of magnitude, but random-k is the cheap test. The harder one — *matched-activation*, where each coalition feature is paired with a same-activation-density non-coalition feature — is the pre-registered control that actually matters. Twenty independent matched draws, each ablating 25 features at the coalition's median activation density (matched to four decimal places: 0.0158 vs 0.0158): the coalition's P(pivot) drop was +0.200; the largest of the twenty matched-null drops was +0.001; the mean was *slightly negative* at −0.002. **The coalition's drop exceeds the largest matched-null draw by 0.199, and beats all 20 draws.** So the effect isn't an artefact of mass-ablating any 25 active features. ([Reproducible run](./matched_activation_null.json).) Other pre-registered controls (specificity tests on adversarial negation sets, interchange patching for sufficiency) haven't yet been done.

**Sufficiency is still asymmetric and I haven't fully retried it.** Clamping the coalition's activations to high values doesn't reliably *produce* the construction in cases where the model wouldn't have used it. The pre-registration committed to one method of sufficiency retrial — interchange patching from a construction-bearing source — and that hasn't been done yet either. The honest reading is that necessity is clean and sufficiency is murky; the murkiness may be a real fact about coalitions, or may be a not-yet-run experiment away from resolving.

**One model. One layer family. One width.** Gemma 2 2B, Gemma Scope canonical SAEs at 16k width. The finding is local to this configuration. The cross-model replication that would tighten the claim — does the same shape of coalition exist in Llama 3 or Mistral or Qwen at L20-equivalent layers? — is on the to-do list and isn't done.

**The coalition is a blunter scalpel than a single feature.** Ablating just feature 3223 on a held-out human-prose corpus (D3) leaves the model's perplexity unchanged — ratio 1.000× baseline. Ablating the full top-25 coalition pushes perplexity to **1.079× baseline** under ablate and 1.123× under clamp-up — small but measurable degradation. Translation: silencing twenty-five features at every position throughout generation does cost the model a little fluency on text it would otherwise predict cleanly. It's still well inside the conventional "scalpel not sledgehammer" gate of < 1.20×, but it's a meaningfully bigger perturbation than the single-feature attack the original Phase 5 measured. The de-slop claim should be read with that in mind: you're not getting the kill for free.

**The strict classifier under-counts both ways.** I've leaned on it because it was hand-validated at 0.857 P/R on a hundred-sentence holdout, but it misses period-form constructions and that misses real model behaviour in both baseline and ablated conditions. Everything in the previous section uses the union of strict + permissive detection; the demo's frontend uses the same regex; the [`reports/m1_rescore_union.json`](./m1_rescore_union.json) artifact reports both numbers side by side. A reader who wants to write an even more permissive detector will probably get an even lower drop, and that's fine — it just means the model reroutes into shapes neither detector catches, which is its own interesting finding.

## The graph as the selection-and-composition layer

The finding above — a coalition of twenty-five features causally implementing a behaviour — is interesting. What I want to convince you of in this section is a different claim: **once you have that coalition stored as a first-class object in a graph, with the rest of the SAE around it, you can do things you genuinely cannot do anywhere else.**

I built three demos into the playground to make this concrete. Each one is a Cypher query under the hood, and each one is impossible without the graph.

### Demo 1 — Surgical de-slop · `concept ∩ behaviour` via set intersection

The full coalition has twenty-five features. But for any given prompt, only a handful of them are *topically* relevant. The graph lets you intersect a prompt's semantic content (via vector search over the 16,384 SAE feature labels) with a named behaviour's feature set — and silence only the overlap.

Type a prompt, click `✨ Surgical de-slop`, and the playground:

1. Embeds your prompt and finds the top-K labels whose meaning matches it
2. Cypher-intersects those features with the `ai-ism` `:Behaviour`'s `INCLUDES` edges
3. Silences only the intersection — a precise subset of the coalition, scoped to your prompt

For *"Discuss the legal and medical implications of AI in healthcare"* the graph returns two: feature 7361 (*statements about legal analysis and recommendations*) and feature 1608 (*legal and health-related terms*). Of the twenty-five-feature coalition, those are the two your prompt actually invokes. Silence them, generate, watch the construction not form.

For *"Explain quantum computing to a child"* it returns zero — the AI-ism coalition isn't topically active for this prompt, so there's nothing to silence. The graph said *leave it alone*. (A vector DB would have happily returned twenty-five irrelevant matches; only a graph with first-class behaviour membership can say "no intersection.")

The Cypher is exposed in the UI (`show Cypher` button):

```cypher
MATCH (b:Behaviour {name: 'ai-ism'})-[r:INCLUDES]->(f:SAEFeature)
  WHERE f.index IN $retrieved AND f.sae_id CONTAINS 'L20/16k'
RETURN f.index AS idx, r.weight AS weight, r.rank AS rank
ORDER BY r.rank
```

That's RAG-for-activations: retrieval is into the model's *internal* concept space, not into a document corpus. The retrieved objects are *features*, not chunks. You compose them into an intervention rather than into a context window.

### Demo 2 — Mix your own chatbot · weighted Cypher union across named behaviours

The playground has four `:Behaviour` nodes seeded: `ai-ism` (the coalition), `bullets` (bullet-point compulsion), `hedging` (caveats), `formal_register` (academic prose). Each is a 25-feature subgraph, weighted by how much each feature contributes.

The mixer gives you a slider per behaviour, 0-100%. The intensity controls how many of each behaviour's top features get pulled into the union. Drag, click Apply, the model regenerates with the union silenced:

```cypher
UNWIND $intensities AS i
MATCH (b:Behaviour {name: i.name})-[r:INCLUDES]->(f:SAEFeature)
  WITH b, f, r ORDER BY r.rank
  LIMIT toInteger(i.intensity * b.coalition_size / 100)
RETURN DISTINCT f.index AS idx, collect(b.name) AS sources
```

`{ai-ism: 100, bullets: 50, hedging: 30}` composes a 45-feature silence-set drawn from three behaviours at proportional depth. The same query mechanism could mix `personality_anxious`, `tone_dry`, `register_clinical` — behaviours are user-defined, named, persistent, composable. **Style as a graph algebra.**

This is the closest thing I've seen to a "model personality control plane" that's actually mechanistic — not a system prompt, not a fine-tune, not RLHF. Sliders that compose subgraphs that drive interventions on running activations.

### Demo 3 — Audit trail · feature provenance as graph paths

Every time the model generates, the playground writes an `:Intervention` node to the graph with `:USED_SOURCE` edges to the things that selected each silenced feature — `preset:top25`, `surgical_deslop:<prompt>`, `community_click:<region>`, `user_click`, `lasso`, `search:<query>`. Each source is in turn `:SELECTED` → `:SAEFeature` for the features it pulled in.

So you can ask, weeks later: *"Why did this run silence feature 3223?"* Answer: a `MATCH (i:Intervention {id: 'i_abc'})-[:USED_SOURCE]->(s)-[:SELECTED]->(f:SAEFeature {index: 3223})` returns the source nodes. **Reproducibility is a path query.**

```cypher
MATCH (i:Intervention {id: $iid})-[:USED_SOURCE]->(s:Source)-[:SELECTED]->(f:SAEFeature)
  OPTIONAL MATCH (f)-[:LABELED_AS {primary:true}]->(l:AutoInterpLabel)
RETURN s.kind, s.label, collect({idx: f.index, label: l.text}) AS features
ORDER BY s.ts
```

The `Why did the model say that?` panel at the bottom of the playground renders this as a source-grouped feature list, with `show Cypher` to expose the underlying query. For glass-box governance — *show me which features your AI silenced, and which lineage decided to silence them* — this is the primitive.

---

### Why this needs a graph

Vector DBs can do retrieval. SQL can do joins. What you can't do without a graph:

- **First-class coalition membership** — *"these twenty-five features collectively implement that behaviour"* as a queryable node with weighted edges. Demo 1's intersection wouldn't work — there's nothing to intersect retrieval *with*.
- **Composable behaviour subgraphs** — Demo 2's union is one Cypher query over arbitrary named coalitions. Without typed edges between behaviours and features, you'd be rebuilding the union in application code per request.
- **Lineage as paths** — Demo 3's audit isn't a join, it's a *traversal* across `(intervention) → (source) → (feature) → (label) → (community)`. Path queries are what graphs are for.

Each demo is one Cypher query of fewer than twelve lines. The graph isn't the visualisation layer; it's the substrate.

## Try it

Everything lives at **`http://127.0.0.1:8765/demo/playground.html`**.

**The story** at `/demo` is a long version of what you just read, with the headline numbers as live cards and a side-by-side baseline-vs-ablated playback you can click through. Four pre-baked examples; the model takes a prompt and writes it normally on the left, then writes it with the twenty-five-feature coalition silenced on the right. Watch the construction die in real time.

**The playground** at `/demo/playground.html` is the full instrument. The three demos above plus everything else: type a concept like "negation" or "code" into the search box and the matching features light up blue on the 16,384-dot map. Click communities to navigate by meaning. Shift-drag to lasso. Alt-click for graph neighbours. Hover any dot for its auto-interp label.

All of it — the live model, the ablation hooks, the graph queries, the per-token activation capture — runs on a single Python daemon you launch with `scripts/probe_run.sh start`.

The code is at [github.com/theohopkinson/not-this-but-that](https://github.com/theohopkinson/not-this-but-that). The Neo4j substrate is on `bolt://localhost:7693`. Every number in this piece is reproducible by running the scripts in `scripts/` against a fresh checkout.

---

## A short footnote on what's where

If you want to know exactly what each tool did:

The **model** is Google's `gemma-2-2b-it`, fp16, Apple Silicon MPS. The **SAE** is Gemma Scope `layer_20/width_16k/canonical`, a JumpReLU SAE trained by Google DeepMind. The **coalition** is the top-25 by per-feature causal attribution to P(pivot) at truncated D1 prefixes (`scripts/pivot_attribution.py`). The **classifier** is `src/classifier/detect_construction` — a regex-plus-spaCy detector blind-validated at **P = 0.80, R = 1.00** on 90 independently-sourced sentences ([Tier 0a](./tier_0a_classifier_blind_eval.md)), above the pre-registered ≥ 0.70 gate. The headline construction-rate numbers in this post are scored with the **union** of that classifier and a permissive regex that catches the cross-sentence staccato form ("isn't X. It's Y") that the dependency-checked classifier misses; both are committed in [`src/classifier/`](../src/classifier/), and the union is what's used by the demo and the offline re-score in [`m1_rescore_union.json`](./m1_rescore_union.json).

The **graph** is Neo4j 5, ingested by `scripts/04_ingest_features.py`: 16,384 `:SAEFeature` nodes, 1.3M `DECODER_SIMILAR` edges (cosine of decoder columns), 349k `CO_ACTIVATES_WITH` edges with PMI and Jaccard, 18 Leiden communities, 16,384 `LABELED_AS` edges to `:AutoInterpLabel` nodes. The demo's UMAP layout pulls community id and label from the graph for every dot.

The **directional ablation** trick that failed on the AI-ism is from Arditi, Obeso, Syed et al., *Refusal in Language Models Is Mediated by a Single Direction* (NeurIPS 2024) — the same recipe that *worked* clean on refusal. The **CAA** steering-vector method that also partially failed is from Rimsky, Gabrieli, Schubert et al. (ACL 2024). The **Hydra Effect** framing is McGrath, Rahtz, Kramár, Mikulik, Legg (arXiv:2307.15771, 2023), originally about self-repair in attention layers during factual recall. None of those papers had reason to expect their methods would partially fail on a rhetorical behaviour; the partial failure is what made this piece worth writing.

The historical aside on antithetic parallelism is from Robert Lowth's *De sacra poesi Hebraeorum* (1753, English trans. George Gregory 1787). The Proverbs translations are Robert Alter's *The Hebrew Bible* (Norton 2019). Lowth would not have predicted any of this and I think he'd have enjoyed it.
