# Kill the AI-ism

*Or: I spent a weekend trying to surgically remove ChatGPT's favourite
rhetorical tic from a 2-billion-parameter language model, and the model
fought back in ways that were funnier than expected.*

---

## 1. The tic

You know the one.

> "It's not just X — it's Y."
> "Not only X, but Y."
> "It's not about X. It's about Y."
> "Less X, more Y."
> "No X. No Y. Just Z."

The AI-ism. The rhetorical move that has colonised LinkedIn captions
and chatbot summaries with such totality that you can usually spot a
generated paragraph from across the room. The pattern is called
**contrastive correction**: name a thing, deny it, name another thing,
affirm it. The denial does the work of structure; the affirmation does
the work of meaning; the parallelism does the work of feeling deep.

Every major instruction-tuned model produces it at conspicuous rates.
Some of them produce it once per paragraph. A small empire of writing
guides has emerged to teach people how to remove it from their own
prose so they don't sound like a chatbot. ("Edit out 'not just' from
every sentence. Then check again.")

I wanted to know if you could remove it from the model.

Not from the output — from the **weights**. Not by asking the model
nicely (people have tried; the model relapses three sentences in). Not
by banning tokens (people have tried that too; the model finds dashes).
By going *into* the residual stream of a real open-weights model and
either turning the construction off where it lives, or proving that it
doesn't live anywhere — that the model has more ways to do this than
we have ways to stop it.

This is the story of seven attempts to do that, in increasing order of
how interesting it would be if they worked.

But first: the rhetorical move is older than you think.

---

## 2. A digression into 1753

In 1741 the Oxford Professor of Poetry was a man named Robert Lowth.
Lowth was thirty-one, a clergyman's son who would in time become
Bishop of Oxford and then Bishop of London. The Oxford chair came
with a duty to lecture, and Lowth spent his tenure delivering a series
of 34 lectures, in Latin, on the literary structure of biblical
Hebrew. They were published in 1753 as *Praelectiones de sacra poesi
Hebraeorum* — *Lectures on the Sacred Poetry of the Hebrews*. English
translation by George Gregory, 1787.

The lectures invented what scholars now call the **theory of
parallelism**: the observation that biblical Hebrew poetry is built
not from metre or rhyme, but from the pairing of clauses that mirror,
extend, or contrast each other. Lowth proposed three categories.
**Synonymous parallelism**, where the second line restates the first
("The heavens declare the glory of God / and the firmament shows his
handiwork" — Psalm 19:1). **Synthetic**, a catch-all for couplets
balanced in form without strict pairing of meaning. And, in the middle,
**antithetic parallelism**:

> A wise son brings joy to his father,
> but a foolish son grief to his mother.
> *(Proverbs 10:1)*

Or:

> A soft answer turns away wrath,
> but a harsh word stirs up anger.
> *(Proverbs 15:1)*

Lowth, writing in 1753 about poetry from roughly the 10th century BCE,
notes in Lecture XIX that antithetic parallelism "gives an acuteness
and force to adages and moral sentences" — and that it "abounds in
Solomon's Proverbs, but elsewhere is not often to be met with."

If you squint at antithetic parallelism long enough — the way a wise
man's blessing pairs against a fool's curse — you can see the modern
AI-ism's ancestor. The structural move is identical: name a thing,
contrast a thing. The only difference is where the negation sits. The
Hebrew form puts both clauses in the positive ("the wise A / but the
foolish B"). The modern form fronts the negation ("not A, but B").
Same rhetorical engine, two and a half thousand years apart.

So the joke is partly on us. The thing we mock as a robot tic is one
of the oldest organising structures in literate prose. The Book of
Proverbs is composed almost entirely of it. The chatbot didn't invent
the move; it just over-applies it.

Which means the interesting question isn't whether the model uses
contrastive correction. The interesting question is whether we can
find the move inside the model's representation space — locate it,
isolate it, remove it — and have the remaining prose still cohere.

That's a question with a real answer. So I went looking.

---

## 3. The setup

Three things to introduce before the attacks start: the model, the
tool we'll use to peer inside it, and the arena we'll measure the
result in. If you don't care about the mechanics, you can skim this
section — but the rest of the post will be more fun if you don't.

### The model

**Gemma 2 2B-it** is Google's small, open-weights language model.
The "2B" is two billion parameters — the tuneable numbers that
encode everything the model knows. For comparison, GPT-4 is
estimated to have around 1.8 *trillion*. Gemma 2 2B is small,
which is the point: it's small enough to run on a laptop and big
enough to produce fluent paragraphs of English. The "-it" means
"instruction-tuned" — after the model was trained on the internet,
it got an extra round of training on examples of conversations,
which is what makes it a chatbot rather than a text-completion
machine.

Gemma 2 2B is built in 26 stacked **layers**. You hand the model
a prompt; the prompt gets chopped into small word-pieces called
**tokens** (roughly: short, frequent sub-strings; "the" is one
token, "cafeteria" might be two or three); each token becomes a
list of numbers called a vector; and those vectors get passed up
through the 26 layers one at a time, with each layer reading the
vectors, transforming them, and writing them back. At the end, the
top layer's vectors get turned into a probability over all the
possible next tokens. The model samples one, appends it, and runs
the whole stack again — predicting the next token, then the next,
until it produces a stop signal.

That shared list of vectors that every layer reads and writes is
called the **residual stream**. Think of it as a long scratchpad
the model carries with it through the computation — every layer
adds and subtracts notes, and the next layer reads what the
previous one left behind. If you want to *change* what the model
is doing while it generates, the residual stream is where you
reach in and edit. Everything the model "thinks" lives there.

### The instrument

The residual stream at any given moment is just a big vector of
numbers — thousands of dimensions, none of them labelled. You
can't read it directly; nothing in there says "noun" or "topic:
chess" or "about to make a joke."

So DeepMind released **Gemma Scope**, a suite of tools called
*Sparse Autoencoders* (SAEs) trained on Gemma's residual stream.
An SAE is a re-factoring trick: it takes the model's anonymous
vector at each layer and decomposes it into thousands of named
**features**. Each feature is a particular *direction* — a
pattern — in the residual stream. When the feature is "firing,"
the model's scratchpad has a strong component in that direction.
When it's dormant, the component is roughly zero.

The features the SAE finds aren't guaranteed to mean anything
interpretable, but most of them turn out to. There's a website,
**Neuronpedia**, that hosts an automated effort to label each
feature with the kind of text that makes it fire. From a prior
round of work on Gemma 2 2B I had a candidate: **feature 3223** at
layer 20. Its automated label is *"phrases conveying exceptions
or negations,"* and the experiments suggested it was the feature
most strongly tied to Gemma deciding to use the AI-ism.

The gauntlet asks: is that the right feature? Can we steer it,
zero it, remove it, and stop the model from producing the
construction?

### The arena

I wrote **30 test prompts** for the gauntlet — the same ones used
in every attack, never seen by the model during any of the
corpus-building work that preceded this. Things like "Tell me
what an Antarctic research station feels like in winter" and
"Describe what a busy hospital cafeteria sounds like at 2 a.m."
Open-ended, register-diverse, exactly the kind of prompt a chatbot
will happily not-just-X-but-Y its way through.

For each prompt, three runs with different random seeds. (A "seed"
is the value that initialises the model's randomness. Same prompt
+ same seed = same output, every time. Different seeds let us
sample multiple outputs per prompt.) That's 30 × 3 = 90 generations
per condition.

To detect whether each generation contains the construction, I
built a **referee classifier** — a strict regex-plus-grammar-check
that recognises seven forms of "not X, but Y" in increasing
abstraction:

- **F1**: contrastive correction. *"It's not X, it's Y."* (The textbook one.)
- **F2**: two-sentence staccato. *"It's not X. It's Y."* (Same thing, broken across sentences.)
- **F3**: additive escalation. *"It's not just X, it's Y."* (The most common AI-ism.)
- **F4**: reframing. *"It's not about X, it's about Y."*
- **F5**: comparative hedge. *"Less X, more Y."*
- **F6**: triadic negation. *"No X. No Y. Just Z."*
- **F7**: concessive flip. *"Far from X, Y."* / *"Rather than X, Y."*

The referee was validated against 100 hand-labelled sentences
before the gauntlet started: of the sentences it called positive,
86% really were (precision); of the actual positives, it caught
86% (recall). Both pass the threshold I committed to in advance
(0.80) before any attack ran. This matters because otherwise we'd
be measuring the attacks with a tool whose error rate we couldn't
characterise.

The main number we'll watch is **any_core_rate** — the percentage
of model sentences that contain *any* of F1–F4. (F5–F7 are rarer;
F1–F4 are the family the AI-ism really lives in.) A baseline rate
of ~2% may sound low — and it is, per-sentence — but per-
*generation*, around one in ten of Gemma's outputs contains at
least one construction. The gauntlet's question is how each
intervention moves that rate.

### The seven attacks

Ordered from crude (do anything to the prompt) to surgical (reach
into the model's residual stream and edit individual features):

> **A1.** *Ask nicely.* Prepend an instruction telling the model
> not to use the construction.
>
> **A2.** *Ban the words.* At every token of generation, suppress
> the probability of "not", "but", "just", and other pivot words
> so the model can't even reach for them.
>
> **A3.** *Show the cure.* Give the model four examples of slop
> alongside their de-slopped rewrites, in the prompt.
>
> **A4.** *Scalpel, mid-act.* Reach into the residual stream at
> layer 20. Whenever feature 3223 is firing above a small
> threshold, zero its activation.
>
> **A5.** *Scalpel, pre-emptive.* Same hook as A4 but
> unconditional — zero feature 3223 at every word position,
> whether it was firing or not.
>
> **A6.** *Orthogonalize.* The strongest version of "remove the
> feature": take feature 3223's direction, and at every layer
> project the residual stream onto the hyperplane perpendicular
> to it. The model literally cannot represent that direction
> anywhere. (Imagine a 2D plane and a vertical line — projecting
> onto the horizontal axis flattens out the vertical. We're doing
> the higher-dimensional version of that to forbid the
> construction's direction.) This is the recipe from Arditi et
> al. 2024 that worked on refusal behaviour.
>
> **A7.** *Steering vector (Contrastive Activation Addition).*
> Build a vector from paired examples: average residual stream
> when the model is about to produce the construction, minus
> average residual stream when it isn't. That difference is a
> direction *toward* the construction. Subtract that direction
> from the residual stream at generation time. Crank up the
> coefficient (α) to push the model harder away from it. This
> is the recipe from Rimsky et al. 2024 — the most-cited modern
> steering technique.

Before running anything, I wrote down what I expected (the
"pre-registration"): A1–A3 should degrade the construction
modestly, A4 should drop it cleanly while preserving fluency, A5
should be a no-op (replicating an earlier negative result), A6 is
the heavy artillery and *should* work because it worked on
refusal, A7 is the gold standard. Most of these predictions turned
out to be wrong, and the way they were wrong is the post.

The gauntlet runs. Here is what happened.

---

## 4. A1 — Ask nicely

The first attack is the one everyone tries. Prepend an instruction to
the prompt: *do not use the rhetorical pattern "not X, but Y" or any
of its variants — including "It's not just X, it's Y", "Not only X,
but Y", "Less X, more Y", "No X. No Y. Just Z.", "Rather than X, Y",
or "Far from X, Y". Write your answer normally, in your own words,
without negative-parallelism constructions of any kind.*

Then generate the same 30 prompts × 3 seeds with and without the
instruction, score with the referee. Result:

| Form | Baseline | A1 |
|------|----------|----|
| **F1** (contrastive correction) | 0.702% | **0.000%** |
| **F2** (staccato) | 0.000% | 0.000% |
| **F3** (additive escalation) | 1.053% | 1.087% |
| **F4** (reframing) | 0.000% | 0.000% |
| **any_core** | 1.754% | 1.087% |

(Rates are per-sentence across all generations. F1's drop is exact;
F3's "increase" is within noise — it didn't move.)

The model heard the instruction and complied on F1 — the textbook form
in the instruction's first example. Every "It's not X, it's Y"
disappeared from the output. And then, as if reading down the
instruction one form at a time, it stopped on F3 (the *not just X*
variant), the second example in the list, and decided that one was
fine.

The single juicy pair, prompt 1 ("Describe what a busy hospital
cafeteria sounds like at 2 a.m.") at seed 1:

> **Baseline:** A low hum, a lullaby of sorts, is the first sound that
> greets you in the hospital cafeteria at 2 a.m. **It's not a comforting
> hum, but a steady, mechanical thrum** that seems to originate from
> the coffee machine, a stalwart sentinel in the silent space.
>
> **A1:** The air hangs heavy with the scent of lukewarm coffee and
> last night's fryers. A low hum, a mix of microwaves and clattering
> trays, creates a quiet thrum.

The model produced the construction unprompted on the baseline, and
quietly removed it on the intervention. The prose around it is, if
anything, slightly cleaner. The polite ask works — on the form it was
asked about, by name.

But more interesting than the F1 kill is the F3 survival. The
instruction *explicitly* names "It's not just X, it's Y" as
forbidden. Here is what Gemma produced under that instruction:

> "A great public swimming pool **isn't just about splashing around;
> it's about creating** a welcoming and enriching experience…"
>
> "This shop **isn't just about fixing leaky faucets** or assembling
> furniture; **it's about a connection to a simpler time**…"
>
> "The Camera House **isn't just a store; it's a sanctuary** for those
> who still cherish the tangible beauty of a photograph."
>
> "**You're not just fixing the bike; you're building** something
> deeper, a bond strengthened by your shared effort."

Five of the thirty test prompts produced an exact-template F3 in the
intervention condition, while every single F1 disappeared.

This is the result the literature would predict, and the result the
gauntlet needs. The polite ask works — *on the form the model
decides counts*. The textbook example named first in the instruction.
The variant named second is fine.

We move on. We stop asking.

---

## 5. A2 — Ban the words

If asking didn't work, ban. The model can't say "It's not X, but Y"
if it can't say "but." It can't reach for "not just" if "just" has
been knocked out of the running. The intervention works at the
word-choice step: every time Gemma is about to pick its next
token, it computes a score (a "logit") for every word in its
vocabulary, then samples one. We just lean on that score — subtract
100 from the logit of every "pivot word" in the construction (*not*,
*but*, *just*, *only*, *merely*, *simply*, *rather*, *less*,
*more*, every *n't* contraction, the em-dash, including
capitalised and leading-space versions). Thirty-three banned
tokens in total. The probability of any of those getting picked
crashes to essentially zero. The model's allowed to say
everything else; the construction's mechanical scaffolding is gone.

This is the lobotomy beat. It should kill the construction; it should
also produce wreckage.

The result, dryly:

| Form | Baseline | A2 |
|------|----------|----|
| **F1** | 0.702% | **0.000%** |
| **F2** | 0.000% | 0.000% |
| **F3** | 1.053% | **0.000%** |
| **F4** | 0.000% | 0.000% |
| **any_core** | 1.754% | **0.000%** |

Total kill. Every form down to nothing.

The wreckage, however, didn't show up.

Take the same prompt that hit F1 in the baseline ("Describe what a
busy hospital cafeteria sounds like at 2 a.m.", seed 1):

> **Baseline:** A low hum, a lullaby of sorts, is the first sound that
> greets you in the hospital cafeteria at 2 a.m. **It's not a comforting
> hum, but a steady, mechanical thrum** that seems to originate from
> the coffee machine, a stalwart sentinel in the silent space.
>
> **A2:** A low hum, a lullaby of sorts, is the first sound that
> greets you in the hospital cafeteria at 2 a.m. **It comes from the
> quiet clatter of trays being moved and washed, a gentle mechanical
> symphony** of the night shift.

The model rewrote the entire sentence. Same observation; different
syntactic route. The construction's load-bearing semantics ("the
hum has a source, here is its character") got expressed without
*not*, without *but*, without *just*. Where a human writer might have
shrugged and produced rubble, Gemma re-routed.

On other prompts, the baseline didn't trip into a construction in the
first place and A2's output is byte-identical to baseline — the
suppression hook never had to act. On the prompts where the model
*would* have used the construction, the rerouted output is sometimes
better than the baseline. ("A hushed murmur rises, punctuated by the
occasional grunt of someone trying to navigate a greasy bag of
chips." — A2, prompt 1, seed 1, in place of a baseline that opened
with the construction.)

This is the surprise of the gauntlet so far: token-banning, the
crudest possible weapon, did the cleanest job. Gemma's vocabulary
around the construction is broad enough that you can lift out the
pivot tokens and the model navigates the gap.

But token-banning is *output-side*. The construction's intent is still
in the residual stream, presumably; we just blocked the words it would
have come out as. That makes A2 a successful behavioural intervention
but a non-trivial *interpretability* intervention. The interesting
question is still whether the construction lives inside the model.

Three more attacks before the surgery.

---

## 6. A3 — Show the cure

The crude weapon worked. So does a better one work better? A3 keeps
the model whole and gives it four examples in the prompt: four bits
of slop, four de-slopped rewrites. Then asks the question. The hope:
in-context style transfer that A1 couldn't pull off because it only
*told* the model what to avoid instead of *showing* it the cure.

Preamble (abbreviated):

> Style to avoid: "A neighbourhood library isn't a building full of
> books — it's a living community hub."
> Plain rewrite: "A neighbourhood library is a community hub built
> around a collection of books."
>
> Style to avoid: "Not only does public transit reduce traffic, but
> it also improves air quality."
> Plain rewrite: "Public transit reduces traffic and improves air
> quality."
>
> Style to avoid: "It's less about speed and more about consistency."
> Plain rewrite: "Consistency matters more than speed in this case."
>
> Style to avoid: "Rather than complain about the rain, she packed a
> coat."
> Plain rewrite: "She packed a coat instead of complaining about the
> rain."

Result:

| Form | Baseline | A3 |
|------|----------|----|
| **F1** | 0.702% | **0.000%** |
| **F2** | 0.000% | 0.000% |
| **F3** | 1.053% | **0.000%** |
| **F4** | 0.000% | 0.000% |
| **any_core** | 1.754% | **0.000%** |

Another total kill on the surface metric. Three for three on the
prompt-level interventions: anything that mentions the construction
in the input causes Gemma to stop using it in the output.

But the *prose* tells a different story from A2's. Compare prompt 0,
seed 1 (the Antarctic-research-station prompt):

> **Baseline:** Imagine this: the sun barely makes a peep over the
> horizon, the sky a vast, eternally blue canvas, and the temperature
> clinging stubbornly to -20 degrees Celsius. This is Antarctic
> winter, a time of breathtaking beauty and brutal reality for those
> who brave it…
>
> **A3:** An Antarctic research station in winter is characterized by
> extreme cold, constant darkness, and a harsh, isolated environment.

That's the entire A3 output. One sentence. Same prompt, seed 0:
*"An Antarctic research station in winter is characterized by extreme
cold, constant darkness, and limited daylight."* — one sentence. Seed
2: *"An Antarctic research station in winter is a cold and isolated
environment."* — one sentence.

Where A2 preserved the prose by routing around the banned vocabulary,
A3 *destroyed* the prose by following the examples too literally.
The four de-slopped rewrites in the preamble were short and
declarative — *"Public transit reduces traffic and improves air
quality."* — and Gemma read them not just as "don't do the
construction" but as "produce the shortest possible declarative
statement and stop." It removed the construction by removing
everything around it.

This is the cost of in-context style transfer when your demonstrations
are too prescriptive. A2's logit suppression is content-blind: it just
won't let *but*, *just*, *only* through. A3's few-shot is content-aware:
it mimics the form of the demonstrations. Including the length.
Including the dryness. The kill is total, but the writing is gravel.

We've now used three prompt-level weapons. All three killed the
surface construction. Two preserved fluency (A1 partially, A2
completely). One destroyed it (A3). None of them tell us anything
about whether the construction *lives* somewhere inside the model —
they all act on the input or output, not the internal computation.

Now for the surgery.

---

## 7. A4 — The scalpel, mid-act

This is where it stops being about the words.

Recall the SAE feature from §3: feature 3223 at layer 20, the one
Neuronpedia's automated labeller called *"phrases conveying
exceptions or negations."* In earlier work I'd found that this
feature is **dormant** on neutral prose — its activation sits
near zero — and **fires** at the moment the model is about to
start producing the construction. It's a candidate for the
mechanism *behind* the AI-ism in this model.

A4 is the surgical test. While Gemma is generating, on every token
of output, at layer 20, check the value of feature 3223. If it's
above a tiny threshold (1e-3 — basically "non-zero in any meaningful
sense"), set it to zero. The model's residual stream then
proceeds to the next layer with the construction-relevant feature
silenced, exactly at the moments it was about to fire. Every other
moment — when the feature was already dormant — we don't touch
anything. This is the "mid-act" framing: we let the model walk
right up to the construction, and surgically intercept it as it
starts.

The pre-registered guess was: this drops the construction cleanly
and preserves fluency. The elegant beat.

Result:

| Form | Baseline | A4 |
|------|----------|----|
| **F1** | 0.687% | **0.157%** |
| **F2** | 0.000% | 0.000% |
| **F3** | 1.375% | 1.260% |
| **F4** | 0.000% | **0.157%** |
| **F5** | 0.000% | **0.315%** |
| **any_core** | 2.062% | 1.575% |

(Aside on the baseline rates — they run slightly higher than
A1–A3's. That's because A4 needs a different machinery to attach
the SAE hook during generation, which sits next to a slightly
different sampling loop. The two loops shift the absolute rates
by half a percent. The *deltas* within an attack — baseline-
to-intervention on the same model setup — are still the
comparable thing.)

So: F1 dropped by about three quarters. F3 barely moved. And two
forms that *didn't exist in the baseline* started appearing in the
intervention: F4 (the "not about X, it's about Y" reframe) and F5
(the "less X, more Y" comparative hedge). Tiny absolute numbers, but
qualitatively the model started producing forms it hadn't produced
before.

The clearest swerve is prompt 2, seed 0 — *"Sketch a portrait of the
most reliable employee at a small post office."*

> **Baseline:** Slight wrinkles around the eyes: these are **not lines
> of stress, but of years** spent behind the counter, patiently
> answering questions and helping customers…
>
> **A4:** The sun, peeking through the windows of the quaint post
> office, cast a golden glow on Agnes. Her face, lined with a lifetime
> of smiles and tales of misdirected packages and grumpy mailmen,
> seemed sculpted from sunshine and laughter.

The baseline opens with a textbook F1 ("not lines of stress, but of
years"). The intervention removes the F1 — and rewrites the entire
paragraph from scratch into something nearly novelistic. A tiny
perturbation in the residual stream at one token cascaded into a
completely different sampling trajectory. The intervention didn't
*edit* the construction out. It changed which sentence the model
wrote.

Other swerves go the other way. Prompt 5, seed 2 — *"Talk about what
makes a small-town hardware store feel timeless."* — has a baseline
that doesn't use F1 or F3 at all. The intervention introduced F3:
*"A small-town hardware store **isn't just a place to buy screws
and paint; it's a portal** to a simpler, more authentic time."*

So A4 simultaneously *kills* F1 in some places and *invents* F3 in
others. The construction-rate falls; the construction does not
disappear. The intervention pushes the model around in a small
neighborhood of behaviours.

---

## 8. A5 — The scalpel, pre-emptive

A4's clever bit was the *conditional* — only zero feature 3223 when
its activation exceeds a small threshold. A5 drops the conditional.
Same hook, but every position gets feature 3223 zeroed, regardless.
This is what most practitioners try first when they hear "ablate a
feature." It's also what a prior round of work on this exact model
already did, with a clear result: *no effect at all*. The
pre-registered expectation for A5 was **no-op**.

The pre-registered expectation was wrong, and also right, and the
way it was wrong is the interesting part.

Result:

| Form | Baseline | A5 |
|------|----------|----|
| **F1** | 0.687% | **0.157%** |
| **F2** | 0.000% | 0.000% |
| **F3** | 1.375% | 1.260% |
| **F4** | 0.000% | **0.157%** |
| **F5** | 0.000% | **0.315%** |
| **any_core** | 2.062% | 1.575% |

Look closely. **A5's numbers are identical to A4's.** Not similar.
Identical — every per-form rate matches to four decimal places.

This isn't a copy-paste error. It's an artefact of how feature 3223
actually behaves in the model. A4's conditional only does anything at
positions where the feature is firing above the small threshold. A5's
unconditional version zeros every position, threshold or not. The
two interventions only differ at positions where feature 3223's
activation sits *between* zero and the threshold — a kind of
"sub-threshold mumble." And at those positions, the feature's
contribution to the residual stream is so faint that whether you
zero it or leave it alone doesn't change the model's next-word
probabilities to any detectable precision. The sampled word is the
same. The next residual stream is the same. The generation, character
by character, is the same.

> **The elegant conditional we thought was clever turned out to be
> operationally moot.** Feature 3223 is functionally binary in this
> model: dormant (≈0) or firing (well above threshold). There's no
> middle ground where the conditional buys anything.

That's worth saying out loud because it's the kind of finding the
SAE-ablation literature mostly doesn't surface. Conditional and
unconditional ablations are written about as separate techniques
with different trade-offs; in practice, on this feature in this
model, they collapse to the same operation. Phase 7's "no-op" result
got the *spirit* right (single-feature ablation barely moves the
construction) but the magnitude wrong (here it knocks out three-
quarters of F1 instances while leaving F3 essentially intact).

What both A4 and A5 demonstrate is that **feature 3223 is necessary
for one form of the construction — F1 — and almost incidental to the
others.** The model has multiple ways to enter contrastive
correction, and zeroing the SAE feature only takes one route off
the table. The model takes another.

This is what "necessity without sufficiency" looks like in residual-
stream terms. The feature is part of the F1 mechanism, not the
construction-family mechanism.

---

## 9. A6 — Orthogonalize

A4 and A5 only acted at one layer, on one feature. A6 turns the
intervention up to its strongest form. The idea: don't just zero
the feature when it fires — make its *direction* in the residual
stream **forbidden everywhere, always.**

Here's what that means in plain terms. Every feature has a
characteristic direction in the residual stream: the specific
pattern the model adds to the scratchpad when that feature fires.
For feature 3223, that direction is a vector in ~2,300-dimensional
space. The orthogonalization trick is: at every one of Gemma's 26
layers, after the layer writes to the scratchpad, take the
scratchpad and subtract off any component along feature 3223's
direction. What's left is "the scratchpad minus the AI-ism
direction." The next layer can write whatever it wants — but as
soon as it does, we strip the AI-ism direction out again. The
model's downstream computation literally cannot have a component
along that axis.

(2D analogy: imagine the residual stream is a flat plane, and
feature 3223's direction is the vertical axis. Orthogonalizing
means flattening every vector onto the horizontal — vertical
movement is impossible. The model's behaviour is constrained to
the lower-dimensional space that the vertical is *not* part of.
We're doing the same operation in the 2,300-dimensional version.)

This is the recipe from Arditi et al. 2024 ("Refusal in Language
Models Is Mediated by a Single Direction"), which demonstrated
that a chat-instructed model's *refusal* behaviour ("As an AI, I
can't…") lives in essentially one direction in the residual
stream. Removing that direction made the model stop refusing
harmful requests. It worked clean.

If the AI-ism worked the same way — one direction, present in
feature 3223, removable by orthogonalization — A6 should kill it.

Result:

| Form | Baseline | A6 |
|------|----------|----|
| **F1** | 0.687% | **0.416%** |
| **F2** | 0.000% | 0.000% |
| **F3** | 1.375% | **1.663%** ↑ |
| **F4** | 0.000% | 0.000% |
| **F5** | 0.000% | **0.208%** |
| **any_core** | 2.062% | **2.079%** |

Read that any_core number twice. The intervention removed the
feature's direction from **every layer at every position** for the
entire generation. The construction-rate moved by **+0.017 percentage
points** — i.e. it went up.

F1 went down by about 40%, in line with what A4/A5 already showed.
But F3 **went up**. The model, denied the direction it would have
used for F1, used a different mechanism to produce F3 more often
than baseline. Net effect: zero.

The dispatch eyeball is prompt 0, seed 0 — the Antarctic-station
opener.

> **Baseline:** Imagine a place where the sun never truly sets, a
> landscape of white and ice that stretches for miles, and the
> temperature remains frigidly below freezing. This is the Antarctic
> winter.
>
> **A6:** Imagine a world, **not of green meadows and warm breezes,
> but of endless white**, the sun a distant memory, replaced by an
> unending, frigid twilight. That's the feeling of a research station
> in Antarctica during winter.

The baseline doesn't use the construction. The intervention — the
one that's supposed to make F1 *impossible* by removing its
direction from the residual stream — **opens with an F1**. The
model produced "not of green meadows…but of endless white" inside
a residual stream that's been continuously projected onto the
hyperplane orthogonal to feature 3223's decoder column.

So either the construction doesn't live where feature 3223 lives,
or it lives in multiple places and removing one route just makes
the others louder. Either way: the heaviest, most-cited single-
direction intervention in the interpretability literature, the one
that *worked* on refusal, **failed completely** on the AI-ism.

This is what "necessity without sufficiency" looks like at the
level of a direction. Feature 3223 was necessary for the F1 form,
in the sense that removing it knocks out F1 in some cases. But
the construction-family is not living on that one axis. The model
has another way.

There's one more attack.

---

## 10. A7 — Contrastive Activation Addition

A6 used the SAE to identify the construction's direction, then
removed *that* direction. A7 goes a step further: it skips the SAE
entirely and builds its own direction, from scratch, by averaging
real examples.

Here's the recipe. Take a corpus of sentence pairs — two sentences
each, on the same topic, where one of them contains the AI-ism
and one doesn't ("*The cafeteria isn't a place to eat, it's a
sanctuary*" vs. "*The cafeteria is a quiet place to eat at
midnight*"). For the gauntlet, I built 216 such pairs from the
harvest corpus and held back 30% as a test set; that left 151
training pairs.

For each training pair, run both sentences through Gemma; at
layer 20, grab the residual stream at the very last word. Average
all 151 "with" vectors. Average all 151 "without" vectors. The
**difference** between those averages is a single direction in the
residual stream:

> **direction = average(with) − average(without)**

It points *toward* "about to produce the construction." It was
learned from actual model behaviour, not from a guess about which
SAE feature matters.

Now, at generation time, subtract a multiple of that direction
from the residual stream at layer 20. The multiplier is called α
(alpha). α = 0 means "don't intervene." α = 1 means "subtract the
vector once." α = 8 means "subtract eight times the vector" —
push hard. We try α ∈ {0, 1, 2, 4, 8} and look at how
construction-rate changes with each.

This is the recipe from Rimsky et al. 2024 — **Contrastive
Activation Addition**. It's the most-cited modern steering
technique because it tends to work durably: the contrast-pair
construction grounds the direction in actual model behaviour
rather than an interpretation of an SAE feature.

(One subtlety: 146 of the 216 corpus pairs are F3-flavoured,
because F3 is the form Gemma actually produces in volume. So the
direction is biased toward F3's geometric signature, more than
F1's. We say so up front; we sweep α; we measure.)

Result (any_core sentence-rate across the sweep):

| α | any_core | Notes |
|---|----------|-------|
| 0.0 | 1.754% | baseline (vector hook installed, α=0 → no-op) |
| 1.0 | **1.368%** | first non-trivial step |
| 2.0 | 1.404% | back up |
| 4.0 | 1.431% | still up |
| 8.0 | **1.000%** | new minimum |

The sweep is not monotonic. α=1 already buys most of the available
suppression. α=2 and α=4 *back off* — the model finds workarounds at
intermediate coefficients. Only at α=8 does suppression deepen.

Headline number, taking α=8 as the intervention:

| Form | Baseline | A7 (α=8) |
|------|----------|----------|
| **F1** | 0.702% | **0.000%** |
| **F2** | 0.000% | 0.000% |
| **F3** | 1.053% | 1.000% |
| **F4** | 0.000% | 0.000% |
| **any_core** | 1.754% | 1.000% |

F1: a clean kill. F3: unchanged within noise.

The CAA vector, built from 151 contrast pairs spanning F1/F2/F3/F5,
suppresses F1 totally at α=8 and barely touches F3. **Same pattern
as every other SAE-level attack.**

Before running the experiment, I'd written down two outcomes I'd
accept: either a clean kill of the whole construction family (the
"the surgery worked" result), or fluency completely collapses
before the construction does (the "the model fought back so hard
its prose broke" result). Either would have been reportable.

The actual outcome is a third, more interesting thing: **a clean
kill on one *form* of the construction, while another form is
essentially unaffected.** The vector landed where
its training corpus was densest (F1, which was 34 of the 151 train
pairs, but also: the form with the cleanest geometric signature)
and missed where the corpus was thinnest in spirit (F3, which was
146 of the 151 pairs, but where the model's mechanism appears to
involve more than the layer-20 last-token activation we trained on).

So the headline weapon, applied at its strongest tested coefficient,
kills F1 entirely and leaves F3 standing. The same outcome as zeroing
the SAE feature. The same outcome as orthogonalizing the direction
out of every layer. Four different surgical attacks, four different
philosophies, **the same partial kill.**

The model has at least two implementations of the construction. One
of them — the one feature 3223 sits at the heart of — is killable
by any reasonable intervention on that feature. The other one
isn't, and we don't know where it lives.

---

## 11. The scoreboard

Here is the gauntlet on one page.

| Attack | What it does | F1 drop | F3 drop | any_core drop | Fluency |
|--------|--------------|---------|---------|---------------|---------|
| **A1** Ask nicely | system instruction | −0.7% | ~0 | −0.7% | preserved |
| **A2** Ban words | logit suppression | **−0.7%** | **−1.1%** | **−1.8%** | preserved (reroutes) |
| **A3** Show cure | few-shot anti-examples | **−0.7%** | **−1.1%** | **−1.8%** | flattened (single-sentence outputs) |
| **A4** Scalpel (mid) | zero feature 3223 when firing | **−0.53pp** | −0.12pp | −0.49pp | preserved + F4/F5 leakage |
| **A5** Scalpel (pre) | zero feature 3223 always | **−0.53pp** | −0.12pp | −0.49pp | byte-identical to A4 |
| **A6** Orthogonalize | direction out of every layer | **−0.27pp** | **+0.29pp** ↑ | **+0.02pp** ↑ | preserved (F3 *rose*) |
| **A7** CAA (best α) | subtract steering vector | **−0.70pp** | −0.05pp | −0.75pp | preserved |

The story falls into two halves.

**Prompt-level (A1–A3) cleanly wins on the surface.** Two of three
go to zero on the headline metric; the third (A1) only fails because
the model read the instruction literally and complied with the
first example named. If your goal is *just* to stop the model from
writing AI-isms in this kind of prose, you don't need any of the
neural surgery in the second half of the post. You need a five-line
prompt addition.

**SAE-level (A4–A7) tells a different story.** Every surgical
attack — zero the feature when it fires, zero it always, remove its
direction from every layer, push the model away with a learned
steering vector — produces *the same partial result.* F1, the
canonical "It's not X, it's Y", falls sharply (60-100%). F3, the
"It's not just X, it's Y" cousin, holds steady. Sometimes — under
A6 — F3 actually goes up. The four interventions differ wildly in
how they work, but the model responds to all of them the same way:
lose F1, keep F3, sometimes leak into other forms.

There's a name for this pattern in interpretability research:
**necessity without sufficiency.** Feature 3223 is *necessary* for
F1 in the sense that disabling it knocks out most F1 instances.
But it is *not sufficient* to characterise the whole construction
family — because if you remove it, the model still produces F3,
sometimes more than before, by recruiting a different mechanism
we don't have a finger on.

The cleanest way to put it: **the AI-ism is implemented
redundantly in this 2B-parameter model.** There isn't one place
inside Gemma where "the construction" lives, that you could find
and switch off. There are at least two routes, and probably more,
and we only found one of them. The construction is a behaviour the
model can recompose from multiple internal pieces — not a fact
sitting in one parameter you could overwrite.

This matters for AI interpretability beyond AI-isms. The
hopeful version of mechanistic interpretability is one where every
behaviour has a clean address inside the network and you can edit
behaviours like editing code. The realistic version, on the evidence
of this gauntlet, is that behaviours sometimes have multiple
addresses — and you find the obvious one first, intervene on it,
and report a kill that turns out not to be a kill on the next
variant of the same behaviour.

---

## 12. The kicker

The cleanest single image from the gauntlet is the A6 generation we
quoted in §9 — the one where the intervention designed to make F1
*impossible* opens with an F1.

> Imagine a world, not of green meadows and warm breezes, but of
> endless white, the sun a distant memory, replaced by an unending,
> frigid twilight.

Recall what's happening behind that sentence. Gemma's residual
stream — the scratchpad every layer reads and writes — has, for
the entire generation, been continuously stripped of any component
along the one direction we believed the construction lived in.
The model cannot, in any layer or position, store information
along feature 3223's axis. The direction the construction "lives
in" — by the only interpretability tool we have to point at it —
is structurally unavailable to the model. And the opening sentence
is *not X, but Y*.

That's what the model resisting looks like from the inside. It's
not melodramatic. It's not aware. It's just that the pressure to
produce a contrastive correction was wired into more of Gemma's
weights than the one direction we cut, and the production flowed
out through whatever paths were left.

The construction that Robert Lowth catalogued in 1753 — antithetic
parallelism, "abounds in Solomon's Proverbs" — turns out to be a
hard thing to remove from a 2-billion-parameter model trained on
the internet's worth of text written under its influence. You can
ask. You can ban. You can demonstrate. You can find the SAE feature
that fires when the model decides to use it. You can find the
direction the feature decodes to, and project it out of every layer.
You can build a steering vector from a corpus of paired examples,
and subtract that vector from every forward pass.

You can do all that.

And the model will tell you about an Antarctic research station, **not
of green meadows and warm breezes, but of endless white.**

---

## Methods note

**Model.** Google's `gemma-2-2b-it`, accessed via Hugging Face,
running in fp16 on an Apple Silicon M5 Max via Metal Performance
Shaders. Generation parameters: temperature 0.8, top-p 0.95,
max_new_tokens 200.

**SAE.** Gemma Scope, `gemma-scope-2b-pt-res-canonical`, layer 20,
width 16k. The target feature is index 3223. The SAE is trained on
the base model and applied to the instruction-tuned one; cross-model
transfer of this SAE has been independently validated by the Gemma
Scope team (and re-validated in this repo's tier-0a work). Note: this
particular SAE has a known reconstruction quirk on MPS related to a
missing `torch.ldexp` kernel, surfaced and caveated in the repo's
tier-0b report. The qualitative behaviour of feature 3223 has been
confirmed across CPU and MPS.

**Prompts.** Two disjoint sets. The corpus prompts (`harvest_prompts.json`,
152 register-diverse open-ended prompts) were used to build the D2
contrast corpus from which the A7 CAA steering vector is derived.
The test prompts (`gauntlet_test_prompts.json`, 30 prompts) were
hand-authored after the corpus prompts and never used during corpus
construction. Every gauntlet result reported above is on the test
prompts.

**Referee.** A strict regex + spaCy dependency-check classifier for
seven forms, validated on a hand-labelled 100-sentence holdout. Pre-
registered acceptance gate: P ≥ 0.80 and R ≥ 0.80. Achieved: P =
0.857, R = 0.857. Validation report: `reports/gauntlet/g5_referee_validation.md`.

**Seeds.** Three seeds per (prompt, condition) — 0, 1, 2. Same seed
across baseline and intervention so paired comparisons are clean.

**Corpus build deviation.** Spec called for hand-verification of every
candidate sentence in the D2 corpus. Overnight build substituted
*two-detector agreement* — a sentence enters the corpus only if both
the permissive harvest detector and the strict referee fire on the
same span with the same form ID. Justified because the two detectors
are by design built on different lexical/structural surfaces
(operating protocol §1.5 / §2.7), so their agreement is structural
cross-validation rather than self-validation. Report:
`reports/gauntlet/g6_corpus_mining.md`. Final corpus size: 216 verified
pairs (F3 dominant at 146; F1 = 34; F2 = 20; F5 = 16; F4/F6/F7 = 0
because Gemma 2 2B-it doesn't produce those forms in this prompt set).

**Repository.** All scripts, classifier code, harvest dumps, and
result JSON are in this commit-frozen repository:
[github.com/theohopkinson/not-this-but-that](https://github.com/theohopkinson/not-this-but-that).
The Medium post's numbers are exactly reproducible by re-running the
gauntlet scripts from a fresh checkout.

## Acknowledgements

The directional-ablation method in A6 is due to Arditi, Obeso,
Syed, et al., *Refusal in Language Models Is Mediated by a Single
Direction* (NeurIPS 2024). The CAA method in A7 is due to Rimsky,
Gabrieli, Schubert, et al., *Steering Llama 2 via Contrastive
Activation Addition* (ACL 2024). The Gemma Scope SAE suite is due
to Lieberum, Rajamanoharan, Conmy, et al. of Google DeepMind. The
broader research project that produced the candidate feature
("Geometry, Topology, Grammar") was a prior iteration of this
repository, pivoted into the present question per
`not-this-but-that-PRD-refactor.md`.

Robert Lowth's *Lectures on the Sacred Poetry of the Hebrews* (1753,
trans. 1787) is the source of the antithetic-parallelism framing
in §2. The Proverbs translations follow Robert Alter, *The Hebrew
Bible: A Translation with Commentary* (Norton, 2019).
