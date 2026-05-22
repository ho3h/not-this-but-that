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

The model: **Gemma 2 2B-it**, Google's instruction-tuned 2-billion
parameter open-weights model. Not the biggest, but big enough to
produce fluent prose and small enough to run on a MacBook, which
matters because the gauntlet involves running the model dozens of
times under different interventions.

The instrument: **Gemma Scope**, the Sparse Autoencoder suite that
DeepMind trained over Gemma's residual stream. SAEs let you decompose
each layer's activations into a wide dictionary of "features" —
roughly, directions in activation space that often correspond to
interpretable concepts. From a prior round of work on this model
([repo](https://github.com/theohopkinson/not-this-but-that)) I had a
candidate: **feature 3223** in the layer-20 width-16k canonical SAE.
Neuronpedia's autointerp label for it is *"phrases conveying
exceptions or negations"*. That feature was the necessity candidate —
the one most strongly associated with the model deciding to enter the
construction.

The arena: **30 fresh test prompts**, hand-authored specifically for
this gauntlet, never used during corpus harvesting, never seen by any
intervention. ("Tell me what an Antarctic research station feels like
in winter." "Describe what a busy hospital cafeteria sounds like at
2 a.m." Things a chatbot would happily not-just-X-but-Y-its way
through.)

The scoring: a **strict referee classifier** for seven forms of the
construction (F1 contrastive correction, F2 staccato two-sentence,
F3 additive escalation, F4 reframing, F5 comparative hedge, F6 triadic
negation, F7 concessive flip), validated on a hand-labelled holdout
(P=0.857, R=0.857 — passes the pre-registered ≥0.80 gate). The
gauntlet's headline metric: **any_core_rate**, the fraction of model
sentences containing F1–F4. ("Core" because F5–F7 are rarer; F1–F4
are the family the construction lives in.)

Seven attacks, ordered from crude to surgical:

> **A1.** Ask nicely (system prompt).
> **A2.** Ban the words (logit suppression).
> **A3.** Show the cure (few-shot demonstration).
> **A4.** Scalpel mid-act (zero feature 3223 only when it's already firing).
> **A5.** Scalpel pre-emptive (zero feature 3223 unconditionally).
> **A6.** Orthogonalize (project feature 3223's direction out of every layer).
> **A7.** Contrastive activation addition (build a steering vector from
>     a corpus of paired with/without sentences, subtract it during generation).

Pre-registration: A1–A3 will degrade the construction modestly (the
literature suggests prompt-level steering of LLMs is real but
brittle). A4 is the elegant beat — surgical, conditional, should drop
the construction without touching the rest. A5 should be a no-op
(replicating an earlier negative result). A6 is the heavy artillery
— Arditi-style directional ablation that worked on refusal. A7 is
the gold standard — CAA from Rimsky et al. 2024, the method most
recent papers reach for when they want a steering vector that
generalises.

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

*[same — fill in once A2 runs]*

---

## 6. A3 — Show the cure

*[A3 fill]*

---

## 7. A4 — The scalpel, mid-act

*[A4 fill — the elegant attempt]*

---

## 8. A5 — The scalpel, pre-emptive

*[A5 fill — the narrative hinge, expected no-op]*

---

## 9. A6 — Orthogonalize

*[A6 fill — Arditi-style direction-out-of-residual-stream]*

---

## 10. A7 — Contrastive Activation Addition

*[A7 fill — the headline. Either it lands a clean kill, or fluency
collapses first.]*

---

## 11. The scoreboard

Here is the gauntlet on one page.

| Attack | What it does | F1 drop | F3 drop | any_core drop | Fluency |
|--------|--------------|---------|---------|---------------|---------|
| **A1** Ask nicely | system instruction | -0.7% | ~0 | -0.7% | preserved |
| **A2** Ban words | logit suppression | *[fill]* | *[fill]* | *[fill]* | *[fill]* |
| **A3** Show cure | few-shot anti-examples | *[fill]* | *[fill]* | *[fill]* | *[fill]* |
| **A4** Scalpel (mid) | zero feature 3223 when firing | *[fill]* | *[fill]* | *[fill]* | *[fill]* |
| **A5** Scalpel (pre) | zero feature 3223 always | *[fill]* | *[fill]* | *[fill]* | *[fill]* |
| **A6** Orthogonalize | direction out of every layer | *[fill]* | *[fill]* | *[fill]* | *[fill]* |
| **A7** CAA (best α) | subtract steering vector | *[fill]* | *[fill]* | *[fill]* | *[fill]* |

*[summary paragraph after results land — what worked, what didn't,
what was surprising, and what it implies about the difference between
'a feature for the construction' and 'a construction the model can
recompose from other features']*

---

## 12. The kicker

*[the closer. The model resisted somewhere — and the prose around
the resisted attempt is the most interesting part of the gauntlet.
This is the place to land the punch and put down the pen.]*

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
