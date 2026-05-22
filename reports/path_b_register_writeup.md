# Path B — what the register finding actually is

*The cross-family replication ran. The headline I would have written before it ran does not survive. The actual story is smaller, more specific, and — if you read what it says rather than what you hoped it'd say — more interesting.*

**Status:** Path B writeup per [`the_explicit_decision.md`](the_explicit_decision.md). Path A (SAE mechanism, feature 3223) is **not** in this document — it is blocked on Tier 0b and is conditional on a verification this codebase couldn't perform. The two stories stay separate.

---

## What the cross-family replication said

I generated 30 D2 prompts × 3 seeds × 150 tokens on Qwen 2.5 7B Instruct, scored every sentence with the strict-mode classifier, and measured the two quantities the Gemma-2-2b-it work pointed at: (a) the C3-share of any_core constructions; (b) H17 median relative position of construction sentences.

| Quantity | Gemma 2 2B-it (reference) | Qwen 2.5 7B Instruct |
|---|---:|---:|
| any_core rate (per sentence) | 1.8 % | **0.79 %** |
| C3 count | 18 of 19 | **0 of 3** |
| C3 share of any_core | **94 %** | **0 %** |
| H17 median pos w/ construction | 0.10 | 0.75 |
| H17 median pos w/o construction | 0.50 | 0.50 |
| H17 Mann-Whitney p | 0.028 | 0.69 |
| H17 direction | opener | "closer" (n=3, no power) |

Qwen produces the construction at half the Gemma rate, and what it does produce is **zero per cent C3**. The 94 %-C3 / opener-position signature is **Gemma-2-2b-it specific, not a cross-family instruct register**.

That is a clean cross-family null. The Path B claim as I'd written it before the run — "instruct-tuning installs a C3-opener register across families" — is false. I am retracting it here rather than narrating around it.

## What's actually going on, after looking

Qwen's outputs are not register-poor. They have the same surface register Gemma-2-2b-it has — **opening-summary + numbered list + bold-headed bullets**. Sample (verbatim):

> Investing in public transit infrastructure can bring numerous benefits to a city, both for its residents and visitors. Here are some key reasons why a city might decide to invest in such infrastructure:
>
> 1. **Reduced Traffic Congestion**: By providing more efficient and accessible public transportation options, cities can reduce the number of cars on the road. …
> 2. **Environmental Benefits**: Public transit systems typically produce fewer greenhouse gas emissions per p[assenger]…

That is the same emphatic / explanatory / listicle register Gemma-2-2b-it uses. The difference is what fills the **opening sentence**. Gemma-2-2b-it opens with the C3 rhetorical move:

> *A good neighborhood library isn't just a place to check out books – it's a vibrant community hub…*

Qwen opens with a declarative summary:

> *Investing in public transit infrastructure can bring numerous benefits to a city…*

Both occupy the "topic sentence of a structured explanation" slot. Gemma fills that slot with C3; Qwen fills it with declarative summary. The **register substrate** — list-form, bold-headed, opening-summary explanation — appears common across instruct models. The **specific stylistic move within the opener** is family-specific.

## The honest claim, after replication

**Within-Gemma-2-2b-it (Tier 0a + Confirmation passed):**
- Construction rate is 4.2× the base rate; non-overlapping CIs.
- The construction is 94 % C3 (minimize-then-elevate) — non-C3 variants are vanishingly rare.
- Construction sentences cluster at the very beginning of generations (median relative position 0.10 vs 0.50 for non-construction sentences; Mann-Whitney p = 0.028 on the held-out Confirmation split, BH α_BH = 0.043).

**Cross-family (Qwen 2.5 7B Instruct, n = 90 generations, 3 constructions of 382 sentences):**
- The 94 % C3 share does NOT replicate (Qwen: 0 % C3 share of any_core).
- The H17 opener effect does NOT replicate (Qwen direction inverts and has no power at n = 3).
- But the surface register *does* appear to be shared (opening-summary + numbered list + bold bullets) — assessed qualitatively, not quantitatively in this campaign.

The honest one-sentence version: **Gemma 2 2B-it inhabits an emphatic explanatory register that is broadly shared across instruct models, but fills the opening-sentence slot of that register with the C3 "not just X — it's Y" construction in a way that is model-family-specific, not cross-family.**

That's the small true thing. It's neither the headline I'd have written before running Qwen nor an utterly null result — it tells you the construction is a *Gemma stylistic signature inside an instruct-shared register*, not a property of instruct-tuning per se.

## What survives, what's retired, what's still open

**Survives:**
- The variant-composition gap within Gemma 2 2B base vs Gemma 2 2B-it. (Phase 2, Tier 0a confirmed via blind-validated classifier.) Real, within-Gemma.
- H17 opener effect within Gemma 2 2B-it. (Discovery → Confirmation → BH-FDR passed.) Real, within-Gemma.
- The methodological discipline: Discovery → Confirmation → cross-model replication did exactly what it was supposed to do. It killed a claim that wouldn't have replicated.

**Retired:**
- "Instruct-tuning installs the C3 register across families." It doesn't, at least not on Qwen 2.5 7B Instruct.
- "Quotation marks predict construction-entry" (H19) — retracted earlier as an apostrophe-in-contraction artifact.

**Open in a way I am NOT going to address by running more replications:**
- Is C3-dominance a Gemma-2-instruct-family property or just Gemma-2-2b-it-specific? The Gemma 2 9B-it test would tell us, but the model is gated on HuggingFace and the current `HF_TOKEN` doesn't have access. The within-family-scale question stays open from this session.
- One cross-family replication on Qwen 2.5 7B Instruct was a clean null. A second cross-family replication on gpt-oss-20b (added 2026-05-21 — see `## gpt-oss-20b replication` below) was also a clean null. Two of three non-Gemma families now triangulate the same answer. Continuing to run more instruct models hoping the cross-family claim resurrects is exactly the false-positive amplifier the protocol exists to prevent; not doing it.

## What this is and isn't

**It is:** an honest narrowed finding. The most defensible version is:

> *Gemma 2 2B-it produces the "not X, but Y" construction at 4.2× its base model's rate, with the C3 variant accounting for 94 % of usage and clustering at the very beginning of generations (median relative position 0.10 vs 0.50). The shared instruct register across families (opening summary + structured bullets) is broader than this; the within-register fill — C3 as the topic sentence — is a Gemma-it stylistic signature, not a general instruct-tuning property. A Qwen 2.5 7B Instruct replication failed to reproduce the C3-share or the opener effect, on n = 3 constructions across 90 generations.*

**It isn't:** a cross-family register claim, a mechanism claim, or a product hypothesis. The de-slop product was already falsified by Phase 7. The mechanism story (Path A — feature 3223 etc.) is in a separate decision document and is conditional on the Tier 0b fix.

## What I'm not doing (deliberately)

- I'm not framing this as "still very promising." The Qwen result is what the protocol calls a kill — for the cross-family claim. The within-Gemma claim survives, but it's a smaller thing.
- I'm not running 6 more models to find one that does replicate the C3 share. That would be drifting toward the false-positive amplifier the protocol exists to prevent.
- I'm not letting Path A creep back in to fill the space the narrowed Path B leaves. Path A is still blocked.
- I'm not claiming the surface-register-is-cross-family observation as a separate finding — it was an eyeball check on Qwen's sample outputs, not a measured replication. If it matters, it can be measured with a separate Discovery campaign on markdown-register features. Until then it's an *observation*, not a *claim*.

## Next steps (in priority order)

1. **Decide whether the narrowed Gemma-only finding is shippable as a short note.** The question is whether *one model with this specific stylistic signature, plus an honest cross-family null* is interesting enough to publish. The answer "it's a small finding, narrowed by replication, with a worked methodology" is itself a small contribution; the Discovery / Confirmation / cross-model-replication protocol is the load-bearing intellectual claim and the C3-as-Gemma-2-2b-it-signature is the empirical demonstration of it working.
2. **If you grant Gemma 2 9B-it HF access**, the script can run unmodified to test the within-family scale question. Useful but not decisive.
3. **Path A stays where it was.** No drift. Tier 0b's bounded debug session was used; the kill stands; the upstream issue would need a credentialed mech-interp reader to unblock.
4. **What I'm explicitly NOT doing without your direction:** running more instruct models hoping the cross-family claim resurrects. The protocol exists to prevent that; respecting it is the credential.
