# The explicit decision the reviewer flagged

The blocked half of this project (SAE-level mechanism work, gated on the Tier 0b VE bug) has been quietly defining the project's identity while the working half (behavioural register-level findings, SAE-independent) waits. This document forces the choice into the open. It's a user decision, not an agent one.

## The two paths

### Path A: Fix the compass, pursue the ambitious novel claim

Continue treating the project as "find the mechanism inside Gemma 2 2B that produces the construction." The deliverable is a candidate finding about SAE feature 3223 (or a small supernode around it), validated by Confirmation on the held-out D1 split and replicated on at least one other model with a Gemma-Scope-style SAE.

**Prerequisites before this path is viable:**

1. **Tier 0b closes.** Specifically: VE for Gemma Scope `gemma-scope-2b-pt-res-canonical / layer_20/width_16k/canonical` is reproduced in this codebase at or near the published number (the canonical SAE is `average_l0_71`; the published VE for this layer is roughly 0.7–0.85 in the Gemma Scope materials). External truth, not the project's own threshold, is the calibration target. My one bounded debug session produced proxy evidence the SAE is functional (L0 = 74 vs canonical 71; cosine 0.83) but couldn't reproduce a positive VE through any of the sae_lens 6.43.0 paths. The honest next move is to file the anomaly upstream and get a credentialed mech-interp reader's input on the recipe — not to keep sweeping scaling factors.

2. **The Phase 4 work gets re-run on the D1-Confirmation split**, with BH-FDR-corrected thresholds, after Tier 0b is closed. Phase 4 currently lives as Discovery, not finding.

3. **At least one of Tiers 1–6 from `pre_registration.yaml`** runs cleanly: ideally Tier 1 (specificity attack — does ablating 3223 also damage plain negation?) and Tier 4 (replication on Gemma 2 9B at the depth-matched layer).

**What you get if it lands:** a clean mechanism-level claim about how one named SAE feature in Gemma 2 2B causally gates the construction's commit. Novel (the necessity-without-sufficiency framing isn't in the prior art the way Kuznetsov 2503.03601 is). Publishable as a mech-interp note. Brittle: it's one model, one feature, and the genealogy half (Phase 6) is harder to make airtight given Gemma's lack of intermediate training checkpoints.

**What it costs:** the Tier 0b debugging is bounded but not predictable. If sae_lens 6.43.0 has a real loader bug for Gemma Scope canonicals, fixing it could require either a sae_lens issue + wait, or moving to a different library. The window between "compass fixed" and "candidate finding" is probably weeks of careful work, with 7 tiers of adversarial gauntlet that can kill at every step.

### Path B: Land the register finding cleanly

Treat the actual result as the behavioural register signature, with the mechanism work explicitly marked as "exploratory, conditional on a verification this codebase couldn't perform."

The thesis: **instruct-tuning installs a register — bold, bulleted, exemplifying, emphatic — and the "not X, but Y" construction lives natively inside that register as the C3 variant.** The supporting evidence is:

- **Phase 2 (Tier 0a confirmed):** Gemma 2 2B-it produces the construction at 1.8 % any_core sentence rate, **94 % of which is C3**. Other base models (Pythia 70M, GPT-2, Gemma 2 2B base) produce it at similar or higher rates but with **C1-dominant** composition (5–25 % C3). The variant-composition shift, not the rate change, is the signal.
- **H17 (Discovery → Confirmation passed, direction known):** in Gemma 2 2B-it, constructions cluster at the *beginning* of generations (median relative position 0.10 vs 0.50, Mann-Whitney p_BH = 0.043 on the Confirmation split). The rhetorical-opener / topic-sentence position. Consistent with the "open with an emphatic frame" register.

**To make this a defensible cross-model claim** the missing piece is replication: do other instruct models show (a) C3-dominant variant composition and (b) the H17 opener effect? If Qwen 2.5 7B Instruct and one other family (e.g. Gemma 2 9B-it for same-family scale, or Llama 3 / Mistral for different family) reproduce both, the register claim becomes a real cross-model behavioural finding. The script is at `scripts/replicate_register_finding.py`.

**What you get if it lands:** a clean, modest, true, *shareable* finding that connects to the markdown-fingerprint literature (Markdown Training Shapes LLM Prose, RLHF/quality detectability). Owes nothing to the contested interpretability layer. The kind of result a credentialed reviewer can read without having to relitigate any methodological controversy first.

**What it costs:** less ambitious than Path A. The register-claim is *descriptive* not *causal* — it tells you the construction lives inside the register, not what produces it inside the network. No product follows from it directly except possibly "if you want to deslop instruct-tuned models, target the opening register" — which is heuristic, not mechanistic.

### Path A + B (the lurking trap)

The trap, the reviewer specifically named: doing both, with the mechanism work quietly defining the project's identity while the register finding is the actual deliverable. The result is that the register finding gets undersold ("not the real result, just the warm-up") and the mechanism work gets oversold ("nearly there, just need to fix the bug"). Both are corrupted by the unresolved blocker.

If both are pursued, they must be pursued *in parallel as two separate projects with two separate writeups,* not as Acts 1 and 2 of one narrative. Path B can ship today. Path A cannot ship until the compass is fixed.

## Decision matrix

|  | A: Mechanism | B: Register |
|---|---|---|
| Status of evidence | Discovery candidate, blocked on Tier 0b | One clean candidate finding (H17) + Tier-0a-confirmed Phase 2 |
| What's left to do | Fix Tier 0b → re-run on D1-conf split → at least one of Tiers 1–6 | Cross-model replication on 2 instruct families |
| Time to ship | weeks, contingent on upstream fixes | bounded; ~1 hour of compute + a writeup |
| Risk of dying mid-process | high (Tier 0b, then 7 kill checks) | low (one quantitative replication question) |
| Reviewer surface | mech-interp; depends on Gemma Scope expertise | behavioural / NLP; broader audience |
| If it lands | novel mechanism claim, brittle | modest cross-model behavioural finding, defensible |

## What I (the agent) am NOT going to do

- Pretend that running cross-model replication forces a decision. Either path benefits from it — Path B leans on it directly; Path A would want it eventually as Tier 4. So running the replication is value-positive regardless.
- Quietly let Path A define the project while Path B languishes. The longer Path A stays blocked, the more "the project is the mechanism" becomes the unstated default, and the easier it is to spend weeks routing around the broken compass instead of shipping what works.
- Decide. The choice between A and B is a product / publication / time-allocation call, not a methodological one. Both are honest. Both can be done in either order. What's not honest is doing them both indistinctly.

## The cheapest thing you can do right now to inform the decision

Run the cross-model replication. ~1 hour of compute. If C3-dominance AND H17 opener-direction replicate on Qwen 2.5 7B Instruct (different family), the register finding is essentially shippable today as Path B. If neither replicates, Path B's window narrows. If only one replicates, you have partial cross-model evidence and the project's centre of gravity stays open.

The script (`scripts/replicate_register_finding.py`) is written and reviewed in this commit. The only thing in the way is the auto-mode classifier blocking the execution — that's a permission, not a methodology, question.
