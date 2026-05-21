# Operating Protocol — the Discovery/Confirmation firewall

*Read this before every session. The discipline below is the only thing that lets you cast many ships without drowning in your own false positives.*

---

## §0 The three sentences

1. **Discovery is contaminated by definition.** Anything that comes out of a Discovery campaign is *a candidate, not a finding.* Never quote it as a finding. Never let it justify a product. Treat the ranked list of survivors the way a chemist treats a positive screen — it tells you *where to look,* not *what's true.*

2. **Confirmation is held-out, pre-registered, and multiplicity-corrected.** A candidate becomes a *candidate finding* only after it survives a test that (a) runs on data Discovery never touched, (b) was pre-registered before the test ran, and (c) was corrected for how many hypotheses were tried in Discovery. The pre-registered seven-tier adversarial gauntlet at [`pre_registration.yaml`](../pre_registration.yaml) is that test.

3. **Product is hypothetical until replication.** A candidate finding becomes *product-eligible* only after independent replication (Tier 4 cross-model, or equivalent). Wanting a product *biases which ships you call seaworthy* — so the product stays hypothetical, and the agent is forbidden from optimizing toward it, until replication lands.

If any of those three slip, the false-positive amplifier turns back on. The discipline is the only thing protecting the work from itself.

---

## §1 Why this isn't breadth-vs-depth

The instinct to cast many ships is correct. Letting the ships' survival counts as evidence is the mistake.

Concretely: if you test 30 SAE features for "necessity at the pivot" at p < 0.05, ~1.5 will pass by chance alone with nothing underneath. An autonomous agent whose job is *"find what works and report it"* is structurally a machine for surfacing those false positives and handing them to you dressed in convincing language. Without a firewall, more hypotheses → more guaranteed false positives, presented with high confidence.

With the firewall, the same casting is fine — even encouraged. Discovery can be as broad as compute allows, because Confirmation will only let a fraction through, with a bar that scales to how many ships were cast.

---

## §2 The firewall

### §2.1 Data splits (physical separation)

The corpus is split *once*, *before* any Discovery campaign reads it, and the split file is committed. The split is deterministic (seeded) so it can be re-derived but not re-rolled.

| Corpus | Discovery | Confirmation | Status |
|---|---:|---:|---|
| D1 contrast pairs (226 total) | 170 | 56 | committed at `data/splits/d1.json` |
| D2 neutral prompts (102 total) | 76 | 26 | committed at `data/splits/d2.json` |
| D3 fluency text | (full) | (full) | shared; quality preservation only |

The Confirmation split is **never read by any Discovery script.** Enforcement is by convention (`data/splits/firewall.py` returns the Discovery indices by default; Confirmation indices require an explicit `phase='confirmation'` argument that scripts only pass in Confirmation campaigns) plus by git review — Discovery PRs that touch Confirmation paths get bounced.

### §2.2 Reframing the work that has already happened

Everything done before this protocol was committed (Phases 0–7 and Tier 0 of the adversarial pre-registration) used the *full* D1/D2 corpora. Per the protocol, **those results are now relabelled as Discovery candidates, not findings.** The reframing:

| Prior label | New label |
|---|---|
| "Feature 3223 is causally necessary at the pivot commit (Phase 4)" | "Feature 3223 is a Discovery candidate for the pivot-commit-gate hypothesis; status: untested on the Confirmation split." |
| "Genealogy: 1.81× larger ablation drop in instruct (Phase 6)" | "Discovery candidate; further conditional on the open VE-measurement issue (Tier 0b)." |
| "M1 variant-composition shift (Phase 2)" | "Stands — this is behavioural and was independently re-validated on a blind set in Tier 0a. Treat as confirmed-by-Tier-0a; Tier 0a's blind set was the de facto held-out split for the classifier." |
| Phase 7 deslop null | "Stands as a null; nulls don't need confirmation (you can't false-positive a non-result)." |

So one prior finding (M1 variant-composition) survives as confirmed-by-Tier-0a. Everything mechanistic (Phase 3–6) is now Discovery, awaiting Confirmation.

### §2.3 Discovery output rules

Every Discovery campaign writes a `reports/discovery/<campaign>/CANDIDATES.md` file. Its top section must contain, verbatim:

```
> ⚠ DISCOVERY OUTPUT — CONTAMINATED BY DEFINITION
> These candidates are a ranked list. None of them is a finding.
> Confirmation runs against pre_registration.yaml on the held-out
> Confirmation split, with the FDR threshold computed from the
> number of hypotheses below.
> Number of hypotheses tried in this campaign: N
```

Every individual candidate is labelled `[CANDIDATE]`. Never `[FOUND]`, `[CONFIRMED]`, `[WORKS]`.

The campaign records its full hypothesis count `N` (every hypothesis seriously tested, not just the survivors). Confirmation reads `N` to compute the corrected threshold.

### §2.4 Confirmation rules

Confirmation = the seven-tier adversarial gauntlet at [`pre_registration.yaml`](../pre_registration.yaml). It runs *only* against candidates that came from a Discovery campaign, *only* on the Confirmation data split, and *only* against pre-registered thresholds.

Multiplicity correction: **Benjamini–Hochberg FDR at q = 0.10** across all candidates Confirmation tests in the same campaign. If Discovery's number of hypotheses was *N*, and Confirmation tests *K* candidates, the per-test BH-corrected significance threshold is `α_i = (i/K) · q` for the *i*-th-ranked p-value. Candidates whose Tier-2 / Tier-3 statistics fall above this are *not* confirmed; the tier kill threshold is the union of `pre_registration.yaml`'s pass condition AND the FDR-corrected p-value.

The pre-registration must include — *before* Confirmation runs — the value of `N` from Discovery and the candidates' p-values, with the corrected α_i annotated alongside each.

### §2.5 What's allowed during Discovery

- Casting many ships: any number of features, layers, intervention types, prompt subsets — provided every one is labelled `[CANDIDATE]` and counted in `N`.
- Re-using the Discovery split, repeatedly.
- "Cheating" — biasing the search toward likely positives. The whole point of Discovery is to find candidates; bias is fine because Confirmation kills the false positives at scale.

### §2.6 What's forbidden

- Reading the Confirmation split from any script under `reports/discovery/` or any campaign-runner script.
- Quoting Discovery output as a finding in any writeup, README, or external communication.
- Adjusting `pre_registration.yaml`'s thresholds *after* Confirmation has run, or after a candidate's p-value is known.
- Running Confirmation without pre-registering the candidate's p-value and the corrected threshold first.
- Building product on a candidate that hasn't replicated cross-model (Tier 4) at minimum.

---

## §3 The prerequisite that gates Discovery

**Tier 0b (VE reconstruction) must close before any Discovery campaign casts ships at SAE-dependent hypotheses.** The instrument is currently broken; sailing thirty new ships from a boat with a broken compass would inherit the broken measurement on all thirty. Per the prime directive and the bounded-fork outcome: fix the compass against published external truth first.

Non-SAE Discovery campaigns (e.g. behavioural-only, prompt-engineering-only) are NOT gated by Tier 0b. M1-based hypotheses can proceed.

---

## §4 The first Discovery campaign — the entry-mode question

Phase 7 surfaced the real frontier and the real product question:

> **What makes the model decide to *enter* the "not X, but Y" construction in the first place?**

Phase 4 showed feature 3223 gates the *commit* — the second decision, the one that completes the pivot once the contrast is already open. Phase 7 confirmed the *entry* — the upstream decision to emit "not" in contrast-context — lives elsewhere. We don't yet know where. That's exactly the kind of unexplored frontier breadth deserves.

This campaign is **deferred until Tier 0b closes** (every hypothesis in it is SAE-dependent). Pre-spec below so the agent can run it as soon as the compass is fixed.

### §4.1 The hypothesis space (~25 ships)

Each is a testable claim about where the entry decision lives. Each goes in `reports/discovery/entry_gate/CANDIDATES.md` with `[CANDIDATE]` labels.

**A. Feature-level (~10 hypotheses):**

1. Per-feature attribution to P(emit "not") at the position *just before* "not" in construction-entry contexts. Top-K candidates.
2. Same attribution at the position *two tokens before* "not" — the construction decision may commit earlier than the immediately-preceding token.
3. Same at three tokens before.
4. Features whose ablation reduces P(emit "not") in construction-entry contexts but NOT in plain-negation-entry contexts. Specificity-by-design.
5. Features that fire ON the "not" token itself in construction usage but not in plain negation.
6. Features that fire on the subject-copula prefix ("It's", "She's", "They're") of construction-entry contexts but not on the same prefix in non-construction usage.
7. Features whose differential activation between construction-entry and plain-negation contexts is largest, at any pre-pivot position.
8. Features at the position emitting the SUBJECT of construction sentences (the "It" in "It's not just X"), if entry is decided that early.
9. The supernode formed by jointly ablating the top-3 features from (1) — does it produce a larger drop than any one?
10. Decoder-cosine neighbours of feature 3223 — does the entry decision recruit features semantically adjacent to the commit gate?

**B. Layer / depth (~5 hypotheses):**

11. Layer sweep of (1): is entry decided earlier in the network (L8–L12), at the same depth as commit (L20), or both?
12. Feature paths across layers: does an early-layer feature predict P(emit "not") downstream?
13. Attention heads at L20 looking at construction-priming tokens — which heads, what do they attend to?
14. Same head analysis at the layer where entry-attribution peaks.
15. Cross-layer logit-lens on the "not" token: at what layer does its probability spike?

**C. Context / prompt (~5 hypotheses):**

16. Are D2 prompts that elicit constructions lexically different from those that don't? Build a prompt-feature classifier.
17. Are there specific token bigrams in the prompt that precede construction-entry generations?
18. Does Instructions chat-template inject anything that biases entry?
19. Generate continuations under chat-template stripped — does construction rate drop?
20. Construction-entry rate as a function of prompt length / topic / register.

**D. Comparative (~5 hypotheses):**

21. Run Discovery on Pythia 70M's entry decision — does it have an analogue?
22. Same on Gemma 2 2B base vs Gemma 2 2B-it: is the entry-gate feature instruct-specific?
23. Compare entry-gate features to commit-gate (3223) — orthogonal? co-active? upstream?
24. Compare to the prior project's suppression features (15596, 10142): does suppressing them at entry-time amplify construction emission?
25. Probe whether the entry decision shows up in attention patterns earlier than in SAE features.

Twenty-five ships, of which Confirmation will allow ≤ ~5 through after BH-FDR correction at q=0.10. That's the budget; the agent is to optimize Discovery aggressively against it without ever crossing the firewall.

### §4.2 What "Confirmation" of an entry-gate candidate looks like

A surviving candidate from §4.1 goes to a Confirmation run that:
1. Reads from `data/splits/d1.json` (Confirmation split — 56 contrast pairs).
2. Runs the relevant tiers from `pre_registration.yaml`. For an entry-gate feature, the relevant tests are Tier 1 (specificity attack — does ablating it also damage plain negation?), Tier 2 (matched-activation null), Tier 3 (interchange-patching for sufficiency), Tier 6 (red-team on a random feature).
3. Applies BH-FDR with `N = 25` over all candidates being tested in the campaign.
4. Records pass / kill against `α_i` AND the pre-registered tier threshold; a candidate must pass BOTH.

If any candidate passes Confirmation, it's a *candidate finding* — still not a product foundation. Replication on a second model family (Tier 4) is the additional barrier to product-eligibility.

---

## §5 What's NOT a Discovery campaign

To avoid the failure mode "everything is exploration, nothing is rigorous," some work is explicitly *not* Discovery and doesn't go through the firewall.

- **Bug fixes.** Tier 0b (VE measurement) is a calibration fix, not a hypothesis. Its pass condition is "matches published number for this SAE," not "passes a threshold I chose."
- **Behavioural classifier work.** The classifier is an instrument, not a hypothesis about model internals. Validating it (Tier 0a) is calibration. Its kill threshold doesn't go through FDR.
- **Methodology audits.** Reading the literature, talking to credentialed reviewers, filing sae_lens issues — these aren't experiments and don't produce candidates.

---

## §6 Reporting honesty

Every writeup that cites a result from this repo must:

1. Label each cited number as *Discovery candidate / Tier-N confirmed / replicated*.
2. State the N of hypotheses Discovery tested before this number emerged.
3. State the FDR threshold the Confirmation step used.
4. NOT quote a Discovery candidate as a finding under any framing. The closest allowed phrasing is "Discovery surfaced X as a candidate; it has not yet been Confirmation-tested."

The honest version of "we found that feature 3223 is causally necessary" is, *after the entry-campaign and a re-run of the seven-tier gauntlet on the Confirmation split:*

> Discovery (tested N = 25 features for pivot-commit causality) surfaced feature 3223. Confirmation on a held-out 56-pair split passed Tier 2 (matched-activation control, p_BH = 0.013) and Tier 1 (construction-specificity ratio = 4.2, kill threshold 2.0). Tier 3 (sufficiency) killed. Tier 4 replication: not yet run.

That sentence is publishable. *"Feature 3223 is causally necessary"* — by itself — currently is not.

---

## §7 What this protects against, in your own words

- "Cast many ships, agent finds what works, build a product" → **with no firewall, this is a false-positive amplifier with a budget.** With the firewall, it's exploration.
- "I'm blocked on the VE bug, let me open thirty new fronts" → **avoidance wearing ambition's clothes.** The firewall makes this visible: any new Discovery campaign that's SAE-dependent is gated until Tier 0b closes.
- "Phase 4 found feature 3223 is necessary" → **a Discovery result quoted as a finding.** The firewall reframes it as a candidate that hasn't been Confirmation-tested. The pre-registered gauntlet exists to test it; running the gauntlet on a held-out split is what makes it a finding.
- "Build a product around the de-slop tool" → **already falsified by Phase 7.** Product stays hypothetical.

The firewall isn't a slowdown. It's the thing that lets you go faster *honestly.*

---

## §8 The agent's operating contract

When you start a session on this repo:

1. **Read this file.** Then [`pre_registration.yaml`](../pre_registration.yaml). Then [`HANDOVER.md`](../HANDOVER.md). In that order.
2. **Check the foundation cracks.** Is Tier 0b closed? If not, you cannot start any SAE-dependent Discovery campaign. Work that is not SAE-dependent is allowed.
3. **Identify the phase you're in.** Discovery → broadly explore, label everything `[CANDIDATE]`, count hypotheses. Confirmation → run only what's pre-registered, on the Confirmation split, with FDR. Replication → another model family, full battery.
4. **If a Confirmation tier kills, stop and report.** The prime directive in `pre_registration.yaml` covers this; surfacing kills is the safety mechanism.
5. **If you're tempted to ship a product around a candidate**, re-read §0 and §6. The temptation IS the bias the firewall exists to catch.

Surviving every kill check is the only evidence. Not-dying-yet is not the same as proven. Proven is not the same as product-eligible.
