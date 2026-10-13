# Confirmation — entry_behavioural campaign

**Discovery campaign:** `reports/discovery/entry_behavioural/CANDIDATES.json` (N = 19 hypotheses tried, K = 7 survived Discovery filter).
**Confirmation data:** D2-Confirmation split — 313 sentences (8 positive, 305 negative). Loaded via `firewall.load_d2(phase='confirmation')`; never read by Discovery.
**Multiplicity correction:** Benjamini–Hochberg FDR at q = 0.10 over K = 7. Per-test α_i = (i/K)·q for the i-th-ranked Confirmation p-value.

## Per-candidate Confirmation verdict

| Rank | ID | Description | Conf p | α_BH | Passes FDR | Direction match | Confounded | Verdict |
|---:|:---|:---|---:|---:|:---:|:---:|:---:|:---|
| 1 | H09 | Sentence contains 'just' / 'merely' / 'simply' / 'only' | 0.0000 | 0.0143 | ✓ | ✓ | ⚠ | passes FDR but CONFOUNDED — not a clean confirmation |
| 2 | H19 | Sentence contains a quotation mark | 0.0002 | 0.0286 | ✓ | ✓ |   | **CONFIRMED** |
| 3 | H17 | Relative sentence position 0-1 (Mann-Whitney U) | 0.0283 | 0.0429 | ✓ | ✓ |   | **CONFIRMED** |
| 4 | H07 | Sentence starts with a subject-copula contraction (It's / That's / …) | 0.0862 | 0.0571 | ✗ | ✓ | ⚠ | KILL |
| 5 | H08 | Sentence starts with a capital pronoun (It / This / They / We) | 0.2632 | 0.0714 | ✗ | ✓ | ⚠ | KILL |
| 6 | H03 | Sentence is in the MIDDLE of its generation | 0.6966 | 0.0857 | ✗ | ✗ |   | KILL |
| 7 | H16 | Generation length in sentences (Mann-Whitney U) | 0.8269 | 0.1000 | ✗ | ✗ |   | KILL |

## Verdict

**2 clean confirmation(s):**
- **H19** — Sentence contains a quotation mark (Conf p = 0.0002, α_BH = 0.0286)
- **H17** — Relative sentence position 0-1 (Mann-Whitney U) (Conf p = 0.0283, α_BH = 0.0429)

These survive the discovery → confirmation → FDR pipeline and are NOT confounded with the classifier's lexicon. They are **candidate findings**: text-level features that predict construction-entry in Gemma 2 2B-it on held-out data, with the FDR-corrected significance threshold applied.

**1 candidate(s) passed FDR but with a confound caveat:**
- H09 — Sentence contains 'just' / 'merely' / 'simply' / 'only' (passes FDR at α = 0.0143, but confounded with classifier lexicon)

These are not clean confirmations. A Confirmation pass on a classifier-lexicon confound just means the classifier on the Confirmation split agrees with itself, which is tautological.

## What this means for the larger thesis

- **The variant-composition shift (Phase 2) stays as confirmed-by-Tier-0a** (94 % C3 in instruct, non-overlapping CIs). That claim is the most robust thing in the repo and survives this Discovery/Confirmation pass.
- **The entry-gate question moves to the SAE-level Discovery campaign** (operating_protocol.md §4), which remains blocked on the Tier 0b VE-measurement issue.
- **The pivot-commit-gate work (feature 3223 etc.) stays as Discovery** until it can be re-run on the D1-Confirmation split with the FDR threshold its own hypothesis count requires.
---

## RETRACTION (added after second look) — H19 was circular too

Reviewer pushed back: was the construction appearing *inside* the quoted span, the way 'just/merely/simply' was *inside* the classifier's C3 lexicon?

I checked. Worse: my H19 detection was sloppier than that. The "quote mark" predicate I implemented was `"'" in s or "\"" in s or "“" in s` — and the `"'"` character matched **the apostrophe inside contractions** (`It's`, `isn't`, `aren't`), not actual quotation marks. Constructions are saturated with contracted copulas because the classifier's C1/C3 regex requires them. So H19 was detecting "uses contractions" by accident, which is just another way of detecting C1/C3.

When I recompute H19 with *real* quotation marks only (`"`, `"`, `"`, leading `'` that isn't inside a word):

- Sentences with construction: 19
- Sentences with *real* quotation marks: 20
- Sentences with BOTH: **0**

H19 collapses to a null. It is not a finding; it was a measurement bug, in the same family as H07 / H08 / H09 — all of them are partly or fully circular with the classifier's surface lexicon.

**Updated verdict — clean confirmations:** 1 of 7, not 2.

| Candidate | Status |
|---|---|
| H09 (just/merely/simply lexicon) | confounded by Discovery's own admission |
| H07 (subject-copula opener) | confounded + Conf KILL |
| H08 (capital pronoun opener) | confounded + Conf KILL |
| **H19 (quote mark)** | **RETRACTED — apostrophe artifact, same family** |
| H03 (middle position) | Conf KILL |
| H16 (generation length) | Conf KILL |
| **H17 (relative position)** | **clean confirmation, direction known** |

## H17 direction: constructions cluster at the BEGINNING

The reviewer floated two interpretations: late = rhetorical kicker (fits register story), diffuse = noise. The data says **neither** — constructions cluster *early*:

| Split | n positive | median pos w/ construction | median pos w/o |
|---|---:|---:|---:|
| Discovery | 11 | 0.250 | 0.500 |
| Confirmation | 8 | 0.100 | 0.500 |

(Relative position 0.0 = first sentence; 1.0 = last sentence.)

Constructions in Gemma-2-2b-it generations appear ~2–5× more often in the first half of the output, with the strongest concentration at the very first sentence. Direction is consistent across Discovery and Confirmation. This is the **rhetorical opener / topic-sentence** position — the "let me open with an emphatic frame" move. Consistent with the register story (instruct-tuning installs an emphatic opening style; C3 is its canonical lexical realisation) but in the *opposite direction* from the reviewer's prior guess. The register story survives; the specific within-register location is "opener," not "closer."

## What this leaves the campaign with

- **One clean candidate finding:** H17 — Gemma-2-2b-it constructions cluster at the beginning of generations.
- **One direction-known interpretation:** opener, not closer; topic-sentence move.
- **Zero other Discovery candidates that aren't confounded with the classifier's lexicon.**
- **A useful retraction recipe.** When a Discovery feature lights up strongly, the next check is: does my detector overlap with the outcome label's detection? H19 should have been caught before Confirmation; it took a reviewer's push to catch it. Worth adding "before announcing a Discovery survivor, check whether the predicate uses any character / lexicon also used by the outcome detector" to the operating protocol's Discovery rules.

