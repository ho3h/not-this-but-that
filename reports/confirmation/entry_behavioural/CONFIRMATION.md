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