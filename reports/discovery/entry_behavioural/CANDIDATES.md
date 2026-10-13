> ⚠ DISCOVERY OUTPUT — CONTAMINATED BY DEFINITION
> These candidates are a ranked list. None of them is a finding.
> Confirmation runs against `pre_registration.yaml` on the held-out
> Confirmation split, with the FDR threshold computed from the
> number of hypotheses below.
> Number of hypotheses tried in this campaign: **19**

# Behavioural entry-gate Discovery campaign

**Question:** What text-level features predict construction-entry in Gemma 2 2B-it generations?

**Data:** Phase 2 generations, D2-Discovery split only (76 prompts × 5 seeds = 380 generations → 726 sentences). D2-Confirmation split (26 prompts) **NEVER read by this script** (see `src/firewall/__init__.py`).

**Outcome:** binary `has_construction` per sentence (11 positive / 715 negative = 1.52% base rate).

**Discovery filter:** raw p ≤ 0.20 → marked [CANDIDATE]. Confirmation will apply Benjamini-Hochberg FDR at q = 0.10 across all 19 hypotheses; the per-test BH α floor is `(i/19) · 0.10` for the i-th ranked p-value.

## Ranked hypotheses

| Rank | ID | Hypothesis | Test | Effect | n cases | p (raw) | α_BH (i/N · q) | Discovery filter |
|---:|:---|:---|:---|:---|---:|---:|---:|:---|
| 1 | H09 | Sentence contains 'just' / 'merely' / 'simply' / 'only' | Fisher exact (greater) | rate true: 20.37%, false: 0.00%, OR=inf | 54 | 0.0000 | 0.0053 | **[CANDIDATE]** |
| 2 | H19 | Sentence contains a quotation mark | Fisher exact (greater) | rate true: 4.21%, false: 0.00%, OR=inf | 261 | 0.0000 | 0.0105 | **[CANDIDATE]** |
| 3 | H07 | Sentence starts with a subject-copula contraction (It's / That's / …) | Fisher exact (greater) | rate true: 11.36%, false: 0.88%, OR=14.44 | 44 | 0.0002 | 0.0158 | **[CANDIDATE]** |
| 4 | H08 | Sentence starts with a capital pronoun (It / This / They / We) | Fisher exact (greater) | rate true: 5.00%, false: 0.68%, OR=7.66 | 140 | 0.0014 | 0.0211 | **[CANDIDATE]** |
| 5 | H16 | Generation length in sentences (Mann-Whitney U) | Mann-Whitney U (two-sided) | median pos: 6.00, median neg: 5.00 | 11/715 | 0.0057 | 0.0263 | **[CANDIDATE]** |
| 6 | H03 | Sentence is in the MIDDLE of its generation | Fisher exact (greater) | rate true: 2.33%, false: 0.59%, OR=4.01 | 387 | 0.0509 | 0.0316 | **[CANDIDATE]** |
| 7 | H17 | Relative sentence position 0-1 (Mann-Whitney U) | Mann-Whitney U (two-sided) | median pos: 0.25, median neg: 0.50 | 11/715 | 0.0893 | 0.0368 | **[CANDIDATE]** |
| 8 | H12 | Prompt starts with 'Discuss' | Fisher exact (greater) | rate true: 2.53%, false: 1.23%, OR=2.08 | 158 | 0.2008 | 0.0421 | (below filter) |
| 9 | H15 | Prompt length in tokens (Mann-Whitney U) | Mann-Whitney U (two-sided) | median pos: 10.00, median neg: 9.00 | 11/715 | 0.3203 | 0.0474 | (below filter) |
| 10 | H11 | Prompt starts with 'Describe' | Fisher exact (greater) | rate true: 1.88%, false: 1.41%, OR=1.33 | 160 | 0.4505 | 0.0526 | (below filter) |
| 11 | H14 | Prompt starts with 'Reflect' | Fisher exact (greater) | rate true: 1.42%, false: 1.54%, OR=0.92 | 141 | 0.6625 | 0.0579 | (below filter) |
| 12 | H13 | Prompt starts with 'Walk' | Fisher exact (greater) | rate true: 1.28%, false: 1.54%, OR=0.83 | 78 | 0.7162 | 0.0632 | (below filter) |
| 13 | H04 | Sentence length in tokens (Mann-Whitney U) | Mann-Whitney U (two-sided) | median pos: 20.00, median neg: 20.00 | 11/715 | 0.7780 | 0.0684 | (below filter) |
| 14 | H01 | Sentence is the FIRST in its generation | Fisher exact (greater) | rate true: 0.57%, false: 1.81%, OR=0.31 | 175 | 0.9530 | 0.0737 | (below filter) |
| 15 | H02 | Sentence is the LAST in its generation | Fisher exact (greater) | rate true: 0.57%, false: 1.81%, OR=0.31 | 175 | 0.9530 | 0.0789 | (below filter) |
| 16 | H06 | Sentence contains bold-emphasis markup (**) | Fisher exact (greater) | rate true: 0.42%, false: 3.56%, OR=0.12 | 473 | 0.9998 | 0.0842 | (below filter) |
| 17 | H05 | Sentence appears inside a bulleted/structured list | Fisher exact (greater) | rate true: 0.00%, false: 2.70%, OR=0.00 | 318 | 1.0000 | 0.0895 | (below filter) |
| 18 | H10 | Prompt starts with 'Explain' | Fisher exact (greater) | rate true: 0.00%, false: 1.76%, OR=0.00 | 101 | 1.0000 | 0.0947 | (below filter) |
| 19 | H18 | Sentence contains an em-dash / en-dash | Fisher exact (greater) | rate true: 0.00%, false: 1.53%, OR=0.00 | 7 | 1.0000 | 0.1000 | (below filter) |

## Discovery filter survivors: 7 of 19

These are the **[CANDIDATE]** hypotheses Confirmation should test on the D2-Confirmation split. Confirmation passes a candidate ONLY IF its p-value on the Confirmation split is below its BH-FDR-corrected α (the column above), AND the candidate's effect direction matches Discovery's.

**The candidates below are not findings. They are leads.** A `[CANDIDATE]` is what survived the Discovery filter on contaminated data; it's a hypothesis worth Confirmation-testing, not evidence of anything.

- **[CANDIDATE] H09** — Sentence contains 'just' / 'merely' / 'simply' / 'only' (raw p = 0.0000)
- **[CANDIDATE] H19** — Sentence contains a quotation mark (raw p = 0.0000)
- **[CANDIDATE] H07** — Sentence starts with a subject-copula contraction (It's / That's / …) (raw p = 0.0002)
- **[CANDIDATE] H08** — Sentence starts with a capital pronoun (It / This / They / We) (raw p = 0.0014)
- **[CANDIDATE] H16** — Generation length in sentences (Mann-Whitney U) (raw p = 0.0057)
- **[CANDIDATE] H03** — Sentence is in the MIDDLE of its generation (raw p = 0.0509)
- **[CANDIDATE] H17** — Relative sentence position 0-1 (Mann-Whitney U) (raw p = 0.0893)

## Confounds you'd be irresponsible to ignore

Two of the surviving candidates are at least partly **circular with the classifier's detection lexicon**, which means they're correlated with construction-presence partly *by construction* of the outcome label, not because they predict the model's decision:

- **H09** (contains *just / merely / simply / only*) — these are the exact lexical hinges the classifier looks for in C3 (`_C3 = … (just|merely|simply) …`). A sentence containing one is *mechanically* more likely to be flagged as a construction. The p-value is real, but the candidate is downgraded to *measuring classifier definition, not entry decision*. Confirmation should still run it, but a Confirmation pass doesn't mean it predicts entry — it means C3's lexicon shows up in C3-positive sentences, which is tautological.

- **H07** (starts with subject-copula contraction *It's / That's /* …) — C1's regex opener is `(it'?s|that'?s|this is|…)\s+not`. A sentence starting with one of these has more *opportunity* to be classified as C1, though it must still contain the construction's structural pivot (`, it's` or `, but`) to be flagged. Less circular than H09 but worth flagging.

- **H08** (starts with capital pronoun) — superset of H07; same caveat, broader.

**The genuinely independent candidates are H03, H16, H17, H19** — sentence position, generation length, relative position, presence of quotation marks. These do not overlap with the classifier's lexicon, so a Confirmation pass on these would mean something real about the entry decision.

## What's next (for the agent picking this up)

1. **Do not quote any candidate above as a finding.** Per the operating protocol §6, the closest allowed phrasing is *"Discovery surfaced X as a candidate; it has not yet been Confirmation-tested."*
2. **Confirmation runs on the D2-Confirmation split** (26 prompts, loaded via `firewall.load_d2(phase='confirmation')`). The Phase 2 generations for those 26 prompts already exist in `reports/phase2_generations_gemma_2b_it.jsonl` — but if you regenerate, use the same sampling params (temperature=0.8, top_p=0.95, max_new_tokens=150, seeds 0-4) so the Confirmation distribution matches Discovery's.
3. **The Confirmation test for each candidate** is the same statistic Discovery used, run on the Confirmation sentences only, with the BH α threshold from the column above. A candidate passes Confirmation only if p_conf ≤ α_BH AND the effect direction matches Discovery's.
4. **Confirmation should explicitly drop or down-weight the confounded candidates (H07, H08, H09).** A circular candidate passing Confirmation is not informative; it would just be measuring the classifier on a new sample.
5. **If no clean candidate survives Confirmation, that is itself reportable.** It means text-level predictors of construction-entry don't replicate out-of-sample at the strict FDR level — entry is harder to predict behaviourally than the variant composition (Phase 2) was. That's a real result, not a failure.