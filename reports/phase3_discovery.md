# Phase 3 — Feature discovery on D1

**Model:** gemma-2-2b · **SAE:** gemma-scope-2b-pt-res-canonical / layer_20/width_16k/canonical (d_sae = 16384)
**Pairs:** 226 (C1=70, C2=63, C3=68, C4=25)
**Position:** last token of each sentence.  
**Stat:** per-feature `(mean(act_with − act_without) / SE)` across all pairs.

## Caveat

The pivot-aligned comparison the PRD (§5 M2) calls for is asymmetric here — the *without* paraphrases have no pivot, by construction. Phase 3 uses the **last-token** position for both sides, which captures the cumulative effect of the construction on the residual stream after the model has finished processing the sentence. This is the same convention as Marks et al. 'Sparse Feature Circuits'. The pivot-specific measurement happens in Phase 4 via M2 (P(pivot token) on `with`-style prompts that have already opened the negation).


## Top features that fire MORE on `with` (construction-recruiting)

| Rank | Feature | t-stat | mean Δ | t·C1 | t·C2 | t·C3 | t·C4 | Label |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 5371 | +24.27 | +12.676 | +18.39 | +6.37 | +21.24 | +13.37 | punctuation and formatting marks, signaling the structure of written t |
| 2 | 3877 | +18.84 | +17.486 | +50.86 | +3.46 | +38.87 | -1.49 | references to personal experiences and reflections related to relation |
| 3 | 9841 | +14.19 | +9.696 | +16.43 | +1.10 | +7.21 | +13.74 | phrases and clauses involving contrasting ideas or situations |
| 4 | 8783 | +13.58 | +7.584 | +9.51 | -1.03 | +13.43 | +5.53 | quotes and statements related to social media interactions and public  |
| 5 | 13611 | +12.22 | +9.091 | +13.09 | -1.39 | +7.93 | +13.23 | punctuation marks and significant pauses in sentences |
| 6 | 6137 | +11.51 | +6.362 | +8.28 | +0.72 | +8.27 | +6.86 | punctuation marks and their context within sentences |
| 7 | 11976 | +11.35 | +4.910 | +8.09 | +5.33 | +7.02 | +1.00 | statements or questions regarding truth and beliefs, often associated  |
| 8 | 6385 | +10.69 | +4.975 | +8.69 | +1.45 | +5.47 | +7.65 | emotional expressions and reactions in conversations |
| 9 | 7569 | +9.51 | +4.950 | +6.34 | +0.55 | +6.50 | +7.98 | events and gatherings related to community activities |
| 10 | 16308 | +9.46 | +3.683 | +3.96 | +1.26 | +7.55 | +12.53 | phrases indicating uncertainty or contrasting statements |
| 11 | 8713 | +9.09 | +4.377 | +3.87 | +2.72 | +5.75 | +14.78 | phrases that indicate addressing or overcoming challenges |
| 12 | 12184 | +8.91 | +3.243 | +7.25 | +3.50 | +2.66 | +4.70 | punctuation marks and sentence endings |
| 13 | 5890 | +8.91 | +4.306 | +2.81 | +4.93 | +5.63 | +6.46 | technical terms and specifications in a structured context |
| 14 | 15295 | +8.86 | +5.574 | +3.04 | -0.07 | +7.28 | +18.47 | markers indicating the beginning of text segments |
| 15 | 9476 | +8.53 | +6.280 | +4.98 | +2.70 | +5.41 | +14.42 | phrases related to personal experiences and expressing clarity in comm |

## Top features that fire MORE on `without` (paraphrase-recruiting)

| Rank | Feature | t-stat | mean Δ | t·C1 | t·C2 | t·C3 | t·C4 | Label |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 3971 | -16.46 | -7.237 | -6.40 | -6.16 | -17.03 | -8.09 | punctuation marks at the end of sentences |
| 2 | 14338 | -15.58 | -7.745 | -8.52 | -8.87 | -8.61 | -4.95 | punctuation marks and sentence endings |
| 3 | 9768 | -13.82 | -3.357 | -8.35 | -4.15 | -9.95 | -7.02 | terms related to control and authority, particularly in political or s |
| 4 | 1264 | -13.14 | -10.649 | -14.04 | +3.17 | -11.53 | -13.57 | quantities and metrics related to various topics |
| 5 | 1692 | -11.04 | -4.721 | -3.80 | -9.76 | -8.59 | -1.56 | legal and technical terminology related to statutes and inventions |
| 6 | 8366 | -10.46 | -4.257 | -7.29 | -3.88 | -5.91 | -3.47 | verbs and their related forms, often related to medical or technical c |
| 7 | 12076 | -10.28 | -6.368 | -4.66 | -7.22 | -4.98 | -4.61 | conjunctions and transitions in arguments |
| 8 | 11135 | -9.11 | -4.819 | -6.07 | -2.40 | -5.88 | -3.72 | conditional and speculative phrases |
| 9 | 12265 | -8.54 | -4.900 | -5.44 | -1.82 | -4.36 | -9.25 | sentences and punctuation |
| 10 | 9909 | -8.12 | -6.146 | -0.97 | -9.48 | -4.74 | -1.55 | references to digital technology and online interactions |
| 11 | 6868 | -7.78 | -3.773 | -3.77 | -5.72 | -4.34 | -1.71 | technical details related to ability and functions associated with equ |
| 12 | 2230 | -7.77 | -2.949 | -3.51 | -4.20 | -4.46 | -3.35 | references to causality and violation of rules or expectations |
| 13 | 2914 | -7.48 | -4.492 | -4.41 | -0.42 | -5.18 | -4.13 | emotional expressions and descriptions |
| 14 | 10836 | -7.21 | -2.541 | -4.08 | -5.45 | -1.73 | -3.02 | punctuation marks and sentence-ending indicators |
| 15 | 6631 | -7.01 | -6.472 | -5.71 | -0.71 | -3.57 | -4.50 | the beginning of a text or important markers in a document |

## Phase 3 kill check

**PASS** — top feature t-stat = +24.27, positive across C1/C2/C3. Candidate for Phase 4 causal validation.