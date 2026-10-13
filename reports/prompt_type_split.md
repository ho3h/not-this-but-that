# Construction rate by prompt type (exploratory, post hoc)

Baseline generations only (no intervention). Hit = negated family ∪ "more than just" cousin.
Prompt labels were assigned from wording alone by an annotator who saw only the shuffled
prompt list, never the generations or hit counts (`data/prompt_type_labels.tsv`).
M = asks what something means or why it matters; P = asks for a process, a scene or facts.

| set | type | prompts | answers | answers with hit | rate | prompts with ≥1 hit |
|---|---|---:|---:|---:|---:|---:|
| D2 (Q5c baseline) | M | 83 | 249 | 25 | 10.0% | 17 |
| D2 (Q5c baseline) | P | 19 | 57 | 3 | 5.3% | 3 |
| D2b (confirmation baseline) | M | 10 | 60 | 4 | 6.7% | 4 |
| D2b (confirmation baseline) | P | 40 | 240 | 1 | 0.4% | 1 |
| **pooled** | **M** | 93 | 309 | 29 | **9.4%** | 21 |
| **pooled** | **P** | 59 | 297 | 4 | **1.3%** | 4 |

Rate ratio M/P: **7.0×**
(95% prompt-clustered bootstrap CI 2.8× to 30.2×).
Prompt-level one-sided Fisher exact test (any hit vs none): p = 0.0077.

The sets differ in their mix (D2 is mostly M; D2b, written later on concrete topics, is mostly P),
so part of the pooled gap is a difference between sets. Stratified by set, the Mantel-Haenszel
rate ratio is **3.0×**; the direction holds within each set.

Caveat: this split was proposed after looking at which prompts elicited the construction.
It is a lead for a pre-registered test, not a confirmed finding.
