# Collateral syntax under top-25 coalition ablation

The coalition was selected for its causal effect on P(pivot), whose
token set includes the comma, " but", " it" and dash tokens. This
table reports what the ablation does to contrastive/clausal syntax
*overall* — not just to the construction — on the committed Q5b/Q5c
generations. Per-generation means; share = fraction of generations
containing the word at least once.

## Q5b primed (n=300)

| measure | baseline | ablated | change |
|---|---:|---:|---:|
| share w/ "but" | 75.0% | 1.7% | -98% |
| share w/ "however" | 1.3% | 0.3% | -75% |
| share w/ "instead" | 1.3% | 2.3% | +75% |
| "but" per gen | 0.90 | 0.02 | -98% |
| commas per gen | 2.01 | 0.63 | -69% |
| em/en-dashes per gen | 0.00 | 0.00 | +0% |
| negations per gen | 0.49 | 0.32 | -36% |
| ",-pivot" bigrams per gen | 1.00 | 0.06 | -94% |
| words per gen | 33 | 30 | -10% |
| sentences per gen | 3.2 | 3.2 | -1% |

## Q5c neutral (n=306)

| measure | baseline | ablated | change |
|---|---:|---:|---:|
| share w/ "but" | 13.4% | 1.3% | -90% |
| share w/ "however" | 2.9% | 0.3% | -89% |
| share w/ "instead" | 0.0% | 0.3% | +0% |
| "but" per gen | 0.14 | 0.01 | -90% |
| commas per gen | 1.52 | 0.58 | -62% |
| em/en-dashes per gen | 0.00 | 0.00 | +0% |
| negations per gen | 0.11 | 0.08 | -24% |
| ",-pivot" bigrams per gen | 0.11 | 0.00 | -97% |
| words per gen | 32 | 32 | -2% |
| sentences per gen | 2.4 | 2.7 | +12% |
