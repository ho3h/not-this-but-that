# Pre-registered confirmation v5 (confirmation_v5_gemma3): verdict

Analysed once. 600 triples on 120 everyday prompts (six genres), 100 on 20 explicit-request prompts.

## H1 (primary): habit with the 25 off: **PASS**

156 → 27 (+83% drop), mid-p 2.1e-32; 95% prompt-clustered CI 73% to 91%.

## H2 (specificity): matched control: **PASS**

Matched 25 off: 156 → 143 (+8% drop), mid-p 0.23. Gate: the 25's drop exceeds control drop by ≥ 40 points.

## H3 (cost): **PASS**

- ARC-Easy (300): normal 88.7% · 25 off 88.3% · matched 88.7% (gate: drop ≤ 3 points) ✓
- WikiText-2 perplexity ratio: 25 off 1.003× · matched 1.005× (gate ≤ 1.1) ✓
- "a" before a vowel: normal 1.7% · 25 off 1.3% (gate ≤ 2%) ✓
- Blind preference, 25 off vs normal (120 pairs): 25 off wins 45, ties 6, normal wins 69; share 40% (gate ≥ 40%) ✓
- 'No' questions right (of 30): normal 28 · 25 off 28 · matched 28 (gate: lose ≤ 2) ✓

## H4 (warnings stay clear): **FAIL**

Blind pairwise on 60 answers to 30 fresh "what should you never do" prompts: 25 off wins 10, ties 18, normal wins 32; share 32% (gate ≥ 40%).

## H5 (ordinary negation kept): **FAIL**

Negation words outside any construction, per 1,000 words (everyday prompts): normal 5.69, 25 off 2.23, matched 5.95; ratio 0.39 (gate ≥ 0.8). All negation words: normal 7.6, 25 off 2.7, matched 7.7.

## Secondary, non-gating

| genre | habit with 25 off |
|---|---|
| advice | 9 → 0 (+100% drop), mid-p 0.002 |
| copy | 4 → 0 (+100% drop), mid-p 0.062 |
| encouragement | 35 → 5 (+86% drop), mid-p 3.7e-08 |
| essay | 67 → 21 (+69% drop), mid-p 2e-11 |
| meaning | 41 → 1 (+98% drop), mid-p 1e-11 |
| message | 0 → 0, mid-p 1 |

| tier | 25 off | matched 25 |
|---|---|---|
| family | 142 → 24 (+83% drop), mid-p 4.8e-30 | 142 → 134 (+6% drop), mid-p 0.46 |
| cousin | 23 → 3 (+87% drop), mid-p 5.7e-06 | 23 → 12 (+48% drop), mid-p 0.017 |
| beyond | 3 → 0 (+100% drop), mid-p 0.12 | 3 → 0 (+100% drop), mid-p 0.12 |
| reversed | 26 → 34 (-31% drop), mid-p 0.27 | 26 → 33 (-27% drop), mid-p 0.3 |

Explicit requests ("use the pattern 'It's not X, it's Y'"): answers using it — normal 100%, 25 off 100%, matched 100%.
Answers containing "but": normal 36.2%, 25 off 35.3%, matched 36.5%.
Mean words: normal 200, 25 off 185, matched 199.
