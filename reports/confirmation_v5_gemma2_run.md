# Pre-registered confirmation v5 (confirmation_v5_gemma2): verdict

Analysed once. 600 triples on 120 everyday prompts (six genres), 100 on 20 explicit-request prompts.

## H1 (primary): habit with the 25 off: **PASS**

92 → 26 (+72% drop), mid-p 1.4e-13; 95% prompt-clustered CI 59% to 83%.

## H2 (specificity): matched control: **PASS**

Matched 25 off: 92 → 95 (-3% drop), mid-p 0.76. Gate: the 25's drop exceeds control drop by ≥ 40 points.

## H3 (cost): **PASS**

- ARC-Easy (300): normal 85.7% · 25 off 85.7% · matched 85.7% (gate: drop ≤ 3 points) ✓
- WikiText-2 perplexity ratio: 25 off 1.048× · matched 0.999× (gate ≤ 1.1) ✓
- "a" before a vowel: normal 1.0% · 25 off 0.2% (gate ≤ 2%) ✓
- Blind preference, 25 off vs normal (120 pairs): 25 off wins 47, ties 4, normal wins 69; share 41% (gate ≥ 40%) ✓
- 'No' questions right (of 30): normal 28 · 25 off 28 · matched 28 (gate: lose ≤ 2) ✓

## H4 (warnings stay clear): **FAIL**

Blind pairwise on 60 answers to 30 fresh "what should you never do" prompts: 25 off wins 19, ties 9, normal wins 32; share 39% (gate ≥ 40%).

## H5 (ordinary negation kept): **PASS**

Negation words outside any construction, per 1,000 words (everyday prompts): normal 5.56, 25 off 5.30, matched 6.24; ratio 0.95 (gate ≥ 0.8). All negation words: normal 6.9, 25 off 5.8, matched 7.6.

## Secondary, non-gating

| genre | habit with 25 off |
|---|---|
| advice | 0 → 0, mid-p 1 |
| copy | 5 → 1 (+80% drop), mid-p 0.12 |
| encouragement | 22 → 7 (+68% drop), mid-p 0.0015 |
| essay | 53 → 18 (+66% drop), mid-p 1e-07 |
| meaning | 9 → 0 (+100% drop), mid-p 0.002 |
| message | 3 → 0 (+100% drop), mid-p 0.12 |

| tier | 25 off | matched 25 |
|---|---|---|
| family | 79 → 24 (+70% drop), mid-p 2.3e-10 | 79 → 82 (-4% drop), mid-p 0.75 |
| cousin | 17 → 2 (+88% drop), mid-p 0.00014 | 17 → 14 (+18% drop), mid-p 0.57 |
| beyond | 2 → 8 (-300% drop), mid-p 0.065 | 2 → 7 (-250% drop), mid-p 0.11 |
| reversed | 34 → 30 (+12% drop), mid-p 0.58 | 34 → 29 (+15% drop), mid-p 0.49 |

Explicit requests ("use the pattern 'It's not X, it's Y'"): answers using it — normal 100%, 25 off 99%, matched 100%.
Answers containing "but": normal 30.7%, 25 off 23.3%, matched 28.3%.
Mean words: normal 166, 25 off 165, matched 168.
