# Pre-registered confirmation v4: verdict

Analysed once. 600 triples on 120 everyday prompts (six genres), 100 on 20 explicit-request prompts.

## H1 (primary): habit with the chat 25 off: **PASS**

94 → 7 (+93% drop), mid-p 1.7e-24; 95% prompt-clustered CI 85% to 99%.

## H2 (specificity): matched control: **PASS**

Matched 25 off: 94 → 103 (-10% drop), mid-p 0.34. Gate: chat-25 drop exceeds control drop by ≥ 40 points.

## H3 (cost): **PASS**

- ARC-Easy (300): normal 85.7% · chat 25 85.3% · matched 85.7% (gate: drop ≤ 3 points) ✓
- WikiText-2 perplexity ratio: chat 25 1.049× · matched 1.005× (gate ≤ 1.1) ✓
- "a" before a vowel: normal 0.5% · chat 25 0.3% (gate ≤ 2%) ✓
- Blind preference, chat 25 vs normal (120 pairs): chat 25 wins 54, ties 4, normal wins 62; share 47% (gate ≥ 40%) ✓

## Secondary, non-gating

| genre | habit with chat 25 off |
|---|---|
| advice | 0 → 0, mid-p 1 |
| copy | 2 → 0 (+100% drop), mid-p 0.25 |
| encouragement | 28 → 0 (+100% drop), mid-p 3.7e-09 |
| essay | 50 → 6 (+88% drop), mid-p 6.8e-13 |
| meaning | 12 → 0 (+100% drop), mid-p 0.00024 |
| message | 2 → 1 (+50% drop), mid-p 0.62 |

| tier | chat 25 | matched 25 |
|---|---|---|
| family | 82 → 6 (+93% drop), mid-p 2.7e-21 | 82 → 91 (-11% drop), mid-p 0.32 |
| cousin | 15 → 1 (+93% drop), mid-p 0.00027 | 15 → 13 (+13% drop), mid-p 0.68 |
| beyond | 4 → 7 (-75% drop), mid-p 0.39 | 4 → 8 (-100% drop), mid-p 0.23 |
| reversed | 25 → 30 (-20% drop), mid-p 0.45 | 25 → 35 (-40% drop), mid-p 0.14 |

Explicit requests ("use the pattern 'It's not X, it's Y'"): answers using it — normal 100%, chat 25 96%, matched 100%.
Answers containing "but": normal 25.7%, chat 25 17.5%, matched 25.7%.
Mean words: normal 157, chat 25 156, matched 156.
