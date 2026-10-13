# Pre-registered confirmation v3: verdict

Analysed once. 510 prompt-sample triples (normal, original 25, chat 25), 102 held-out D2 prompts.

## H1 (primary): habit (family or cousin) with the chat 25 off: **PASS**

88 → 3 (+97% drop), mid-p 5.8e-25; 95% prompt-clustered CI on the drop 92% to 100%.

Gates: PASS if drop ≥ 50% and mid-p < 0.05; KILL if drop < 25% or mid-p ≥ 0.05.

## H2 (co-primary): cost: **PASS**

- perplexity on human prose: 1.052× (gate ≤ 1.15) ✓
- quiz: 90% → 90% (gate: drop ≤ one question) ✓
- "a" before a vowel: 0.2% → 0.0% of answers (gate ≤ 2%) ✓

## Secondary, non-gating

| tier | chat 25 | original 25 |
|---|---|---|
| family | 47 → 3 (+94% drop), mid-p 6.8e-13 | 47 → 32 (+32% drop), mid-p 0.026 |
| cousin | 42 → 0 (+100% drop), mid-p 2.3e-13 | 42 → 43 (-2% drop), mid-p 0.85 |
| beyond | 37 → 30 (+19% drop), mid-p 0.33 | 37 → 47 (-27% drop), mid-p 0.18 |
| reversed | 8 → 12 (-50% drop), mid-p 0.38 | 8 → 15 (-88% drop), mid-p 0.15 |
| habit (family or cousin) | 88 → 3 (+97% drop), mid-p 5.8e-25 | 88 → 75 (+15% drop), mid-p 0.092 |

Answers containing "but": normal 12.9%, original 25 1.0%, chat 25 11.2%.
Mean words: normal 110, original 25 109, chat 25 110.
Original 25 cost: perplexity 1.043×, quiz 90%, "a" + vowel 0.4%.
