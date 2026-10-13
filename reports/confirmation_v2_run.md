# Pre-registered confirmation v2: verdict

Analysed once, stages [1], 1200 pairs (800 on M prompts, 400 on P prompts).

## H1 (primary): family tier on M prompts: **KILL**

Baseline 85/800 → coalition off 80/800; relative drop 5.9% (95% prompt-clustered CI -17% to 23%); McNemar mid-p 0.634 (b=57, c=52).

Gates: PASS if drop ≥ 30% and mid-p < 0.05; KILL if drop < 15% or mid-p ≥ 0.05.

## H2 (secondary): prompt type: **CONFIRMED**

Baseline family-or-cousin rate: M 23.4%, P 0.2%; ratio 93.5× (95% CI 24.7× to 115.0×). Gate: ratio ≥ 2.0 and CI lower bound > 1.

## Secondary, non-gating

| tier (M prompts unless noted) | baseline | coalition off | rel drop | mid-p |
|---|---:|---:|---:|---:|
| textbook (v1) | 71/800 | 36/800 | 49% | 8.72e-05 |
| cousin | 113/800 | 127/800 | -12% | 0.0722 |
| family + cousin | 187/800 | 199/800 | -6% | 0.294 |
| family, all prompts | 86/1200 | 81/1200 | 6% | 0.637 |
| family, P prompts | 1/400 | 1/400 | 0% | 1 |

Share of M answers containing "but": 6.5% → 0.6%. Mean words per M answer: 111 → 111.
