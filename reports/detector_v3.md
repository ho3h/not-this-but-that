# Detector v3: sensitivity re-score

Pre-registered verdicts stand as registered (frozen v2 detector). This re-scores every set with v3,
which fixes the edge cases in `src/classifier/detect_v3.py`. Counts are answers containing the form;
"normal → switched off", McNemar mid-p on the pairs.

## Confirmation v2, meaning prompts (pre-registered, chat) (800 pairs)

| tier | v2 (registered) | v3 |
|---|---|---|
| family | 85 → 80 (-6%), p 0.63 | 86 → 84 (-2%), p 0.85 |
| cousin | 113 → 127 (+12%), p 0.072 | 161 → 170 (+6%), p 0.39 |
| family or cousin | 187 → 199 (+6%), p 0.29 | 234 → 244 (+4%), p 0.44 |
| reversed (separate tier) | n/a | 26 → 27 (+4%), p 0.87 |
| beyond (separate tier) | n/a | 63 → 82 (+30%), p 0.052 |

## Confirmation v2, process/scene prompts (400 pairs)

| tier | v2 (registered) | v3 |
|---|---|---|
| family | 1 → 1 (+0%), p 1 | 2 → 1 (-50%), p 0.62 |
| cousin | 0 → 0, p 1 | 0 → 0, p 1 |
| family or cousin | 1 → 1 (+0%), p 1 | 2 → 1 (-50%), p 0.62 |
| reversed (separate tier) | n/a | 0 → 0, p 1 |
| beyond (separate tier) | n/a | 0 → 0, p 1 |

## First round, everyday prompts (Q5c) (306 pairs)

| tier | v2 (registered) | v3 |
|---|---|---|
| family | 18 → 10 (-44%), p 0.012 | 18 → 10 (-44%), p 0.012 |
| cousin | 10 → 11 (+10%), p 0.62 | 13 → 15 (+15%), p 0.45 |
| family or cousin | 28 → 21 (-25%), p 0.021 | 31 → 25 (-19%), p 0.092 |
| reversed (separate tier) | n/a | 0 → 2, p 0.25 |
| beyond (separate tier) | n/a | 7 → 6 (-14%), p 0.69 |

## First round, primed, completion only (Q5b) (300 pairs)

| tier | v2 (registered) | v3 |
|---|---|---|
| family | 37 → 18 (-51%), p 0.0045 | 33 → 18 (-45%), p 0.02 |
| cousin | 0 → 0, p 1 | 2 → 0 (-100%), p 0.25 |
| family or cousin | 37 → 18 (-51%), p 0.0045 | 35 → 18 (-49%), p 0.0079 |
| reversed (separate tier) | n/a | 5 → 2 (-60%), p 0.29 |
| beyond (separate tier) | n/a | 1 → 0 (-100%), p 0.5 |

## Confirmation 1, concrete prompts (D2b) (300 pairs)

| tier | v2 (registered) | v3 |
|---|---|---|
| family | 3 → 1 (-67%), p 0.38 | 3 → 1 (-67%), p 0.38 |
| cousin | 2 → 3 (+50%), p 0.5 | 3 → 3 (+0%), p 1 |
| family or cousin | 5 → 4 (-20%), p 0.69 | 6 → 4 (-33%), p 0.45 |
| reversed (separate tier) | n/a | 0 → 0, p 1 |
| beyond (separate tier) | n/a | 3 → 3 (+0%), p 1 |

## Prompt type (normal answers, family or cousin)

- v2: meaning 187/800, process/scene 1/400
- v3: meaning 234/800, process/scene 2/400

## Precision audit

A sample of the hits v3 adds over v2 (and of the v1 hits it drops) was labelled by an annotator
who saw only the text excerpts, never the condition (`data/detector_v3_audit_items.json`,
`data/detector_v3_audit_labels.tsv`, summary in `detector_v3_audit.json`):

- added cousin hits: 40 of 40 confirmed;
- added family hits: 10 of 17 confirmed in the first draft. All 7 misses came from two patterns
  (a staccato second sentence opening with This/These, and a loose "no longer" form), which were
  removed; the remaining additions in the sample are 4 of 4 confirmed. A narrow replacement
  ("This wasn't just X. This was Y.", same demonstrative in both sentences) was added afterwards
  and is not separately audited;
- dropped v1 hits (epistemic hedges such as "I'm not sure what you mean, but"): 4 of 4 were false positives.

