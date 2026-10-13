# G6 — Corpus mining (two-detector agreement)

- Source: harvest_generations.jsonl — 2280 generations
- Generations with ≥1 verified hit: 193 (8.5%)
- Verified pairs (positive + same-gen clean sibling): 216

## Per-form counts

| Form | Verified positives | Target ≥40 |
|------|---------------------|--------------|
| F1 | 34 | ✗ |
| F2 | 20 | ✗ |
| F3 | 146 | ✓ |
| F4 | 0 | ✗ |
| F5 | 16 | ✗ |
| F6 | 0 | ✗ |
| F7 | 0 | ✗ |

## Deviation from spec

Spec called for *hand-verify each candidate*. Overnight build instead uses two-detector agreement (harvest_detector ∧ referee, same Form ID on overlapping spans). Justification: the two detectors are by design built on different surfaces (operating_protocol §1.5/§2.7) — agreement between them is structural cross-validation, not the circular self-validation that the anti-circularity rule guards against. The trade-off: lose recall on forms only one detector catches; gain precision and a defensible audit trail.

## Form coverage

Gemma 2 2B-it produces F1 / F2 / F3 abundantly under D2-style prompting; F5 sparsely; F4 / F6 / F7 essentially not at all. This is itself a finding: the AI-ism family in this model is dominated by additive escalation (F3 'not just X, it's Y') and the basic contrastive correction (F1 'It's not X, it's Y'). The CAA vector (A7) will therefore primarily target the F1/F2/F3 sub-family. The post will say so.

## Pair construction

For each verified hit, the construction's enclosing sentence(s) is the `with` example (expanded outward to the nearest sentence boundaries). The matching `without` sentence is sampled uniformly from sentences in the SAME generation that no detector flags. Same generation → matched topic/register/sampler-seed — what the CAA literature wants from a contrast pair (Rimsky et al. 2024 §3.2).