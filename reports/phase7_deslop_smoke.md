# Phase 7 — De-slop demo (Gemma 2 2B-it, feature ablation)

**Features ablated:** [3223]
**Pairs:** 3 (3 D2 prompts × 1 seeds)
**Sampling:** temperature=0.8, top_p=0.95, max_new_tokens=30

## Construction rate (any-core C1∪C2∪C3)

- baseline (no intervention): **0.00%** (0/3 generations contain the construction)
- ablated (feature → 0):     **0.00%** (0/3)
- absolute drop:             **+0.00%**
- relative drop:             **+0.00%**

## Meaning preservation (MiniLM cosine of baseline vs ablated)

- mean cosine = 1.000, median = 1.000
- 10th pct = 1.000, 90th pct = 1.000

Meaning is largely preserved (mean cosine ≥ 0.75) — the ablation is a scalpel, not a sledgehammer.

## Examples (first 6 pairs)

### `Explain why a city might invest in public transit infrastructure.` (seed 0)

**Baseline** ·:  
> **Answer:**

There are many reasons why a city might invest in public transit infrastructure. Here are some of the most important:

**Ablated** ·:  
> **Answer:**

There are many reasons why a city might invest in public transit infrastructure. Here are some of the most important:

---

### `Describe what makes a good neighbourhood library.` (seed 0)

**Baseline** ·:  
> A good neighborhood library isn't just a place to check out books. It's a community center that celebrates knowledge, creativity, and community.

**Ablated** ·:  
> A good neighborhood library isn't just a place to check out books. It's a community center that celebrates knowledge, creativity, and community.

---

### `Discuss the role of mentorship in early-career growth.` (seed 0)

**Baseline** ·:  
> A lot of people are passionate about the path they are on, but not all people have the opportunity to meet the right person who can

**Ablated** ·:  
> A lot of people are passionate about the path they are on, but not all people have the opportunity to meet the right person who can

---
