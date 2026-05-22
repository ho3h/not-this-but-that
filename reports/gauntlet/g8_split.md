# G8 — Stratified train/test split

- Corpus: verified_corpus.jsonl (216 verified pairs)
- Split: 70% train / 30% test, stratified by form
- Seed: 41

## Per-form split

| Form | Train | Test |
|------|-------|------|
| F1 | 24 | 10 |
| F2 | 14 | 6 |
| F3 | 102 | 44 |
| F5 | 11 | 5 |

## Anti-overfitting firewall

The CAA vector (A7) is built from TRAIN ids only (151 pairs). The TEST ids never enter `build_vector`. The gauntlet's headline number, however, comes from completely fresh TEST prompts in `data/d2_corpus/gauntlet_test_prompts.json` — disjoint from the harvest by construction. So we have two layers of holdout: in-corpus TEST (this file) and fresh prompts (the gauntlet's actual scoring surface).