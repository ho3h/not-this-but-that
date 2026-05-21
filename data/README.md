# `data/` — corpus layout

Per PRD §7 the project uses three disjoint corpora:

| File | Purpose | Populated by |
|---|---|---|
| `D1_contrast_pairs.jsonl` | ≥200 minimal pairs (with vs without the construction, meaning-matched). For SAE feature discovery via differential activation at the pivot. **Every pair hand-checked.** | Phase 1 |
| `D2_neutral_prompts.json`  | ≥100 open prompts that invite prose but don't beg for the construction. Used for M1 (spontaneous construction rate) and Phase 4 clamp-up targets. | Phase 2 |
| `D3_fluency.txt`           | A few thousand tokens of clean human prose, never seen during intervention design. Phase 5 perplexity reference. | Phase 5 (one-off) |

`labels_cache.json` is the Neuronpedia autointerp cache for Gemma 2 2B feature
labels — kept at top level because the labelling apparatus is reused.

`staging/` and `synthetic/` are the neograph substrate's parquet caches and
synthetic-prompt outputs from the previous project. Kept because the substrate
(see `src/neograph/`) reuses them for cross-model relation work in §9.

`legacy/` contains the capital-city / weekday / arithmetic prompts and the
old per-model label caches from the grammar-layer-era project. Preserved for
reproducibility of the previous writeups, not on the active path.
