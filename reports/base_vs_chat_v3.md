# Chat vs base model, detector v3 (exploratory)

Phase 2 generations, 50 D2 prompts, 250 answers per model.

| model | answers with family or cousin |
|---|---:|
| gemma-2-2b (base) | 11 / 250 |
| gemma-2-2b-it (chat) | 52 / 250 |

Ratio chat/base: **4.7×** (95% prompt-clustered bootstrap CI 2.4× to 12.2×).

The earlier 12× figure (`phase2_baseline.md`) counts only the strict C3 form per sentence and rests
on very few base-model events; this is the broader and steadier count.
