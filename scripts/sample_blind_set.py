"""Tier 0a — sample sentences for blind classifier validation.

Pulls 30 human sentences from D3 (Phase 5 perplexity corpus, never seen by
the classifier) and 30 AI sentences from Phase 2 Gemma-2-2b-it generations
(scored by the classifier to produce M1, but the regex was never tuned on
them).

Outputs `data/classifier_blind_set.jsonl` containing ONLY the sentences
and their source — no classifier output, no labels yet. The agent labels
those by hand in `data/classifier_blind_labels.jsonl` BEFORE running the
classifier. Both files are committed before any P/R computation.

Sampling is deterministic (rng seed = 17) so the set is reproducible.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
D3 = REPO / "data" / "D3_fluency.txt"
GEMMA_IT = REPO / "reports" / "phase2_generations_gemma_2b_it.jsonl"
OUT = REPO / "data" / "classifier_blind_set.jsonl"

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“])")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in SENT_SPLIT.split(text) if len(s.strip()) > 30]


def main() -> None:
    rng = np.random.default_rng(17)

    # --- HUMAN: D3 ---
    d3 = D3.read_text()
    # strip the meta header
    d3_body = d3.split("---", 1)[1] if "---" in d3 else d3
    human_sents = []
    for para in d3_body.split("\n\n"):
        para = para.strip()
        if len(para) < 50 or para.startswith("PRD"):
            continue
        human_sents.extend(split_sentences(para))
    human_idx = rng.choice(len(human_sents), size=30, replace=False)
    human_sample = [human_sents[i] for i in sorted(human_idx)]

    # --- AI: Gemma-2-2b-it generations ---
    ai_lines = [json.loads(line) for line in GEMMA_IT.read_text().splitlines()
                if line.strip()]
    ai_sents = []
    for row in ai_lines:
        for s in split_sentences(row["generation"]):
            # filter out markdown headings and short fragments
            if s.startswith(("*", "#", "-")) or len(s) < 40:
                continue
            ai_sents.append(s)
    print(f"AI sentence pool: {len(ai_sents)}")
    ai_idx = rng.choice(len(ai_sents), size=30, replace=False)
    ai_sample = [ai_sents[i] for i in sorted(ai_idx)]

    out = []
    for i, s in enumerate(human_sample):
        out.append({"id": f"H{i:02d}", "source": "human_d3", "text": s})
    for i, s in enumerate(ai_sample):
        out.append({"id": f"A{i:02d}", "source": "ai_gemma_2b_it_d2", "text": s})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        f.write(json.dumps({
            "_meta": "Tier 0a blind set. Hand-label in classifier_blind_labels.jsonl BEFORE classifier runs.",
            "_n_human": len(human_sample), "_n_ai": len(ai_sample),
            "_rng_seed": 17,
        }) + "\n")
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} sentences to {OUT}")


if __name__ == "__main__":
    main()
