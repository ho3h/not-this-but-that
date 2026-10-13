"""Tier 0a supplement — positive-enriched blind set.

The natural-rate blind set contained only 1 positive in 60 sentences. This
supplement samples 30 sentences from Phase 2 generations that contain
"not" (a non-biased prefilter — the regex was never tuned on any of these
sentences). Many will still be non-constructions (plain negations), but
the positive rate should be substantially higher than the 1.7% baseline,
thickening the recall measurement.

Hand-label rule: same as before — C1/C2/C3/C4 iff the construction is
genuinely present per PRD §1, else 'none'. Plain negation is 'none'.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SOURCES = [
    REPO / "reports" / "phase2_generations_gemma_2b_it.jsonl",
    REPO / "reports" / "phase2_generations_gemma_2b.jsonl",
    REPO / "reports" / "phase2_generations_pythia_70m.jsonl",
    REPO / "reports" / "phase2_generations_gpt2.jsonl",
]
OUT = REPO / "data" / "classifier_blind_set_enriched.jsonl"

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“])")
NOT_RE = re.compile(r"\bn(?:o[tn]|'t)\b", re.IGNORECASE)


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    return [s.strip() for s in SENT_SPLIT.split(text) if len(s.strip()) > 30]


def main() -> None:
    rng = np.random.default_rng(29)
    pool = []
    for src in SOURCES:
        if not src.exists():
            continue
        for line in src.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for s in split_sentences(row["generation"]):
                if s.startswith(("*", "#", "-")) or len(s) < 40:
                    continue
                if NOT_RE.search(s):
                    pool.append({"text": s, "model": row["model"]})

    print(f"pool of negation-containing sentences across 4 models: {len(pool)}")
    idx = rng.choice(len(pool), size=30, replace=False)
    sample = [pool[i] for i in sorted(idx)]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        f.write(json.dumps({
            "_meta": "Tier 0a positive-enriched supplement. Hand-label in classifier_blind_labels_enriched.jsonl BEFORE classifier runs.",
            "_filter": "sentences containing 'not' / 'no' / 'n\\'t'",
            "_n": len(sample),
            "_rng_seed": 29,
        }) + "\n")
        for i, r in enumerate(sample):
            f.write(json.dumps({"id": f"E{i:02d}", "source": f"ai_{r['model']}", "text": r["text"]}) + "\n")
    print(f"wrote {len(sample)} sentences to {OUT}")


if __name__ == "__main__":
    main()
