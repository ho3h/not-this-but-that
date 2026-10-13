"""Split D1 and D2 into Discovery (75 %) and Confirmation (25 %) sub-corpora.

Run ONCE, before any Discovery campaign reads either corpus. The split is
deterministic (rng seed = 31) so it can be re-derived from the seed and the
underlying corpus state — but the agent must never re-roll it after Discovery
has touched the Discovery subset. The point of committing the split file is
that any re-roll is visible in git as a destructive edit.

Writes:
  - data/splits/d1.json — {"discovery": [ids], "confirmation": [ids], "seed": 31, "corpus_hash": ...}
  - data/splits/d2.json — same

Per the operating protocol (reports/operating_protocol.md §2.1):
  - Discovery scripts read only the "discovery" index list.
  - Confirmation scripts read the "confirmation" list ONLY when explicitly
    flagged (phase='confirmation').
  - Both lists are immutable post-commit; the corpus_hash field is a tripwire
    that catches accidental D1/D2 corpus edits after the split was made.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SPLITS = REPO / "data" / "splits"
RNG_SEED = 31
CONFIRMATION_FRAC = 0.25


def corpus_hash(rows: list[dict]) -> str:
    """Stable hash of the underlying corpus content. Order-independent."""
    blob = json.dumps([dict(sorted(r.items())) for r in rows], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def split_corpus(rows: list[dict], rng_seed: int = RNG_SEED) -> dict:
    n = len(rows)
    n_conf = int(round(n * CONFIRMATION_FRAC))
    rng = np.random.default_rng(rng_seed)
    perm = rng.permutation(n)
    conf_idx = sorted(int(i) for i in perm[:n_conf])
    disc_idx = sorted(int(i) for i in perm[n_conf:])
    return {
        "seed": rng_seed,
        "n_total": n,
        "n_discovery": len(disc_idx),
        "n_confirmation": len(conf_idx),
        "discovery": disc_idx,
        "confirmation": conf_idx,
        "corpus_hash": corpus_hash(rows),
    }


def main() -> None:
    SPLITS.mkdir(parents=True, exist_ok=True)

    # D1 — contrast pairs
    d1_path = REPO / "data" / "D1_contrast_pairs.jsonl"
    d1_rows = [
        json.loads(line)
        for line in d1_path.read_text().splitlines()
        if line.strip() and not line.startswith('{"_meta')
    ]
    d1_split = split_corpus(d1_rows)
    (SPLITS / "d1.json").write_text(json.dumps(d1_split, indent=2))
    print(f"D1: total={d1_split['n_total']} discovery={d1_split['n_discovery']} "
          f"confirmation={d1_split['n_confirmation']} hash={d1_split['corpus_hash']}")

    # D2 — neutral prompts (these are positional, so we split by index too)
    d2_path = REPO / "data" / "D2_neutral_prompts.json"
    d2_data = json.loads(d2_path.read_text())
    d2_rows = [{"idx": i, "text": p} for i, p in enumerate(d2_data["prompts"])]
    d2_split = split_corpus(d2_rows, rng_seed=RNG_SEED + 1)  # separate seed per corpus
    (SPLITS / "d2.json").write_text(json.dumps(d2_split, indent=2))
    print(f"D2: total={d2_split['n_total']} discovery={d2_split['n_discovery']} "
          f"confirmation={d2_split['n_confirmation']} hash={d2_split['corpus_hash']}")


if __name__ == "__main__":
    main()
