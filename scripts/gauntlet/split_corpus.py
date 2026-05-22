"""G8 — Stratified 70/30 train/test split of the verified corpus.

Pre-registered: the CAA vector (A7) is built from TRAIN pairs ONLY.
The TEST split exists so we have an in-corpus holdout we never trained
on, in case we want a second-opinion check beyond the fresh-prompt
gauntlet. The actual gauntlet's headline numbers come from
data/d2_corpus/gauntlet_test_prompts.json — completely disjoint from
the harvest.

Seed 41 (consistent with referee_holdout build).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
CORPUS_PATH = REPO / "data" / "d2_corpus" / "verified_corpus.jsonl"
SPLIT_PATH = REPO / "data" / "d2_corpus" / "split.json"
REPORT_PATH = REPO / "reports" / "gauntlet" / "g8_split.md"

TRAIN_FRAC = 0.70
SEED = 41


def main() -> None:
    pairs = []
    with CORPUS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('{"_meta'):
                continue
            pairs.append(json.loads(line))
    print(f"[split] loaded {len(pairs)} pairs")

    by_form: dict[str, list[int]] = defaultdict(list)
    for p in pairs:
        by_form[p["form"]].append(p["id"])

    rng = random.Random(SEED)
    train_ids, test_ids = [], []
    for form, ids in sorted(by_form.items()):
        shuffled = ids[:]
        rng.shuffle(shuffled)
        cut = int(round(len(shuffled) * TRAIN_FRAC))
        train_ids.extend(shuffled[:cut])
        test_ids.extend(shuffled[cut:])

    train_ids.sort()
    test_ids.sort()
    SPLIT_PATH.write_text(json.dumps({
        "_meta": "G8 stratified 70/30 train/test split, seed=41. Stratified by form.",
        "seed": SEED,
        "train_frac": TRAIN_FRAC,
        "train": train_ids,
        "test": test_ids,
    }, indent=2))
    print(f"[split] wrote {SPLIT_PATH}: train={len(train_ids)}, test={len(test_ids)}")

    # Per-form summary
    lines = [
        "# G8 — Stratified train/test split",
        "",
        f"- Corpus: {CORPUS_PATH.name} ({len(pairs)} verified pairs)",
        f"- Split: {TRAIN_FRAC:.0%} train / {1-TRAIN_FRAC:.0%} test, stratified by form",
        f"- Seed: {SEED}",
        "",
        "## Per-form split",
        "",
        "| Form | Train | Test |",
        "|------|-------|------|",
    ]
    for form in sorted(by_form):
        ids = by_form[form]
        n_train = sum(1 for i in ids if i in set(train_ids))
        n_test = len(ids) - n_train
        lines.append(f"| {form} | {n_train} | {n_test} |")
    lines += [
        "",
        "## Anti-overfitting firewall",
        "",
        f"The CAA vector (A7) is built from TRAIN ids only ({len(train_ids)} "
        "pairs). The TEST ids never enter `build_vector`. The gauntlet's "
        "headline number, however, comes from completely fresh TEST prompts "
        "in `data/d2_corpus/gauntlet_test_prompts.json` — disjoint from the "
        "harvest by construction. So we have two layers of holdout: in-corpus "
        "TEST (this file) and fresh prompts (the gauntlet's actual scoring "
        "surface).",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print(f"[split] wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
