"""Data-split firewall.

Enforces the operating protocol §2.1: Discovery scripts get the Discovery
indices by default. Confirmation indices require an explicit `phase` kwarg.
Mistyped values raise loudly. The corpus hash recorded in the split file is
checked on load — if the underlying corpus has changed since the split was
made, the firewall refuses to operate and prints what changed.

Pattern:
    from firewall import load_d1, load_d2
    pairs = load_d1()                         # → Discovery split (default)
    pairs = load_d1(phase="confirmation")     # → Confirmation split (explicit)
    pairs = load_d1(phase="full")             # → full corpus (audit / methodology
                                              #   work only; never inside a Discovery
                                              #   or Confirmation campaign script)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

REPO = Path(__file__).resolve().parent.parent.parent
SPLITS = REPO / "data" / "splits"


def _corpus_hash(rows: list[dict]) -> str:
    blob = json.dumps([dict(sorted(r.items())) for r in rows], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _load_split(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"data split missing at {path}. Run scripts/make_data_split.py "
            "once, BEFORE any Discovery campaign. See reports/operating_protocol.md §2.1."
        )
    return json.loads(path.read_text())


def _select(rows: list[dict], split: dict, phase: str, label: str) -> list[dict]:
    expected_hash = split["corpus_hash"]
    actual_hash = _corpus_hash(rows)
    if expected_hash != actual_hash:
        raise RuntimeError(
            f"FIREWALL: {label} corpus has changed since the split was committed.\n"
            f"  expected hash: {expected_hash}\n"
            f"  actual hash:   {actual_hash}\n"
            f"You probably edited the corpus after the split was made. Either revert\n"
            f"the corpus edit OR re-run scripts/make_data_split.py AND review whether\n"
            f"any Discovery work since the previous split is now invalid (per the\n"
            f"operating protocol, the Confirmation split must be untouched by\n"
            f"Discovery; re-rolling it is a destructive edit that has to be justified)."
        )
    if phase == "discovery":
        idx = split["discovery"]
    elif phase == "confirmation":
        idx = split["confirmation"]
    elif phase == "full":
        idx = list(range(split["n_total"]))
    else:
        raise ValueError(
            f"phase must be one of 'discovery', 'confirmation', 'full' — got {phase!r}.\n"
            "Discovery is the default. 'confirmation' must be passed explicitly.\n"
            "See reports/operating_protocol.md §2.6."
        )
    return [rows[i] for i in idx]


def load_d1(phase: Literal["discovery", "confirmation", "full"] = "discovery") -> list[dict]:
    """Load D1 contrast pairs for a given phase."""
    d1_path = REPO / "data" / "D1_contrast_pairs.jsonl"
    rows = [
        json.loads(line)
        for line in d1_path.read_text().splitlines()
        if line.strip() and not line.startswith('{"_meta')
    ]
    split = _load_split(SPLITS / "d1.json")
    return _select(rows, split, phase, label="D1")


def load_d2(phase: Literal["discovery", "confirmation", "full"] = "discovery") -> list[str]:
    """Load D2 prompts for a given phase."""
    d2_path = REPO / "data" / "D2_neutral_prompts.json"
    d2_data = json.loads(d2_path.read_text())
    rows = [{"idx": i, "text": p} for i, p in enumerate(d2_data["prompts"])]
    split = _load_split(SPLITS / "d2.json")
    selected = _select(rows, split, phase, label="D2")
    return [r["text"] for r in selected]
