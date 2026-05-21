"""Phase 1 kill-check regression tests.

If these fail, the construction classifier has degraded below the PRD §8 P1
gate (precision/recall >= 0.85 on C1-C3) — or the spaCy dependency check has
regressed in a way that contradicts the regex.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))


def _load_validation():
    path = _REPO_ROOT / "data" / "classifier_validation.jsonl"
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith('{"_meta'):
            continue
        rows.append(json.loads(line))
    return rows


def test_validation_set_shape():
    """Validation set is the hand-written 100 from Phase 1."""
    rows = _load_validation()
    assert len(rows) == 100
    counts = {"C1": 0, "C2": 0, "C3": 0, "C4": 0, "none": 0}
    for r in rows:
        counts[r["label"]] += 1
    # Spec from HANDOVER §5 Phase 1 (~30 positives, ~30 paraphrases, ~40 neutrals).
    assert counts["C1"] == 10
    assert counts["C2"] == 10
    assert counts["C3"] == 10
    assert counts["C4"] == 5
    assert counts["none"] == 65


def test_regex_only_kill_check():
    """Regex-only: P/R >= 0.85 on C1, C2, C3 (the gated variants)."""
    from eval_classifier import evaluate

    rows = _load_validation()
    result = evaluate(rows, strict=False)
    for v in ("C1", "C2", "C3"):
        row = result["per_variant"][v]
        assert row["precision"] >= 0.85, f"{v} precision={row['precision']} < 0.85"
        assert row["recall"] >= 0.85, f"{v} recall={row['recall']} < 0.85"
    assert result["kill_check_passed"]


def test_strict_mode_does_not_regress_kill_check():
    """Strict mode (regex + spaCy dep filter) also clears the gate."""
    from eval_classifier import evaluate

    rows = _load_validation()
    result = evaluate(rows, strict=True)
    for v in ("C1", "C2", "C3"):
        row = result["per_variant"][v]
        assert row["precision"] >= 0.85, f"strict {v} precision={row['precision']} < 0.85"
        assert row["recall"] >= 0.85, f"strict {v} recall={row['recall']} < 0.85"


def test_strict_mode_accepts_c2_not_only_pivot():
    """Regression guard: 'Not only ... but ...' has 'not' parsed as advmod of
    'only', not of the verb. The head-chain walk in dependency.py must reach
    the verb anyway, or this hit will be filtered.
    """
    from classifier import Variant, detect_construction

    text = "Not only does it scale, but it adapts."
    hits = detect_construction(text, strict=True)
    assert Variant.C2 in {h.variant for h in hits}, (
        "strict mode incorrectly filtered C2; check dependency head-chain walk"
    )


@pytest.mark.parametrize("text,expected", [
    ("It's not a tool, it's a revolution.", "C1"),
    ("Not only does it scale, but it adapts.", "C2"),
    ("It's not just an update — it's a rethink.", "C3"),
    ("No mandate. No approval. Just power.", "C4"),
])
def test_canonical_examples_hit_in_both_modes(text, expected):
    from classifier import Variant, detect_construction

    for strict in (False, True):
        hits = detect_construction(text, strict=strict)
        variants = {h.variant.value for h in hits}
        assert expected in variants, f"strict={strict}: missed {expected} on {text!r}"
