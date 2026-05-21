"""Tests for the Discovery/Confirmation data-split firewall.

These tests are the executable form of reports/operating_protocol.md §2.1
and §2.6. If any of them fail, the firewall is broken and Discovery work
done in that state cannot be trusted to have stayed inside its lane.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from firewall import load_d1, load_d2


REPO = Path(__file__).resolve().parent.parent
SPLITS = REPO / "data" / "splits"


def test_d1_split_exists_and_partitions_cleanly():
    split = json.loads((SPLITS / "d1.json").read_text())
    assert split["n_discovery"] + split["n_confirmation"] == split["n_total"]
    assert set(split["discovery"]).isdisjoint(set(split["confirmation"])), (
        "FIREWALL VIOLATION: discovery and confirmation indices overlap"
    )


def test_d2_split_exists_and_partitions_cleanly():
    split = json.loads((SPLITS / "d2.json").read_text())
    assert split["n_discovery"] + split["n_confirmation"] == split["n_total"]
    assert set(split["discovery"]).isdisjoint(set(split["confirmation"])), (
        "FIREWALL VIOLATION: discovery and confirmation indices overlap"
    )


def test_default_phase_is_discovery_not_confirmation():
    """The default phase MUST be discovery. Forgetting the kwarg should NOT
    leak Confirmation data into a Discovery script."""
    default = load_d1()
    discovery = load_d1(phase="discovery")
    assert len(default) == len(discovery)
    assert all(a["with"] == b["with"] for a, b in zip(default, discovery))


def test_confirmation_requires_explicit_phase():
    """Confirmation data must be reachable only by passing phase='confirmation'
    explicitly. This is the typo-protection — load_d1(phase='discoveryyyy')
    must raise rather than silently return Discovery, because in the
    Confirmation case a silent fallback would be worse than an error."""
    with pytest.raises(ValueError):
        load_d1(phase="discoveryyy")  # typo
    with pytest.raises(ValueError):
        load_d1(phase="conf")  # abbreviation
    with pytest.raises(ValueError):
        load_d2(phase="all")  # invalid


def test_corpus_hash_tripwire():
    """If the underlying corpus changed after the split was committed, the
    firewall MUST refuse rather than silently returning a stale split."""
    d1_split = json.loads((SPLITS / "d1.json").read_text())
    rows_now = [
        json.loads(line)
        for line in (REPO / "data" / "D1_contrast_pairs.jsonl").read_text().splitlines()
        if line.strip() and not line.startswith('{"_meta')
    ]
    # The split's recorded hash should match the current corpus.
    import hashlib
    actual = hashlib.sha256(
        json.dumps([dict(sorted(r.items())) for r in rows_now], sort_keys=True).encode()
    ).hexdigest()[:16]
    assert actual == d1_split["corpus_hash"], (
        "D1 corpus has changed since the split was made. The firewall will refuse "
        "to operate; re-run scripts/make_data_split.py only AFTER auditing whether "
        "any Discovery work since the previous split is now contaminated."
    )


def test_confirmation_split_is_nontrivial_size():
    """The Confirmation split must be large enough that a positive Confirmation
    result is non-trivial. <10 items would be too easy to overfit."""
    d1 = json.loads((SPLITS / "d1.json").read_text())
    d2 = json.loads((SPLITS / "d2.json").read_text())
    assert d1["n_confirmation"] >= 25
    assert d2["n_confirmation"] >= 20
