"""Dependency-based false-positive filter for the construction classifier.

PRD §5 M1: "regex hinge-detector + a light dependency check to kill false
positives." Stub for Phase 1. The intended check: the matched "not" should be
a child of a verb that is also the head of the post-pivot clause — i.e. the
negation actually scopes over the contrasted material rather than being a
parenthetical or list-item.

Phase 1 plugs in a spaCy parse and a hand-labelled validation set.
"""

from __future__ import annotations

from classifier.detect import Hit


def is_genuine(text: str, hit: Hit) -> bool:
    """Return True if the hit is a real construction (not a false positive).

    Phase 0 stub: every regex hit is considered genuine. Phase 1 replaces this
    with a parse-aware check before the precision/recall validation gate.
    """
    return True
