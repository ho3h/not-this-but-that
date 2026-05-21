"""Construction classifier (M1).

Detects the four PRD variants of negative-parallelism (C1-C4) in a candidate
sentence. The detector is intentionally a thin regex layer on top of a
dependency-based sanity check; the regex catches the lexical hinge, the
parser kills false positives where the hinge appears outside the construction.

Public API:
    detect_construction(text) -> list[Hit]
    rate(texts) -> dict[variant, float]
"""

from __future__ import annotations

from classifier.detect import Hit, Variant, detect_construction, rate

__all__ = ["Hit", "Variant", "detect_construction", "rate"]
