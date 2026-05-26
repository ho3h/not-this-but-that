"""Construction classifier v2 — union of strict + permissive.

The v1 classifier (`detect.py`) catches single-sentence constructions via
regex hinges + spaCy dependency check, but misses F2 staccato — the
cross-sentence form ("isn't X. It's Y"). The model uses F2 staccato about
as often as the single-sentence form in instruction-tuned generation, so
v1's coverage was structurally biased.

v2 adds a permissive regex layer that catches F2 staccato (and a few other
across-sentence variants). The union is the "everything we can see" detector
used in the Medium post's headline numbers.

API mirrors v1 to allow drop-in use:
    detect_construction_v2(text) -> list[Hit]
    has_construction(text) -> bool
    rate_v2(texts) -> dict[variant, float]
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from classifier.detect import Hit as HitV1, Variant, detect_construction


@dataclass(frozen=True)
class Hit:
    """v2 hit. Adds the detector source so callers can disambiguate."""
    variant: Variant
    span: tuple[int, int]
    hinge: str
    detector: str  # "strict" | "permissive"


# Permissive patterns — identical to the JS frontend at web/demo/playground.js
# so the post, the demo, and the offline re-score all agree.
_PERMISSIVE = [
    # F1/F3 same-sentence with comma/dash pivot
    (re.compile(
        r"\b(is|are|isn'?t|aren'?t|was|were|wasn'?t|weren'?t|don'?t|doesn'?t|don)"
        r"\s+(?:not\s+)?(?:just\s+)?[^.,;:!?\n]{1,80}"
        r"[,;—–\-]\s*(?:it'?s?|they'?re?|they|he'?s?|she'?s?|we'?re?|but\s+|but\b)",
        re.IGNORECASE,
    ), Variant.C1, "same-sentence comma/dash"),

    # F2 staccato — cross-sentence "isn't/not just X. It's Y."
    (re.compile(
        r"\b(?:is|are|isn'?t|aren'?t|was|were|wasn'?t|weren'?t|don'?t|doesn'?t)"
        r"\s+(?:not|just)\s+(?:just\s+)?[^.!?\n]{1,80}"
        r"[.!?]\s*(?:It'?s?|They'?re?|He'?s?|She'?s?|We'?re?|But\s+|Rather|Instead)",
    ), Variant.C3, "cross-sentence staccato"),

    # F4/F5 less/more or not about
    (re.compile(
        r"(?:\bless\b\s+[^.,;:!?\n]{1,40}\s*[,;—–\-]\s*more\b"
        r"|\bnot\s+about\b\s+[^.,;:!?\n]{1,40}\s*[,;—–\-]\s*"
        r"(?:it'?s?\s+about|about))",
        re.IGNORECASE,
    ), Variant.C1, "less/more or not-about reframing"),
]


def detect_permissive(text: str) -> list[Hit]:
    """Permissive-only detection. Returns Hit objects with detector='permissive'."""
    hits: list[Hit] = []
    for pat, var, hinge_name in _PERMISSIVE:
        for m in pat.finditer(text):
            hits.append(Hit(variant=var, span=(m.start(), m.end()),
                             hinge=hinge_name, detector="permissive"))
    return hits


def detect_construction_v2(text: str, *, strict: bool = False) -> list[Hit]:
    """Union of strict (v1) + permissive. `strict` flag is forwarded to v1."""
    out: list[Hit] = []
    # v1 strict
    for h in detect_construction(text, strict=strict):
        out.append(Hit(variant=h.variant, span=h.span, hinge=h.hinge,
                        detector="strict"))
    # v2 permissive
    out.extend(detect_permissive(text))
    return out


def has_construction(text: str, *, strict: bool = False) -> bool:
    """True if either detector finds a core (C1/C2/C3) construction."""
    for h in detect_construction_v2(text, strict=strict):
        if h.variant in (Variant.C1, Variant.C2, Variant.C3):
            return True
    return False


def rate_v2(texts) -> dict[str, float]:
    """Fraction of texts containing each variant under the union detector."""
    counts = {v.value: 0 for v in Variant}
    n = 0
    for t in texts:
        n += 1
        seen = {h.variant for h in detect_construction_v2(t)}
        for v in seen:
            counts[v.value] += 1
    if n == 0:
        return {v: 0.0 for v in counts}
    return {v: c / n for v, c in counts.items()}


__all__ = ["Hit", "Variant", "detect_construction_v2", "has_construction",
           "detect_permissive", "rate_v2"]
