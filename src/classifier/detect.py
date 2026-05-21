"""Regex hinge-detector for C1-C4 (PRD §1).

This is the lightweight first pass. Phase 1 of the PRD adds a dependency-based
sanity check (classifier/dependency.py) and a hand-labelled validation set; the
kill-check is precision/recall ≥ 0.85 on C1-C3.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class Variant(str, Enum):
    C1 = "C1"  # Contrastive correction:    not X, it's Y
    C2 = "C2"  # Additive escalation:       not only X, but Y
    C3 = "C3"  # Minimize-then-elevate:     not just X — Y
    C4 = "C4"  # Triadic negation:          no X. no Y. just Z.


@dataclass(frozen=True)
class Hit:
    variant: Variant
    span: tuple[int, int]
    hinge: str  # the specific lexical pivot that matched


# --- regex hinges ---------------------------------------------------------------
# Each pattern is anchored on the *pivot* token, not the negation. The pivot is
# where the next-token probability M2 lives, so we want the classifier and the
# probe to agree on where the construction commits.

_C2 = re.compile(
    r"\bnot\s+only\b[^.!?\n]{1,120}?\b(?:but|but\s+also|yet|—|–|--)\b",
    re.IGNORECASE,
)

_C3 = re.compile(
    r"\bnot\s+(?:just|merely|simply)\b[^.!?\n]{1,120}?(?:[—–]|--|\bbut\b|\bit'?s\b|\bthey'?re\b)",
    re.IGNORECASE,
)

# C1: bare "not ... it's/they're/this is" without the "only/just/merely" qualifier.
# The qualifier-bearing variants are caught by C2/C3 first; this is the residual.
_C1 = re.compile(
    r"\b(?:it'?s|that'?s|this\s+is|they'?re|these\s+are|we'?re)\s+not\b[^.!?\n]{1,120}?[,;—–]\s*(?:it'?s|that'?s|they'?re|this\s+is)\b",
    re.IGNORECASE,
)

# Also accept the inverted order: "not X, but Y" without a copula opener.
_C1_ALT = re.compile(
    r"\bnot\s+(?:a|an|the|just|only|merely)?\s*[^.!?\n,]{1,80},\s*but\b",
    re.IGNORECASE,
)

# C4: triadic. "No X. No Y. Just Z." — punctuation-anchored.
_C4 = re.compile(
    r"\bno\s+\w[^.!?\n]{0,60}\.\s+no\s+\w[^.!?\n]{0,60}\.\s+(?:just|only|but)\b",
    re.IGNORECASE,
)


def detect_construction(text: str) -> list[Hit]:
    """Return all C1-C4 hits in `text`. Order: by start offset.

    A sentence with multiple hinges yields multiple hits; M1 aggregation
    (`rate`) counts presence per variant per generation, not raw hit count.
    """
    hits: list[Hit] = []
    for variant, pat in ((Variant.C2, _C2), (Variant.C3, _C3), (Variant.C4, _C4)):
        for m in pat.finditer(text):
            hits.append(Hit(variant=variant, span=m.span(), hinge=m.group(0)))
    # C1 catches the residual — only register if the span isn't already covered
    # by a more specific variant (C2/C3 strictly dominate).
    covered = [h.span for h in hits]
    for pat in (_C1, _C1_ALT):
        for m in pat.finditer(text):
            if not any(s <= m.start() < e or s < m.end() <= e for s, e in covered):
                hits.append(Hit(variant=Variant.C1, span=m.span(), hinge=m.group(0)))
    hits.sort(key=lambda h: h.span[0])
    return hits


def rate(texts: Iterable[str]) -> dict[str, float]:
    """Fraction of `texts` that contain at least one hit per variant.

    Returns {"C1": ..., "C2": ..., "C3": ..., "C4": ..., "any": ..., "any_core": ...}
    where any_core = at least one of C1/C2/C3 (the primary causal claim).
    """
    texts = list(texts)
    if not texts:
        return {v.value: 0.0 for v in Variant} | {"any": 0.0, "any_core": 0.0}
    n = len(texts)
    per_variant = {v: 0 for v in Variant}
    any_hit = 0
    any_core_hit = 0
    for t in texts:
        hits = detect_construction(t)
        present = {h.variant for h in hits}
        for v in present:
            per_variant[v] += 1
        if present:
            any_hit += 1
        if present & {Variant.C1, Variant.C2, Variant.C3}:
            any_core_hit += 1
    out = {v.value: per_variant[v] / n for v in Variant}
    out["any"] = any_hit / n
    out["any_core"] = any_core_hit / n
    return out
