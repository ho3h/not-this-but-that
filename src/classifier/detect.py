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

# Lexical building blocks (named so the patterns read closer to English).
#
# A *negation opener* commits to the construction: "X is not", "X isn't",
# "it's not", "we're not". The Phase 0 regex only handled subject-copula
# contractions ("it's not"); Phase 1 also accepts copula-with-n't contractions
# ("X isn't", "we aren't").
_NEG = (
    r"(?:"
    r"\b(?:is|are|was|were|am)\s+not\b"
    r"|\b(?:is|are|was|were)n'?t\b"
    r"|\b(?:it|that|this|he|she|we|they|these|those|there)'?s\s+not\b"
    r"|\b(?:we|they|you)'?re\s+not\b"
    r")"
)

# A *copula pivot* starts the contrasted clause. Phase 0 only listed "it's"
# and "they're"; Phase 1 covers the full subject-copula contraction family,
# the bare "this is / these are" forms, AND bare "<subj> <copula>" pairs
# (uncontracted: "she is", "he was", "they are"). The uncontracted forms
# appear in D1's hand-written pairs ("She is not merely a researcher, she is
# a translator of ideas.") and in any prose with a more formal register.
_PIVOT = (
    r"(?:"
    r"\b(?:it|that|this|he|she|we|they|these|those|there)'?s\b"
    r"|\b(?:we|they|you)'?re\b"
    r"|\b(?:he|she|it|this|that|there)\s+(?:is|was)\b"
    r"|\b(?:we|they|these|those|you)\s+(?:are|were)\b"
    r")"
)

# C2 — additive escalation. "Not only X, but Y" is canonical; "Not only X, [also] Y"
# is the same construction without "but". The pivot alternation accepts both.
_C2 = re.compile(
    r"\bnot\s+only\b[^.!?\n]{1,120}?\b(?:but|but\s+also|yet|also|—|–|--)\b",
    re.IGNORECASE,
)

# C3 — minimize-then-elevate. Accept both "not just" and "isn't just".
_C3 = re.compile(
    rf"\b(?:not|(?:is|are|was|were)n'?t)\s+(?:just|merely|simply)\b"
    rf"[^.!?\n]{{1,120}}?"
    rf"(?:[—–]|--|\bbut\b|{_PIVOT})",
    re.IGNORECASE,
)

# C1 — contrastive correction. Copula-bearing negation opener, then a
# comma/semicolon/em-dash pivot, then a copula clause.
_C1 = re.compile(
    rf"{_NEG}[^.!?\n]{{1,120}}?[,;—–]\s*{_PIVOT}",
    re.IGNORECASE,
)

# C1 alternative — "not X, but Y" / "not X but Y" (no copula opener).
# Comma optional; the "but" pivot is required.
_C1_ALT = re.compile(
    r"\bnot\s+(?:a|an|the|just|only|merely)?\s*[^.!?\n,]{1,80},?\s*but\b",
    re.IGNORECASE,
)

# C4: triadic. "No X. No Y. Just Z." — punctuation-anchored.
_C4 = re.compile(
    r"\bno\s+\w[^.!?\n]{0,60}\.\s+no\s+\w[^.!?\n]{0,60}\.\s+(?:just|only|but)\b",
    re.IGNORECASE,
)


def detect_construction(text: str, *, strict: bool = False) -> list[Hit]:
    """Return all C1-C4 hits in `text`. Order: by start offset.

    A sentence with multiple hinges yields multiple hits; M1 aggregation
    (`rate`) counts presence per variant per generation, not raw hit count.

    When `strict=True`, every regex hit is run through `dependency.is_genuine`
    (spaCy parse). The dependency check only rejects — never adds — so strict
    mode trades a small precision gain for ~5ms/sentence parse cost. Phase 2
    M1 runs default to strict; the Phase 1 kill check uses regex-only because
    that's the gate the PRD specifies and strict mode is additive on top.
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
            ms, me = m.span()
            # half-open interval intersection: overlap iff max(starts) < min(ends).
            # The Phase 0 check only caught partial overlap from one side and missed
            # the case where C1's span fully contains C3's (e.g. "It's not just an
            # update — it's a rethink." — C1 (0..end) ⊇ C3 (5..end)).
            if not any(max(ms, s) < min(me, e) for s, e in covered):
                hits.append(Hit(variant=Variant.C1, span=(ms, me), hinge=m.group(0)))
    hits.sort(key=lambda h: h.span[0])

    if strict:
        from classifier.dependency import is_genuine

        hits = [h for h in hits if is_genuine(text, h)]

    return hits


def rate(texts: Iterable[str], *, strict: bool = False) -> dict[str, float]:
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
        hits = detect_construction(t, strict=strict)
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
