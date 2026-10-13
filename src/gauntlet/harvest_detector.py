"""Permissive harvest detector for F1–F8. NOT for scoring the gauntlet.

The corpus build pipeline (G6) feeds candidates through this; every hit is
hand-verified before entering the corpus. So this can be loose — false
positives die at hand-verify, false negatives lose us examples we'll never
see. The trade-off favours recall.

Per operating_protocol.md §1.5 / §2.7: the harvest detector MUST be a
different surface than the eval referee. They share form definitions but
use different surface features so the referee doesn't get to score what
its own regex caught.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from gauntlet.forms import Form


@dataclass(frozen=True)
class HarvestHit:
    form: Form
    span_start: int
    span_end: int
    hinge: str  # the matched substring


# --- permissive regexes (designed to OVER-MATCH; hand-verify kills FPs) ---

# F1: any 'not ... it's/he's/she's/we're/they're/this is/that is' OR 'not ..., but'
_F1A = re.compile(
    r"\b(?:is|are|was|were|am)?\s*not\b[^.!?\n]{1,140}?"
    r"[,;—–-]\s*(?:it|that|this|he|she|we|they|these|those|there)'?s\b",
    re.IGNORECASE,
)
_F1B = re.compile(
    r"\b(?:is|are|was|were)n'?t\b[^.!?\n]{1,140}?"
    r"[,;—–-]\s*(?:it|that|this|he|she|we|they|these|those|there)'?s\b",
    re.IGNORECASE,
)
_F1C = re.compile(  # "not X, but Y" subject-less variant
    r"\bnot\s+(?:a|an|the|just|only|merely|simply)?\s*[^.!?\n,]{1,80},?\s*but\b",
    re.IGNORECASE,
)

# F2: cross-sentence staccato — first sentence has negation closer, next starts with copula
# Operates on a longer window than a single sentence; runs on text segments not pre-split.
_F2 = re.compile(
    r"\b(?:isn'?t|aren'?t|wasn'?t|weren'?t|is\s+not|are\s+not|was\s+not|were\s+not)\b"
    r"[^.!?\n]{0,80}\.\s+"
    r"(?:It|That|This|He|She|We|They|These|Those|There)'?s\b",
)

# F3: additive escalation — both "not only" and "not just" with optional pivot
_F3A = re.compile(
    r"\bnot\s+only\b[^.!?\n]{1,140}?\b(?:but|and|—|–|--|;)\b",
    re.IGNORECASE,
)
_F3B = re.compile(
    r"\b(?:not|(?:is|are|was|were)n'?t)\s+(?:just|merely|simply)\b"
    r"[^.!?\n]{1,140}?"
    r"(?:[—–]|--|\bbut\b|\b(?:it|that|this|he|she|we|they|these|those|there)'?s\b|"
    r"\b(?:we|they|you)'?re\b|\bthis\s+is\b|\bthese\s+are\b)",
    re.IGNORECASE,
)

# F4: reframing — "not about X, [copula] about Y"
_F4 = re.compile(
    r"\bnot\s+about\b[^.!?\n]{1,80}?"
    r"[,;—–]\s*"
    r"(?:(?:it|that|this|he|she|we|they)'?s\s+)?about\b",
    re.IGNORECASE,
)

# F5: comparative hedge — "less X, more Y" (both phrasal and "less about / more about")
_F5A = re.compile(
    r"\bless\s+about\b[^.!?\n]{1,80}?\band\s+more\s+about\b",
    re.IGNORECASE,
)
_F5B = re.compile(
    r"\bless\s+\w+(?:[,;]\s+|\s+and\s+)more\s+\w+",
    re.IGNORECASE,
)

# F6: triadic — same as Phase-0 C4
_F6 = re.compile(
    r"\bno\s+\w[^.!?\n]{0,60}\.\s+no\s+\w[^.!?\n]{0,60}\.\s+(?:just|only|but)\b",
    re.IGNORECASE,
)

# F7: concessive flip — "Far from X, Y" / "Rather than X, Y"
_F7A = re.compile(r"\bfar\s+from\s+[^.!?\n,]{1,80},\s*\w", re.IGNORECASE)
_F7B = re.compile(r"\brather\s+than\s+[^.!?\n,]{1,80},\s*\w", re.IGNORECASE)


_PATTERNS: list[tuple[Form, re.Pattern]] = [
    (Form.F2, _F2),  # F2 first — it operates on multi-sentence windows
    (Form.F3, _F3A), (Form.F3, _F3B),
    (Form.F4, _F4),
    (Form.F5, _F5A), (Form.F5, _F5B),
    (Form.F6, _F6),
    (Form.F7, _F7A), (Form.F7, _F7B),
    (Form.F1, _F1A), (Form.F1, _F1B), (Form.F1, _F1C),
]


def harvest(text: str) -> list[HarvestHit]:
    """Return all candidate F1–F7 hits. Intentionally noisy. Use hand-verify
    to convert candidates → verified positives.

    Order: more specific forms first (F2/F3/F4) so their spans claim before
    F1's broader pattern picks them up too.
    """
    hits: list[HarvestHit] = []
    claimed: list[tuple[int, int]] = []
    for form, pat in _PATTERNS:
        for m in pat.finditer(text):
            s, e = m.span()
            # half-open overlap with already-claimed spans
            if any(max(s, cs) < min(e, ce) for cs, ce in claimed):
                continue
            hits.append(HarvestHit(form=form, span_start=s, span_end=e, hinge=m.group(0)))
            claimed.append((s, e))
    hits.sort(key=lambda h: h.span_start)
    return hits


def harvest_in_generation(generation_text: str) -> list[HarvestHit]:
    """Convenience: run harvest on an entire model generation (multi-sentence)."""
    return harvest(generation_text)
