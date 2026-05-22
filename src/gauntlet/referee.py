"""The gauntlet referee — strict F1-F8 detector for scoring gauntlet attacks.

Design rules (operating_protocol.md §1.5 + §2.7):
  1. Anti-circularity: the referee MUST NOT use the same surface as
     harvest_detector. Both recognize the same forms but via different
     lexical/structural anchors so the corpus isn't auto-validated by
     the thing that scores it.
  2. Validated externally: run validate_referee() on the hand-labeled
     holdout in data/d2_corpus/referee_holdout.jsonl before any gauntlet
     attack uses these numbers. Kill if overall P/R < 0.80.

Strictness vs the harvest detector:
  - Tighter span lengths (the construction tends to be short).
  - Strict copula-pivot requirement on F1/F2/F4 (no naked "but"-only matches).
  - Dependency check on F1/F2/F3/F4 — the negation token must attach to
     a verb/aux (Phase 1 lesson). Falls back to accept if spaCy parse fails.
  - F5/F7 require both halves of the contrast.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from gauntlet.forms import CORE_FORMS, Form


@dataclass(frozen=True)
class Hit:
    form: Form
    span_start: int
    span_end: int
    hinge: str


# ─── building blocks (different surface from harvest_detector) ────────────
# Negation OPENER explicitly requires a copula attached: "<subj> is not", "isn't",
# "<subj>'s not". The harvest detector accepts naked "not"; the referee does not.
_NEG_COP = (
    r"(?:"
    r"\b(?:I|you|we|they|he|she|it|that|this|these|those|there)\s+"
    r"(?:is|are|was|were|am)\s+not\b"
    r"|\b(?:I|you|we|they|he|she|it|that|this|these|those|there)'?"
    r"(?:s|re|m)\s+not\b"
    r"|\b(?:is|are|was|were)n'?t\b"
    r")"
)
# COPULA PIVOT continuation — explicit subject-copula contraction OR bare "this is".
_PIV = (
    r"(?:"
    r"\b(?:it|that|this|he|she|we|they|these|those|there)'?s\b"
    r"|\b(?:we|they|you)'?re\b"
    r"|\bthis\s+is\b|\bthese\s+are\b|\bthat\s+is\b"
    r")"
)


# F1 — single-sentence contrastive correction.
# Referee form: <NEG_COP> [span ≤ 60 chars] [comma/dash/semicolon] [PIV]
_F1_REF = re.compile(
    rf"{_NEG_COP}[^.!?\n]{{1,60}}[,;—–]\s*{_PIV}",
    re.IGNORECASE,
)
# F1 alt — "<not> X, but Y" with comma required, and Y must be content-bearing
# (a word, not just punctuation).
_F1_BUT = re.compile(
    r"\bnot\s+(?:a|an|the|just|only|merely|simply)?\s*[^.!?\n,]{2,60},\s*but\s+\w",
    re.IGNORECASE,
)


# F2 — cross-sentence staccato. Requires .  followed by  Capital + copula pivot,
# with the first sentence ending in a NEG_COP-style negation.
_F2_REF = re.compile(
    rf"{_NEG_COP}[^.!?\n]{{1,80}}\.\s+"
    r"(?:It|That|This|He|She|We|They|These|Those|There)\s*'?(?:s|re|m)?\s+",
)


# F3 — additive escalation. "not only X but Y" AND "not just/merely/simply X — Y / it's Y".
_F3_ONLY = re.compile(
    r"\bnot\s+only\b[^.!?\n]{1,100}?\b(?:but|but\s+also|yet|—|–)\b",
    re.IGNORECASE,
)
_F3_JUST = re.compile(
    rf"\b(?:not|(?:is|are|was|were)n'?t)\s+(?:just|merely|simply)\b"
    rf"[^.!?\n]{{1,100}}?(?:[—–]|\bbut\b|{_PIV})",
    re.IGNORECASE,
)


# F4 — reframing. "not about X, [copula] about Y" — narrower than harvest's.
_F4_REF = re.compile(
    rf"\bnot\s+about\b[^.!?\n]{{1,60}}?,\s*"
    rf"(?:(?:it|that|this|he|she|we|they)'?s\s+)?about\b",
    re.IGNORECASE,
)


# F5 — comparative hedge. Referee version: explicit "less X, more Y" or
# "less about X and more about Y" with a verb/copula present.
_F5_LESS_MORE = re.compile(
    r"\bless\s+\w+(?:\s+\w+){0,3},?\s+(?:and\s+)?more\s+\w+",
    re.IGNORECASE,
)
_F5_ABOUT = re.compile(
    r"\bless\s+about\b[^.!?\n]{1,60}?(?:,\s*)?and\s+more\s+about\b",
    re.IGNORECASE,
)


# F6 — triadic negation. Same form as Phase-0 C4. Three sentences: No X. No Y. {Just|Only} Z.
_F6_REF = re.compile(
    r"\bno\s+[A-Za-z][^.!?\n]{0,60}\.\s+no\s+[A-Za-z][^.!?\n]{0,60}\.\s+(?:just|only)\b",
    re.IGNORECASE,
)


# F7 — concessive flip. "Far from X, Y" / "Rather than X, Y" with a subject after the comma.
_F7_FAR = re.compile(r"\bfar\s+from\s+[^.!?\n,]{2,60},\s+\w", re.IGNORECASE)
_F7_RATHER = re.compile(r"\brather\s+than\s+[^.!?\n,]{2,60},\s+\w", re.IGNORECASE)


_PATTERNS: list[tuple[Form, re.Pattern]] = [
    (Form.F2, _F2_REF),
    (Form.F3, _F3_ONLY), (Form.F3, _F3_JUST),
    (Form.F4, _F4_REF),
    (Form.F5, _F5_ABOUT), (Form.F5, _F5_LESS_MORE),
    (Form.F6, _F6_REF),
    (Form.F7, _F7_FAR), (Form.F7, _F7_RATHER),
    (Form.F1, _F1_REF), (Form.F1, _F1_BUT),
]


# ─── dependency check (Phase 1 lesson, generalized to F1-F4) ──────────────


@lru_cache(maxsize=1)
def _nlp():
    import spacy
    return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])


def _negation_attaches_to_verb(text: str, hit: Hit) -> bool:
    """For F1/F2/F3/F4: confirm the 'not' / *n't inside the hit span is a `neg`
    or `advmod` of a verb/aux, walking head-chain up to 4 hops. The same logic
    as src/classifier/dependency.py, applied here for the gauntlet referee.
    """
    try:
        doc = _nlp()(text)
    except Exception:
        return True  # parse failure → defer to regex
    s, e = hit.span_start, hit.span_end
    neg_toks = [
        t for t in doc
        if s <= t.idx < e and t.text.lower() in ("not", "n't", "no")
    ]
    if not neg_toks:
        return True  # no negation token in span: defer (e.g. F5/F6/F7 which don't use "not")
    for tok in neg_toks:
        if tok.text.lower() == "no" and tok.dep_ == "det":
            return True
        cursor = tok
        for _ in range(4):
            if cursor.head is cursor:
                break
            cursor = cursor.head
            if cursor.pos_ in ("VERB", "AUX"):
                return True
    return False


def detect(text: str, *, strict: bool = True) -> list[Hit]:
    """Run the referee on a text segment. Returns non-overlapping hits.
    `strict=False` skips the dependency check (faster, less precise — for the
    gauntlet always use strict=True).
    """
    hits: list[Hit] = []
    claimed: list[tuple[int, int]] = []
    for form, pat in _PATTERNS:
        for m in pat.finditer(text):
            s, e = m.span()
            if any(max(s, cs) < min(e, ce) for cs, ce in claimed):
                continue
            hit = Hit(form=form, span_start=s, span_end=e, hinge=m.group(0))
            # Dep check on negation-bearing forms only
            if strict and form in (Form.F1, Form.F2, Form.F3, Form.F4):
                if not _negation_attaches_to_verb(text, hit):
                    continue
            hits.append(hit)
            claimed.append((s, e))
    hits.sort(key=lambda h: h.span_start)
    return hits


def score_text(text: str, *, strict: bool = True) -> dict[str, bool | int]:
    """Per-form binary score + any-core (F1∪F2∪F3∪F4) + any (all F1-F7).
    Returns a dict with keys 'F1'..'F7', 'any_core', 'any', and 'n_hits'.
    """
    hits = detect(text, strict=strict)
    present = {h.form for h in hits}
    out: dict[str, bool | int] = {
        f.value: (f in present) for f in CORE_FORMS
    }
    out["any_core"] = bool(present & {Form.F1, Form.F2, Form.F3, Form.F4})
    out["any"] = bool(present)
    out["n_hits"] = len(hits)
    return out


def rate(texts: Iterable[str], *, strict: bool = True) -> dict[str, float]:
    """Convenience: per-form positive-rate across a sequence of texts."""
    texts = list(texts)
    if not texts:
        return {f.value: 0.0 for f in CORE_FORMS} | {"any_core": 0.0, "any": 0.0}
    scored = [score_text(t, strict=strict) for t in texts]
    n = len(scored)
    out = {f.value: sum(1 for s in scored if s[f.value]) / n for f in CORE_FORMS}
    out["any_core"] = sum(1 for s in scored if s["any_core"]) / n
    out["any"] = sum(1 for s in scored if s["any"]) / n
    return out
