"""Construction classifier v3: v2 plus the edge cases found in the October 2026 audit.

v2 (`detect_v2.py`) stays frozen: it is the detector the pre-registered
confirmations were scored with. v3 is used for sensitivity analyses and for the
demo highlighter. Audit and before/after counts: reports/detector_v3.md.

Changes from v2:

- Curly quotes and apostrophes are normalised before matching (v2 only knew
  ASCII apostrophes, so "It’s not X, it’s Y" scored zero).
- Family (negated) additions:
  - "This wasn't just X. This was Y." (same demonstrative in both sentences);
  - "not just/merely/only/simply X, they mend Y" with a pronoun pivot and any verb;
  - "no longer just X; it's Y";
  - "less about/like/of X and more about/like/of Y".
- Cousin ("more than just X. It's Y") additions:
  - contracted copulas ("it's more than just", "they're more than just");
  - degree words ("far / so much / a lot more than just");
  - any verb before "more than just|merely|simply" ("can be", "serve as", "deliver");
  - fronted and appositive forms ("More than just X, it's Y"; "X, more than just Y, is Z").
- v1 hits on epistemic hedges ("I'm not sure what you mean, but") are dropped.
- Two tiers reported separately and never folded into the family or cousin,
  because they are fuzzier: `reversed` ("Y, not X", "Y, not just X",
  "Y rather than X") and `beyond` ("goes beyond X").

API: detect_v3(text) -> dict[str, list[Hit]] with keys family, cousin,
reversed, beyond; has_family / has_cousin helpers; normalise().
"""

from __future__ import annotations

import re

from classifier.detect import detect_construction
from classifier.detect_v2 import Hit, Variant, detect_more_than_just, detect_permissive

_CURLY = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})
_PRON = r"(?:it|they|he|she|we|you|this|that|these|those)"
_PIVOT_ANY = rf"{_PRON}(?:'s|'re|\s+\w+)\b"  # pronoun + contraction or any verb
_TAIL_STACCATO_V3 = (
    r"(?:It'?s?|That'?s|They'?re|They|He'?s?|She'?s?|We'?re|We|But|Rather|Instead"
    r"|This|These|Those|There'?s?)\b"
)
_NEG_OPENER = (
    r"(?:(?:is|are|was|were)n'?t|(?:is|are|was|were|am)\s+not"
    r"|(?:[Ii]t|[Tt]hat|[Tt]his|[Hh]e|[Ss]he|[Tt]here)'?s\s+not"
    r"|(?:[Ww]e|[Tt]hey|[Yy]ou)'?re\s+not"
    r"|(?:[Dd]o|[Dd]oes|[Dd]id)(?:n'?t|\s+not)\s+(?:just|merely|simply|only|necessarily|really|mean))"
    r"(?!\s+(?:sure|certain|aware|convinced)\b)"
)

FAMILY_EXTRA = [
    # This wasn't just X. This was Y. (same demonstrative subject in both sentences)
    (re.compile(r"\b(This|That)\s+(?:isn't|wasn't|is\s+not|was\s+not)\s+(?:just\s+|merely\s+|simply\s+|about\s+)?"
                r"[^.!?\n]{1,80}[.!?]\s*\1\s+(?:is|was|'s)\b"), "This wasn't X. This was Y."),
    # not just X, they mend Y (pronoun + any verb after the pivot)
    (re.compile(r"\bnot\s+(?:just|merely|only|simply)\b[^.!?\n;,—–]{1,80}[,;—–]\s*" + _PIVOT_ANY, re.I),
     "not just X, pronoun + verb"),
    # no longer (just) X; it's Y
    (re.compile(r"\bno\s+longer\s+(?:just|merely|simply)\b[^.!?\n]{1,80}[,;—–.]\s*(?:it'?s|they'?re|that'?s|it\s+is|they\s+are)\b", re.I),
     "no longer just X; it's Y"),
    # less about X and more about Y
    (re.compile(r"\bless\s+(?:about|like|of)\b[^.!?\n]{1,80}?\b(?:and|but|,|;|—|–)?\s*more\s+(?:about|like|of)\b", re.I),
     "less about X, more about Y"),
]

_MTJ_HEAD = r"\bmore\s+than\s+(?:just|merely|simply)\b"
_COP = r"(?:\b(?:is|are|was|were|be|been|being)|'s|'re)"
_DEGREE = r"(?:(?:so\s+much|far|much|a\s+lot|a\s+whole\s+lot|far\s+much|way)\s+)?"
COUSIN_EXTRA = [
    # copula (incl. contracted) + degree + more than (just) X, pivot
    (re.compile(_COP + r"\s+" + _DEGREE + r"more\s+than\s+(?:just\s+|merely\s+|simply\s+)?[^.,;:!?\n]{1,80}[,;—–\-]\s*"
                r"(?:it'?s?|that'?s|they'?re|they|he'?s?|she'?s?|we'?re|we|but|rather|instead|this|these)\b", re.I),
     "copula/degree more than X, pivot"),
    # copula (incl. contracted) + degree + more than (just) X. It's Y.
    (re.compile(_COP + r"\s+" + _DEGREE + r"more\s+than\s+(?:just\s+|merely\s+|simply\s+)?[^.!?\n]{1,80}[.!?]\s*"
                + _TAIL_STACCATO_V3), "copula/degree more than X. It's Y."),
    # any verb + more than just X, pivot / . It's Y   (just|merely|simply required)
    (re.compile(_MTJ_HEAD + r"[^.!?\n;,—–]{1,80}[,;—–]\s*" + _PIVOT_ANY, re.I), "more than just X, pronoun + verb"),
    (re.compile(_MTJ_HEAD + r"[^.!?\n]{1,80}[.!?]\s*" + _TAIL_STACCATO_V3), "more than just X. It's Y."),
    # appositive: X, more than just Y, is Z
    (re.compile(r",\s*more\s+than\s+(?:just|merely|simply)\b[^.!?\n,]{1,60},\s*(?:is|are|was|were|has|have|becomes?)\b", re.I),
     "X, more than just Y, is Z"),
]

REVERSED = [
    (re.compile(r"\w[^.!?\n,;]{2,80},\s+not\s+(?:just\s+|merely\s+|only\s+|simply\s+)?(?:a|an|the|just|merely|only|simply|about)\b", re.I),
     "Y, not X"),
    (re.compile(r"\brather\s+than\b", re.I), "Y rather than X"),
]
BEYOND = [
    (re.compile(r"\b(?:go|goes|going|gone|went|reach|reaches|extend|extends|extending|stretch|stretches)\s+"
                r"(?:far\s+|well\s+|way\s+)?beyond\b", re.I), "goes beyond X"),
]


# v1's "not X ... but" pattern also fires on epistemic hedges ("I'm not sure what
# you mean, but ..."), which v2's permissive layer already excluded.
_EPISTEMIC = re.compile(r"not\s+(?:sure|certain|aware|convinced|clear)\b", re.I)


def normalise(text: str) -> str:
    return text.translate(_CURLY)


def _scan(text: str, patterns, variant=Variant.C1, detector="v3") -> list[Hit]:
    hits = []
    for pat, name in patterns:
        for m in pat.finditer(text):
            hits.append(Hit(variant=variant, span=m.span(), hinge=name, detector=detector))
    return hits


def detect_v3(text: str) -> dict[str, list[Hit]]:
    t = normalise(text)
    family = [Hit(h.variant, h.span, h.hinge, "v1") for h in detect_construction(t, strict=False)
              if not _EPISTEMIC.match(h.hinge)]
    family += detect_permissive(t)
    family += _scan(t, FAMILY_EXTRA)
    cousin = detect_more_than_just(t) + _scan(t, COUSIN_EXTRA, Variant.C3)
    return {
        "family": family,
        "cousin": cousin,
        "reversed": _scan(t, REVERSED),
        "beyond": _scan(t, BEYOND),
    }


def has_family(text: str) -> bool:
    return bool(detect_v3(text)["family"])


def has_cousin(text: str) -> bool:
    return bool(detect_v3(text)["cousin"])


# Audit (reports/detector_v3_audit.json, labels blind to condition): of a sample of
# hits v3 adds over v2, 40/40 cousin hits were confirmed. Family additions were
# 10/17 before two low-precision patterns (staccato with a This/These second
# sentence; a loose "no longer" form) were removed; every unconfirmed hit came
# from those two. All 4 sampled v1 hedge hits that v3 drops were false positives.
