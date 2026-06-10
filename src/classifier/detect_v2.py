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
# and web/demo/app.js so the post, the demo, and the offline re-score all agree.
#
# 2026-06-09 revision: negation is now MANDATORY in the same-sentence and
# staccato patterns. The original patterns accepted a bare copula with
# `(?:not\s+)?` optional, which made "Running is a wonderful activity that
# provides numerous health benefits, but…" a hit — an ordinary concessive,
# not a contrastive correction. The original staccato pattern was also blind
# to the plain period-form ("isn't X. It's Y") because it required a literal
# not/just after the negated verb. The pronoun tails gained \b so "He" can no
# longer prefix-match "Here's". Three FP classes found in the audit are
# excluded by construction: first-person openers ("I'm not sure…" — epistemic
# hedging dominates), bare do-support ("didn't know. He…" — narration, only
# minimizer/mean forms kept), and epistemic predicates ("wasn't sure what I
# meant. He…"). Audit + before/after counts: reports/permissive_fix_audit.md.

# Negated openers shared by the same-sentence and staccato patterns. v1
# (`detect.py`) already covers most copula negation same-sentence; the
# permissive layer adds do-support correction forms ("doesn't just X, it Y",
# "doesn't mean X. It means Y") and the cross-sentence variants.
# The (?!…) lookahead drops epistemic-state predicates, which hedge rather
# than correct.
#
# DEFINITIONAL BOUNDARY (2026-06-09): the construction family is
# negation-anchored — that's what the pre-registration defined (C1-C4) and
# what the strict classifier was blind-validated on. The affirmative
# "more than just X(,;.) it's/they're Y" minimizer is a rhetorical COUSIN,
# not a member: it denies nothing. It is detected separately
# (`detect_more_than_just`) and reported as its own tier in the analyses,
# because the ablated model reroutes into it heavily. Folding it into the
# family mid-project would change the validated definition; ignoring it
# would hide the escape route. So: family and cousin, two numbers.
_NEG_OPENER = (
    r"(?:"
    r"(?:is|are|was|were)n'?t"                                   # isn't / weren't
    r"|(?:is|are|was|were|am)\s+not"                              # is not
    r"|(?:it|that|this|he|she|there)'?s\s+not"                    # it's not
    r"|(?:we|they|you)'?re\s+not"                                 # they're not
    r"|(?:do|does|did)(?:n'?t|\s+not)\s+(?:just|merely|simply|only|necessarily|really|mean)"
    r")"
    r"(?!\s+(?:sure|certain|aware|convinced)\b)"
)

# Pivot tails. \b after the group — without it "He" prefix-matched "Here's"
# and "it" matched "items". Bare pronouns (it/they/he/she/we) stay in the
# alternation: "doesn't replace teachers, it empowers them".
_TAIL_SAME = (
    r"(?:it'?s?|that'?s|they'?re|they|he'?s?|she'?s?|we'?re|we"
    r"|but|rather|instead)\b"
)
_TAIL_STACCATO = (
    r"(?:It'?s?|That'?s|They'?re|They|He'?s?|She'?s?|We'?re|We"
    r"|But|Rather|Instead)\b"
)

_PERMISSIVE = [
    # F1/F3 same-sentence with comma/dash pivot — negation required
    (re.compile(
        r"\b" + _NEG_OPENER +
        r"\s+(?:just\s+|merely\s+|simply\s+|only\s+)?[^.,;:!?\n]{1,80}"
        r"[,;—–\-]\s*" + _TAIL_SAME,
        re.IGNORECASE,
    ), Variant.C1, "same-sentence comma/dash"),

    # F2 staccato — cross-sentence "isn't X. It's Y." — negation required.
    # Case-sensitive tail anchors the second sentence's capitalised start;
    # the opener list carries its own capitalised subject variants.
    (re.compile(
        r"\b(?:"
        r"(?:is|are|was|were)n'?t"
        r"|(?:is|are|was|were|am)\s+not"
        r"|(?:[Ii]t|[Tt]hat|[Tt]his|[Hh]e|[Ss]he|[Tt]here)'?s\s+not"
        r"|(?:[Ww]e|[Tt]hey|[Yy]ou)'?re\s+not"
        r"|(?:[Dd]o|[Dd]oes|[Dd]id)(?:n'?t|\s+not)\s+(?:just|merely|simply|only|necessarily|really|mean)"
        r")"
        r"(?!\s+(?:sure|certain|aware|convinced)\b)"
        r"\s+(?:just\s+|merely\s+|simply\s+|only\s+|about\s+)?[^.!?\n]{1,80}"
        r"[.!?]\s*" + _TAIL_STACCATO,
    ), Variant.C3, "cross-sentence staccato"),

    # F4/F5 less/more or not about — unchanged (both arms already encode the
    # contrast lexically; "less X, more Y" needs no negator).
    (re.compile(
        r"(?:\bless\b\s+[^.,;:!?\n]{1,40}\s*[,;—–\-]\s*more\b"
        r"|\bnot\s+about\b\s+[^.,;:!?\n]{1,40}\s*[,;—–\-]\s*"
        r"(?:it'?s?\s+about|about))",
        re.IGNORECASE,
    ), Variant.C1, "less/more or not-about reframing"),
]


# The affirmative "more than just" cousin — comma/semicolon form and
# period form, symmetric. Not part of the construction family (see the
# definitional-boundary note above); reported as its own tier.
_MTJ = [
    (re.compile(
        r"\b(?:is|are|was|were)\s+(?:much\s+)?more\s+than\s+(?:just\s+|merely\s+|simply\s+)?"
        r"[^.,;:!?\n]{1,80}[,;—–\-]\s*" + _TAIL_SAME,
        re.IGNORECASE,
    ), "more-than-just (same-sentence)"),
    (re.compile(
        r"\b(?:is|are|was|were)\s+(?:much\s+)?more\s+than\s+(?:just\s+|merely\s+|simply\s+)?"
        r"[^.!?\n]{1,80}[.!?]\s*" + _TAIL_STACCATO,
    ), "more-than-just (cross-sentence)"),
]


def detect_permissive(text: str) -> list[Hit]:
    """Permissive-only detection. Returns Hit objects with detector='permissive'."""
    hits: list[Hit] = []
    for pat, var, hinge_name in _PERMISSIVE:
        for m in pat.finditer(text):
            hits.append(Hit(variant=var, span=(m.start(), m.end()),
                             hinge=hinge_name, detector="permissive"))
    return hits


def detect_more_than_just(text: str) -> list[Hit]:
    """The affirmative cousin, detected separately from the family."""
    hits: list[Hit] = []
    for pat, hinge_name in _MTJ:
        for m in pat.finditer(text):
            hits.append(Hit(variant=Variant.C3, span=(m.start(), m.end()),
                             hinge=hinge_name, detector="mtj-cousin"))
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
           "detect_permissive", "detect_more_than_just", "rate_v2"]
