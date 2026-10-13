"""Dependency-based false-positive filter for the construction classifier.

PRD §5 M1: "regex hinge-detector + a light dependency check to kill false
positives." The check is intentionally narrow — it only *rejects* a regex hit
when the parse confidently shows the construction isn't there. On any
ambiguity (parse failure, missing token, unexpected structure) the regex hit
is accepted; the regex is the source of truth, the parser only filters.

The filter looks for:
  - The negation token (a "not" or a "*n't" suffix) inside the hit span.
  - That token's syntactic role: it must be a `neg`/`advmod` dependency on a
    verb (`AUX` or `VERB`). If the parser shows it as something else (e.g., a
    fixed-multiword adverb in a non-construction context, or a token inside a
    hyphenated compound), the hit is rejected.

The default in `detect_construction` is regex-only; pass `strict=True` to
apply this filter. Phase 2 M1 runs on actual model generations should default
to strict because that's where adversarial parse-ambiguous inputs appear.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from classifier.detect import Hit


@lru_cache(maxsize=1)
def _nlp():
    """Lazy spaCy load — kept off the import-time path so the regex stays cheap.

    The model is `en_core_web_sm` (downloaded by Phase 1 install). We disable
    components we don't need (`ner`, `lemmatizer`) to keep parses fast — only
    `tagger`, `attribute_ruler`, and `parser` matter for the negation check.
    """
    import spacy

    return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])


def is_genuine(text: str, hit: "Hit") -> bool:
    """Return True if the regex hit survives the dependency check.

    Conservative: any parse failure or unrecognised structure → True (accept).
    Only returns False when the parse shows the negation token is *not* a
    verb-attached neg/advmod dependency.
    """
    try:
        doc = _nlp()(text)
    except Exception:
        return True

    ms, me = hit.span
    # spaCy splits contractions like "isn't" into "is" + "n't"; both forms are
    # caught by checking lowercase token text.
    neg_tokens = [
        t for t in doc
        if ms <= t.idx < me and t.text.lower() in ("not", "n't", "no")
    ]
    if not neg_tokens:
        # Regex hit but no recognisable negation token in span — defer to regex.
        return True

    for tok in neg_tokens:
        # "no" in C4's "No X. No Y. Just Z." attaches as `det` to a noun.
        if tok.text.lower() == "no" and tok.dep_ == "det":
            return True
        # Walk up the head chain: a genuine construction's negation token
        # eventually reaches a verb/aux. In C2's "not only does it scale,
        # but it adapts.", spaCy parses "not" → advmod → "only" → advmod →
        # "does" (the aux). Direct-attach check misses that; chain-walk
        # catches it. Cap depth at 4 to avoid runaway on weird parses.
        cursor = tok
        for _ in range(4):
            if cursor.head is cursor:  # root
                break
            cursor = cursor.head
            if cursor.pos_ in ("VERB", "AUX"):
                return True

    # No verb/aux ancestor in the head chain — the negation isn't part of a
    # clausal construction. Reject.
    return False
