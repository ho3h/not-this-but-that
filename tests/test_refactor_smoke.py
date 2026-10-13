"""Phase 0 done-check: new modules import, classifier hits the four canonical
PRD examples (and misses a paraphrase that doesn't use the construction).

This is the trivial smoke test that gates `not-this-but-that` Phase 0.
"""

from __future__ import annotations


def test_modules_import():
    import classifier  # noqa: F401
    import deslop  # noqa: F401
    import genealogy  # noqa: F401
    import quality  # noqa: F401
    import steering  # noqa: F401


def test_classifier_hits_canonical_examples():
    from classifier import Variant, detect_construction

    cases = [
        ("It's not a tool, it's a revolution.", Variant.C1),
        ("Not only does it scale, but it adapts.", Variant.C2),
        ("It's not just an update — it's a rethink.", Variant.C3),
        ("No mandate. No approval. Just power.", Variant.C4),
    ]
    for text, expected in cases:
        hits = detect_construction(text)
        variants = {h.variant for h in hits}
        assert expected in variants, (
            f"expected {expected.value} hit on {text!r}, got variants={variants}"
        )


def test_classifier_misses_plain_paraphrase():
    from classifier import detect_construction

    plain = "This feature reflects a philosophy."
    assert detect_construction(plain) == [], (
        "plain copular paraphrase should not match any of C1-C4"
    )


def test_rate_aggregates():
    from classifier import rate

    texts = [
        "It's not a tool, it's a revolution.",  # C1
        "Not only does it scale, but it adapts.",  # C2
        "This is a plain sentence.",  # none
    ]
    r = rate(texts)
    assert r["any_core"] == 2 / 3
    assert r["any"] == 2 / 3
    assert r["C1"] >= 1 / 3
    assert r["C2"] >= 1 / 3
