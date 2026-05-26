"""Construction classifier.

v1 (`detect`): regex hinges + spaCy dependency check. Catches single-sentence
constructions (C1/C2/C3). Misses F2 staccato (across-sentence "isn't X. It's Y.").

v2 (`detect_v2`): union of v1 strict + a permissive regex layer that catches
F2 staccato (and a couple of other across-sentence variants). The "everything
we can see" detector used in the Medium post's headline numbers. Same patterns
as the JS detector in web/demo/playground.js — they all agree.

Public API:
    detect_construction(text)        -> list[Hit]      # v1
    rate(texts)                       -> dict[variant, float]
    detect_construction_v2(text)     -> list[Hit]      # v2 union
    has_construction(text)            -> bool           # v2 any-core boolean
    rate_v2(texts)                    -> dict[variant, float]
"""

from __future__ import annotations

from classifier.detect import Hit, Variant, detect_construction, rate
from classifier.detect_v2 import (
    detect_construction_v2,
    detect_permissive,
    has_construction,
    rate_v2,
)

__all__ = ["Hit", "Variant", "detect_construction", "rate",
           "detect_construction_v2", "detect_permissive",
           "has_construction", "rate_v2"]
