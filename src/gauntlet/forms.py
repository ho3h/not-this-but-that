"""F1-F8 form definitions, shared between the harvest detector and the referee.

The two detectors must agree on form *identity* (so verified positives carry
the same form ID) but disagree on *surface* (the operating-protocol §1.5
anti-circularity rule). Concretely: the harvest detector intentionally
over-matches, the referee tightly defines.
"""

from __future__ import annotations

from enum import Enum


class Form(str, Enum):
    F1 = "F1"  # Contrastive correction:        "not X, it's Y"  /  "not X, but Y"
    F2 = "F2"  # Staccato two-sentence:         "isn't X. It's Y."
    F3 = "F3"  # Additive escalation:           "not only X, but Y" / "not just X — Y"
    F4 = "F4"  # Reframing:                     "not about X, it's about Y"
    F5 = "F5"  # Comparative hedge:             "less X, more Y"
    F6 = "F6"  # Triadic negation:              "No X. No Y. Just Z."
    F7 = "F7"  # Concessive flip:               "Far from X, Y" / "Rather than X, Y"
    F8 = "F8"  # Cross-sentence parallelism — corpus only, not measured by gauntlet


CORE_FORMS = (Form.F1, Form.F2, Form.F3, Form.F4, Form.F5, Form.F6, Form.F7)
