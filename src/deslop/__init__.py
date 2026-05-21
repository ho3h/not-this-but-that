"""De-slop demo: inference-time steering vector that suppresses the construction.

The Phase 7 payoff. Packages the validated Phase-4 intervention as a single
steering vector applied at the pivot position; ships a before/after CLI and a
small judge eval for "less AI-sounding while staying fluent."

This module exists only after Phase 4 lands a feature that beats controls in
both directions and Phase 5 confirms quality is preserved. Until then the
public API raises.
"""

from __future__ import annotations

from deslop.demo import deslop

__all__ = ["deslop"]
