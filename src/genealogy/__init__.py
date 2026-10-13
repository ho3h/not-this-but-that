"""Base-vs-instruct comparison of the validated construction feature (Phase 6).

The novelty bet: gemma-2-2b vs gemma-2-2b-it on identical prompts. Outcome
worth chasing — "feature exists in base, lies dormant; instruct-tuning amplifies
its recruitment." Caveat: Gemma Scope SAEs are trained on the *base* model;
running them on instruct is a known transfer issue, so reconstruction quality
must be verified before any instruct-side number is trusted.
"""

from __future__ import annotations

from genealogy.transfer import reconstruction_quality

__all__ = ["reconstruction_quality"]
