"""Held-out perplexity under an (optionally intervened) model.

D3 corpus (PRD §7) is a few thousand tokens of clean human prose, never seen
during intervention design. Phase 5 measures perplexity twice — baseline and
intervened — and reports the ratio. A blow-up is the M3 fluency-failure signal.
"""

from __future__ import annotations

from typing import Callable


def perplexity(
    text: str,
    forward: Callable[[str], float],
) -> float:
    """Return exp(mean negative log-likelihood) of `text` under `forward`.

    `forward(text)` must return the average NLL per token (nats) — this lets
    Phase 5 share the same harness between baseline and hook-installed runs.
    Phase 0 stub.
    """
    raise NotImplementedError("Phase 5: per-token NLL → exp(mean).")
