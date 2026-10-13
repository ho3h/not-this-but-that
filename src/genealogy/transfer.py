"""Verify SAE reconstruction quality on the instruct model before trusting it.

Gemma Scope is trained on `gemma-2-2b` (base). Applying the SAE encoder/decoder
to `gemma-2-2b-it` activations is a transfer assumption that needs an empirical
check. PRD §8 Phase 6: "verify reconstruction quality on the instruct model
before trusting any instruct-side number. Document the caveat prominently."

Phase 6 measures:
  - explained variance (fraction of activation variance the SAE recovers)
  - mean L2 reconstruction error vs base-model baseline
  - top-k feature-firing overlap on identical prompts

If reconstruction degrades meaningfully on instruct, every Phase 6 instruct-side
claim has to be hedged with this caveat in the writeup.
"""

from __future__ import annotations


def reconstruction_quality(model_name: str, sae_release: str, sae_id: str) -> dict:
    """Return a dict of reconstruction-quality metrics on `model_name`.

    Phase 6 stub.
    """
    raise NotImplementedError("Phase 6.")
