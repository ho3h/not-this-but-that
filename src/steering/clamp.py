"""Feature-level clamp hooks for ablation (down) and steering (up).

Adapts the zero/mean-ablation hook pattern from scripts/load_bearing_topk.py
into a reusable bidirectional primitive. Phase 4 uses both:
  - `direction='ablate'` to measure M1/M2 drop
  - `direction='clamp_up'` to measure M1 rise on neutral prompts
Controls (random-k, bottom-k) are size-matched feature lists fed to the same
hook factory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class ClampSpec:
    feature_idx: int
    direction: Literal["ablate", "clamp_up"]
    value: float = 0.0  # for clamp_up: the activation level to fix to


def make_clamp_hook(specs: list[ClampSpec], position: int = -1):
    """Build a TransformerLens hook that applies all `specs` at `position`.

    Phase 0 stub: returns a hook that does nothing if specs is empty. Phase 4
    fills in the post-encode write to the SAE feature index. The hook signature
    matches sae-lens 4.x — note the `**kwargs` to absorb TL's `hook=` kwarg.
    """

    def hook(activations: torch.Tensor, **kwargs) -> torch.Tensor:
        if not specs:
            return activations
        raise NotImplementedError(
            "steering.clamp.make_clamp_hook: Phase 4 work. See "
            "scripts/load_bearing_topk.py for the ablation pattern to adapt."
        )

    return hook
