"""SAE-feature clamping for bidirectional causal validation (Phase 4).

`steer_features(features, direction)` is the new harness:
  - direction='ablate'  → set feature activation to zero at the pivot position
  - direction='clamp_up' → fix feature activation to a target value
The grammar-layer-era zero-ablation harness measured one direction. Phase 4
requires both, against random-k and bottom-k controls.
"""

from __future__ import annotations

from steering.clamp import ClampSpec, make_clamp_hook

__all__ = ["ClampSpec", "make_clamp_hook"]
