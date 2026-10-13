"""Guards the MLX RoPE workaround in scripts/v2_mlx.py.

With MLX 0.31.2, mx.fast.rope gives wrong rotations for batch rows after the
first when L == 1 (one decode step). _rope_folded folds the batch into the head
axis; every row must match the batch-size-1 result exactly.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))


@pytest.mark.parametrize("L", [1, 7])
def test_folded_rope_matches_single_row(L):
    from v2_mlx import _rope_folded

    rope = nn.RoPE(256, traditional=False, base=10000)
    x1 = mx.random.normal((1, 8, L, 256)).astype(mx.bfloat16)
    ref = rope(x1, offset=20)
    out = _rope_folded(rope, mx.concatenate([x1] * 4, axis=0), offset=20)
    for i in range(4):
        assert float(mx.abs(out[i] - ref[0]).max()) == 0.0


@pytest.mark.parametrize("L", [1, 7])
@pytest.mark.parametrize("scaling", [None, {"rope_type": "linear", "factor": 8.0}])
def test_gemma3_folded_rope_matches_single_row(L, scaling):
    """Gemma 3's local and global RoPE modules, wrapped by g3_mlx.FoldedRope."""
    from mlx_lm.models.rope_utils import initialize_rope
    from g3_mlx import FoldedRope

    base = 10_000.0 if scaling is None else 1_000_000.0
    rope = FoldedRope(initialize_rope(dims=256, base=base, traditional=False,
                                      max_position_embeddings=131072, scaling_config=scaling))
    x1 = mx.random.normal((1, 8, L, 256)).astype(mx.bfloat16)
    ref = rope.inner(x1, offset=20)
    out = rope(mx.concatenate([x1] * 4, axis=0), offset=20)
    for i in range(4):
        assert float(mx.abs(out[i] - ref[0]).max()) == 0.0
