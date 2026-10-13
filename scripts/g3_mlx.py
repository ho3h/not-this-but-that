"""Gemma 3 engine: the v2_mlx_multi procedure ported to gemma-3-*-it with Gemma Scope 2.

Same intervention as the Gemma 2 work: each wrapped decoder layer subtracts
row_mask * JumpReLU(h W_enc + b_enc) W_dec for a union feature set U (the SAE's
reconstruction error stays in), every position except BOS. A negative mask value
scales the features up instead.

Differences from Gemma 2:
- mlx-lm's Gemma 3 is a multimodal wrapper; the decoder layers live at
  model.language_model.model.layers (model.layers is a property that returns them).
- Gemma 3 uses a separate RoPE module per layer type (local / global). Each is
  wrapped in FoldedRope, the same batch-into-heads fold as v2_mlx._rope_folded,
  to dodge the MLX batched-decode RoPE bug (tests/test_mlx_rope_batch.py).
- Weights stay in bf16 (Gemma 3 activations overflow fp16).
- SAEs: Gemma Scope 2 instruct residual SAEs, release gemma-scope-2-{size}-it-res,
  id layer_{L}_width_16k_l0_medium (the Neuronpedia-labelled ones).

Sampling, chat tokens, row masks and end ids are shared with the Gemma 2 engine
(v2_mlx_multi.sample_rows / set_row_masks, v2_mlx.chat_tokens / end_ids), which
work on any mlx-lm model and tokenizer.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

import v2_mlx_multi as MM

SAE_ID = "layer_{layer}_width_16k_l0_medium"


def release(name: str) -> str:
    size = name.split("gemma-3-")[1].split("-")[0]  # "4b", "27b"
    return f"gemma-scope-2-{size}-it-res"


class FoldedRope(nn.Module):
    """RoPE with the batch folded into the head axis: (B, H, L, D) -> (1, B*H, L, D)."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def __call__(self, x, offset=0):
        B, H, L, D = x.shape
        return self.inner(x.reshape(1, B * H, L, D), offset=offset).reshape(B, H, L, D)


def fold_ropes(model) -> None:
    for layer in model.layers:
        attn = layer.self_attn
        if not isinstance(attn.rope, FoldedRope):
            attn.rope = FoldedRope(attn.rope)


class Capture(nn.Module):
    """Wraps one decoder layer and keeps its last output (instance-level, so other
    layers of the same class are untouched)."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.h = None

    def __call__(self, x, mask=None, cache=None):
        h = self.layer(x, mask, cache)
        self.h = h
        return h


def capture(model, layer: int) -> Capture:
    layers = model.language_model.model.layers
    if not isinstance(layers[layer], Capture):
        layers[layer] = Capture(layers[layer])
    return layers[layer]


def release_capture(model, layer: int) -> None:
    layers = model.language_model.model.layers
    if isinstance(layers[layer], Capture):
        layers[layer] = layers[layer].layer


def load_sae(name: str, layer: int):
    from sae_lens import SAE

    sae = SAE.from_pretrained(release=release(name), sae_id=SAE_ID.format(layer=layer), device="cpu")
    return sae[0] if isinstance(sae, tuple) else sae


def load_multi(name: str, union_by_layer: dict[int, list[int]], fold: bool = True):
    """Load gemma-3-*-it through mlx-lm and wrap each layer in union_by_layer with a
    masked switch-off over its union set (MM.SAEMaskedAblate, unchanged)."""
    import torch
    from mlx_lm import load as mlx_load

    model, tok = mlx_load(name)
    if fold:
        fold_ropes(model)
    layers = model.language_model.model.layers
    to_mx = lambda t: mx.array(t.detach().float().cpu().numpy())  # noqa: E731
    wraps = {}
    for layer, feats in union_by_layer.items():
        sae = load_sae(name, layer)
        idx = torch.tensor(sorted(feats))
        w = MM.SAEMaskedAblate(layers[layer], to_mx(sae.W_enc[:, idx]), to_mx(sae.b_enc[idx]),
                               to_mx(sae.threshold[idx]), to_mx(sae.W_dec[idx]))
        w.order = sorted(feats)
        layers[layer] = w
        wraps[layer] = w
    return model, tok, wraps
