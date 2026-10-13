"""Multi-layer, multi-condition batches for the dose-response experiment.

Kept separate from v2_mlx.py, which is frozen by hash for the pre-registered
confirmation v2. Reuses its patched attention (RoPE batch fix) and helpers.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn

import v2_engine as E
from v2_mlx import _patch_gemma2, end_ids


class SAEMaskedAblate(nn.Module):
    """Like SAEAblate, but each batch row switches off its own subset of a
    union feature set U at this layer: row_mask is (B, |U|) with 1.0 = off."""

    def __init__(self, layer, W_enc_U, b_enc_U, thr_U, W_dec_U):
        super().__init__()
        self.layer = layer
        self.W_enc_U, self.b_enc_U, self.thr_U, self.W_dec_U = W_enc_U, b_enc_U, thr_U, W_dec_U
        self.row_mask = None

    def __call__(self, x, mask=None, cache=None):
        h = self.layer(x, mask, cache)
        if self.row_mask is None:
            return h
        hf = h.astype(mx.float32)
        pre = hf @ self.W_enc_U + self.b_enc_U
        acts = mx.where(pre > self.thr_U, pre, 0.0) * self.row_mask[:, None, :]
        contrib = acts @ self.W_dec_U
        if h.shape[1] > 1:  # prefill: leave BOS alone
            pos = mx.concatenate([mx.zeros((1, 1, 1)), mx.ones((1, h.shape[1] - 1, 1))], axis=1)
            contrib = contrib * pos
        return (hf - contrib).astype(h.dtype)


def load_multi(name: str, union_by_layer: dict[int, list[int]]):
    """Wrap each layer in union_by_layer with a masked ablation over its union set."""
    import torch
    from mlx_lm import load as mlx_load
    from sae_lens import SAE

    _patch_gemma2()
    model, tok = mlx_load(name)
    wraps = {}
    to_mx = lambda t: mx.array(t.detach().float().cpu().numpy())  # noqa: E731
    for layer, feats in union_by_layer.items():
        sae = SAE.from_pretrained(release=E.SAE_RELEASE, sae_id=f"layer_{layer}/width_16k/canonical", device="cpu")
        sae = sae[0] if isinstance(sae, tuple) else sae
        idx = torch.tensor(sorted(feats))
        w = SAEMaskedAblate(model.model.layers[layer], to_mx(sae.W_enc[:, idx]), to_mx(sae.b_enc[idx]),
                            to_mx(sae.threshold[idx]), to_mx(sae.W_dec[idx]))
        w.order = sorted(feats)
        model.model.layers[layer] = w
        wraps[layer] = w
    return model, tok, wraps


def set_row_masks(wraps: dict, doses: list[dict[int, set]], n: int) -> None:
    """Rows are dose-major: dose d occupies rows d*n .. d*n+n-1."""
    for layer, w in wraps.items():
        pos = {f: i for i, f in enumerate(w.order)}
        m = np.zeros((len(doses) * n, len(w.order)), dtype=np.float32)
        for d, dose in enumerate(doses):
            for f in dose.get(layer, ()):
                m[d * n:(d + 1) * n, pos[f]] = 1.0
        w.row_mask = mx.array(m)


def sample_rows(model, tok, prompt_tokens: list[int], U: "mx.array", *, max_new_tokens: int,
                temperature: float, top_p: float, prefilter_k: int = 1024) -> list[str]:
    """Decode len(U) rows from one prompt; row r uses uniforms U[r] (inverse-CDF top-p)."""
    from mlx_lm.models.cache import make_prompt_cache

    B = U.shape[0]
    cache = make_prompt_cache(model)
    logits = model(mx.array([prompt_tokens] * B), cache=cache)[:, -1, :]
    ends = end_ids(tok)
    finished = np.zeros(B, dtype=bool)
    out = [[] for _ in range(B)]
    for t in range(max_new_tokens):
        probs = mx.softmax(logits.astype(mx.float32) / max(temperature, 1e-6), axis=-1)
        part = mx.argpartition(-probs, kth=prefilter_k - 1, axis=-1)[:, :prefilter_k]
        pv = mx.take_along_axis(probs, part, axis=-1)
        order = mx.argsort(-pv, axis=-1)
        topv = mx.take_along_axis(pv, order, axis=-1)
        topi = mx.take_along_axis(part, order, axis=-1)
        csum = mx.cumsum(topv, axis=-1)
        if bool(mx.any(csum[:, -1] < top_p)):
            topi = mx.argsort(-probs, axis=-1)
            topv = mx.take_along_axis(probs, topi, axis=-1)
            csum = mx.cumsum(topv, axis=-1)
        keep = (csum - topv) < top_p
        filt = mx.where(keep, topv, 0.0)
        cdf = mx.cumsum(filt, axis=-1) / mx.sum(filt, axis=-1, keepdims=True)
        choice = mx.minimum(mx.sum(cdf < U[:, t:t + 1], axis=-1), cdf.shape[-1] - 1)
        nxt = mx.take_along_axis(topi, choice[:, None], axis=-1).squeeze(-1)
        nxt_np = np.array(nxt)
        for r in range(B):
            if not finished[r]:
                if int(nxt_np[r]) in ends:
                    finished[r] = True
                else:
                    out[r].append(int(nxt_np[r]))
        if finished.all():
            break
        logits = model(nxt[:, None], cache=cache)[:, -1, :]
    return [tok.decode(o) for o in out]
