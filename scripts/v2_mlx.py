"""MLX engine for paired generation with the layer-20 coalition switched off.

Same intervention as v2_engine.PairedHook in error-term mode, ported to MLX for
speed on Apple Silicon (Hugging Face's Gemma 2 decode step carries ~0.4 s of
fixed overhead on MPS regardless of batch size; MLX does not).

- Only the switched-off features are encoded: JumpReLU activations are
  computed per feature, so a_C = JumpReLU(x @ W_enc[:, C] + b_enc[C]) is exact
  without touching the other 16,359 features. The residual becomes
  x - a_C @ W_dec[C]; everything the SAE doesn't explain is kept.
- Never applied at BOS (position 0; prompts are unpadded, one per call).
- Each call decodes n normal and n intervened rows for one prompt. Row k of
  each half draws the same uniform at every step (inverse-CDF over the top-p
  nucleus), so a pair stays token-identical until the intervention changes
  the next-token distribution. Uniforms come from numpy, seeded per prompt.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn

import v2_engine as E


def _rope_folded(rope, x, offset=0):
    """Apply RoPE with the batch folded into the head axis: (B, H, L, D) -> (1, B*H, L, D).

    With MLX 0.31.2, mx.fast.rope returns wrong rotations for every batch row
    after the first when L == 1 (the decode step), so batched generation drifts
    by several logits from row 1 onward; making the input contiguous produces
    NaNs instead. The kernel is correct for batch size 1, and every row shares
    the same positions here (prompts are unpadded), so folding is exact.
    """
    B, H, L, D = x.shape
    return rope(x.reshape(1, B * H, L, D), offset=offset).reshape(B, H, L, D)


def _gemma2_attention_batched(self, x, mask=None, cache=None):
    """mlx-lm 0.31.3's Gemma 2 attention, with RoPE applied via _rope_folded."""
    B, L, D = x.shape
    queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
    keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
    offset = cache.offset if cache is not None else 0
    queries = _rope_folded(self.rope, queries, offset)
    keys = _rope_folded(self.rope, keys, offset)
    if cache is not None:
        keys, values = cache.update_and_fetch(keys, values)
    queries = queries * self.scale
    if self.repeats > 1:  # broadcast keys/values over query groups (no copies)
        queries = queries.reshape(B, self.n_kv_heads, self.repeats, L, self.head_dim)
        keys = mx.expand_dims(keys, 2)
        values = mx.expand_dims(values, 2)
    scores = queries @ keys.swapaxes(-1, -2)
    scores = mx.tanh(scores / self.attn_logit_softcapping) * self.attn_logit_softcapping
    if mask is not None:
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.array(mx.finfo(scores.dtype).min, scores.dtype))
        else:
            scores = scores + mask
    scores = mx.softmax(scores, precise=True, axis=-1)
    output = scores @ values
    if self.repeats > 1:
        output = output.reshape(B, self.n_heads, L, self.head_dim)
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(output)


def _patch_gemma2():
    from mlx_lm.models import gemma2

    gemma2.Attention.__call__ = _gemma2_attention_batched


class SAEAblate(nn.Module):
    def __init__(self, layer, W_enc_C, b_enc_C, thr_C, W_dec_C):
        super().__init__()
        self.layer = layer
        self.W_enc_C, self.b_enc_C, self.thr_C, self.W_dec_C = W_enc_C, b_enc_C, thr_C, W_dec_C
        self.row_mask = None  # (B,) float32, 1.0 on rows to intervene

    def __call__(self, x, mask=None, cache=None):
        h = self.layer(x, mask, cache)
        if self.row_mask is None:
            return h
        hf = h.astype(mx.float32)
        pre = hf @ self.W_enc_C + self.b_enc_C
        acts = mx.where(pre > self.thr_C, pre, 0.0)
        contrib = acts @ self.W_dec_C  # (B, L, D)
        rm = self.row_mask[:, None, None]
        if h.shape[1] > 1:  # prefill: leave BOS alone
            pos = mx.concatenate([mx.zeros((1, 1, 1)), mx.ones((1, h.shape[1] - 1, 1))], axis=1)
            rm = rm * pos
        return (hf - rm * contrib).astype(h.dtype)


def load(name: str = "google/gemma-2-2b-it", features: list[int] | None = None):
    """Load an MLX model, wrap layer LAYER with the SAE ablation for `features`."""
    import torch
    from mlx_lm import load as mlx_load

    _patch_gemma2()
    model, tok = mlx_load(name)
    sae = E.load_sae()
    feats = sorted(features or E.COALITION_25)
    idx = torch.tensor(feats)
    to_mx = lambda t: mx.array(t.detach().float().cpu().numpy())  # noqa: E731
    W_enc_C = to_mx(sae.W_enc[:, idx])
    b_enc_C = to_mx(sae.b_enc[idx])
    thr_C = to_mx(sae.threshold[idx])
    W_dec_C = to_mx(sae.W_dec[idx])
    wrap = SAEAblate(model.model.layers[E.LAYER], W_enc_C, b_enc_C, thr_C, W_dec_C)
    model.model.layers[E.LAYER] = wrap
    return model, tok, wrap


def chat_tokens(tok, prompt: str) -> list[int]:
    return tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)


def end_ids(tok) -> set[int]:
    ids = {tok.eos_token_id}
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(eot, int) and eot >= 0:
        ids.add(eot)
    return ids


def generate_paired(model, tok, wrap, prompt_tokens: list[int], seed: int, *, n: int,
                    max_new_tokens: int = 150, temperature: float = 0.8, top_p: float = 0.95,
                    prefilter_k: int = 1024, intervene: bool = True) -> dict:
    B = 2 * n
    U = mx.array(np.random.default_rng(seed).random((n, max_new_tokens), dtype=np.float64).astype(np.float32))
    U = mx.concatenate([U, U], axis=0)  # rows k and n+k share uniforms
    wrap.row_mask = mx.array([0.0] * n + [1.0 if intervene else 0.0] * n)
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    x = mx.array([prompt_tokens] * B)
    logits = model(x, cache=cache)[:, -1, :]
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
        if bool(mx.any(csum[:, -1] < top_p)):  # nucleus wider than the shortlist: exact sort
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
    texts = [tok.decode(o) for o in out]
    return {"normal": texts[:n], "intervened": texts[n:]}
