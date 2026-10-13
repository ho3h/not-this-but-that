"""Engine v2: Hugging Face transformers + the Gemma Scope layer-20 SAE.

Fixes three things in the v1 daemon path (scripts/probe_daemon.py):

1. Chat template. Prompts to gemma-2-2b-it are wrapped in the model's chat
   template, and generation stops at <end_of_turn>. v1 fed raw text plus BOS.
2. Error term. mode="ablate" subtracts only the switched-off features' decoder
   contributions from the layer-20 residual, so the baseline is the unmodified
   model. v1 replaced the residual with the SAE reconstruction in BOTH
   conditions (no error term); that behaviour is kept as mode="splice" so old
   numbers can be reproduced and compared.
3. Speed. KV-cached generation through model.generate, several samples per
   call, instead of a full forward pass per token.

The SAE is never applied at the BOS position: Gemma Scope wasn't trained on
it, and its residual norm is orders of magnitude larger than other tokens'.
"""

from __future__ import annotations

import os
import pathlib
import re

import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
LAYER = 20
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_20/width_16k/canonical"
COALITION_25 = [
    347, 1250, 1608, 2137, 2184, 2282, 3223, 4197, 4516, 6336, 6759, 7361,
    8530, 8820, 9606, 9816, 9909, 10822, 11864, 12506, 12524, 12767, 12898,
    12923, 14401,
]
CORE_2 = [3223, 9909]


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _hf_token() -> str | None:
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO / ".env")
    except ImportError:
        pass
    return os.environ.get("HF_TOKEN")


def load_model(name: str, dtype=torch.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, token=_hf_token())
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype, attn_implementation="eager", token=_hf_token()
    ).to(device())
    model.eval()
    return tok, model


def load_sae():
    from sae_lens import SAE

    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=device())
    sae = sae[0] if isinstance(sae, tuple) else sae
    sae.eval()
    return sae


class Intervention:
    """Forward hook on decoder layer LAYER.

    mode="none"   leave the residual alone
    mode="ablate" h - sum_{i in features} a_i(h) * W_dec[i]   (error term kept)
    mode="splice" decode(encode(h) with features zeroed)       (v1 behaviour)
    """

    def __init__(self, sae, features: list[int] | None, mode: str):
        assert mode in ("none", "ablate", "splice")
        self.sae = sae
        self.mode = mode
        self.idx = None
        if features:
            self.idx = torch.tensor(sorted(features), device=sae.W_dec.device, dtype=torch.long)
            self.W_sub = sae.W_dec[self.idx].detach()  # (k, d_model)

    def _apply(self, h: torch.Tensor, skip_first: bool) -> torch.Tensor:
        x = h[:, 1:] if skip_first else h
        if x.shape[1] == 0:
            return h
        xf = x.float()
        with torch.no_grad():
            acts = self.sae.encode(xf)
            if self.mode == "splice":
                if self.idx is not None:
                    acts[..., self.idx] = 0.0
                new = self.sae.decode(acts)
            else:  # ablate
                new = xf - acts[..., self.idx] @ self.W_sub
        new = new.to(h.dtype)
        if skip_first:
            return torch.cat([h[:, :1], new], dim=1)
        return new

    def __call__(self, module, inputs, output):
        if self.mode == "none" or (self.mode == "ablate" and self.idx is None):
            return output
        h = output[0] if isinstance(output, tuple) else output
        # Prefill passes include BOS at position 0; decode steps (seq len 1) never do.
        new = self._apply(h, skip_first=h.shape[1] > 1)
        return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new


def attach(model, intervention: Intervention):
    return model.model.layers[LAYER].register_forward_hook(intervention)


def chat_ids(tok, prompt: str) -> torch.Tensor:
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt"
    )
    if not isinstance(ids, torch.Tensor):  # BatchEncoding in some versions
        ids = ids["input_ids"]
    return ids


def generate(model, tok, input_ids: torch.Tensor, *, n: int, seed: int, max_new_tokens: int,
             temperature: float = 0.8, top_p: float = 0.95) -> list[str]:
    """n samples for one prompt. Same seed and shapes give a paired RNG stream across conditions."""
    end_ids = [tok.eos_token_id]
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(eot, int) and eot >= 0 and eot != tok.unk_token_id:
        end_ids.append(eot)
    ids = input_ids.to(model.device).repeat(n, 1)
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(
            ids, attention_mask=torch.ones_like(ids), do_sample=True, temperature=temperature,
            top_p=top_p, max_new_tokens=max_new_tokens, eos_token_id=end_ids,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        )
    return [tok.decode(row[ids.shape[1]:], skip_special_tokens=True) for row in out]


class PairedHook:
    """Layer-LAYER hook for paired batches: ablates `features` (error term kept)
    on the rows flagged in `row_mask`, never at padding or BOS positions."""

    def __init__(self, sae, features: list[int] | None, mode: str = "ablate"):
        assert mode in ("ablate", "splice")
        self.sae, self.mode = sae, mode
        self.idx = None
        if features:
            self.idx = torch.tensor(sorted(features), device=sae.W_dec.device, dtype=torch.long)
            self.W_sub = sae.W_dec[self.idx].detach()
        self.row_mask = None      # (R,) bool: rows to intervene on
        self.prefill_mask = None  # (R, L) bool: real, non-BOS prompt positions

    def __call__(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if self.row_mask is None or not bool(self.row_mask.any()):
            return output
        if self.mode == "ablate" and self.idx is None:
            return output
        if h.shape[1] > 1:
            pos = self.prefill_mask & self.row_mask[:, None]
        else:
            pos = self.row_mask[:, None].expand(-1, 1)
        xf = h.float()
        with torch.no_grad():
            acts = self.sae.encode(xf)
            if self.mode == "splice":
                if self.idx is not None:
                    acts[..., self.idx] = 0.0
                new = self.sae.decode(acts)
            else:
                new = xf - acts[..., self.idx] @ self.W_sub
        out = torch.where(pos[..., None], new, xf).to(h.dtype)
        return (out,) + tuple(output[1:]) if isinstance(output, tuple) else out


def _end_ids(tok) -> list[int]:
    ids = [tok.eos_token_id]
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(eot, int) and eot >= 0 and eot != tok.unk_token_id:
        ids.append(eot)
    return ids


def generate_paired(model, tok, sae, prompt_ids: list[torch.Tensor], seeds: list[int], *, n: int,
                    features: list[int], mode: str = "ablate", max_new_tokens: int = 150,
                    temperature: float = 0.8, top_p: float = 0.95, prefilter_k: int = 1024):
    """n normal and n intervened samples for each prompt, decoded in one batch.

    Common random numbers: the normal and intervened rows of sample k of a prompt
    draw the same uniform at every step (inverse-CDF sampling over the top-p
    nucleus), so a pair stays token-identical until the intervention changes the
    next-token distribution. Uniforms come from a CPU generator seeded per prompt,
    so runs are reproducible regardless of batch composition.

    Returns [{"normal": [n texts], "intervened": [n texts]} for each prompt].
    """
    dev = model.device
    G = len(prompt_ids)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    L = max(int(p.shape[-1]) for p in prompt_ids)
    rows, masks, partner_u, intervene = [], [], [], []
    U = []
    for g, (p, seed) in enumerate(zip(prompt_ids, seeds)):
        gen = torch.Generator().manual_seed(seed)
        U.append(torch.rand(n, max_new_tokens, generator=gen))
        p = p.view(-1)
        for cond in (0, 1):
            for k in range(n):
                rows.append(torch.cat([torch.full((L - p.numel(),), pad, dtype=p.dtype), p]))
                masks.append(torch.cat([torch.zeros(L - p.numel(), dtype=torch.long),
                                        torch.ones(p.numel(), dtype=torch.long)]))
                partner_u.append((g, k))
                intervene.append(cond == 1)
    ids = torch.stack(rows).to(dev)
    am = torch.stack(masks).to(dev)
    R = ids.shape[0]
    U = torch.stack(U)  # (G, n, T)
    u_rows = torch.stack([U[g, k] for g, k in partner_u]).to(dev)  # (R, T)

    first_real = (am.cumsum(-1) == 1) & (am == 1)  # BOS position per row
    hook = PairedHook(sae, features, mode)
    hook.row_mask = torch.tensor(intervene, device=dev)
    hook.prefill_mask = (am == 1) & ~first_real
    handle = model.model.layers[LAYER].register_forward_hook(hook)

    end = torch.tensor(_end_ids(tok), device=dev)
    finished = torch.zeros(R, dtype=torch.bool, device=dev)
    outs = torch.full((R, max_new_tokens), pad, dtype=torch.long, device=dev)
    pos = (am.cumsum(-1) - 1).clamp(min=0)
    try:
        with torch.no_grad():
            res = model(input_ids=ids, attention_mask=am, position_ids=pos, use_cache=True)
            cache = res.past_key_values
            last_pos = pos[:, -1]
            for t in range(max_new_tokens):
                logits = res.logits[:, -1].float() / max(temperature, 1e-6)
                probs = torch.softmax(logits, dim=-1)
                topv, topi = probs.topk(prefilter_k, dim=-1)
                csum = topv.cumsum(-1)
                if bool((csum[:, -1] < top_p).any()):  # nucleus wider than the shortlist: exact sort
                    topv, topi = probs.sort(dim=-1, descending=True)
                    csum = topv.cumsum(-1)
                keep = (csum - topv) < top_p
                filt = topv * keep
                cdf = filt.cumsum(-1) / filt.sum(-1, keepdim=True)
                choice = (cdf < u_rows[:, t:t + 1]).sum(-1).clamp(max=cdf.shape[-1] - 1)
                nxt = topi.gather(-1, choice[:, None]).squeeze(-1)
                nxt = torch.where(finished, torch.full_like(nxt, pad), nxt)
                outs[:, t] = nxt
                finished |= torch.isin(nxt, end)
                if bool(finished.all()):
                    break
                am = torch.cat([am, torch.ones(R, 1, dtype=am.dtype, device=dev)], dim=1)
                last_pos = last_pos + 1
                res = model(input_ids=nxt[:, None], attention_mask=am, position_ids=last_pos[:, None],
                            past_key_values=cache, use_cache=True)
                cache = res.past_key_values
    finally:
        handle.remove()

    texts = []
    for r in range(R):
        row = outs[r].tolist()
        cut = next((i for i, x in enumerate(row) if x in set(end.tolist())), len(row))
        texts.append(tok.decode(row[:cut], skip_special_tokens=True))
    result = []
    for g in range(G):
        base = g * 2 * n
        result.append({"normal": texts[base:base + n], "intervened": texts[base + n:base + 2 * n]})
    return result


_CURLY = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def normalise(text: str) -> str:
    """The detectors match ASCII apostrophes only; normalise before scoring."""
    return text.translate(_CURLY)


def score(text: str) -> dict:
    import sys

    sys.path.insert(0, str(REPO / "src"))
    from classifier.detect import detect_construction
    from classifier.detect_v2 import detect_construction_v2, detect_more_than_just

    t = normalise(text)
    return {
        "strict": bool(detect_construction(t, strict=False)),
        "family": bool(detect_construction_v2(t)),
        "cousin": bool(detect_more_than_just(t)),
        "but": bool(re.search(r"\bbut\b", t, re.I)),
        "words": len(re.findall(r"\b\w[\w'-]*\b", t)),
    }
