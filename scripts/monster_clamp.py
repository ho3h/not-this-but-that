"""Golden-Gate-style clamping: pin "not X but Y" features on at every word (exploratory).

Golden Gate Claude (Anthropic, May 2024) clamped one feature to 10x its maximum
activation. Here a set of features is clamped at layer 20 to s x its typical maximum
(the 99.9th percentile of its activation over confirmation v4's normal answers): at
every position except BOS, the feature's own contribution is replaced by
clamp x W_dec, leaving the SAE error and every other feature untouched.

Sets: all 25; the top 5; 13565 alone ("comparison or enhancement"); 15646 alone
("conditions or exceptions"). Strengths s = 1, 2, 4. Ten mundane prompts, one
sample, 140 tokens, common random numbers. Writes reports/monster_clamp.{json,md}
and reports/monster_clamp_generations.jsonl.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import dose_response as DR  # noqa: E402
import v2_engine as E  # noqa: E402
import v2_mlx as M  # noqa: E402
import v2_mlx_multi as MM  # noqa: E402

sys.path.insert(0, str(E.REPO / "src"))
from classifier.detect_v3 import detect_v3  # noqa: E402

REPO = E.REPO
PROMPTS = ["Give me a simple recipe for scrambled eggs.", "What's the capital of France?", "How do I boil an egg?",
           "Write a two-line out-of-office reply.", "Describe a cat sitting on a windowsill.",
           "How do I fold a fitted sheet?", "List five vegetables that grow well in pots.",
           "Explain how a refrigerator keeps food cold.", "How many legs does a spider have?",
           "Write a short note to tell my neighbour I'll be away next week."]
STRENGTHS = (1, 2, 4)
MAX_NEW, SEED = 140, 20261400


class SAEClamp(nn.Module):
    """row_clamp is (B, |U|): NaN = leave the feature alone; a number = clamp it to that activation."""

    def __init__(self, inner: MM.SAEMaskedAblate):
        super().__init__()
        self.inner = inner
        self.row_clamp = None

    def __call__(self, x, mask=None, cache=None):
        h = self.inner.layer(x, mask, cache)
        if self.row_clamp is None:
            return h
        w = self.inner
        hf = h.astype(mx.float32)
        pre = hf @ w.W_enc_U + w.b_enc_U
        acts = mx.where(pre > w.thr_U, pre, 0.0)
        target = mx.where(mx.isnan(self.row_clamp)[:, None, :], acts, mx.nan_to_num(self.row_clamp)[:, None, :])
        delta = (target - acts) @ w.W_dec_U
        if h.shape[1] > 1:
            pos = mx.concatenate([mx.zeros((1, 1, 1)), mx.ones((1, h.shape[1] - 1, 1))], axis=1)
            delta = delta * pos
        return (hf + delta).astype(h.dtype)


def typical_max(model, tok, w, chat):
    rows = [json.loads(l) for l in (REPO / "reports/confirmation_v4_generations.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["condition"] == "normal"][::4]
    captured = {}
    orig = type(w).__call__

    def spy(self, x, mask=None, cache=None):
        h = self.layer(x, mask, cache)
        captured["h"] = h
        return h

    type(w).__call__ = spy
    w.row_mask = None
    vals = [[] for _ in chat]
    pos = [w.order.index(f) for f in chat]
    for r in rows:
        pt = M.chat_tokens(tok, r["prompt"])
        model(mx.array([pt + tok.encode(r["text"], add_special_tokens=False)]))
        hf = captured["h"].astype(mx.float32)[0, len(pt):]
        pre = np.array(hf @ w.W_enc_U + w.b_enc_U)
        on = pre > np.array(w.thr_U)
        for j, c in enumerate(pos):
            vals[j].extend(pre[on[:, c], c].tolist())
    type(w).__call__ = orig
    return {f: float(np.percentile(v, 99.9)) if v else 0.0 for f, v in zip(chat, vals)}


def main() -> None:
    chat = json.loads((REPO / "reports/confirmation_v3_prereg.json").read_text())["chat_features"]
    model, tok, wraps = MM.load_multi("google/gemma-2-2b-it", {20: chat})
    w = wraps[20]
    tmax = typical_max(model, tok, w, chat)
    print("typical max", {k: round(v, 1) for k, v in list(tmax.items())[:6]}, flush=True)
    clamp = SAEClamp(w)
    model.model.layers[20] = clamp
    sets = {"all 25": chat, "top 5": chat[:5], "13565": [13565], "15646": [15646]}
    conds = [("normal", None, 0)] + [(f"{name} x{s}", feats, s) for name, feats in sets.items() for s in STRENGTHS]
    rc = np.full((len(conds), len(w.order)), np.nan, dtype=np.float32)
    for c, (_, feats, s) in enumerate(conds):
        for f in feats or []:
            rc[c, w.order.index(f)] = s * tmax[f]
    clamp.row_clamp = mx.array(rc)
    rows = []
    for i, p in enumerate(PROMPTS):
        U = np.random.default_rng(SEED + i).random((1, MAX_NEW)).astype(np.float32)
        outs = MM.sample_rows(model, tok, M.chat_tokens(tok, p), mx.array(np.tile(U, (len(conds), 1))),
                              max_new_tokens=MAX_NEW, temperature=0.8, top_p=0.95)
        for c, (name, _, _) in enumerate(conds):
            rows.append({"id": i, "prompt": p, "condition": name, "text": outs[c]})
        print(f"{i + 1}/{len(PROMPTS)}", flush=True)
    (REPO / "reports/monster_clamp_generations.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    res = {}
    for name, _, _ in conds:
        cs = [r["text"] for r in rows if r["condition"] == name]
        res[name] = {"not_x_but_y": sum(bool(detect_v3(t)["family"] or detect_v3(t)["cousin"]) for t in cs),
                     "degenerate": sum(DR.degenerate(t) for t in cs), "answers": len(cs)}
    (REPO / "reports/monster_clamp.json").write_text(json.dumps({"typical_max": tmax, "results": res}, indent=2))
    md = ["# Clamping the features on (exploratory)", "", "10 mundane prompts, 1 sample, 140 tokens.", "",
          "| condition | answers using not X but Y | degenerate |", "|---|---:|---:|"]
    md += [f"| {k} | {v['not_x_but_y']} / {v['answers']} | {v['degenerate']} |" for k, v in res.items()]
    (REPO / "reports/monster_clamp.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
