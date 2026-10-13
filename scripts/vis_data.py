"""Data for the v5 explainer visuals (reports/figures/v5/*.json), Gemma 2 2B-it.

1. fork_samples.json: 300 continuations of "A city's coffee culture" (answering "Explain how a
   city's coffee culture shapes its character."), sampled normally and with the 25 chat-chosen
   features ablated, 40 tokens each, T=1.0, top-p 0.95. Row i and row 300+i share their random
   numbers, so each pair differs only by the ablation.
2. lightup.json: one normal answer sentence that uses "not X but Y", with the activation of each of
   the 25 features at every token (layer 20, Gemma Scope), plus each feature's typical maximum
   (99.9th percentile over confirmation v4 normal answers, from reports/monster_clamp.json) and
   the feature map coordinates.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_mlx as M  # noqa: E402
import v2_mlx_multi as MM  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from classifier.detect_v3 import detect_v3  # noqa: E402

OUT = REPO / "reports/figures/v5"
OUT.mkdir(parents=True, exist_ok=True)
PROMPT = "Explain how a city's coffee culture shapes its character."
STEM = "A city's coffee culture"
N, MAX_NEW, SEED = 300, 40, 20261012


def main() -> None:
    chat = json.loads((REPO / "reports/confirmation_v3_prereg.json").read_text())["chat_features"]
    model, tok, wraps = MM.load_multi("google/gemma-2-2b-it", {20: chat})
    w = wraps[20]
    prefix = M.chat_tokens(tok, PROMPT) + tok.encode(STEM, add_special_tokens=False)

    MM.set_row_masks(wraps, [{}, {20: set(chat)}], N)
    U = np.random.default_rng(SEED).random((N, MAX_NEW)).astype(np.float32)
    outs = MM.sample_rows(model, tok, prefix, mx.array(np.vstack([U, U])), max_new_tokens=MAX_NEW,
                          temperature=1.0, top_p=0.95)
    rows = [{"condition": "normal" if i < N else "25 ablated", "pair": i % N, "text": t} for i, t in enumerate(outs)]
    for r in rows:
        first = re.split(r"(?<=[.!?])\s", STEM + r["text"])[0]
        d = detect_v3(first)
        r["first_sentence"] = first
        r["construction"] = bool(d["family"] or d["cousin"])
    (OUT / "fork_samples.json").write_text(json.dumps({"prompt": PROMPT, "stem": STEM, "rows": rows}, indent=1))
    for c in ("normal", "25 ablated"):
        print(c, sum(r["construction"] for r in rows if r["condition"] == c), "/", N, "first sentences use it", flush=True)

    # one normal sentence of the canonical shape, for the light-up animation
    cands = [r for r in rows if r["condition"] == "normal" and re.match(r" isn't just [^.;,]{3,60}[;,] it's ", r["text"])]
    pick = min(cands, key=lambda r: len(r["first_sentence"])) if cands else None
    if pick is None:
        print("no canonical normal sentence found", flush=True)
        return
    text = pick["first_sentence"]
    MM.set_row_masks(wraps, [{}], 1)
    w.row_mask = None
    ids = M.chat_tokens(tok, PROMPT) + tok.encode(text, add_special_tokens=False)
    start = len(M.chat_tokens(tok, PROMPT))
    captured = {}
    orig = type(w).__call__

    def spy(self, x, mask=None, cache=None):
        h = self.layer(x, mask, cache)
        captured["h"] = h
        return h

    type(w).__call__ = spy
    model(mx.array([ids]))
    type(w).__call__ = orig
    hf = captured["h"].astype(mx.float32)[0, start:]
    pre = hf @ w.W_enc_U + w.b_enc_U
    acts = np.array(mx.where(pre > w.thr_U, pre, 0.0))
    tmax = json.loads((REPO / "reports/monster_clamp.json").read_text())["typical_max"]
    layout = json.loads((REPO / "web/demo/layout.json").read_text())["features"]
    xy = {f["idx"]: [f["x"], f["y"]] for f in layout}
    (OUT / "lightup.json").write_text(json.dumps({
        "prompt": PROMPT, "text": text, "tokens": [tok.decode([t]) for t in ids[start:]],
        "features": w.order, "acts": acts.round(3).tolist(),
        "typical_max": [tmax[str(f)] if str(f) in tmax else tmax[f] for f in w.order],
        "xy": [xy[f] for f in w.order]}, indent=1))
    print("light-up sentence:", text, flush=True)


if __name__ == "__main__":
    main()
