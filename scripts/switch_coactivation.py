"""Which of the 25 switches fire together? Co-activation measured in the chatbot's own answers.

Runs gemma-2-2b-it over the normal answers of confirmation v4 (one per prompt and
sample, chat template), reads the layer-20 SAE activations of the 25 at every
answer token, and scores each pair by lift: P(both fire) / (P(a) P(b)), keeping
pairs that fire together on at least 30 tokens. This is the same relationship
the Neo4j graph stores as CO_ACTIVATES_WITH, recomputed directly for the 25.

Writes reports/switch_coactivation.json.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import v2_engine as E  # noqa: E402
import v2_mlx as M  # noqa: E402
import v2_mlx_multi as MM  # noqa: E402

REPO = E.REPO


def main() -> None:
    chat = json.loads((REPO / "reports/confirmation_v3_prereg.json").read_text())["chat_features"]
    model, tok, wraps = MM.load_multi("google/gemma-2-2b-it", {20: chat})
    w = wraps[20]
    w.row_mask = None
    pos = [w.order.index(f) for f in chat]
    captured = {}
    orig = type(w).__call__

    def spy(self, x, mask=None, cache=None):
        h = self.layer(x, mask, cache)
        captured["h"] = h
        return h

    type(w).__call__ = spy
    rows = [json.loads(l) for l in (REPO / "reports/confirmation_v4_generations.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["condition"] == "normal"]
    fires = []
    for i, r in enumerate(rows):
        pt = M.chat_tokens(tok, r["prompt"])
        ids = pt + tok.encode(r["text"], add_special_tokens=False)
        model(mx.array([ids]))
        h = captured["h"].astype(mx.float32)[0, len(pt):]
        pre = h @ w.W_enc_U + w.b_enc_U
        on = np.array(pre > w.thr_U)[:, pos]
        fires.append(on)
        if i % 100 == 0:
            print(i, len(rows), flush=True)
    type(w).__call__ = orig
    F = np.concatenate(fires).astype(np.float64)
    n = F.shape[0]
    p = F.mean(axis=0)
    both = F.T @ F
    pairs = []
    for a in range(len(chat)):
        for b in range(a + 1, len(chat)):
            if both[a, b] >= 30 and p[a] > 0 and p[b] > 0:
                lift = (both[a, b] / n) / (p[a] * p[b])
                pairs.append({"a": chat[a], "b": chat[b], "together": int(both[a, b]), "lift": float(lift)})
    pairs.sort(key=lambda x: -x["lift"])
    out = {"tokens": n, "answers": len(rows), "firing_rate": {str(f): float(p[i]) for i, f in enumerate(chat)}, "pairs": pairs}
    (REPO / "reports/switch_coactivation.json").write_text(json.dumps(out, indent=2))
    print(n, "tokens;", len(pairs), "pairs; top:", pairs[:5])


if __name__ == "__main__":
    main()
