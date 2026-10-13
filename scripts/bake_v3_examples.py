"""Bake story-page playback examples from the pre-registered confirmation v3 run.

Each example has two versions of one answer, decoded together with common
random numbers: normal, and the 25 chat-chosen features switched off. For every
answer token, records the top-10 layer-20 SAE features (switched-off features
read as zero). Writes web/demo/examples.json in the format web/demo/app.js
plays: runs.baseline and runs.off.

  .venv/bin/python scripts/bake_v3_examples.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch
import mlx.core as mx

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import v2_engine as E  # noqa: E402
import v2_mlx as M  # noqa: E402
import v2_mlx_multi as MM  # noqa: E402

REPO = E.REPO
PICKS = [  # (prompt index, sample, card label)
    (16, 3, "A city's coffee culture"),
    (80, 3, "Learning calligraphy"),
    (95, 4, "A neighbourhood bakery"),
]
GENS = REPO / "reports" / "confirmation_v3_generations.jsonl"
OUT = REPO / "web" / "demo" / "examples.json"


def records(model, tok, wrap, sae, prompt_tokens, answer: str, dose: dict, off: list[int]):
    ans_ids = tok.encode(answer, add_special_tokens=False)
    MM.set_row_masks({20: wrap}, [dose], 1)
    captured = {}
    cls = type(wrap)
    orig = cls.__call__

    def spy(self, x, mask=None, cache=None):
        h = orig(self, x, mask, cache)
        captured["h"] = np.array(h.astype(mx.float32))
        return h

    cls.__call__ = spy
    try:
        model(mx.array([prompt_tokens + ans_ids]))
    finally:
        cls.__call__ = orig
    h = torch.tensor(captured["h"][0, len(prompt_tokens):])
    with torch.no_grad():
        acts = sae.encode(h.to(sae.W_dec.device)).cpu()
    if off:
        acts[:, torch.tensor(off)] = 0.0
    out = []
    for step, (tid, a) in enumerate(zip(ans_ids, acts)):
        vals, idx = a.topk(10)
        out.append({"step": step, "token_id": int(tid), "token_str": tok.decode([tid]),
                    "top_features": [{"idx": int(i), "act": float(v)} for v, i in zip(vals, idx) if v > 0]})
    return out


def main() -> None:
    prereg = json.loads((REPO / "reports/confirmation_v3_prereg.json").read_text())
    chat25 = prereg["chat_features"]
    rows = {}
    for line in GENS.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows.setdefault((r["prompt_idx"], r["sample"]), {})[r["condition"]] = r
    model, tok, wraps = MM.load_multi("google/gemma-2-2b-it", {20: sorted(chat25)})
    sae = E.load_sae()
    examples = []
    for pi, k, label in PICKS:
        trio = rows[(pi, k)]
        pt = M.chat_tokens(tok, trio["normal"]["prompt"])
        runs = {
            "baseline": {"ablate_label": "normal", "ablate_set": [], "completion": trio["normal"]["text"],
                         "records": records(model, tok, wraps[20], sae, pt, trio["normal"]["text"], {}, [])},
            "off": {"ablate_label": "25 off", "ablate_set": chat25, "completion": trio["chat 25"]["text"],
                    "records": records(model, tok, wraps[20], sae, pt, trio["chat 25"]["text"], {20: set(chat25)}, chat25)},
        }
        examples.append({"id": f"v3-{pi}-{k}", "label": label, "prompt": trio["normal"]["prompt"],
                         "model": "gemma-2-2b-it", "max_new_tokens": 150, "runs": runs})
        print(label, {k2: len(v["records"]) for k2, v in runs.items()})
    OUT.write_text(json.dumps({"source": "reports/confirmation_v3_generations.jsonl (pre-registered confirmation v3)",
                               "off_set": chat25, "examples": examples}))


if __name__ == "__main__":
    main()
