"""Bake story-page playback examples from the pre-registered confirmation v2 run.

Takes chosen (prompt id, sample) pairs from reports/confirmation_v2_generations.jsonl
and records, for every answer token, the top-10 layer-20 SAE features (normal run:
unmodified model; features-off run: the coalition switched off, so its features
read as zero). Writes web/demo/examples.json in the format web/demo/app.js plays.

  .venv/bin/python scripts/bake_v2_examples.py
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

PICKS = [  # (prompt id, sample, card label)
    (11, 6, "A village duck pond"),
    (22, 5, "A town tug-of-war"),
    (21, 9, "A seaside beach hut"),
]
GENS = E.REPO / "reports" / "confirmation_v2_generations.jsonl"
OUT = E.REPO / "web" / "demo" / "examples.json"


def records(model, tok, wrap, sae, prompt_tokens, answer: str, intervene: bool):
    ans_ids = tok.encode(answer, add_special_tokens=False)
    captured = {}
    cls = type(wrap)
    orig = cls.__call__

    def spy(self, x, mask=None, cache=None):
        h = orig(self, x, mask, cache)
        captured["h"] = np.array(h.astype(mx.float32))
        return h

    cls.__call__ = spy
    wrap.row_mask = mx.array([1.0 if intervene else 0.0])
    try:
        model(mx.array([prompt_tokens + ans_ids]))
    finally:
        cls.__call__ = orig
        wrap.row_mask = None
    h = torch.tensor(captured["h"][0, len(prompt_tokens):])
    with torch.no_grad():
        acts = sae.encode(h.to(sae.W_dec.device)).cpu()
    if intervene:
        acts[:, torch.tensor(E.COALITION_25)] = 0.0
    out = []
    for step, (tid, a) in enumerate(zip(ans_ids, acts)):
        vals, idx = a.topk(10)
        out.append({"step": step, "token_id": int(tid), "token_str": tok.decode([tid]),
                    "top_features": [{"idx": int(i), "act": float(v)} for v, i in zip(vals, idx) if v > 0]})
    return out


def main() -> None:
    rows = {(r["id"], r["sample"]): r for r in map(json.loads, GENS.read_text().splitlines()) if r["stage"] == 1}
    model, tok, wrap = M.load("google/gemma-2-2b-it", E.COALITION_25)
    sae = E.load_sae()
    examples = []
    for pid, k, label in PICKS:
        r = rows[(pid, k)]
        pt = M.chat_tokens(tok, r["prompt"])
        base = {"ablate_label": "normal", "ablate_set": [], "completion": r["baseline"],
                "records": records(model, tok, wrap, sae, pt, r["baseline"], False)}
        off = {"ablate_label": "all 25 features off", "ablate_set": E.COALITION_25, "completion": r["coalition"],
               "records": records(model, tok, wrap, sae, pt, r["coalition"], True)}
        examples.append({"id": f"v2-{pid}-{k}", "label": label, "prompt": r["prompt"], "model": "gemma-2-2b-it",
                         "max_new_tokens": 150, "seed": f"{20261002 + 10 * pid + 1}/{k}",
                         "runs": {"baseline": base, "top25": off, "minimal": off}})
        print(label, len(base["records"]), len(off["records"]))
    OUT.write_text(json.dumps({"source": "reports/confirmation_v2_generations.jsonl (pre-registered confirmation v2)",
                               "top25_set": E.COALITION_25, "minimal_set": E.CORE_2, "examples": examples}))


if __name__ == "__main__":
    main()
