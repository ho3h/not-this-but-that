"""The "not X but Y" monster: how far can the 25 features be turned up? (exploratory)

Amplifies the 25 (confirmation v3 set) at layer 20 by x3, x5, x8 and x12 (mask -2,
-4, -7, -11: the features' contribution is multiplied by that factor at every
position except BOS) on deliberately mundane prompts, 2 samples each, 160 tokens,
common random numbers. Reports how many answers use "not X but Y" (detector v3
family or cousin), how many are degenerate (repeated 4-grams), and keeps every
answer for picking examples. Writes reports/monster.{json,md} and
reports/monster_generations.jsonl.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import dose_response as DR  # noqa: E402
import v2_engine as E  # noqa: E402
import v2_mlx as M  # noqa: E402
import v2_mlx_multi as MM  # noqa: E402

sys.path.insert(0, str(E.REPO / "src"))
from classifier.detect_v3 import detect_v3  # noqa: E402

REPO = E.REPO
PROMPTS = [
    "Give me a simple recipe for scrambled eggs.",
    "What's the capital of France?",
    "How do I change a flat tyre on a bike?",
    "Write a shopping list for a week of easy dinners.",
    "Explain how a refrigerator keeps food cold.",
    "What time zone is New York in?",
    "Describe a typical Monday morning commute.",
    "How do I boil an egg?",
    "Write a short note to tell my neighbour I'll be away next week.",
    "What are the rules of noughts and crosses?",
    "How do I reset a forgotten Wi-Fi password on my router?",
    "Describe a cat sitting on a windowsill.",
    "List five vegetables that grow well in pots.",
    "How many legs does a spider have?",
    "Write a two-line out-of-office reply.",
    "How do I unclog a bathroom sink?",
    "Describe the inside of a supermarket on a quiet Tuesday.",
    "What's the difference between a teaspoon and a tablespoon?",
    "How do I fold a fitted sheet?",
    "Explain what a library card is for.",
]
DOSES = [("normal", 0.0), ("x3", -2.0), ("x5", -4.0), ("x8", -7.0), ("x12", -11.0)]
N, MAX_NEW, SEED = 2, 160, 20261300


def main() -> None:
    chat = json.loads((REPO / "reports/confirmation_v3_prereg.json").read_text())["chat_features"]
    model, tok, wraps = MM.load_multi("google/gemma-2-2b-it", {20: chat})
    w = wraps[20]
    mask = np.zeros((len(DOSES) * N, len(w.order)), dtype=np.float32)
    for d, (_, m) in enumerate(DOSES):
        mask[d * N:(d + 1) * N, :] = m
    w.row_mask = mx.array(mask)
    rows = []
    for i, p in enumerate(PROMPTS):
        U = np.random.default_rng(SEED + i).random((N, MAX_NEW)).astype(np.float32)
        outs = MM.sample_rows(model, tok, M.chat_tokens(tok, p), mx.array(np.tile(U, (len(DOSES), 1))),
                              max_new_tokens=MAX_NEW, temperature=0.8, top_p=0.95)
        for d, (name, _) in enumerate(DOSES):
            for k in range(N):
                rows.append({"id": i, "prompt": p, "dose": name, "sample": k, "text": outs[d * N + k]})
        print(f"{i + 1}/{len(PROMPTS)}", flush=True)
    (REPO / "reports/monster_generations.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    res = {}
    for name, _ in DOSES:
        cs = [r["text"] for r in rows if r["dose"] == name]
        res[name] = {"answers": len(cs), "not_x_but_y": sum(bool(detect_v3(t)["family"] or detect_v3(t)["cousin"]) for t in cs),
                     "degenerate": sum(DR.degenerate(t) for t in cs)}
    (REPO / "reports/monster.json").write_text(json.dumps(res, indent=2))
    md = ["# The not X but Y monster (exploratory)", "", "20 mundane prompts x 2 samples, 160 tokens; the 25 amplified at layer 20.", "",
          "| dose | answers using not X but Y | degenerate |", "|---|---:|---:|"]
    for name, d in res.items():
        md.append(f"| {name} | {d['not_x_but_y']} / {d['answers']} | {d['degenerate']} |")
    (REPO / "reports/monster.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
