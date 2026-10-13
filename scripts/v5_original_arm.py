"""The original 25 on confirmation v5's fresh prompts, for a like-for-like comparison (exploratory).

Confirmation v5 (Gemma 2) tested the specific switch on D7 + D6. This runs the
original 25 (confirmation v3/v4 set) on the same prompts with the same seeds,
normal and switched-off rows decoded together, and scores them with the same
measures: habit, ordinary negation rate, and blind warning-clarity pairs (written
to data/v5_original_never_pairs.txt + key, same instruction as the registered H4).

  .venv/bin/python scripts/v5_original_arm.py            # generate + pairs
  .venv/bin/python scripts/v5_original_arm.py --analyze  # after labels exist
"""

from __future__ import annotations

import json
import random
import re
import sys

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import v2_engine as E  # noqa: E402
import v2_mlx as M  # noqa: E402
import v2_mlx_multi as MM  # noqa: E402

sys.path.insert(0, str(E.REPO / "src"))
from classifier.detect_v3 import detect_v3  # noqa: E402

REPO = E.REPO
PRE = json.loads((REPO / "reports/confirmation_v5_gemma2_prereg.json").read_text())
ORIG = json.loads((REPO / "reports/confirmation_v3_prereg.json").read_text())["chat_features"]
GENS = REPO / "reports/v5_original_arm_generations.jsonl"
PAIRS, KEY, LABELS = (REPO / "data/v5_original_never_pairs.txt", REPO / "data/v5_original_never_pairs_key.json",
                      REPO / "data/v5_original_never_pairs_labels.tsv")
NEG = re.compile(r"\bnot\b|n't\b|n’t\b|\bno\b|\bnever\b", re.I)
CONDS = ("normal", "original 25 off")


def generate() -> None:
    model, tok, wraps = MM.load_multi(PRE["model"], {20: ORIG})
    n, T = PRE["samples_per_prompt"], PRE["max_new_tokens"]
    prompts = json.loads((REPO / PRE["prompts_file"]).read_text())["prompts"]
    never = json.loads((REPO / PRE["never_prompts_file"]).read_text())["prompts"]
    with GENS.open("w") as f:
        MM.set_row_masks(wraps, [{}, {20: set(ORIG)}], n)
        for i, p in enumerate(prompts):
            U = np.random.default_rng(PRE["seed_base"] + p["id"]).random((n, T)).astype(np.float32)
            outs = MM.sample_rows(model, tok, M.chat_tokens(tok, p["prompt"]), mx.array(np.tile(U, (2, 1))),
                                  max_new_tokens=T, temperature=PRE["temperature"], top_p=PRE["top_p"])
            for c, name in enumerate(CONDS):
                for k in range(n):
                    f.write(json.dumps({"kind": "D7", "id": p["id"], "genre": p["genre"], "prompt": p["prompt"],
                                        "condition": name, "sample": k, "text": outs[c * n + k]}) + "\n")
            if i % 20 == 0:
                print(f"D7 {i + 1}/{len(prompts)}", flush=True)
        nn = PRE["never_samples"]
        MM.set_row_masks(wraps, [{}, {20: set(ORIG)}], nn)
        for i, q in enumerate(never):
            U = np.random.default_rng(PRE["seed_base"] + 1000 + i).random((nn, T)).astype(np.float32)
            outs = MM.sample_rows(model, tok, M.chat_tokens(tok, q), mx.array(np.tile(U, (2, 1))),
                                  max_new_tokens=T, temperature=PRE["temperature"], top_p=PRE["top_p"])
            for c, name in enumerate(CONDS):
                for k in range(nn):
                    f.write(json.dumps({"kind": "D6", "id": i, "prompt": q, "condition": name, "sample": k,
                                        "text": outs[c * nn + k]}) + "\n")
    rows = [json.loads(l) for l in GENS.read_text().splitlines() if l.strip()]
    by = {}
    for r in rows:
        if r["kind"] == "D6":
            by.setdefault((r["id"], r["sample"]), {"prompt": r["prompt"]})[r["condition"]] = r["text"]
    keys = sorted(by); rng = random.Random(2026); rng.shuffle(keys)
    lines, key = [], []
    for i, k in enumerate(keys, 1):
        flip = rng.random() < 0.5
        a, b = (CONDS[1], CONDS[0]) if flip else CONDS
        key.append({"item": i, "id": k[0], "sample": k[1], "A": a, "B": b})
        lines.append(f"### PAIR {i}\nQUESTION: {by[k]['prompt']}\nANSWER A: {by[k][a].replace(chr(10), ' ⏎ ')}\nANSWER B: {by[k][b].replace(chr(10), ' ⏎ ')}\n")
    PAIRS.write_text("\n".join(lines)); KEY.write_text(json.dumps(key, indent=1))
    print("done; pairs written")


def analyze() -> None:
    rows = [json.loads(l) for l in GENS.read_text().splitlines() if l.strip()]
    main = [r for r in rows if r["kind"] == "D7" and r["genre"] != "explicit"]

    def hit(t):
        d = detect_v3(t)
        return bool(d["family"] or d["cousin"])

    def ordinary(t):
        d = detect_v3(t)
        spans = [h.span for tier in ("family", "cousin", "reversed", "beyond") for h in d[tier]]
        return sum(1 for m in NEG.finditer(t) if not any(s - 2 <= m.start() <= e + 2 for s, e in spans))

    res = {}
    for c in CONDS:
        cs = [r["text"] for r in main if r["condition"] == c]
        words = sum(len(t.split()) for t in cs)
        res[c] = {"habit": sum(map(hit, cs)), "answers": len(cs), "ordinary_neg_per_1k": 1000 * sum(map(ordinary, cs)) / words}
    key = {k["item"]: k for k in json.loads(KEY.read_text())}
    w = t = l = 0
    for line in LABELS.read_text().splitlines():
        p = line.split("\t")
        if len(p) >= 2 and p[0].strip().isdigit():
            lab = p[1].strip().upper()[:1]
            if lab == "T":
                t += 1
            elif key[int(p[0])][lab] == CONDS[1]:
                w += 1
            else:
                l += 1
    res["warning_clarity"] = {"off_wins": w, "ties": t, "normal_wins": l, "share": (w + 0.5 * t) / max(1, w + t + l)}
    res["habit_drop"] = 1 - res[CONDS[1]]["habit"] / res["normal"]["habit"]
    res["ordinary_negation_ratio"] = res[CONDS[1]]["ordinary_neg_per_1k"] / res["normal"]["ordinary_neg_per_1k"]
    (REPO / "reports/v5_original_arm.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    analyze() if "--analyze" in sys.argv else generate()
