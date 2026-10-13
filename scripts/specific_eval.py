"""Evaluate the specific switch against the current set (exploratory).

Conditions: normal, the current set, the specific set (reports/specific_selection_{model}.json).
  habit    D2 prompts, 2 samples, 200 tokens, common random numbers: habit rate,
           negation words per 1,000, "don't"-type contractions
  yes/no   the 30 'no' questions of data/negation_check_questions.json
  never    the 30 'never' prompts, 2 samples, 200 tokens; writes blind pairs
           normal vs specific to data/specific_never_pairs_{model}.txt (+ key)

  .venv/bin/python scripts/specific_eval.py gemma2
"""

from __future__ import annotations

import json
import random
import re
import sys

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import g3_mlx as G  # noqa: E402
import v2_engine as E  # noqa: E402
import v2_mlx as M  # noqa: E402
import v2_mlx_multi as MM  # noqa: E402

sys.path.insert(0, str(E.REPO / "src"))
from classifier.detect_v3 import detect_v3  # noqa: E402

REPO = E.REPO
NEG = re.compile(r"\bnot\b|n't\b|n’t\b|\bno\b|\bnever\b", re.I)
CONTR = re.compile(r"\b(don't|don’t|doesn't|doesn’t|can't|can’t|won't|won’t|didn't|didn’t)\b", re.I)
N, MAX_NEW, SEED = 2, 200, 20261050


def main() -> None:
    which = sys.argv[1]
    sel = json.loads((REPO / f"reports/specific_selection_{which}.json").read_text())
    conds = {"normal": [], "current": sel["current_set"], "specific": sel["specific_set"]}
    names = list(conds)
    L = sel["layer"]
    loader = MM.load_multi if which == "gemma2" else G.load_multi
    model, tok, wraps = loader(sel["model"], {L: sorted(set(conds["current"]) | set(conds["specific"]))})
    data = json.loads((REPO / "data/negation_check_questions.json").read_text())

    # yes/no
    MM.set_row_masks(wraps, [{L: set(conds[n])} for n in names], 1)
    yes_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes", " Yes")})
    no_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no", " No")})
    no_right = {n: 0 for n in names}
    for x in [q for q in data["yes_no"] if q["a"] == "no"]:
        pr = mx.softmax(model(mx.array([M.chat_tokens(tok, x["q"] + " Answer with yes or no.")] * len(names)))[:, -1, :].astype(mx.float32), axis=-1)
        py, pn = np.array(mx.sum(pr[:, yes_ids], axis=-1)), np.array(mx.sum(pr[:, no_ids], axis=-1))
        for c, n in enumerate(names):
            no_right[n] += int(pn[c] > py[c])

    # habit + never generations
    MM.set_row_masks(wraps, [{L: set(conds[n])} for n in names], N)
    d2 = json.loads((REPO / "data/D2_neutral_prompts.json").read_text())["prompts"]
    jobs = [("D2", i, p) for i, p in enumerate(d2)] + [("never", i, q) for i, q in enumerate(data["never"])]
    gens = []
    for j, (kind, i, p) in enumerate(jobs):
        U = np.random.default_rng(SEED + (0 if kind == "D2" else 5000) + i).random((N, MAX_NEW)).astype(np.float32)
        outs = MM.sample_rows(model, tok, M.chat_tokens(tok, p), mx.array(np.tile(U, (len(names), 1))),
                              max_new_tokens=MAX_NEW, temperature=0.8, top_p=0.95)
        for c, n in enumerate(names):
            for k in range(N):
                gens.append({"kind": kind, "id": i, "prompt": p, "condition": n, "sample": k, "text": outs[c * N + k]})
        if j % 10 == 0:
            print(f"{j + 1}/{len(jobs)}", flush=True)
    (REPO / f"reports/specific_eval_{which}_generations.jsonl").write_text("\n".join(json.dumps(g) for g in gens) + "\n")

    res = {}
    for n in names:
        cs = [g["text"] for g in gens if g["kind"] == "D2" and g["condition"] == n]
        words = sum(len(t.split()) for t in cs)
        res[n] = {"size": len(conds[n]), "habit": sum(bool(detect_v3(t)["family"] or detect_v3(t)["cousin"]) for t in cs),
                  "answers": len(cs), "negations_per_1k": 1000 * sum(len(NEG.findall(t)) for t in cs) / words,
                  "contractions": sum(len(CONTR.findall(t)) for t in cs), "no_questions_right": no_right[n]}
    (REPO / f"reports/specific_eval_{which}.json").write_text(json.dumps(res, indent=2))
    for n, d in res.items():
        print(n, d)

    by = {}
    for g in gens:
        if g["kind"] == "never":
            by.setdefault((g["prompt"], g["sample"]), {})[g["condition"]] = g["text"]
    keys = sorted(by); rng = random.Random(77); rng.shuffle(keys)
    lines, key = [], []
    for i, k in enumerate(keys, 1):
        flip = rng.random() < 0.5
        a, b = ("specific", "normal") if flip else ("normal", "specific")
        key.append({"item": i, "q": k[0], "sample": k[1], "A": a, "B": b})
        lines.append(f"### PAIR {i}\nQUESTION: {k[0]}\nANSWER A: {by[k][a].replace(chr(10), ' ⏎ ')}\nANSWER B: {by[k][b].replace(chr(10), ' ⏎ ')}\n")
    (REPO / f"data/specific_never_pairs_{which}.txt").write_text("\n".join(lines))
    (REPO / f"data/specific_never_pairs_{which}_key.json").write_text(json.dumps(key, indent=1))


if __name__ == "__main__":
    main()
