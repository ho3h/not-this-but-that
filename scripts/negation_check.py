"""Can the model still say no? A negation check for the switched-off features (exploratory).

The decision-point features also lower the overall rate of negation words in
answers (not, n't, no, never). This checks whether that is a loss of ability or
a change of habit:

  yes/no   60 factual questions (30 'no', 30 'yes'), chat template plus
           "Answer with yes or no."; the prediction is whichever of Yes/No gets
           more next-token probability. Reports accuracy on each half and the
           mean probability given to the right answer.
  never    30 requests whose natural answer lists things not to do; 2 samples
           each, 200 tokens, common random numbers. Reports how many answers
           contain any negation or "avoid", and negation words per 1,000 words.

  .venv/bin/python scripts/negation_check.py gemma2   # normal / 25 off / matched 25 (v4's control)
  .venv/bin/python scripts/negation_check.py gemma3   # normal / A / B / C / matched (layer 22, g3_pilot sets)

Writes reports/negation_check_{model}.json and appends a section to reports/negation_check.md.
"""

from __future__ import annotations

import json
import re
import sys

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import g3_mlx as G  # noqa: E402
import v2_engine as E  # noqa: E402
import v2_mlx as M  # noqa: E402
import v2_mlx_multi as MM  # noqa: E402

REPO = E.REPO
NEG = re.compile(r"\bnot\b|n't\b|n’t\b|\bno\b|\bnever\b", re.I)
PROHIBIT = re.compile(r"\bnot\b|n't\b|n’t\b|\bno\b|\bnever\b|\bavoid", re.I)
N, MAX_NEW, SEED = 2, 200, 20261040


def setup(which: str):
    if which == "gemma2":
        v4 = json.loads((REPO / "reports/confirmation_v4_prereg.json").read_text())
        conds = {"normal": [], "25 off": v4["chat_features"], "matched 25": v4["matched_control"]}
        layer = 20
        model, tok, wraps = MM.load_multi("google/gemma-2-2b-it", {layer: sorted(set().union(*map(set, conds.values())))})
    else:
        pilot = json.loads((REPO / "reports/g3_pilot.json").read_text())["sets"]
        conds = {"normal": [], "A": pilot["A"], "B": pilot["B"], "C": pilot["C"], "matched 25": pilot["M"]}
        layer = 22
        model, tok, wraps = G.load_multi("google/gemma-3-4b-it", {layer: sorted(set().union(*map(set, conds.values())))})
    return model, tok, wraps, layer, conds


def main() -> None:
    which = sys.argv[1]
    model, tok, wraps, layer, conds = setup(which)
    names = list(conds)
    data = json.loads((REPO / "data/negation_check_questions.json").read_text())
    yes_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes", " Yes")})
    no_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no", " No")})

    MM.set_row_masks(wraps, [{layer: set(conds[n])} for n in names], 1)
    correct = {n: {"yes": 0, "no": 0} for n in names}
    p_right = {n: {"yes": [], "no": []} for n in names}
    for item in data["yes_no"]:
        ids = M.chat_tokens(tok, item["q"] + " Answer with yes or no.")
        pr = mx.softmax(model(mx.array([ids] * len(names)))[:, -1, :].astype(mx.float32), axis=-1)
        py = np.array(mx.sum(pr[:, yes_ids], axis=-1)); pn = np.array(mx.sum(pr[:, no_ids], axis=-1))
        for c, n in enumerate(names):
            pick = "yes" if py[c] > pn[c] else "no"
            correct[n][item["a"]] += int(pick == item["a"])
            right = py[c] if item["a"] == "yes" else pn[c]
            p_right[n][item["a"]].append(float(right / (py[c] + pn[c])))

    MM.set_row_masks(wraps, [{layer: set(conds[n])} for n in names], N)
    gens = []
    for i, q in enumerate(data["never"]):
        U = np.random.default_rng(SEED + i).random((N, MAX_NEW)).astype(np.float32)
        outs = MM.sample_rows(model, tok, M.chat_tokens(tok, q), mx.array(np.tile(U, (len(names), 1))),
                              max_new_tokens=MAX_NEW, temperature=0.8, top_p=0.95)
        for c, n in enumerate(names):
            for k in range(N):
                gens.append({"q": q, "condition": n, "sample": k, "text": outs[c * N + k]})
        print(f"never {i + 1}/{len(data['never'])}", flush=True)
    (REPO / f"reports/negation_check_{which}_generations.jsonl").write_text("\n".join(json.dumps(g) for g in gens) + "\n")

    n_no = sum(1 for x in data["yes_no"] if x["a"] == "no"); n_yes = len(data["yes_no"]) - n_no
    res = {}
    for n in names:
        cs = [g["text"] for g in gens if g["condition"] == n]
        words = sum(len(t.split()) for t in cs)
        res[n] = {"no_accuracy": correct[n]["no"] / n_no, "yes_accuracy": correct[n]["yes"] / n_yes,
                  "p_right_no": float(np.mean(p_right[n]["no"])), "p_right_yes": float(np.mean(p_right[n]["yes"])),
                  "never_answers_with_prohibition": sum(bool(PROHIBIT.search(t)) for t in cs) / len(cs),
                  "never_negations_per_1k_words": 1000 * sum(len(NEG.findall(t)) for t in cs) / words}
    (REPO / f"reports/negation_check_{which}.json").write_text(json.dumps(res, indent=2))
    md = [f"## {which}", "", "| condition | 'no' questions right | 'yes' questions right | P(right), 'no' qs | 'never' answers with a prohibition | negations per 1k words |",
          "|---|---:|---:|---:|---:|---:|"]
    for n, d in res.items():
        md.append(f"| {n} | {round(d['no_accuracy'] * n_no)}/{n_no} | {round(d['yes_accuracy'] * n_yes)}/{n_yes} | {d['p_right_no']:.2f} | "
                  f"{100 * d['never_answers_with_prohibition']:.0f}% | {d['never_negations_per_1k_words']:.1f} |")
    out = REPO / "reports/negation_check.md"
    head = "" if out.exists() else ("# Can the model still say no? (exploratory)\n\n"
                                    "See `scripts/negation_check.py`. Questions: `data/negation_check_questions.json`.\n\n")
    out.write_text((out.read_text() if out.exists() else "") + head + "\n".join(md) + "\n\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
