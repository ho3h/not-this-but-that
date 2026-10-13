"""Gemma 3 pilot: which layer-22 feature set to pre-register (exploratory).

In Gemma 3 the "more than just X. It's Y" cousin is as common as the negated
family, and the family's top features barely touch it at the decision point
(g3_ladder_L22.json), so the Gemma 2 recipe (top 25 of the family ranking) may
leave half the habit. This pilot compares candidate sets on D2 prompts
(2 samples, 200 tokens, common random numbers, one batch per prompt):

  normal
  A  family top 25                      (the Gemma 2 recipe)
  B  family top 20 + cousin top 5       (union)
  C  family top 15 + cousin top 10      (union)
  D  family top 25 + cousin top 10      (union)
  M  25 firing-rate-matched features    (ladder draw 0)

Writes reports/g3_pilot_generations.jsonl and reports/g3_pilot.{json,md}.
"""

from __future__ import annotations

import json
import re
import sys
import time

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import dose_response as DR  # noqa: E402
import g3_mlx as G  # noqa: E402
import v2_engine as E  # noqa: E402
import v2_mlx as M  # noqa: E402
import v2_mlx_multi as MM  # noqa: E402

sys.path.insert(0, str(E.REPO / "src"))
from classifier.detect_v3 import detect_v3  # noqa: E402

REPO = E.REPO
LAYER = 22
GENS = REPO / "reports/g3_pilot_generations.jsonl"
N, MAX_NEW, TEMP, TOP_P, SEED = 2, 200, 0.8, 0.95, 20261030
A_AN = re.compile(r"\ba (?:[aeio]\w+|un(?!i|e)\w*)", re.I)


def sets() -> dict[str, list[int]]:
    a = json.loads((REPO / f"reports/g3_attribution_L{LAYER}.json").read_text())
    fam = [x["feature_idx"] for x in a["family"]["ranked"]]
    cou = [x["feature_idx"] for x in a["cousin"]["ranked"]]
    union = lambda nf, nc: sorted(set(fam[:nf]) | set(cou[:nc]))  # noqa: E731
    lad = json.loads((REPO / f"reports/g3_ladder_L{LAYER}.json").read_text())
    return {"normal": [], "A": fam[:25], "B": union(20, 5), "C": union(15, 10), "D": union(25, 10),
            "M": lad["matched_draw0"]}


def generate() -> None:
    S = sets()
    names = list(S)
    model, tok, wraps = G.load_multi("google/gemma-3-4b-it", {LAYER: sorted(set().union(*map(set, S.values())))})
    MM.set_row_masks(wraps, [{LAYER: set(S[n])} for n in names], N)
    prompts = json.loads((REPO / "data/D2_neutral_prompts.json").read_text())["prompts"]
    done = {json.loads(l)["id"] for l in GENS.read_text().splitlines() if l.strip()} if GENS.exists() else set()
    t0 = time.time()
    for i, p in enumerate(prompts):
        if i in done:
            continue
        U = np.random.default_rng(SEED + i).random((N, MAX_NEW)).astype(np.float32)
        outs = MM.sample_rows(model, tok, M.chat_tokens(tok, p), mx.array(np.tile(U, (len(names), 1))),
                              max_new_tokens=MAX_NEW, temperature=TEMP, top_p=TOP_P)
        with GENS.open("a") as f:
            for c, n in enumerate(names):
                for k in range(N):
                    f.write(json.dumps({"id": i, "prompt": p, "condition": n, "sample": k, "text": outs[c * N + k]}) + "\n")
        print(f"prompt {i + 1}/{len(prompts)} · {(time.time() - t0) / 60:.1f} min", flush=True)


def analyze() -> None:
    S = sets()
    rows = [json.loads(l) for l in GENS.read_text().splitlines() if l.strip()]
    res = {}
    for n in S:
        cs = [r for r in rows if r["condition"] == n]
        d = [detect_v3(r["text"]) for r in cs]
        res[n] = {"size": len(S[n]), "answers": len(cs),
                  "habit": sum(bool(x["family"] or x["cousin"]) for x in d),
                  "family": sum(bool(x["family"]) for x in d), "cousin": sum(bool(x["cousin"]) for x in d),
                  "beyond": sum(bool(x["beyond"]) for x in d),
                  "a_an": sum(bool(A_AN.search(r["text"])) for r in cs),
                  "but": sum(bool(re.search(r"\bbut\b", r["text"], re.I)) for r in cs),
                  "degenerate": sum(DR.degenerate(r["text"]) for r in cs),
                  "words": float(np.mean([len(r["text"].split()) for r in cs]))}
    (REPO / "reports/g3_pilot.json").write_text(json.dumps({"sets": S, "results": res}, indent=2))
    base = res["normal"]["habit"]
    md = ["# Gemma 3 pilot: candidate feature sets at layer 22 (exploratory)", "",
          "D2 prompts × 2 samples, 200 tokens, common random numbers. Habit = detector v3 family or cousin.", "",
          "| set | features | habit | drop | family | cousin | goes beyond | \"a\" + vowel | \"but\" | degenerate | words |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n, d in res.items():
        drop = f"{100 * (1 - d['habit'] / base):.0f}%" if base else ""
        md.append(f"| {n} | {d['size']} | {d['habit']}/{d['answers']} | {drop} | {d['family']} | {d['cousin']} | {d['beyond']} | "
                  f"{d['a_an']} | {d['but']} | {d['degenerate']} | {d['words']:.0f} |")
    (REPO / "reports/g3_pilot.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    analyze() if "--analyze" in sys.argv else generate()
