"""Self-priming: does one "not X but Y" make the next more likely? (exploratory, Gemma 2 2B-it)

Each of 40 prompts (data/priming_pairs.json; 20 process-type, 20 meaning-type, all from the
held-out turn-up set) gets its answer started for the model with one of three openers that
carry the same content:

  construction  "{subject} isn't just {x}; it's {y}."
  plain         "{subject} is {y}."
  both          "{subject} is {x}, and it's {y}."   (mentions x too, with no contrast)

The model then writes the rest (8 samples, 200 new tokens, T=0.8, top-p 0.95). Each sample
uses the same uniforms in every condition. Only the continuation is scored, with detector v3
(family or cousin). Prediction stated before running: construction > both and construction >
plain; both ≈ plain. (It came out the other way; see reports/priming.md.) "Next sentence" hits
start within the continuation's first 200 characters. Writes reports/priming.{json,md} and
reports/priming_generations.jsonl.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_engine as E  # noqa: E402
import v2_mlx as M  # noqa: E402
import v2_mlx_multi as MM  # noqa: E402

sys.path.insert(0, str(E.REPO / "src"))
from classifier.detect_v3 import detect_v3  # noqa: E402

REPO = E.REPO
N, MAX_NEW, SEED, BOOT = 8, 200, 20261009, 5000
CONDS = ("construction", "plain", "both")


def opener(it: dict, cond: str) -> str:
    s, x, y = it["subject"], it["x"], it["y"]
    return {"construction": f"{s} isn't just {x}; it's {y}.", "plain": f"{s} is {y}.",
            "both": f"{s} is {x}, and it's {y}."}[cond]


NEXT = 200  # characters: a hit starting this early is in the sentence right after the opener


def hits(text: str) -> int:
    d = detect_v3(text)
    return len(d["family"]) + len(d["cousin"])


def first_hit(text: str) -> int | None:
    d = detect_v3(text)
    return min((h.span[0] for h in d["family"] + d["cousin"]), default=None)


def generate() -> list[dict]:
    from mlx_lm import load as mlx_load

    M._patch_gemma2()
    model, tok = mlx_load("google/gemma-2-2b-it")
    items = json.loads((REPO / "data/priming_pairs.json").read_text())["items"]
    rows = []
    for it in items:
        U = mx.array(np.random.default_rng(SEED + it["id"]).random((N, MAX_NEW)).astype(np.float32))
        for cond in CONDS:
            op = opener(it, cond)
            prefix = M.chat_tokens(tok, it["prompt"]) + tok.encode(op, add_special_tokens=False)
            outs = MM.sample_rows(model, tok, prefix, U, max_new_tokens=MAX_NEW, temperature=0.8, top_p=0.95)
            for k, text in enumerate(outs):
                rows.append({"id": it["id"], "type": it["type"], "prompt": it["prompt"], "condition": cond,
                             "sample": k, "opener": op, "text": text, "hits": hits(text)})
        print(f"{it['id'] + 1}/{len(items)}", flush=True)
    return rows


def summarise(rows: list[dict]) -> dict:
    ids = sorted({r["id"] for r in rows})
    by = {(r["id"], r["condition"]): [] for r in rows}
    for r in rows:
        by[(r["id"], r["condition"])].append(r["hits"] > 0)
    rate = lambda pid, c: float(np.mean(by[(pid, c)]))  # noqa: E731
    rng = np.random.default_rng(SEED)
    out = {}
    for typ in ("all", "P", "M"):
        sel = [i for i in ids if typ == "all" or rows[[r["id"] for r in rows].index(i)]["type"] == typ]
        per = {c: np.array([rate(i, c) for i in sel]) for c in CONDS}
        res = {c: {"answers_with_hit": int(sum(sum(by[(i, c)]) for i in sel)), "answers": N * len(sel),
                   "rate": float(per[c].mean()),
                   "next_sentence": sum(1 for r in rows if r["condition"] == c and r["id"] in sel
                                        and r["first"] is not None and r["first"] < NEXT),
                   "median_first_char": float(np.median([r["first"] for r in rows if r["condition"] == c
                                                         and r["id"] in sel and r["first"] is not None] or [np.nan]))}
               for c in CONDS}
        for a, b in (("construction", "plain"), ("construction", "both"), ("both", "plain")):
            boots = []
            for _ in range(BOOT):
                s = rng.integers(0, len(sel), len(sel))
                den = per[b][s].mean()
                boots.append(per[a][s].mean() / den if den > 0 else np.inf)
            fin = np.array([v for v in boots if np.isfinite(v)])
            ratio = per[a].mean() / per[b].mean() if per[b].mean() > 0 else float("inf")
            res[f"{a}/{b}"] = {"ratio": ratio, "ci95": [float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))],
                               "boot_infinite": int(len(boots) - len(fin))}
        out[typ] = res
    return out


def main() -> None:
    gen_path = REPO / "reports/priming_generations.jsonl"
    if "--rescore" in sys.argv and gen_path.exists():
        rows = [json.loads(l) for l in gen_path.read_text().splitlines() if l.strip()]
        for r in rows:
            r["hits"] = hits(r["text"])
    else:
        rows = generate()
        gen_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    for r in rows:
        r["first"] = first_hit(r["text"])
    res = summarise(rows)
    (REPO / "reports/priming.json").write_text(json.dumps(res, indent=2))
    md = ["# Self-priming: what happens after one \"not X but Y\"? (exploratory)", "",
          "Gemma 2 2B-it. 40 held-out prompts, the answer started with one of three openers carrying the same "
          "content; 8 samples each, 200 new tokens. Only the continuation is scored (detector v3, family or cousin).", "",
          "| prompts | opener | continuations using not X but Y | rate | in the next sentence | median position of first use (characters) |",
          "|---|---|---:|---:|---:|---:|"]
    for typ, label in (("all", "all 40"), ("P", "20 process"), ("M", "20 meaning")):
        for c in CONDS:
            v = res[typ][c]
            md.append(f"| {label} | {c} | {v['answers_with_hit']} / {v['answers']} | {100 * v['rate']:.1f}% | "
                      f"{v['next_sentence']} | {v['median_first_char']:.0f} |")
    md += ["", "Rate ratios (95% prompt-clustered bootstrap CI):", ""]
    for typ, label in (("all", "all 40"), ("P", "20 process"), ("M", "20 meaning")):
        for k in ("construction/plain", "construction/both", "both/plain"):
            v = res[typ][k]
            md.append(f"- {label}, {k}: **{v['ratio']:.2f}×** ({v['ci95'][0]:.2f}× to {v['ci95'][1]:.2f}×)")
    md += ["", "## Reading", "",
           "The prediction (one use primes more) was wrong. After the construction opener the model rarely "
           "uses it again, and when it does, it is far down the answer. After a plain grand claim "
           "(\"X is Y\"), it usually supplies the missing foil in the very next sentence (\"It's not just "
           "about books anymore, it's about community\"). The construction behaves less like a contagion and "
           "more like a slot: an answer about meaning gets roughly one, and a grand claim without a foil "
           "invites it. For scale, the same 40 prompts with no opener (turn-up run, 2 samples each) used it "
           "in 11 of 40 whole answers to meaning prompts and 0 of 40 to process prompts."]
    (REPO / "reports/priming.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
