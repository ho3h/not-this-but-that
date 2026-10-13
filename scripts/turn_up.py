"""Turn the 25 up as well as off: one dial, both directions (exploratory).

If the chat-chosen 25 carry the habit, scaling their contribution up should make
it more common, and scaling 25 firing-rate-matched features up by the same
factor should not. Uses the frozen SAEMaskedAblate wrapper unchanged: a row mask
value m gives h - m * contrib, so m = 1 switches the features off and m = -(k-1)
multiplies their contribution by k. BOS is never touched.

Prompts: D2 (102) and D2b (50), none used to choose the features; each is
labelled M (meaning/value) or P (process/scene/facts) in
data/prompt_type_labels.tsv, blind to outcomes. Two samples per prompt with
common random numbers across conditions, chat template, 200 tokens,
temperature 0.8, top_p 0.95.

Conditions: 25 off, normal, 25 x2, 25 x3, matched 25 x3 (the frozen
matched_control set from confirmation v4).

  .venv/bin/python scripts/turn_up.py            # generate (resumable)
  .venv/bin/python scripts/turn_up.py --analyze  # writes reports/turn_up.{json,md}
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
import v2_engine as E  # noqa: E402
import v2_mlx as M  # noqa: E402
import v2_mlx_multi as MM  # noqa: E402

sys.path.insert(0, str(E.REPO / "src"))
from classifier.detect_v3 import detect_v3  # noqa: E402

REPO = E.REPO
GENS = REPO / "reports/turn_up_generations.jsonl"
N, MAX_NEW, TEMP, TOP_P, SEED = 2, 200, 0.8, 0.95, 20261010
CONDS = [("25 off", "chat", 1.0), ("normal", None, 0.0), ("25 x2", "chat", -1.0),
         ("25 x3", "chat", -2.0), ("matched 25 x3", "matched", -2.0)]


def prompts() -> list[dict]:
    labels = {}
    for line in (REPO / "data/prompt_type_labels.tsv").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        _, lab, text = line.split("\t", 2)
        labels[text.strip()] = lab
    out = []
    for src in ("D2_neutral_prompts.json", "D2b_confirmation_prompts.json"):
        for p in json.loads((REPO / "data" / src).read_text())["prompts"]:
            text = p if isinstance(p, str) else p.get("prompt") or p.get("text")
            out.append({"id": len(out), "set": src.split("_")[0], "type": labels[text.strip()], "prompt": text})
    return out


def generate() -> None:
    v4 = json.loads((REPO / "reports/confirmation_v4_prereg.json").read_text())
    sets = {"chat": v4["chat_features"], "matched": v4["matched_control"]}
    union = sorted(set(sets["chat"]) | set(sets["matched"]))
    model, tok, wraps = MM.load_multi("google/gemma-2-2b-it", {20: union})
    w = wraps[20]
    pos = {f: i for i, f in enumerate(w.order)}
    mask = np.zeros((len(CONDS) * N, len(union)), dtype=np.float32)
    for c, (_, which, m) in enumerate(CONDS):
        if which:
            for f in sets[which]:
                mask[c * N:(c + 1) * N, pos[f]] = m
    w.row_mask = mx.array(mask)
    done = set()
    if GENS.exists():
        done = {json.loads(l)["id"] for l in GENS.read_text().splitlines() if l.strip()}
    ps = prompts()
    t0 = time.time()
    for i, p in enumerate(ps):
        if p["id"] in done:
            continue
        U = np.random.default_rng(SEED + p["id"]).random((N, MAX_NEW)).astype(np.float32)
        outs = MM.sample_rows(model, tok, M.chat_tokens(tok, p["prompt"]), mx.array(np.tile(U, (len(CONDS), 1))),
                              max_new_tokens=MAX_NEW, temperature=TEMP, top_p=TOP_P)
        with GENS.open("a") as f:
            for c, (name, _, _) in enumerate(CONDS):
                for k in range(N):
                    f.write(json.dumps({**p, "condition": name, "sample": k, "text": outs[c * N + k]}) + "\n")
        print(f"prompt {i + 1}/{len(ps)} · {(time.time() - t0) / 60:.1f} min", flush=True)


def analyze() -> None:
    rows = [json.loads(l) for l in GENS.read_text().splitlines() if l.strip()]
    names = [c for c, _, _ in CONDS]

    def habit(t):
        d = detect_v3(t)
        return bool(d["family"] or d["cousin"])

    def boot_ci(ids, by, cond, n_boot=5000):
        rng = np.random.default_rng(0)
        ids = np.array(sorted(ids)); vals = []
        for _ in range(n_boot):
            s = rng.choice(ids, len(ids))
            hits = sum(by[(int(i), cond)][0] for i in s); n = sum(by[(int(i), cond)][1] for i in s)
            vals.append(hits / n)
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    res = {}
    for typ in ("all", "M", "P"):
        sub = [r for r in rows if typ == "all" or r["type"] == typ]
        by = {}
        for r in sub:
            k = (r["id"], r["condition"]); h, n = by.get(k, (0, 0))
            by[k] = (h + habit(r["text"]), n + 1)
        ids = {r["id"] for r in sub}
        res[typ] = {}
        for c in names:
            cs = [r for r in sub if r["condition"] == c]
            res[typ][c] = {
                "answers": len(cs),
                "habit": sum(habit(r["text"]) for r in cs),
                "habit_ci": boot_ci(ids, by, c),
                "but": float(np.mean([bool(re.search(r"\bbut\b", r["text"], re.I)) for r in cs])),
                "degenerate": float(np.mean([DR.degenerate(r["text"]) for r in cs])),
                "words": float(np.mean([len(r["text"].split()) for r in cs])),
            }
    (REPO / "reports/turn_up.json").write_text(json.dumps(res, indent=2))
    md = ["# Turning the 25 up as well as off (exploratory)", "",
          "D2 + D2b prompts (none used to choose the features), 2 samples each, common random numbers across",
          "conditions, chat template, 200 tokens. Habit = detector v3 family or cousin. x2 / x3 multiply the",
          "features' layer-20 contribution; matched 25 = confirmation v4's frozen firing-rate-matched control.",
          "CIs are 95% prompt-clustered bootstrap.", ""]
    for typ, title in (("all", "All prompts"), ("M", "Meaning/value prompts"), ("P", "Process/scene prompts")):
        md += [f"## {title}", "", "| condition | answers | with the habit | 95% CI | \"but\" | degenerate | words |",
               "|---|---:|---:|---:|---:|---:|---:|"]
        for c in names:
            d = res[typ][c]
            md.append(f"| {c} | {d['answers']} | {d['habit']} ({100 * d['habit'] / d['answers']:.1f}%) | "
                      f"{100 * d['habit_ci'][0]:.1f}–{100 * d['habit_ci'][1]:.1f}% | {100 * d['but']:.0f}% | "
                      f"{100 * d['degenerate']:.0f}% | {d['words']:.0f} |")
        md.append("")
    (REPO / "reports/turn_up.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    analyze() if "--analyze" in sys.argv else generate()
