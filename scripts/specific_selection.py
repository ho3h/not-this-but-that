"""A more specific switch: features that drive the habit but not ordinary negation (exploratory).

The decision-point features also carry ordinary negation (reports/negation_check.md).
This scores every strong habit candidate on a second set of decision points: the
token before an ordinary negation ("you don't need", "it can't", "never leave"...)
in the same normal selection answers, outside any construction. It records each
feature's drop in P(that negation token) alone, keeping an unmodified row in
every batch.

Selection rule (fixed before any evaluation): a feature is "specific" if switching
it off alone lowers the ordinary-negation probability by less than 5% of baseline.
Gemma 2: the top 25 specific features by family score (mirrors the original 25).
Gemma 3: the top 20 specific by family score plus the top 5 specific by cousin
score (mirrors set B).

  .venv/bin/python scripts/specific_selection.py gemma2
  .venv/bin/python scripts/specific_selection.py gemma3

Writes reports/specific_selection_{model}.json.
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

sys.path.insert(0, str(E.REPO / "src"))
from classifier.detect_v3 import detect_v3, normalise  # noqa: E402

REPO = E.REPO
NEG_WORD = re.compile(r"\s(not|don't|doesn't|can't|won't|never|isn't|aren't|shouldn't|didn't|no)\b", re.I)
HABITISH = re.compile(r"^\s*\S+\s+(?:just|only|merely|simply|about)\b", re.I)
N_PREFIX, THRESH = 40, 0.05


def config(which: str) -> dict:
    if which == "gemma2":
        rows = [json.loads(l) for l in (REPO / "reports/confirmation_v2_generations.jsonl").read_text().splitlines() if l.strip()]
        answers = [(r["prompt"], r["baseline"]) for r in rows]
        attrib = json.loads((REPO / "reports/reroute_attribution.json").read_text())
        current = json.loads((REPO / "reports/confirmation_v3_prereg.json").read_text())["chat_features"]
        return {"name": "google/gemma-2-2b-it", "layer": 20, "answers": answers, "attrib": attrib,
                "current": current, "n_family": 25, "n_cousin": 0, "loader": MM.load_multi}
    rows = [json.loads(l) for l in (REPO / "reports/g3_baseline_generations.jsonl").read_text().splitlines() if l.strip()]
    answers = [(r["prompt"], r["text"]) for r in rows if r["set"] == "D4" and r["sample"] == 0]
    attrib = json.loads((REPO / "reports/g3_attribution_L22.json").read_text())
    current = json.loads((REPO / "reports/g3_pilot.json").read_text())["sets"]["B"]
    return {"name": "google/gemma-3-4b-it", "layer": 22, "answers": answers, "attrib": attrib,
            "current": current, "n_family": 20, "n_cousin": 5, "loader": G.load_multi}


def ordinary_prefixes(answers) -> list[dict]:
    """Cut each answer before its first negation that is not part of a construction."""
    out, used = [], set()
    for prompt, text in answers:
        if prompt in used:
            continue
        t = normalise(text)
        d = detect_v3(t)
        spans = [h.span for h in d["family"] + d["cousin"] + d.get("reversed", []) + d.get("beyond", [])]
        for m in NEG_WORD.finditer(t):
            a = m.start()
            if a < 20 or any(s - 5 <= a <= e + 5 for s, e in spans) or HABITISH.match(t[a:a + 40]):
                continue
            out.append({"prompt": prompt, "answer_prefix": t[:a], "word": " " + m.group(1)})
            used.add(prompt)
            break
        if len(out) >= N_PREFIX:
            break
    return out


def main() -> None:
    which = sys.argv[1]
    C = config(which)
    fam = C["attrib"]["family"]["ranked"]; cou = C["attrib"]["cousin"]["ranked"]
    cands = sorted({x["feature_idx"] for x in fam[:100]} | {x["feature_idx"] for x in cou[:60]} | set(C["current"]))
    model, tok, wraps = C["loader"](C["name"], {C["layer"]: cands})
    w = wraps[C["layer"]]
    pos = {f: i for i, f in enumerate(w.order)}
    prefixes = ordinary_prefixes(C["answers"])
    print(len(prefixes), "ordinary-negation prefixes;", len(cands), "candidates", flush=True)
    drops = {f: [] for f in cands}
    base = []
    for it in prefixes:
        ids = M.chat_tokens(tok, it["prompt"]) + tok.encode(it["answer_prefix"], add_special_tokens=False)
        tid = tok.encode(it["word"], add_special_tokens=False)[0]
        for s in range(0, len(cands), 63):
            fs = cands[s:s + 63]
            m = np.zeros((len(fs) + 1, len(cands)), dtype=np.float32)
            for r, f in enumerate(fs, 1):
                m[r, pos[f]] = 1.0
            w.row_mask = mx.array(m)
            p = np.array(mx.softmax(model(mx.array([ids] * (len(fs) + 1)))[:, -1, :].astype(mx.float32), axis=-1)[:, tid])
            if s == 0:
                base.append(float(p[0]))
            for f, pf in zip(fs, p[1:]):
                drops[f].append(float(p[0] - pf))
        print(f"prefix {len(base)}/{len(prefixes)} · P {base[-1]:.2f}", flush=True)
    w.row_mask = None
    b = float(np.mean(base))
    rel = {f: float(np.mean(d)) / b for f, d in drops.items()}
    specific = {f for f in cands if rel[f] < THRESH}
    sel_f = [x["feature_idx"] for x in fam if x["feature_idx"] in specific][:C["n_family"]]
    sel_c = [x["feature_idx"] for x in cou if x["feature_idx"] in specific and x["feature_idx"] not in sel_f][:C["n_cousin"]]
    selected = sel_f + sel_c
    res = {"model": C["name"], "layer": C["layer"], "threshold": THRESH, "n_prefixes": len(prefixes),
           "baseline_p_negation": b, "relative_negation_drop": {str(f): rel[f] for f in cands},
           "current_set": C["current"], "current_removed_as_unspecific": [f for f in C["current"] if f not in specific],
           "specific_set": selected, "examples": prefixes[:5]}
    (REPO / f"reports/specific_selection_{which}.json").write_text(json.dumps(res, indent=2))
    print("baseline P(negation)", round(b, 3))
    print("current set, features dropped as unspecific:", [(f, round(rel[f], 2)) for f in C["current"] if f not in specific])
    print("specific set:", selected)


if __name__ == "__main__":
    main()
