"""Do people vote for "not X but Y"? Human votes from LMArena battles (exploratory).

Data: lmarena-ai/arena-human-preference-55k (Hugging Face, 2024): one prompt, two anonymous
model answers, and a human vote (A, B or tie). Each answer is scored with detector v3
(family or cousin). The test uses battles where exactly one answer contains the construction
and the vote is not a tie: how often does that answer win?

Longer answers both win more and contain more of everything, so the win rate is also
reported within bins of the two answers' length ratio, and from a logistic regression of
"A wins" on (construction in A minus in B) and log(length A / length B). CIs: 2,000 battle
bootstrap resamples. Also reports each model's rate of answers using the construction.
Writes reports/why_arena.{json,md}.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from classifier.detect_v3 import detect_v3  # noqa: E402

SEED, BOOT = 20261009, 2000


def answer_text(cell: str) -> str:
    try:
        parts = json.loads(cell)
    except (json.JSONDecodeError, TypeError):
        return str(cell)
    return "\n\n".join(p for p in parts if isinstance(p, str))


def has(text: str) -> bool:
    d = detect_v3(text)
    return bool(d["family"] or d["cousin"])


def logistic(X: np.ndarray, y: np.ndarray, iters: int = 50) -> np.ndarray:
    w = np.zeros(X.shape[1])
    for _ in range(iters):  # Newton-Raphson
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p)
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(X.shape[1])
        w += np.linalg.solve(H, g)
    return w


def main() -> None:
    from datasets import load_dataset

    ds = load_dataset("lmarena-ai/arena-human-preference-55k", split="train")
    rows, per_model = [], defaultdict(lambda: [0, 0])
    for r in ds:
        a, b = answer_text(r["response_a"]), answer_text(r["response_b"])
        ha, hb = has(a), has(b)
        per_model[r["model_a"]][0] += ha; per_model[r["model_a"]][1] += 1
        per_model[r["model_b"]][0] += hb; per_model[r["model_b"]][1] += 1
        if r["winner_tie"] or ha == hb:
            continue
        rows.append({"cons_wins": int((ha and r["winner_model_a"]) or (hb and r["winner_model_b"])),
                     "a_wins": int(r["winner_model_a"]), "d": int(ha) - int(hb),
                     "loglen": float(np.log((len(a) + 1) / (len(b) + 1)))})
    n = len(rows)
    wins = np.array([x["cons_wins"] for x in rows])
    rng = np.random.default_rng(SEED)
    boot = [wins[rng.integers(0, n, n)].mean() for _ in range(BOOT)]
    # length-ratio bins, from the construction answer's side
    side = np.array([x["loglen"] * x["d"] for x in rows])
    bins = [(-np.inf, -0.25), (-0.25, 0.25), (0.25, np.inf)]
    by_bin = {f"{lo:+.2f}..{hi:+.2f}": {"battles": int(((side > lo) & (side <= hi)).sum()),
                                        "win_rate": float(wins[(side > lo) & (side <= hi)].mean())}
              for lo, hi in bins if ((side > lo) & (side <= hi)).any()}
    X = np.column_stack([np.ones(n), [x["d"] for x in rows], [x["loglen"] for x in rows]])
    y = np.array([x["a_wins"] for x in rows])
    w = logistic(X, y)
    wb = []
    for _ in range(BOOT):
        s = rng.integers(0, n, n)
        wb.append(logistic(X[s], y[s])[1])
    models = {m: {"rate": h / t, "answers": t} for m, (h, t) in per_model.items() if t >= 300}
    res = {"battles_one_sided": n, "construction_win_rate": float(wins.mean()),
           "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
           "by_length_ratio_bin": by_bin,
           "logit_construction": float(w[1]), "logit_construction_ci95": [float(np.percentile(wb, 2.5)), float(np.percentile(wb, 97.5))],
           "odds_ratio_length_adjusted": float(np.exp(w[1])), "logit_loglen": float(w[2]),
           "models": dict(sorted(models.items(), key=lambda kv: -kv[1]["rate"]))}
    (REPO / "reports/why_arena.json").write_text(json.dumps(res, indent=2))
    md = ["# Do people vote for \"not X but Y\"? (exploratory)", "",
          "LMArena human-preference battles (55k). Battles where exactly one answer uses the construction "
          "(detector v3, family or cousin) and the vote is not a tie.", "",
          f"- Battles: {n}",
          f"- Construction answer wins: **{100 * wins.mean():.1f}%** (95% CI {100 * res['ci95'][0]:.1f}% to {100 * res['ci95'][1]:.1f}%)",
          f"- Length-adjusted odds ratio: **{res['odds_ratio_length_adjusted']:.2f}** "
          f"(95% CI {np.exp(res['logit_construction_ci95'][0]):.2f} to {np.exp(res['logit_construction_ci95'][1]):.2f})", "",
          "| construction answer's length vs the other (log ratio) | battles | win rate |", "|---|---:|---:|"]
    md += [f"| {k} | {v['battles']} | {100 * v['win_rate']:.1f}% |" for k, v in by_bin.items()]
    md += ["", "Share of each model's answers using the construction (models with ≥300 answers):", "",
           "| model | answers | rate |", "|---|---:|---:|"]
    md += [f"| {m} | {v['answers']} | {100 * v['rate']:.1f}% |" for m, v in list(res["models"].items())[:25]]
    (REPO / "reports/why_arena.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
