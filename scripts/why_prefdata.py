"""Does preference-training data reward "not X but Y"? Chosen vs rejected answers (exploratory).

Two public preference datasets, each a list of (prompt, chosen answer, rejected answer):
  - UltraFeedback, binarised and cleaned (argilla, 2023): ratings by GPT-4 (AI feedback); 61k pairs.
  - OLMo 2 7B preference mix (allenai, 2025): the data OLMo 2's DPO stage trained on; judged by
    an LLM; a seeded sample of 100k pairs.
Each answer is scored with detector v3 (family or cousin). Reported per dataset: the share of
chosen and of rejected answers using the construction; among pairs where exactly one answer
uses it, how often that answer is the chosen one; and the same win rate adjusted for length
(logistic fit of "construction answer chosen" on log(length ratio), evaluated at equal
length). CIs: 2,000 pair bootstrap resamples. Writes reports/why_prefdata.{json,md}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from classifier.detect_v3 import detect_v3  # noqa: E402

SEED, BOOT, OLMO_SAMPLE = 20261011, 2000, 100_000
DATASETS = [("UltraFeedback (GPT-4 ratings)", "argilla/ultrafeedback-binarized-preferences-cleaned", None),
            ("OLMo 2 preference mix", "allenai/olmo-2-1124-7b-preference-mix", OLMO_SAMPLE)]


def last_assistant(msgs) -> str:
    if isinstance(msgs, str):
        return msgs
    for m in reversed(msgs):
        if m.get("role") == "assistant":
            return m.get("content") or ""
    return ""


def has(text: str) -> bool:
    d = detect_v3(text)
    return bool(d["family"] or d["cousin"])


def fit_intercept(x: np.ndarray, y: np.ndarray, iters: int = 40) -> float:
    X = np.column_stack([np.ones(len(x)), x])
    w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(2)
        w += np.linalg.solve(H, X.T @ (y - p))
    return float(1 / (1 + np.exp(-w[0])))


def analyse(name: str, repo: str, sample: int | None) -> dict:
    from datasets import load_dataset

    ds = load_dataset(repo, split="train")
    if sample and len(ds) > sample:
        ds = ds.shuffle(seed=SEED).select(range(sample))
    hc = hr = 0
    one_sided = []  # (construction answer chosen?, log length ratio construction/other)
    for r in ds:
        c, j = last_assistant(r["chosen"]), last_assistant(r["rejected"])
        if not c or not j:
            continue
        a, b = has(c), has(j)
        hc += a; hr += b
        if a != b:
            cons, other = (c, j) if a else (j, c)
            one_sided.append((int(a), float(np.log((len(cons) + 1) / (len(other) + 1)))))
    n = len(ds)
    y = np.array([o[0] for o in one_sided]); x = np.array([o[1] for o in one_sided])
    rng = np.random.default_rng(SEED)
    boot_raw, boot_adj = [], []
    for _ in range(BOOT):
        s = rng.integers(0, len(y), len(y))
        boot_raw.append(y[s].mean()); boot_adj.append(fit_intercept(x[s], y[s]))
    return {"pairs": n, "chosen_share": hc / n, "rejected_share": hr / n, "one_sided_pairs": len(y),
            "construction_chosen": float(y.mean()),
            "ci95": [float(np.percentile(boot_raw, 2.5)), float(np.percentile(boot_raw, 97.5))],
            "construction_chosen_equal_length": fit_intercept(x, y),
            "ci95_equal_length": [float(np.percentile(boot_adj, 2.5)), float(np.percentile(boot_adj, 97.5))],
            "construction_answer_longer": float((x > 0).mean())}


def main() -> None:
    res = {name: analyse(name, repo, sample) for name, repo, sample in DATASETS}
    (REPO / "reports/why_prefdata.json").write_text(json.dumps(res, indent=2))
    md = ["# Does preference data reward \"not X but Y\"? (exploratory)", "",
          "Chosen vs rejected answers in public preference datasets. Detector v3, family or cousin.", "",
          "| dataset | pairs | chosen using it | rejected using it | one-sided pairs | construction answer chosen | at equal length |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for name, v in res.items():
        md.append(f"| {name} | {v['pairs']:,} | {100 * v['chosen_share']:.1f}% | {100 * v['rejected_share']:.1f}% | "
                  f"{v['one_sided_pairs']:,} | **{100 * v['construction_chosen']:.1f}%** "
                  f"({100 * v['ci95'][0]:.1f}–{100 * v['ci95'][1]:.1f}) | {100 * v['construction_chosen_equal_length']:.1f}% "
                  f"({100 * v['ci95_equal_length'][0]:.1f}–{100 * v['ci95_equal_length'][1]:.1f}) |")
    md += ["", "50% would mean the judges are indifferent. \"At equal length\" removes the advantage longer answers "
           "get, since answers that use the construction tend to be longer."]
    (REPO / "reports/why_prefdata.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
