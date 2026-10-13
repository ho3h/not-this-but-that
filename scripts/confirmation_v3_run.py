"""Pre-registered confirmation v3: can 25 chat-chosen features squash the habit, cheaply?

Reads every parameter from reports/confirmation_v3_prereg.json and refuses to
run or analyse if any frozen file's SHA-256 differs. Three conditions per
prompt, decoded together with common random numbers (scripts/v2_mlx_multi.py):
normal, the original coalition (secondary arm), and the 25 chat-chosen
features (primary arm). Held-out prompts: the 102 D2 prompts, never used to
choose the chat features.

  .venv/bin/python scripts/confirmation_v3_run.py            # generate + cost checks
  .venv/bin/python scripts/confirmation_v3_run.py --analyze  # verdict, once
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from math import comb

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
PREREG = REPO / "reports" / "confirmation_v3_prereg.json"
GENS = REPO / "reports" / "confirmation_v3_generations.jsonl"
COST = REPO / "reports" / "confirmation_v3_cost.json"
OUT_JSON = REPO / "reports" / "confirmation_v3_run.json"
OUT_MD = REPO / "reports" / "confirmation_v3_run.md"
A_AN = re.compile(r"\ba (?:[aeio]\w+|un(?!i|e)\w*)", re.I)


def check_frozen(prereg: dict) -> None:
    bad = [rel for rel, want in prereg["frozen_sha256"].items()
           if hashlib.sha256((REPO / rel).read_bytes()).hexdigest() != want]
    if bad:
        sys.exit("frozen files changed since registration: " + ", ".join(bad))


def conditions(prereg: dict):
    return [("normal", {}), ("original 25", {20: set(prereg["original_coalition"])}),
            ("chat 25", {20: set(prereg["chat_features"])})]


def prompts(prereg: dict) -> list[str]:
    d = json.loads((REPO / prereg["prompts_file"]).read_text())
    d = d if isinstance(d, list) else d.get("prompts", d)
    return [p if isinstance(p, str) else (p.get("prompt") or p.get("text")) for p in d]


def generate(prereg: dict) -> None:
    conds = conditions(prereg)
    union = {20: sorted(set(prereg["original_coalition"]) | set(prereg["chat_features"]))}
    model, tok, wraps = MM.load_multi(prereg["model"], union)
    n = prereg["samples_per_prompt"]
    MM.set_row_masks(wraps, [d for _, d in conds], n)
    done = set()
    if GENS.exists():
        done = {json.loads(l)["prompt_idx"] for l in GENS.read_text().splitlines() if l.strip()}
    t0 = time.time()
    ps = prompts(prereg)
    for i, p in enumerate(ps):
        if i in done:
            continue
        U = np.random.default_rng(prereg["seed_base"] + i).random((n, prereg["max_new_tokens"])).astype(np.float32)
        outs = MM.sample_rows(model, tok, M.chat_tokens(tok, p), mx.array(np.tile(U, (len(conds), 1))),
                              max_new_tokens=prereg["max_new_tokens"], temperature=prereg["temperature"],
                              top_p=prereg["top_p"])
        with GENS.open("a") as f:
            for c, (name, _) in enumerate(conds):
                for k in range(n):
                    f.write(json.dumps({"prompt_idx": i, "prompt": p, "condition": name, "sample": k,
                                        "text": outs[c * n + k]}) + "\n")
        print(f"prompt {i + 1}/{len(ps)} · {(time.time() - t0) / 60:.1f} min", flush=True)
    ppl = DR.perplexities(model, tok, wraps, conds)
    acc = DR.quiz(model, tok, wraps, conds)
    COST.write_text(json.dumps({name: {"perplexity_ratio": ppl[c] / ppl[0], "quiz_accuracy": acc[c]}
                                for c, (name, _) in enumerate(conds)}, indent=2))
    print("done. Run with --analyze for the verdict.")


def midp(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    pmf = lambda i: comb(n, i) / 2 ** n  # noqa: E731
    return float(min(1.0, 2.0 * (sum(pmf(i) for i in range(k)) + 0.5 * pmf(k))))


def analyze(prereg: dict) -> None:
    rows = [json.loads(l) for l in GENS.read_text().splitlines() if l.strip()]
    cost = json.loads(COST.read_text())
    pairs = {}
    for r in rows:
        pairs.setdefault((r["prompt_idx"], r["sample"]), {})[r["condition"]] = r["text"]
    keys = sorted(pairs)

    def hit(t, tier):
        x = detect_v3(t)
        return bool(x["family"] or x["cousin"]) if tier == "habit" else bool(x[tier])

    def compare(arm: str, tier: str):
        b = sum(1 for k in keys if hit(pairs[k]["normal"], tier) and not hit(pairs[k][arm], tier))
        c = sum(1 for k in keys if not hit(pairs[k]["normal"], tier) and hit(pairs[k][arm], tier))
        base = sum(hit(pairs[k]["normal"], tier) for k in keys)
        arm_n = sum(hit(pairs[k][arm], tier) for k in keys)
        return {"n": len(keys), "normal": base, "arm": arm_n,
                "rel_drop": (1 - arm_n / base) if base else None, "b": b, "c": c, "mid_p": midp(b, c)}

    def boot(arm: str, n_boot: int = 10_000):
        by = {}
        for k in keys:
            by.setdefault(k[0], []).append((hit(pairs[k]["normal"], "habit"), hit(pairs[k][arm], "habit")))
        ids = list(by)
        rng = random.Random(0)
        vals = []
        for _ in range(n_boot):
            s = [rng.choice(ids) for _ in ids]
            base = sum(x for i in s for x, _ in by[i])
            arm_n = sum(y for i in s for _, y in by[i])
            if base:
                vals.append(1 - arm_n / base)
        vals.sort()
        return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]

    g = prereg["gates"]
    h1 = compare("chat 25", "habit")
    h1["ci95"] = boot("chat 25")
    rd = h1["rel_drop"] or 0.0
    if rd >= g["pass_rel_drop"] and h1["mid_p"] < g["alpha"]:
        effect = "PASS"
    elif rd < g["kill_rel_drop"] or h1["mid_p"] >= g["alpha"]:
        effect = "KILL"
    else:
        effect = "PARTIAL"
    a_an = {name: sum(bool(A_AN.search(pairs[k][name])) for k in keys) / len(keys) for name in ("normal", "original 25", "chat 25")}
    cost_checks = {
        "perplexity_ratio": cost["chat 25"]["perplexity_ratio"] <= g["max_perplexity_ratio"],
        "quiz": (cost["normal"]["quiz_accuracy"] - cost["chat 25"]["quiz_accuracy"]) <= g["max_quiz_drop"] + 1e-9,
        "a_an_errors": a_an["chat 25"] <= g["max_a_an_rate"],
    }
    cost_verdict = "PASS" if all(cost_checks.values()) else "FAIL"
    but = {name: sum(bool(re.search(r"\bbut\b", pairs[k][name], re.I)) for k in keys) / len(keys)
           for name in ("normal", "original 25", "chat 25")}
    words = {name: sum(len(pairs[k][name].split()) for k in keys) / len(keys) for name in ("normal", "original 25", "chat 25")}
    secondary = {tier: {"chat 25": compare("chat 25", tier), "original 25": compare("original 25", tier)}
                 for tier in ("family", "cousin", "beyond", "reversed")}
    secondary["habit"] = {"original 25": compare("original 25", "habit")}
    result = {"H1_habit_chat25": h1, "H1_verdict": effect, "cost": cost, "a_an_rate": a_an,
              "cost_checks": cost_checks, "cost_verdict": cost_verdict, "but_share": but, "words": words,
              "secondary": secondary}
    OUT_JSON.write_text(json.dumps(result, indent=2))
    fmt = lambda t: f"{t['normal']} → {t['arm']}" + (f" ({100 * t['rel_drop']:+.0f}% drop)" if t["rel_drop"] is not None else "") + f", mid-p {t['mid_p']:.2g}"  # noqa: E731
    md = ["# Pre-registered confirmation v3: verdict", "",
          f"Analysed once. {len(keys)} prompt-sample triples (normal, original 25, chat 25), 102 held-out D2 prompts.", "",
          f"## H1 (primary): habit (family or cousin) with the chat 25 off: **{effect}**", "",
          f"{fmt(h1)}; 95% prompt-clustered CI on the drop {100 * h1['ci95'][0]:.0f}% to {100 * h1['ci95'][1]:.0f}%.", "",
          f"Gates: PASS if drop ≥ {100 * g['pass_rel_drop']:.0f}% and mid-p < {g['alpha']}; KILL if drop < {100 * g['kill_rel_drop']:.0f}% or mid-p ≥ {g['alpha']}.", "",
          f"## H2 (co-primary): cost: **{cost_verdict}**", "",
          f"- perplexity on human prose: {cost['chat 25']['perplexity_ratio']:.3f}× (gate ≤ {g['max_perplexity_ratio']}) {'✓' if cost_checks['perplexity_ratio'] else '✗'}",
          f"- quiz: {100 * cost['normal']['quiz_accuracy']:.0f}% → {100 * cost['chat 25']['quiz_accuracy']:.0f}% (gate: drop ≤ one question) {'✓' if cost_checks['quiz'] else '✗'}",
          f"- \"a\" before a vowel: {100 * a_an['normal']:.1f}% → {100 * a_an['chat 25']:.1f}% of answers (gate ≤ {100 * g['max_a_an_rate']:.0f}%) {'✓' if cost_checks['a_an_errors'] else '✗'}", "",
          "## Secondary, non-gating", "",
          "| tier | chat 25 | original 25 |", "|---|---|---|"]
    for tier in ("family", "cousin", "beyond", "reversed"):
        md.append(f"| {tier} | {fmt(secondary[tier]['chat 25'])} | {fmt(secondary[tier]['original 25'])} |")
    md.append(f"| habit (family or cousin) | {fmt(h1)} | {fmt(secondary['habit']['original 25'])} |")
    md += ["", f"Answers containing \"but\": normal {100 * but['normal']:.1f}%, original 25 {100 * but['original 25']:.1f}%, chat 25 {100 * but['chat 25']:.1f}%.",
           f"Mean words: normal {words['normal']:.0f}, original 25 {words['original 25']:.0f}, chat 25 {words['chat 25']:.0f}.",
           f"Original 25 cost: perplexity {cost['original 25']['perplexity_ratio']:.3f}×, quiz {100 * cost['original 25']['quiz_accuracy']:.0f}%, \"a\" + vowel {100 * a_an['original 25']:.1f}%."]
    OUT_MD.write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    prereg = json.loads(PREREG.read_text())
    check_frozen(prereg)
    if args.analyze:
        analyze(prereg)
    else:
        generate(prereg)


if __name__ == "__main__":
    main()
