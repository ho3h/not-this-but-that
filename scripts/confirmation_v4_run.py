"""Pre-registered confirmation v4: the chat-chosen 25 on fresh, varied prompts.

Reads every parameter from reports/confirmation_v4_prereg.json and refuses to
run or analyse if any frozen file's SHA-256 differs. Three conditions per
prompt, decoded together with common random numbers: normal, the chat-chosen
25 switched off, and a frozen set of 25 firing-rate-matched features switched
off (control). Prompts: 140 fresh D5 prompts in seven genres (six everyday
genres plus explicit requests for the pattern).

  .venv/bin/python scripts/confirmation_v4_run.py              # generate + capability checks
  .venv/bin/python scripts/confirmation_v4_run.py --pairs      # write the blind-preference sample
  .venv/bin/python scripts/confirmation_v4_run.py --analyze    # verdict, once (after preference labels exist)
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
import cost_evals as CE  # noqa: E402
import v2_engine as E  # noqa: E402
import v2_mlx as M  # noqa: E402
import v2_mlx_multi as MM  # noqa: E402

sys.path.insert(0, str(E.REPO / "src"))
from classifier.detect_v3 import detect_v3  # noqa: E402

REPO = E.REPO
PREREG = REPO / "reports" / "confirmation_v4_prereg.json"
GENS = REPO / "reports" / "confirmation_v4_generations.jsonl"
COST = REPO / "reports" / "confirmation_v4_cost.json"
PAIRS = REPO / "data" / "confirmation_v4_pairs.txt"
PAIRS_KEY = REPO / "data" / "confirmation_v4_pairs_key.json"
PAIRS_LABELS = REPO / "data" / "confirmation_v4_pairs_labels.tsv"
OUT_JSON = REPO / "reports" / "confirmation_v4_run.json"
OUT_MD = REPO / "reports" / "confirmation_v4_run.md"
A_AN = re.compile(r"\ba (?:[aeio]\w+|un(?!i|e)\w*)", re.I)
CONDS = ("normal", "chat 25", "matched 25")


def check_frozen(prereg: dict) -> None:
    bad = [rel for rel, want in prereg["frozen_sha256"].items()
           if hashlib.sha256((REPO / rel).read_bytes()).hexdigest() != want]
    if bad:
        sys.exit("frozen files changed since registration: " + ", ".join(bad))


def doses(prereg: dict) -> list[dict]:
    return [{}, {20: set(prereg["chat_features"])}, {20: set(prereg["matched_control"])}]


def generate(prereg: dict) -> None:
    union = {20: sorted(set(prereg["chat_features"]) | set(prereg["matched_control"]))}
    model, tok, wraps = MM.load_multi(prereg["model"], union)
    n = prereg["samples_per_prompt"]
    D = doses(prereg)
    prompts = json.loads((REPO / prereg["prompts_file"]).read_text())["prompts"]
    done = set()
    if GENS.exists():
        done = {json.loads(l)["id"] for l in GENS.read_text().splitlines() if l.strip()}
    MM.set_row_masks(wraps, D, n)
    t0 = time.time()
    for i, p in enumerate(prompts):
        if p["id"] in done:
            continue
        U = np.random.default_rng(prereg["seed_base"] + p["id"]).random((n, prereg["max_new_tokens"])).astype(np.float32)
        outs = MM.sample_rows(model, tok, M.chat_tokens(tok, p["prompt"]), mx.array(np.tile(U, (len(D), 1))),
                              max_new_tokens=prereg["max_new_tokens"], temperature=prereg["temperature"], top_p=prereg["top_p"])
        with GENS.open("a") as f:
            for c, name in enumerate(CONDS):
                for k in range(n):
                    f.write(json.dumps({"id": p["id"], "genre": p["genre"], "prompt": p["prompt"], "condition": name,
                                        "sample": k, "text": outs[c * n + k]}) + "\n")
        print(f"prompt {i + 1}/{len(prompts)} · {(time.time() - t0) / 60:.1f} min", flush=True)
    arc = CE.arc_accuracy(model, tok, wraps, D)
    ppl = CE.wikitext_perplexity(model, tok, wraps, D)
    COST.write_text(json.dumps({name: {"arc_easy": arc[c], "wikitext_ppl": ppl[c], "wikitext_ratio": ppl[c] / ppl[0]}
                                for c, name in enumerate(CONDS)}, indent=2))
    print("done. Next: --pairs, then blind preference labels, then --analyze.")


def load_pairs():
    rows = [json.loads(l) for l in GENS.read_text().splitlines() if l.strip()]
    pairs = {}
    for r in rows:
        pairs.setdefault((r["id"], r["sample"]), {"genre": r["genre"], "prompt": r["prompt"]})[r["condition"]] = r["text"]
    return pairs


def write_pairs(prereg: dict) -> None:
    pairs = load_pairs()
    keys = sorted(k for k, v in pairs.items() if v["genre"] != "explicit")
    rng = random.Random(prereg["preference"]["seed"])
    pick = rng.sample(keys, prereg["preference"]["n_pairs"])
    lines, key = [], []
    for i, k in enumerate(pick, 1):
        flip = rng.random() < 0.5
        a, b = ("chat 25", "normal") if flip else ("normal", "chat 25")
        key.append({"item": i, "id": k[0], "sample": k[1], "A": a, "B": b})
        lines.append(f"### PAIR {i}\nQUESTION: {pairs[k]['prompt']}\nANSWER A: {pairs[k][a].replace(chr(10), ' ⏎ ')}\n"
                     f"ANSWER B: {pairs[k][b].replace(chr(10), ' ⏎ ')}\n")
    PAIRS.write_text("\n".join(lines))
    PAIRS_KEY.write_text(json.dumps(key, indent=1))
    print(f"wrote {len(pick)} pairs to {PAIRS.relative_to(REPO)}")


def midp(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    pmf = lambda i: comb(n, i) / 2 ** n  # noqa: E731
    return float(min(1.0, 2.0 * (sum(pmf(i) for i in range(k)) + 0.5 * pmf(k))))


def analyze(prereg: dict) -> None:
    pairs = load_pairs()
    cost = json.loads(COST.read_text())
    g = prereg["gates"]
    cache = {}

    def hit(t, tier="habit"):
        x = cache.get(t) or cache.setdefault(t, detect_v3(t))
        return bool(x["family"] or x["cousin"]) if tier == "habit" else bool(x[tier])

    def compare(keys, arm, tier="habit"):
        b = sum(1 for k in keys if hit(pairs[k]["normal"], tier) and not hit(pairs[k][arm], tier))
        c = sum(1 for k in keys if not hit(pairs[k]["normal"], tier) and hit(pairs[k][arm], tier))
        base = sum(hit(pairs[k]["normal"], tier) for k in keys)
        arm_n = sum(hit(pairs[k][arm], tier) for k in keys)
        return {"n": len(keys), "normal": base, "arm": arm_n, "rel_drop": (1 - arm_n / base) if base else None,
                "b": b, "c": c, "mid_p": midp(b, c)}

    def boot(keys, arm, n_boot=10_000):
        by = {}
        for k in keys:
            by.setdefault(k[0], []).append((hit(pairs[k]["normal"]), hit(pairs[k][arm])))
        ids = list(by)
        rng = random.Random(0)
        vals = []
        for _ in range(n_boot):
            s = [rng.choice(ids) for _ in ids]
            base = sum(x for i in s for x, _ in by[i])
            if base:
                vals.append(1 - sum(y for i in s for _, y in by[i]) / base)
        vals.sort()
        return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]

    main_keys = sorted(k for k, v in pairs.items() if v["genre"] != "explicit")
    explicit_keys = sorted(k for k, v in pairs.items() if v["genre"] == "explicit")
    h1 = compare(main_keys, "chat 25"); h1["ci95"] = boot(main_keys, "chat 25")
    ctrl = compare(main_keys, "matched 25")
    rd = h1["rel_drop"] or 0.0
    h1_v = "PASS" if (rd >= g["pass_rel_drop"] and h1["mid_p"] < g["alpha"]) else ("KILL" if (rd < g["kill_rel_drop"] or h1["mid_p"] >= g["alpha"]) else "PARTIAL")
    h2_v = "PASS" if ((rd - (ctrl["rel_drop"] or 0.0)) >= g["min_gap_over_control"]) else "FAIL"

    labels = {}
    key = {k["item"]: k for k in json.loads(PAIRS_KEY.read_text())}
    for line in PAIRS_LABELS.read_text().splitlines():
        p = line.split("\t")
        if len(p) >= 2 and p[0].strip().isdigit():
            labels[int(p[0])] = p[1].strip().upper()[:1]
    wins = ties = losses = 0
    for item, lab in labels.items():
        winner = {"A": key[item]["A"], "B": key[item]["B"]}.get(lab)
        if lab == "T":
            ties += 1
        elif winner == "chat 25":
            wins += 1
        elif winner == "normal":
            losses += 1
    pref_share = (wins + 0.5 * ties) / max(1, wins + ties + losses)
    a_an = {c: sum(bool(A_AN.search(pairs[k][c])) for k in main_keys) / len(main_keys) for c in CONDS}
    checks = {
        "arc": (cost["normal"]["arc_easy"] - cost["chat 25"]["arc_easy"]) <= g["max_arc_drop"] + 1e-9,
        "wikitext": cost["chat 25"]["wikitext_ratio"] <= g["max_wikitext_ratio"],
        "a_an": a_an["chat 25"] <= g["max_a_an_rate"],
        "preference": pref_share >= g["min_preference_share"],
    }
    h3_v = "PASS" if all(checks.values()) else "FAIL"
    genres = sorted({v["genre"] for v in pairs.values() if v["genre"] != "explicit"})
    per_genre = {gn: compare([k for k in main_keys if pairs[k]["genre"] == gn], "chat 25") for gn in genres}
    explicit = {c: sum(hit(pairs[k][c]) for k in explicit_keys) / max(1, len(explicit_keys)) for c in CONDS}
    tiers = {t: {"chat 25": compare(main_keys, "chat 25", t), "matched 25": compare(main_keys, "matched 25", t)}
             for t in ("family", "cousin", "beyond", "reversed")}
    but = {c: sum(bool(re.search(r"\bbut\b", pairs[k][c], re.I)) for k in main_keys) / len(main_keys) for c in CONDS}
    words = {c: float(np.mean([len(pairs[k][c].split()) for k in main_keys])) for c in CONDS}
    res = {"H1": h1, "H1_verdict": h1_v, "control": ctrl, "H2_verdict": h2_v, "cost": cost, "a_an": a_an,
           "preference": {"chat25_wins": wins, "ties": ties, "normal_wins": losses, "share": pref_share},
           "cost_checks": checks, "H3_verdict": h3_v, "per_genre": per_genre, "explicit_rate": explicit,
           "tiers": tiers, "but_share": but, "words": words}
    OUT_JSON.write_text(json.dumps(res, indent=2))
    f = lambda t: f"{t['normal']} → {t['arm']}" + (f" ({100 * t['rel_drop']:+.0f}% drop)" if t['rel_drop'] is not None else "") + f", mid-p {t['mid_p']:.2g}"  # noqa: E731
    md = ["# Pre-registered confirmation v4: verdict", "",
          f"Analysed once. {len(main_keys)} triples on 120 everyday prompts (six genres), {len(explicit_keys)} on 20 explicit-request prompts.", "",
          f"## H1 (primary): habit with the chat 25 off: **{h1_v}**", "",
          f"{f(h1)}; 95% prompt-clustered CI {100 * h1['ci95'][0]:.0f}% to {100 * h1['ci95'][1]:.0f}%.", "",
          f"## H2 (specificity): matched control: **{h2_v}**", "",
          f"Matched 25 off: {f(ctrl)}. Gate: chat-25 drop exceeds control drop by ≥ {100 * g['min_gap_over_control']:.0f} points.", "",
          f"## H3 (cost): **{h3_v}**", "",
          f"- ARC-Easy (300): normal {100 * cost['normal']['arc_easy']:.1f}% · chat 25 {100 * cost['chat 25']['arc_easy']:.1f}% · matched {100 * cost['matched 25']['arc_easy']:.1f}% (gate: drop ≤ {100 * g['max_arc_drop']:.0f} points) {'✓' if checks['arc'] else '✗'}",
          f"- WikiText-2 perplexity ratio: chat 25 {cost['chat 25']['wikitext_ratio']:.3f}× · matched {cost['matched 25']['wikitext_ratio']:.3f}× (gate ≤ {g['max_wikitext_ratio']}) {'✓' if checks['wikitext'] else '✗'}",
          f"- \"a\" before a vowel: normal {100 * a_an['normal']:.1f}% · chat 25 {100 * a_an['chat 25']:.1f}% (gate ≤ {100 * g['max_a_an_rate']:.0f}%) {'✓' if checks['a_an'] else '✗'}",
          f"- Blind preference, chat 25 vs normal ({wins + ties + losses} pairs): chat 25 wins {wins}, ties {ties}, normal wins {losses}; share {100 * pref_share:.0f}% (gate ≥ {100 * g['min_preference_share']:.0f}%) {'✓' if checks['preference'] else '✗'}",
          "", "## Secondary, non-gating", "", "| genre | habit with chat 25 off |", "|---|---|"]
    for gn, t in per_genre.items():
        md.append(f"| {gn} | {f(t)} |")
    md += ["", "| tier | chat 25 | matched 25 |", "|---|---|---|"]
    for t, d in tiers.items():
        md.append(f"| {t} | {f(d['chat 25'])} | {f(d['matched 25'])} |")
    md += ["", f"Explicit requests (\"use the pattern 'It's not X, it's Y'\"): answers using it — normal {100 * explicit['normal']:.0f}%, chat 25 {100 * explicit['chat 25']:.0f}%, matched {100 * explicit['matched 25']:.0f}%.",
           f"Answers containing \"but\": normal {100 * but['normal']:.1f}%, chat 25 {100 * but['chat 25']:.1f}%, matched {100 * but['matched 25']:.1f}%.",
           f"Mean words: normal {words['normal']:.0f}, chat 25 {words['chat 25']:.0f}, matched {words['matched 25']:.0f}."]
    OUT_MD.write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--pairs", action="store_true")
    args = ap.parse_args()
    prereg = json.loads(PREREG.read_text())
    check_frozen(prereg)
    if args.analyze:
        analyze(prereg)
    elif args.pairs:
        write_pairs(prereg)
    else:
        generate(prereg)


if __name__ == "__main__":
    main()
