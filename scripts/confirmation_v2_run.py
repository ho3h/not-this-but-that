"""Pre-registered confirmation v2: chat-formatted answers, error term preserved.

Reads every parameter from reports/confirmation_v2_prereg.json (coalition,
prompts file and its sha256, sampling, stages, gates). Refuses to run unless
the pre-registration, the prompts and the detector files are committed and
unmodified, and the prompts file matches the registered hash.

Event-driven design (registered): stage 1 draws `samples_per_stage` answers
per prompt and condition. If the baseline family events on M prompts reach
`target_baseline_events_M`, the analysis runs on stage 1. Otherwise stage 2
draws the same number again and the analysis runs once on both stages. The
stage decision reads baseline outputs only. No other interim looks: the
console shows progress, never rates.

  .venv/bin/python scripts/confirmation_v2_run.py            # generate (checkpointed)
  .venv/bin/python scripts/confirmation_v2_run.py --analyze  # verdict, once

Writes reports/confirmation_v2_generations.jsonl and, on --analyze,
reports/confirmation_v2_run.{json,md}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from math import comb

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import v2_engine as E  # noqa: E402

REPO = E.REPO
PREREG = REPO / "reports" / "confirmation_v2_prereg.json"
GENS = REPO / "reports" / "confirmation_v2_generations.jsonl"
OUT_JSON = REPO / "reports" / "confirmation_v2_run.json"
OUT_MD = REPO / "reports" / "confirmation_v2_run.md"
FROZEN_PATHS = [
    "reports/confirmation_v2_prereg.json", "reports/confirmation_v2_prereg.md",
    "src/classifier/detect.py", "src/classifier/detect_v2.py", "scripts/v2_engine.py", "scripts/v2_mlx.py",
    "scripts/confirmation_v2_run.py",
]


def check_frozen(prereg: dict) -> None:
    """Every frozen file must match the SHA-256 recorded at registration."""
    prompts_path = REPO / prereg["prompts_file"]
    digest = hashlib.sha256(prompts_path.read_bytes()).hexdigest()
    if digest != prereg["prompts_sha256"]:
        sys.exit(f"prompts file hash {digest[:12]} != registered {prereg['prompts_sha256'][:12]}")
    bad = []
    for rel, want in prereg["frozen_sha256"].items():
        got = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
        if got != want:
            bad.append(rel)
    if bad:
        sys.exit("frozen files changed since registration: " + ", ".join(bad))


def load_rows() -> list[dict]:
    if not GENS.exists():
        return []
    return [json.loads(line) for line in GENS.read_text().splitlines() if line.strip()]


def seed_for(prereg: dict, pid: int, stage: int) -> int:
    return prereg["seed_base"] + pid * 10 + stage


def generate_stage(prereg: dict, stage: int, model, tok, wrap, done: set) -> None:
    import v2_mlx as M

    prompts = json.loads((REPO / prereg["prompts_file"]).read_text())["prompts"]
    n = prereg["samples_per_stage"]
    t0 = time.time()
    for i, p in enumerate(prompts):
        if (p["id"], stage) in done:
            continue
        out = M.generate_paired(model, tok, wrap, M.chat_tokens(tok, p["prompt"]),
                                seed=seed_for(prereg, p["id"], stage), n=n,
                                max_new_tokens=prereg["max_new_tokens"],
                                temperature=prereg["temperature"], top_p=prereg["top_p"])
        with GENS.open("a") as f:
            for k in range(n):
                f.write(json.dumps({"id": p["id"], "type": p["type"], "prompt": p["prompt"],
                                    "stage": stage, "sample": k,
                                    "baseline": out["normal"][k],
                                    "coalition": out["intervened"][k]}) + "\n")
        el = time.time() - t0
        print(f"stage {stage} · prompt {i + 1}/{len(prompts)} · {el / 60:.1f} min", flush=True)


def baseline_M_events(rows: list[dict]) -> int:
    return sum(1 for r in rows if r["type"] == "M" and E.score(r["baseline"])["family"])


def midp(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    pmf = lambda i: comb(n, i) / 2 ** n  # noqa: E731
    return float(min(1.0, 2.0 * (sum(pmf(i) for i in range(k)) + 0.5 * pmf(k))))


def boot_rel_drop(rows: list[dict], key: str, n_boot: int = 10_000, seed: int = 0):
    by = {}
    for r in rows:
        by.setdefault(r["id"], []).append(r)
    ids = list(by)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        s = [rng.choice(ids) for _ in ids]
        b = sum(x[f"b_{key}"] for i in s for x in by[i])
        a = sum(x[f"a_{key}"] for i in s for x in by[i])
        if b:
            vals.append(1 - a / b)
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]) if vals else None


def boot_ratio(rows: list[dict], n_boot: int = 10_000, seed: int = 1):
    by = {"M": {}, "P": {}}
    for r in rows:
        by[r["type"]].setdefault(r["id"], []).append(r["b_fc"])
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        rate = {}
        for t, d in by.items():
            ids = list(d)
            s = [rng.choice(ids) for _ in ids]
            hits = sum(sum(d[i]) for i in s)
            tot = sum(len(d[i]) for i in s)
            rate[t] = hits / tot
        if rate["P"] > 0:
            vals.append(rate["M"] / rate["P"])
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]) if vals else None


def analyze(prereg: dict) -> None:
    rows = load_rows()
    stages = sorted({r["stage"] for r in rows})
    scored = []
    for r in rows:
        b, a = E.score(r["baseline"]), E.score(r["coalition"])
        scored.append({"id": r["id"], "type": r["type"],
                       "b_fam": int(b["family"]), "a_fam": int(a["family"]),
                       "b_strict": int(b["strict"]), "a_strict": int(a["strict"]),
                       "b_cou": int(b["cousin"]), "a_cou": int(a["cousin"]),
                       "b_fc": int(b["family"] or b["cousin"]), "a_fc": int(a["family"] or a["cousin"]),
                       "b_but": int(b["but"]), "a_but": int(a["but"]),
                       "b_words": b["words"], "a_words": a["words"]})

    def tier(sub, key):
        bb = sum(1 for x in sub if x[f"b_{key}"] and not x[f"a_{key}"])
        cc = sum(1 for x in sub if not x[f"b_{key}"] and x[f"a_{key}"])
        base, abl = sum(x[f"b_{key}"] for x in sub), sum(x[f"a_{key}"] for x in sub)
        return {"n": len(sub), "baseline": base, "ablated": abl,
                "rel_drop": (1 - abl / base) if base else None,
                "b": bb, "c": cc, "mid_p": midp(bb, cc)}

    M = [x for x in scored if x["type"] == "M"]
    P = [x for x in scored if x["type"] == "P"]
    h1 = tier(M, "fam")
    h1["ci95"] = boot_rel_drop(M, "fam")
    g = prereg["gates"]
    rd, p = h1["rel_drop"] or 0.0, h1["mid_p"]
    if rd >= g["pass_rel_drop"] and p < g["alpha"]:
        verdict = "PASS"
    elif rd < g["kill_rel_drop"] or p >= g["alpha"]:
        verdict = "KILL"
    else:
        verdict = "REPLICATED DIRECTION, SMALLER MAGNITUDE"

    rate = lambda sub, k: sum(x[k] for x in sub) / len(sub)  # noqa: E731
    ratio = rate(M, "b_fc") / rate(P, "b_fc") if rate(P, "b_fc") else None
    ci_ratio = boot_ratio(scored)
    h2_ok = ratio is not None and ratio >= g["h2_min_ratio"] and ci_ratio and ci_ratio[0] > 1
    result = {
        "stages_run": stages, "pairs": len(scored),
        "H1_family_on_M": h1, "H1_verdict": verdict,
        "H2_rate_M": rate(M, "b_fc"), "H2_rate_P": rate(P, "b_fc"),
        "H2_ratio": ratio, "H2_ci95": ci_ratio, "H2_verdict": "CONFIRMED" if h2_ok else "NOT CONFIRMED",
        "secondary": {
            "strict_on_M": tier(M, "strict"), "cousin_on_M": tier(M, "cou"),
            "family_plus_cousin_on_M": tier(M, "fc"), "family_on_all": tier(scored, "fam"),
            "family_on_P": tier(P, "fam"),
            "but_share_M": [rate(M, "b_but"), rate(M, "a_but")],
            "words_mean_M": [sum(x["b_words"] for x in M) / len(M), sum(x["a_words"] for x in M) / len(M)],
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))
    s = result["secondary"]
    md = [
        "# Pre-registered confirmation v2: verdict", "",
        f"Analysed once, stages {stages}, {len(scored)} pairs ({len(M)} on M prompts, {len(P)} on P prompts).", "",
        f"## H1 (primary): family tier on M prompts: **{verdict}**", "",
        f"Baseline {h1['baseline']}/{h1['n']} → coalition off {h1['ablated']}/{h1['n']}; "
        f"relative drop {100 * rd:.1f}% (95% prompt-clustered CI "
        + (f"{100 * h1['ci95'][0]:.0f}% to {100 * h1['ci95'][1]:.0f}%" if h1["ci95"] else "n/a")
        + f"); McNemar mid-p {p:.3g} (b={h1['b']}, c={h1['c']}).", "",
        f"Gates: PASS if drop ≥ {100 * g['pass_rel_drop']:.0f}% and mid-p < {g['alpha']}; "
        f"KILL if drop < {100 * g['kill_rel_drop']:.0f}% or mid-p ≥ {g['alpha']}.", "",
        f"## H2 (secondary): prompt type: **{result['H2_verdict']}**", "",
        f"Baseline family-or-cousin rate: M {100 * result['H2_rate_M']:.1f}%, P {100 * result['H2_rate_P']:.1f}%; "
        f"ratio " + (f"{ratio:.1f}×" if ratio else "undefined")
        + (f" (95% CI {ci_ratio[0]:.1f}× to {ci_ratio[1]:.1f}×)" if ci_ratio else "")
        + f". Gate: ratio ≥ {g['h2_min_ratio']} and CI lower bound > 1.", "",
        "## Secondary, non-gating", "",
        "| tier (M prompts unless noted) | baseline | coalition off | rel drop | mid-p |", "|---|---:|---:|---:|---:|",
    ]
    for name, t in (("textbook (v1)", s["strict_on_M"]), ("cousin", s["cousin_on_M"]),
                    ("family + cousin", s["family_plus_cousin_on_M"]),
                    ("family, all prompts", s["family_on_all"]), ("family, P prompts", s["family_on_P"])):
        rdx = f"{100 * t['rel_drop']:.0f}%" if t["rel_drop"] is not None else "n/a"
        md.append(f"| {name} | {t['baseline']}/{t['n']} | {t['ablated']}/{t['n']} | {rdx} | {t['mid_p']:.3g} |")
    md += ["", f"Share of M answers containing \"but\": {100 * s['but_share_M'][0]:.1f}% → {100 * s['but_share_M'][1]:.1f}%. "
               f"Mean words per M answer: {s['words_mean_M'][0]:.0f} → {s['words_mean_M'][1]:.0f}."]
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
        return
    import v2_mlx as M

    model, tok, wrap = M.load(prereg["model"], prereg["coalition"])
    done = {(r["id"], r["stage"]) for r in load_rows()}
    generate_stage(prereg, 1, model, tok, wrap, done)
    events = baseline_M_events([r for r in load_rows() if r["stage"] == 1])
    if events >= prereg["target_baseline_events_M"]:
        print(f"stage 1 reached the event target; analysis runs on stage 1 only.")
        return
    print("stage 1 below the event target; running stage 2 as registered.")
    generate_stage(prereg, 2, model, tok, wrap, done)
    print("done. Run with --analyze for the verdict.")


if __name__ == "__main__":
    main()
