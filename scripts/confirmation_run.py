"""Pre-registered confirmation run on the fresh D2b prompts.

Reads ALL parameters from reports/confirmation_prereg.json — coalition,
prompts, seeds, sampling config, decision rules. Nothing is derived at run
time and the daemon's attribution endpoint is deliberately not consulted.

Mirrors the Q5b/Q5c protocol: daemon `generate` with the SAE spliced in for
both conditions, per-pair checkpointing (crash-safe resume). Unlike the
exploratory runners it does NOT print running construction rates — the
pre-registration commits to a single analysis at full n, so the console
shows progress only.

Run (daemon must be up: scripts/probe_run.sh start):
  .venv/bin/python scripts/confirmation_run.py          # ~300 pairs, hours on MPS
  .venv/bin/python scripts/confirmation_run.py --analyze  # verdict at full n

Writes reports/confirmation_run.json (rows) and, on --analyze at full n,
reports/confirmation_run.md (the verdict against the pre-registered gates).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.request

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

PROBE = "http://127.0.0.1:8765/probe"
PREREG = REPO / "reports" / "confirmation_prereg.json"
OUT = REPO / "reports" / "confirmation_run.json"
OUT_MD = REPO / "reports" / "confirmation_run.md"


def call(body, timeout=300):
    req = urllib.request.Request(
        PROBE, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def generate(prereg, prompt, ablate, seed):
    return call({
        "cmd": "generate", "model": prereg["model"], "prompt": prompt,
        "ablate": ablate, "max_new_tokens": prereg["max_new_tokens"],
        "temperature": prereg["temperature"], "top_p": prereg["top_p"],
        "seed": seed,
    })["result"]["completion"]


def run(prereg):
    prompts = json.loads((REPO / prereg["prompts_file"]).read_text())["prompts"]
    coalition = prereg["coalition"]
    seeds = prereg["seeds"]

    state = json.loads(OUT.read_text()) if OUT.exists() else {
        "prereg": prereg, "rows": []}
    done = {(r["prompt_idx"], r["seed"]) for r in state["rows"]}
    target = len(prompts) * len(seeds)
    print(f"confirmation run: {len(state['rows'])}/{target} pairs done", flush=True)

    t0 = time.perf_counter()
    for pi, prompt in enumerate(prompts):
        for s in seeds:
            if (pi, s) in done:
                continue
            try:
                base = generate(prereg, prompt, None, s)
                abl = generate(prereg, prompt, coalition, s)
            except Exception as e:
                print(f"ERROR pi={pi} s={s}: {e} — exiting (resumable)", flush=True)
                sys.exit(2)
            state["rows"].append({
                "prompt_idx": pi, "prompt": prompt, "seed": s,
                "baseline": base, "ablated": abl,
            })
            OUT.write_text(json.dumps(state, indent=2))
            n = len(state["rows"])
            rate = n / max(time.perf_counter() - t0, 1)
            # progress only — no running construction rates (pre-registered
            # single analysis; see confirmation_prereg.md)
            print(f"  [{n:>3}/{target}] pi={pi:>2} s={s}  ({rate*60:.2f} pair/min)",
                  flush=True)
    print("generation complete — run with --analyze for the verdict", flush=True)


def analyze(prereg):
    from m1_stats_reanalysis import bootstrap_clustered, mcnemar_midp, paired_table, score

    from classifier import detect_construction

    state = json.loads(OUT.read_text())
    rows = state["rows"]
    target = prereg["n_pairs"]
    if len(rows) < target:
        print(f"refusing to analyze at n={len(rows)} < pre-registered n={target} "
              "(no interim looks)")
        sys.exit(1)

    scored = score(rows, "prompt_idx")
    n = len(scored)
    nb = sum(1 for r in scored if r["baseline_hit"])
    na = sum(1 for r in scored if r["ablated_hit"])
    rel = (nb - na) / nb if nb else 0.0
    mc = paired_table(scored)
    p = mc["mcnemar_midp_p_value_two_sided"]
    boot = bootstrap_clustered(scored, n_boot=10000, seed=42)

    def strict_hit(t):
        return any(h.variant.value in ("C1", "C2", "C3")
                   for h in detect_construction(t, strict=False))
    sb = sum(1 for r in rows if strict_hit(r["baseline"]))
    sa = sum(1 for r in rows if strict_hit(r["ablated"]))

    pr = prereg["pass_rule"]; kr = prereg["kill_rule"]
    if rel >= pr["min_rel_drop"] and p < pr["max_midp"]:
        verdict = "PASS"
    elif rel < kr["rel_drop_below"] or p >= kr["or_midp_at_least"]:
        verdict = "KILL"
    else:
        verdict = "REPLICATED DIRECTION, SMALLER MAGNITUDE"

    md = [
        "# Pre-registered confirmation run — verdict",
        "",
        f"Analyzed once at n = {n} pairs per `confirmation_prereg.md` "
        "(registered 2026-06-09).",
        "",
        f"- Union: baseline {nb}/{n} ({nb/n:.2%}) → ablated {na}/{n} ({na/n:.2%})",
        f"- Relative drop: **{rel:.1%}** "
        f"(95% prompt-clustered bootstrap CI [{boot['relative_drop_ci95'][0]:+.1%}, "
        f"{boot['relative_drop_ci95'][1]:+.1%}])",
        f"- McNemar mid-p (two-sided): **{p:.4g}** "
        f"(kills {mc['baseline_only']}, leaks {mc['ablated_only']})",
        f"- Strict detector (secondary, non-gating): {sb} → {sa}",
        "",
        f"## Verdict against the pre-registered gates: **{verdict}**",
        "",
        f"(PASS requires rel drop ≥ {pr['min_rel_drop']:.0%} and "
        f"mid-p < {pr['max_midp']}; KILL if rel drop < {kr['rel_drop_below']:.0%} "
        f"or mid-p ≥ {kr['or_midp_at_least']}.)",
    ]
    OUT_MD.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n→ {OUT_MD}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if not PREREG.exists():
        print("reports/confirmation_prereg.json missing — register first.")
        sys.exit(1)
    prereg = json.loads(PREREG.read_text())
    if args.analyze:
        analyze(prereg)
    else:
        run(prereg)


if __name__ == "__main__":
    main()
