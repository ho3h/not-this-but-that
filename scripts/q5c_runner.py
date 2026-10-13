"""Q5c runner — full D2 sustained-ablation run, checkpoints after every pair.

Detach via: nohup .venv/bin/python scripts/q5c_runner.py > /tmp/q5c.log 2>&1 &
Resumable: re-running reads `reports/q5c_d2_high_power.json` and continues
from the next undone (prompt_idx, seed).
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
import urllib.request

PROBE = "http://127.0.0.1:8765/probe"
OUT = pathlib.Path("reports/q5c_d2_high_power.json")
SEEDS = [0, 1, 2]
MAX_NEW = 50            # smaller than the prior 100-token mistake
TEMPERATURE = 0.8
TOP_P = 0.95
N_PROMPTS_TARGET = 102  # full D2: 102 × 3 = 306 pairs


def call(body, timeout=600):
    req = urllib.request.Request(
        PROBE, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def gen_one(prompt, ablate, seed):
    return call(
        {"cmd": "generate", "model": "it", "prompt": prompt, "ablate": ablate,
         "max_new_tokens": MAX_NEW, "temperature": TEMPERATURE, "top_p": TOP_P,
         "seed": seed},
        timeout=300,
    )["result"]["completion"]


def main():
    d2 = json.loads(pathlib.Path("data/D2_neutral_prompts.json").read_text())["prompts"]
    top25 = call({"cmd": "attribution", "top_n": 25, "kind": "promote"})["result"]["features"]

    if OUT.exists():
        state = json.loads(OUT.read_text())
    else:
        state = {"top25": top25, "max_new_tokens": MAX_NEW, "seeds": SEEDS,
                 "model": "it", "rows": []}

    from classifier import detect_construction
    done_keys = {(r["prompt_idx"], r["seed"]) for r in state["rows"]}
    n_target_pairs = N_PROMPTS_TARGET * len(SEEDS)
    print(f"already done: {len(state['rows'])} pairs; target {n_target_pairs}", flush=True)

    t0 = time.perf_counter()
    for pi in range(N_PROMPTS_TARGET):
        if pi >= len(d2):
            print(f"  hit end of D2 ({len(d2)} prompts)")
            break
        prompt = d2[pi]
        for s in SEEDS:
            if (pi, s) in done_keys:
                continue
            try:
                base_text = gen_one(prompt, None, s)
                abl_text = gen_one(prompt, top25, s)
            except Exception as e:
                print(f"  ERROR at pi={pi} s={s}: {e}", flush=True)
                # If daemon's down, exit cleanly so we can resume
                sys.exit(2)
            b_hits = detect_construction(base_text, strict=False)
            a_hits = detect_construction(abl_text, strict=False)
            state["rows"].append({
                "prompt_idx": pi, "prompt": prompt, "seed": s,
                "baseline": base_text, "ablated": abl_text,
                "baseline_variants": sorted({h.variant.value for h in b_hits}),
                "ablated_variants": sorted({h.variant.value for h in a_hits}),
                "baseline_core": any(h.variant.value in ("C1", "C2", "C3") for h in b_hits),
                "ablated_core": any(h.variant.value in ("C1", "C2", "C3") for h in a_hits),
            })
            OUT.write_text(json.dumps(state, indent=2))
            n = len(state["rows"])
            b_rate = sum(1 for r in state["rows"] if r["baseline_core"]) / n
            a_rate = sum(1 for r in state["rows"] if r["ablated_core"]) / n
            elapsed = time.perf_counter() - t0
            rate = n / max(elapsed, 1)
            fb = "✓" if state["rows"][-1]["baseline_core"] else "·"
            fa = "✓" if state["rows"][-1]["ablated_core"] else "·"
            print(f"  [{n:>3}/{n_target_pairs}] pi={pi:>2} s={s}  "
                  f"base{fb} {state['rows'][-1]['baseline_variants']!s:<14} "
                  f"abl{fa} {state['rows'][-1]['ablated_variants']!s:<14}  "
                  f"running: base={b_rate:.2%} abl={a_rate:.2%}  "
                  f"({rate*60:.2f} pair/min)", flush=True)

    n = len(state["rows"])
    b_rate = sum(1 for r in state["rows"] if r["baseline_core"]) / max(n, 1)
    a_rate = sum(1 for r in state["rows"] if r["ablated_core"]) / max(n, 1)
    print(f"\nDONE  n={n}  base={b_rate:.2%}  abl={a_rate:.2%}  drop={b_rate-a_rate:+.2%}  "
          f"elapsed={time.perf_counter()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
