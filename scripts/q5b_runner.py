"""Q5b expansion runner — D1 continuation, primed prompts, top-25 ablation.

Same methodology as Q5c but on D1 truncated 'with'-prompts (the model is
*primed* for the construction). Resumable: re-reads
reports/q5b_d1_continuation.json and continues from undone (prefix_idx, seed).

Target: 100 prefixes × 3 seeds = 300 pairs (vs original 40 × 2 = 80).
"""
from __future__ import annotations
import json, pathlib, re, sys, time, urllib.request

PROBE = "http://127.0.0.1:8765/probe"
OUT = pathlib.Path("reports/q5b_d1_continuation.json")
SEEDS = [0, 1, 2]
MAX_NEW = 50
TEMPERATURE = 0.8
TOP_P = 0.95
N_PREFIX_TARGET = 100

PIVOT_RE = re.compile(
    r"[,;—–\-]\s*(?:it|that|this|he|she|we|they|these|those|there|but|also|yet)\b",
    re.IGNORECASE,
)


def call(body, timeout=600):
    req = urllib.request.Request(PROBE, data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def truncate_to_pivot(text):
    m = PIVOT_RE.search(text)
    return text[:m.start()].rstrip() if m else None


def gen_one(prompt, ablate, seed):
    return call(
        {"cmd": "generate", "model": "it", "prompt": prompt, "ablate": ablate,
         "max_new_tokens": MAX_NEW, "temperature": TEMPERATURE, "top_p": TOP_P, "seed": seed},
        timeout=300,
    )["result"]["completion"]


def main():
    # Load all D1 prefixes
    prefixes = []
    for line in pathlib.Path("data/D1_contrast_pairs.jsonl").read_text().splitlines():
        if not line.strip() or line.startswith('{"_meta'):
            continue
        d = json.loads(line)
        pref = truncate_to_pivot(d["with"])
        if pref:
            prefixes.append({"variant": d["variant"], "prefix": pref})
    print(f"D1 prefixes available: {len(prefixes)}; target {N_PREFIX_TARGET}", flush=True)

    top25 = call({"cmd": "attribution", "top_n": 25, "kind": "promote"})["result"]["features"]

    if OUT.exists():
        state = json.loads(OUT.read_text())
        print(f"resuming with {len(state.get('rows', []))} prior pairs on disk", flush=True)
    else:
        state = {"top25": top25, "max_new_tokens": MAX_NEW, "seeds": SEEDS,
                 "model": "it", "rows": []}

    from classifier import detect_construction
    done = {(r["prefix_idx"], r["seed"]) for r in state["rows"]}

    t0 = time.perf_counter()
    n_done_at_start = len(done)
    for pi in range(min(N_PREFIX_TARGET, len(prefixes))):
        p = prefixes[pi]
        for s in SEEDS:
            if (pi, s) in done:
                continue
            try:
                base = gen_one(p["prefix"], None, s)
                abl = gen_one(p["prefix"], top25, s)
            except Exception as e:
                print(f"  ERROR pi={pi} s={s}: {e}", flush=True)
                sys.exit(2)
            b_hits = detect_construction(base, strict=False)
            a_hits = detect_construction(abl, strict=False)
            state["rows"].append({
                "prefix_idx": pi, "prefix": p["prefix"], "variant": p["variant"], "seed": s,
                "baseline": base, "ablated": abl,
                "baseline_variants": sorted({h.variant.value for h in b_hits}),
                "ablated_variants": sorted({h.variant.value for h in a_hits}),
                "baseline_core": any(h.variant.value in ("C1", "C2", "C3") for h in b_hits),
                "ablated_core": any(h.variant.value in ("C1", "C2", "C3") for h in a_hits),
            })
            OUT.write_text(json.dumps(state, indent=2))
            n = len(state["rows"])
            elapsed = time.perf_counter() - t0
            rate = max(n - n_done_at_start, 1) / max(elapsed, 1)
            print(f"  [{n:>3}] pi={pi:>2} s={s}  "
                  f"base{'✓' if state['rows'][-1]['baseline_core'] else '·'} "
                  f"abl{'✓' if state['rows'][-1]['ablated_core'] else '·'}  "
                  f"({rate*60:.2f} pair/min)", flush=True)

    n = len(state["rows"])
    nb = sum(1 for r in state["rows"] if r["baseline_core"])
    na = sum(1 for r in state["rows"] if r["ablated_core"])
    print(f"\nDONE  n={n}  baseline {nb}/{n} = {nb/n:.2%}  ablated {na}/{n} = {na/n:.2%}", flush=True)


if __name__ == "__main__":
    main()
