"""Q7-full ladder runner — call daemon's ladder command at any layer.

Usage:
    .venv/bin/python scripts/q7full_ladder.py --layer 12
    .venv/bin/python scripts/q7full_ladder.py --layer 25

Reads `reports/pivot_attribution_L{layer}.json` (or _L20.json for layer 20),
builds the {5,10,25,50,75,100} ladder, sends it to the daemon with
`sae_layer: <N>`, saves to `reports/asymptote_ladder_L{layer}.json`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.request

PROBE = "http://127.0.0.1:8765/probe"


def call(body, timeout=3600):
    req = urllib.request.Request(
        PROBE, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--sizes", nargs="+", type=int, default=[5, 10, 25, 50, 75, 100])
    ap.add_argument("--max-samples", type=int, default=80)
    ap.add_argument("--n-random", type=int, default=3)
    args = ap.parse_args()

    repo = pathlib.Path(__file__).resolve().parent.parent
    attrib_path = (repo / "reports" /
                   ("pivot_attribution.json" if args.layer == 20
                    else f"pivot_attribution_L{args.layer}.json"))
    if not attrib_path.exists():
        print(f"ERROR: {attrib_path} not found. Run pivot_attribution.py "
              f"--sae-layer {args.layer} first.", file=sys.stderr)
        sys.exit(1)

    data = json.loads(attrib_path.read_text())
    full = data.get("full_ranked_by_score") or data["top_promotes_pivot"]
    full = sorted(full, key=lambda r: -r["scored"])

    conditions = {}
    for n in args.sizes:
        feats = [int(r["feature_idx"]) for r in full[:n]]
        if len(feats) < n:
            print(f"  WARNING: only {len(feats)} features available for size {n}")
        conditions[f"attrib_top{n}"] = feats

    print(f"Layer {args.layer}: built {len(conditions)} conditions "
          f"({list(conditions.keys())})")
    print(f"Calling daemon ladder with max_samples={args.max_samples}, "
          f"n_random_per_size={args.n_random}, sae_layer={args.layer}…")
    t0 = time.perf_counter()
    resp = call({
        "cmd": "ladder",
        "conditions": conditions,
        "n_random_per_size": args.n_random,
        "max_samples": args.max_samples,
        "sae_layer": args.layer,
        "seed": 11,
    })
    print(f"  done in {time.perf_counter()-t0:.0f}s")

    if not resp.get("ok"):
        print(f"ERROR: {resp.get('error')}", file=sys.stderr)
        sys.exit(2)

    out_path = repo / "reports" / f"asymptote_ladder_L{args.layer}.json"
    out_path.write_text(json.dumps(resp, indent=2))

    r = resp["result"]
    print(f"\nBaseline P(pivot) at L{args.layer}: {r['baseline_mean_p_pivot']:.4f}")
    print(f"\n=== Layer {args.layer} ladder ===")
    print(f"{'cond':<22} {'n':>4} {'p':>8} {'drop':>8} {'rel':>8} {'null_max':>10}")
    for n in args.sizes:
        name = f"attrib_top{n}"
        if name not in r["by_condition"]:
            continue
        s = r["by_condition"][name]
        null = r["random_by_size"].get(str(n)) or r["random_by_size"].get(n) or {}
        nm = null.get("max", 0.0)
        print(f"{name:<22} {s['n_features']:>4} {s['mean_p_pivot']:>8.4f} "
              f"{s['mean_drop']:>+8.4f} {s['rel_drop']:>+8.2%} {nm:>+10.4f}")
    print(f"\n→ saved {out_path}")


if __name__ == "__main__":
    main()
