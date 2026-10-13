"""Matched-activation null control for the top-25 coalition.

The pre-registration commits to this control over random-k. For each feature
in top-25, we sample a non-coalition feature with similar activation density.
That gives a same-size null set whose features are equally "active" overall
but not causally implicated. If the coalition still substantially beats the
matched-activation null, the effect isn't an artefact of mass-ablating any
25 active features.

Procedure:
  1. Pull activation densities for all 16k features from Neo4j.
  2. For each f in top-25, find the K=10 nearest non-coalition features by
     |density - f.density|. Sample one per draw.
  3. Run N independent matched draws (pre-registered n=20). Each is its own
     ablation set of size 25.
  4. measure_pivot for each draw via the daemon. Compare distribution of
     drops to coalition's drop.

Resumable: if reports/matched_activation_null.json exists, picks up the
remaining draws (same RNG seed → same draws). Useful for extending from
n=5 to n=20 without redoing work.

Writes reports/matched_activation_null.json.
"""
from __future__ import annotations
import argparse, json, pathlib, time, urllib.request
import numpy as np

PROBE = "http://127.0.0.1:8765/probe"
REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "reports" / "matched_activation_null.json"
DEFAULT_N_DRAWS = 20  # pre-registered
K_NEIGHBOURS = 10     # keep at 10 to preserve prior draws under same seed
SEED = 42


def _write_partial(top25, matched_draws, density_by_idx, n_draws, coal_res,
                    matched_results, coalition_med_density):
    """Snapshot current state to disk so a kill leaves usable partial data."""
    matched_drops = np.array([m["mean_drop"] for m in matched_results]) if matched_results else np.array([])
    summary = {
        "coalition_drop": coal_res["mean_drop"] if coal_res else None,
        "n_draws_done": len(matched_results),
        "matched_null_mean": float(matched_drops.mean()) if matched_drops.size else None,
        "matched_null_std": float(matched_drops.std(ddof=1)) if matched_drops.size > 1 else None,
        "matched_null_max": float(matched_drops.max()) if matched_drops.size else None,
        "coalition_beats_every_matched_draw": (
            bool(coal_res["mean_drop"] > matched_drops.max())
            if coal_res and matched_drops.size else None
        ),
    }
    out = {
        "n_draws": n_draws,
        "n_draws_done": len(matched_results),
        "k_neighbours_per_anchor": K_NEIGHBOURS,
        "seed": SEED,
        "coalition": top25,
        "matched_draws": matched_draws[:len(matched_results)],
        "median_activation_density": {
            "coalition": coalition_med_density,
            "matched_draws": [
                float(np.median([density_by_idx[i] for i in d]))
                for d in matched_draws[:len(matched_results)]
            ],
        },
        "coalition_result": coal_res,
        "matched_results": matched_results,
        "summary": summary,
    }
    OUT.write_text(json.dumps(out, indent=2))


def call(body, timeout=600):
    req = urllib.request.Request(PROBE, data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-draws", type=int, default=DEFAULT_N_DRAWS)
    args = ap.parse_args()
    N_DRAWS = args.n_draws

    # 1. Get coalition
    top25 = call({"cmd": "attribution", "top_n": 25, "kind": "promote"})["result"]["features"]
    coalition_set = set(top25)
    print(f"coalition (top-25): {top25}")

    # 2. Pull all feature densities from Neo4j via daemon graph_cypher
    rows = call({"cmd": "graph_cypher",
                  "query": "MATCH (f:SAEFeature) WHERE f.sae_id CONTAINS 'L20/16k' "
                           "RETURN f.index AS idx, f.activation_density AS d",
                  "params": {}})["result"]["rows"]
    density_by_idx = {int(r["idx"]): float(r["d"] or 0.0) for r in rows}
    print(f"loaded {len(density_by_idx)} feature densities from Neo4j")

    # 3. For each coalition feature, find K nearest by density (excluding coalition)
    coalition_densities = {f: density_by_idx[f] for f in top25}
    candidates_by_anchor: dict[int, list[int]] = {}
    for f in top25:
        target = coalition_densities[f]
        # Sort non-coalition features by |density - target|
        candidates = sorted(
            (i for i in density_by_idx if i not in coalition_set),
            key=lambda i: abs(density_by_idx[i] - target),
        )[:K_NEIGHBOURS]
        candidates_by_anchor[f] = candidates
    print(f"prepared {K_NEIGHBOURS} matched candidates per anchor (median |Δdensity|: "
          f"{np.median([abs(density_by_idx[candidates_by_anchor[f][0]] - density_by_idx[f]) for f in top25]):.5f})")

    # 4. Sample N draws — each draw picks one match per anchor (replacement allowed across draws)
    rng = np.random.default_rng(SEED)
    matched_draws = []
    used_features_per_draw = []  # for diagnostics
    for d in range(N_DRAWS):
        # Random pick from the K candidates for each anchor, no repeats within draw
        draw = []
        used = set()
        for f in top25:
            opts = [c for c in candidates_by_anchor[f] if c not in used]
            if not opts:
                opts = candidates_by_anchor[f]  # fallback if exhausted
            pick = int(rng.choice(opts))
            draw.append(pick)
            used.add(pick)
        matched_draws.append(sorted(draw))
        used_features_per_draw.append(draw)

    # Sanity-check the matching quality
    def median_density(idxs):
        return float(np.median([density_by_idx[i] for i in idxs]))
    coalition_med_density = median_density(top25)
    print(f"\nmedian activation density:")
    print(f"  coalition  : {coalition_med_density:.4f}")
    for i, draw in enumerate(matched_draws):
        print(f"  match draw {i}: {median_density(draw):.4f}")

    # 5. Measure for coalition + each matched draw — RESUMABLE
    def measure(ablate, label):
        t0 = time.perf_counter()
        r = call({"cmd": "measure_pivot", "ablate": ablate, "max_samples": 80})["result"]
        elapsed = time.perf_counter() - t0
        return {
            "label": label,
            "ablate": ablate,
            "baseline_mean": r["baseline_mean"],
            "ablated_mean": r["ablated_mean"],
            "mean_drop": r["mean_drop"],
            "rel_drop": r["rel_drop"],
            "elapsed_s": round(elapsed, 1),
        }

    # Load prior state if any (so we can extend n=5 → n=20 without redoing)
    prior = json.loads(OUT.read_text()) if OUT.exists() else {}
    coal_res = prior.get("coalition_result")
    if coal_res is None:
        print(f"\nmeasuring coalition (top-25)…")
        coal_res = measure(top25, "coalition (top-25)")
        print(f"  drop = {coal_res['mean_drop']:+.4f}  rel = {coal_res['rel_drop']:+.2%}  ({coal_res['elapsed_s']:.0f}s)")
    else:
        print(f"\ncoalition already measured: drop = {coal_res['mean_drop']:+.4f}  rel = {coal_res['rel_drop']:+.2%}")

    matched_results = list(prior.get("matched_results", []))
    print(f"prior matched draws on disk: {len(matched_results)}")
    for i, draw in enumerate(matched_draws):
        if i < len(matched_results):
            continue
        print(f"measuring matched draw {i+1}/{N_DRAWS}…")
        r = measure(draw, f"matched_draw_{i}")
        matched_results.append(r)
        print(f"  drop = {r['mean_drop']:+.4f}  rel = {r['rel_drop']:+.2%}  ({r['elapsed_s']:.0f}s)")
        # Checkpoint each draw
        _write_partial(top25, matched_draws, density_by_idx, N_DRAWS, coal_res, matched_results, coalition_med_density)

    # 6. Stats
    matched_drops = np.array([m["mean_drop"] for m in matched_results])
    coalition_drop = coal_res["mean_drop"]
    null_mean = float(matched_drops.mean())
    null_std = float(matched_drops.std(ddof=1)) if len(matched_drops) > 1 else 0.0
    null_max = float(matched_drops.max())
    null_min = float(matched_drops.min())
    # Report concrete numbers; avoid Infinity ratios when null_mean is non-positive.
    # The headline comparison is coalition_drop vs largest matched draw.
    gap = coalition_drop - null_max
    beats_all = coalition_drop > null_max
    print(f"\n=== Summary ===")
    print(f"  coalition drop          : {coalition_drop:+.4f} ({coal_res['rel_drop']:+.2%})")
    print(f"  matched-null drops      : {[f'{x:+.4f}' for x in matched_drops]}")
    print(f"  matched-null mean       : {null_mean:+.4f}")
    print(f"  matched-null max        : {null_max:+.4f}")
    print(f"  matched-null min        : {null_min:+.4f}")
    print(f"  coalition − largest     : {gap:+.4f}  (coalition exceeds largest matched draw by this much)")
    print(f"  coalition vs all draws  : {'BEATS' if beats_all else 'does NOT beat'} every matched draw")

    out = {
        "n_draws": N_DRAWS,
        "k_neighbours_per_anchor": K_NEIGHBOURS,
        "seed": SEED,
        "coalition": top25,
        "matched_draws": matched_draws,
        "median_activation_density": {
            "coalition": coalition_med_density,
            "matched_draws": [median_density(d) for d in matched_draws],
        },
        "coalition_result": coal_res,
        "matched_results": matched_results,
        "summary": {
            "coalition_drop": coalition_drop,
            "matched_null_mean": null_mean,
            "matched_null_std": null_std,
            "matched_null_max": null_max,
            "matched_null_min": null_min,
            "coalition_minus_largest_matched": gap,
            "coalition_beats_every_matched_draw": beats_all,
        },
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
