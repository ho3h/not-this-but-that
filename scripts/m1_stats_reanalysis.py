"""Bootstrap CIs + paired McNemar exact tests on the existing Q5b/Q5c/Q5d JSONs.

Adds proper inferential statistics to the M1 results that the post quotes as
point estimates. Two new statistics per probe:

  1. Prompt-clustered bootstrap 95% CI on construction rates and on the
     absolute drop (resamples PROMPTS, since seeds within a prompt aren't
     independent — paired-prompt design). 10,000 resamples.
  2. McNemar's mid-p exact paired test on the per-pair binary outcomes
     (baseline-construction × ablated-construction discordant pairs).
     Mid-p variant per Fagerland, Lydersen & Laake 2013.

Uses the v2 union classifier so numbers match the post.

Writes reports/m1_stats_reanalysis.json + a short markdown summary.
"""
from __future__ import annotations
import json, pathlib
from math import comb
import numpy as np

from classifier import has_construction

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT_JSON = REPO / "reports" / "m1_stats_reanalysis.json"
OUT_MD = REPO / "reports" / "m1_stats_reanalysis.md"


def score(rows: list[dict], prompt_key: str) -> list[dict]:
    """Return list of {prompt_id, baseline_hit, ablated_hit} per pair."""
    out = []
    for r in rows:
        out.append({
            "prompt_id": r[prompt_key],
            "seed": r.get("seed"),
            "baseline_hit": has_construction(r["baseline"]),
            "ablated_hit": has_construction(r["ablated"]),
        })
    return out


def bootstrap_clustered(rows: list[dict], n_boot: int = 10000, seed: int = 42) -> dict:
    """Prompt-clustered bootstrap. Resample prompt_ids with replacement; for
    each draw, take ALL pairs sharing the sampled prompt_ids. Compute
    baseline_rate, ablated_rate, absolute_drop, relative_drop.
    """
    rng = np.random.default_rng(seed)
    by_prompt: dict = {}
    for r in rows:
        by_prompt.setdefault(r["prompt_id"], []).append(r)
    prompt_ids = list(by_prompt.keys())
    n_p = len(prompt_ids)

    b_rates, a_rates, abs_drops, rel_drops = [], [], [], []
    for _ in range(n_boot):
        picks = rng.choice(prompt_ids, size=n_p, replace=True)
        sampled = []
        for p in picks:
            sampled.extend(by_prompt[p])
        n = len(sampled)
        nb = sum(1 for s in sampled if s["baseline_hit"])
        na = sum(1 for s in sampled if s["ablated_hit"])
        br = nb / n
        ar = na / n
        b_rates.append(br); a_rates.append(ar)
        abs_drops.append(br - ar)
        rel_drops.append((br - ar) / br if br > 0 else 0.0)

    def ci(arr, p_lo=2.5, p_hi=97.5):
        a = np.array(arr)
        return float(np.percentile(a, p_lo)), float(np.percentile(a, p_hi)), float(a.mean())

    b_lo, b_hi, b_mean = ci(b_rates)
    a_lo, a_hi, a_mean = ci(a_rates)
    abs_lo, abs_hi, abs_mean = ci(abs_drops)
    rel_lo, rel_hi, rel_mean = ci(rel_drops)

    return {
        "n_boot": n_boot,
        "n_prompts": n_p,
        "baseline_rate_mean": b_mean, "baseline_rate_ci95": [b_lo, b_hi],
        "ablated_rate_mean": a_mean,  "ablated_rate_ci95": [a_lo, a_hi],
        "absolute_drop_mean": abs_mean, "absolute_drop_ci95": [abs_lo, abs_hi],
        "relative_drop_mean": rel_mean, "relative_drop_ci95": [rel_lo, rel_hi],
    }


def mcnemar_midp(b: int, c: int) -> float:
    """Mid-p McNemar exact, two-sided.

    `b` = number of pairs where baseline=1, ablated=0 (favouring "drop").
    `c` = number of pairs where baseline=0, ablated=1.
    Returns mid-p two-sided p-value.

    Reference: Fagerland, Lydersen & Laake (2013), BMC Med Res Methodol.
    The mid-p exact correction: P = 2 * (sum_{k=0}^{min(b,c)-1} P(X=k))
                                    + P(X=min(b,c))   under X ~ Bin(b+c, 0.5)
    where the outer factor of 2 makes it two-sided.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # P(X=i) under Binomial(n, 0.5)
    def pmf(i):
        return comb(n, i) / (2 ** n)
    tail = sum(pmf(i) for i in range(k))
    midp_one = tail + 0.5 * pmf(k)
    return float(min(1.0, 2.0 * midp_one))


def paired_table(rows: list[dict]) -> dict:
    a = sum(1 for r in rows if     r["baseline_hit"] and     r["ablated_hit"])
    b = sum(1 for r in rows if     r["baseline_hit"] and not r["ablated_hit"])
    c = sum(1 for r in rows if not r["baseline_hit"] and     r["ablated_hit"])
    d = sum(1 for r in rows if not r["baseline_hit"] and not r["ablated_hit"])
    return {
        "both_yes": a, "baseline_only": b, "ablated_only": c, "both_no": d,
        "discordant_total": b + c,
        "mcnemar_midp_p_value_two_sided": mcnemar_midp(b, c),
    }


def main():
    sources = [
        ("Q5b primed",        "reports/q5b_d1_continuation.json",   "prefix_idx"),
        ("Q5c neutral",       "reports/q5c_d2_high_power.json",     "prompt_idx"),
        ("Q5d minimal (n=16)","reports/q5d_minimal_set_d1.json",    "prefix_idx"),
        ("Q5d minimal n=120", "reports/q5d_minimal_set_d1_n120.json","prefix_idx"),
    ]
    results = {}
    for name, path, key in sources:
        d = json.loads((REPO / path).read_text())
        rows = score(d["rows"], key)
        n = len(rows)
        nb = sum(1 for r in rows if r["baseline_hit"])
        na = sum(1 for r in rows if r["ablated_hit"])
        print(f"\n=== {name} (n={n}) ===")
        print(f"  raw rates  : baseline {nb}/{n} = {nb/n:.2%}, ablated {na}/{n} = {na/n:.2%}")
        boot = bootstrap_clustered(rows, n_boot=10000, seed=42)
        mc = paired_table(rows)
        print(f"  bootstrap  : baseline CI95 [{boot['baseline_rate_ci95'][0]:.2%}, {boot['baseline_rate_ci95'][1]:.2%}], "
              f"ablated CI95 [{boot['ablated_rate_ci95'][0]:.2%}, {boot['ablated_rate_ci95'][1]:.2%}]")
        print(f"               absolute drop CI95 [{boot['absolute_drop_ci95'][0]:+.4f}, {boot['absolute_drop_ci95'][1]:+.4f}]")
        print(f"               relative drop CI95 [{boot['relative_drop_ci95'][0]:+.2%}, {boot['relative_drop_ci95'][1]:+.2%}]")
        print(f"  McNemar    : discordant pairs b={mc['baseline_only']} (kills) c={mc['ablated_only']} (leaks); "
              f"mid-p two-sided = {mc['mcnemar_midp_p_value_two_sided']:.6f}")
        results[name] = {"n": n, "baseline_hits": nb, "ablated_hits": na,
                          "bootstrap": boot, "mcnemar": mc}

    out = {
        "method_note": (
            "Per-prompt clustered bootstrap (10,000 resamples) of construction "
            "rates and drops, scored with the v2 union detector (strict + "
            "permissive); McNemar's mid-p exact paired test (Fagerland, "
            "Lydersen & Laake 2013) on per-pair discordant outcomes."
        ),
        "by_probe": results,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))

    md = [
        "# M1 — bootstrap CIs + paired McNemar",
        "",
        "Re-analysis of the Q5b/Q5c/Q5d generations (saved JSON) with proper",
        "inferential statistics. Scoring uses the v2 union detector",
        "(strict + permissive — see `src/classifier/detect_v2.py`).",
        "",
        "| probe | n | baseline rate | ablated rate | absolute drop (95% CI) | rel drop (95% CI) | McNemar mid-p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        b = r["bootstrap"]; m = r["mcnemar"]
        md.append(
            f"| {name} | {r['n']} | "
            f"{r['baseline_hits']}/{r['n']} = {r['baseline_hits']/r['n']:.2%} | "
            f"{r['ablated_hits']}/{r['n']} = {r['ablated_hits']/r['n']:.2%} | "
            f"{b['absolute_drop_mean']:+.4f} [{b['absolute_drop_ci95'][0]:+.4f}, {b['absolute_drop_ci95'][1]:+.4f}] | "
            f"{b['relative_drop_mean']:+.2%} [{b['relative_drop_ci95'][0]:+.2%}, {b['relative_drop_ci95'][1]:+.2%}] | "
            f"{m['mcnemar_midp_p_value_two_sided']:.4g} |"
        )
    md.append("")
    md.append("Per-probe paired tables (b = baseline-hit & ablated-clean; c = baseline-clean & ablated-hit):")
    md.append("")
    for name, r in results.items():
        m = r["mcnemar"]
        md.append(f"- **{name}**: both yes {m['both_yes']}, baseline-only {m['baseline_only']}, "
                  f"ablated-only {m['ablated_only']}, both no {m['both_no']}; "
                  f"discordant {m['discordant_total']}.")
    OUT_MD.write_text("\n".join(md))
    print(f"\n→ {OUT_MD}")
    print(f"→ {OUT_JSON}")


if __name__ == "__main__":
    main()
