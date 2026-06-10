"""Held-out re-slices of the committed eval JSONs — no new compute.

Selection/evaluation independence audit. The deployed top-25 coalition was
selected by per-feature attribution over the FIRST 40 D1 prefixes
(reports/pivot_attribution_n40.json == reports/pivot_attribution.json,
promote-ranked). That makes three slices of the existing evals cleanly
out-of-sample w.r.t. coalition selection:

  1. Q5c on the 26 confirmation-split D2 prompts (data/splits/d2.json) —
     doubly held out: D2 never entered selection, and these 26 prompts were
     reserved by the project's own operating protocol.
  2. Q5b prefixes 40-99 (180 pairs) — never used to select the deployed
     coalition.
  3. Q5b prefixes 80-99 (60 pairs) — additionally disjoint from the n=80 and
     n=100 attribution scans, i.e. never touched by ANY attribution run.

Also reports a prompt-level sign test on the full Q5c (collapse seeds within
a prompt; count prompts that changed status) since seeds within a prompt are
correlated and pair-level McNemar treats them as independent.

Scored with the v2 union detector (post 2026-06-09 fix) + strict alongside.
Writes reports/heldout_reslice.{md,json}.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from m1_stats_reanalysis import bootstrap_clustered, mcnemar_midp, paired_table, score

from classifier import detect_construction

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT_MD = REPO / "reports" / "heldout_reslice.md"
OUT_JSON = REPO / "reports" / "heldout_reslice.json"


def strict_hit(text: str) -> bool:
    hits = detect_construction(text, strict=False)
    return any(h.variant.value in ("C1", "C2", "C3") for h in hits)


def analyse(rows: list[dict], key: str, label: str) -> dict:
    scored = score(rows, key)
    n = len(scored)
    nb = sum(1 for r in scored if r["baseline_hit"])
    na = sum(1 for r in scored if r["ablated_hit"])
    mc = paired_table(scored)
    boot = bootstrap_clustered(scored, n_boot=10000, seed=42)
    sb = sum(1 for r in rows if strict_hit(r["baseline"]))
    sa = sum(1 for r in rows if strict_hit(r["ablated"]))
    res = {
        "label": label, "n": n,
        "union": {"baseline": nb, "ablated": na,
                   "rel_drop": (nb - na) / nb if nb else None,
                   "mcnemar_midp": mc["mcnemar_midp_p_value_two_sided"],
                   "kills": mc["baseline_only"], "leaks": mc["ablated_only"],
                   "rel_drop_ci95": boot["relative_drop_ci95"]},
        "strict": {"baseline": sb, "ablated": sa,
                    "rel_drop": (sb - sa) / sb if sb else None},
    }
    print(f"{label}: union {nb}/{n} → {na}/{n} "
          f"(rel {res['union']['rel_drop']:.1%} CI [{boot['relative_drop_ci95'][0]:+.1%}, {boot['relative_drop_ci95'][1]:+.1%}], "
          f"mid-p {mc['mcnemar_midp_p_value_two_sided']:.4g}); strict {sb} → {sa}")
    return res


def prompt_level_sign_test(rows: list[dict], key: str) -> dict:
    """Collapse seeds within a prompt: prompt is a 'hit' if any seed hit."""
    scored = score(rows, key)
    by_prompt: dict = {}
    for r in scored:
        d = by_prompt.setdefault(r["prompt_id"], {"b": False, "a": False})
        d["b"] = d["b"] or r["baseline_hit"]
        d["a"] = d["a"] or r["ablated_hit"]
    b = sum(1 for d in by_prompt.values() if d["b"] and not d["a"])
    c = sum(1 for d in by_prompt.values() if not d["b"] and d["a"])
    both = sum(1 for d in by_prompt.values() if d["b"] and d["a"])
    p = mcnemar_midp(b, c)
    return {"n_prompts": len(by_prompt), "changed_to_clean": b,
            "changed_to_hit": c, "hit_in_both": both, "sign_test_midp": p}


def main():
    q5b = json.loads((REPO / "reports/q5b_d1_continuation.json").read_text())["rows"]
    q5c = json.loads((REPO / "reports/q5c_d2_high_power.json").read_text())["rows"]
    conf = set(json.loads((REPO / "data/splits/d2.json").read_text())["confirmation"])

    results = {}
    results["q5c_confirmation_split"] = analyse(
        [r for r in q5c if r["prompt_idx"] in conf], "prompt_idx",
        "Q5c neutral, 26 confirmation-split prompts (78 pairs)")
    results["q5b_nonselection_40plus"] = analyse(
        [r for r in q5b if r["prefix_idx"] >= 40], "prefix_idx",
        "Q5b primed, prefixes 40-99 — outside the n=40 selection set (180 pairs)")
    results["q5b_nonselection_80plus"] = analyse(
        [r for r in q5b if r["prefix_idx"] >= 80], "prefix_idx",
        "Q5b primed, prefixes 80-99 — outside every attribution scan (60 pairs)")
    results["q5c_full_prompt_level"] = prompt_level_sign_test(q5c, "prompt_idx")
    pl = results["q5c_full_prompt_level"]
    print(f"Q5c prompt-level (n={pl['n_prompts']} prompts): "
          f"{pl['changed_to_clean']} prompts → clean, {pl['changed_to_hit']} → hit, "
          f"sign-test mid-p {pl['sign_test_midp']:.4g}")

    OUT_JSON.write_text(json.dumps(results, indent=2))

    md = [
        "# Held-out re-slices (selection/evaluation independence)",
        "",
        "The deployed coalition was selected on the first 40 D1 prefixes",
        "(`pivot_attribution_n40.json`, promote-ranked top-25). These slices",
        "of the committed eval JSONs are out-of-sample w.r.t. that selection.",
        "Union = v2 detector post-2026-06-09 fix; strict = v1 non-strict.",
        "",
        "| slice | n | union baseline → ablated | rel drop (95% CI) | mid-p | strict |",
        "|---|---:|---|---|---:|---|",
    ]
    for k in ("q5c_confirmation_split", "q5b_nonselection_40plus", "q5b_nonselection_80plus"):
        r = results[k]
        u = r["union"]; s = r["strict"]
        ci = u["rel_drop_ci95"]
        md.append(
            f"| {r['label']} | {r['n']} | {u['baseline']} → {u['ablated']} | "
            f"{u['rel_drop']:.1%} [{ci[0]:+.1%}, {ci[1]:+.1%}] | "
            f"{u['mcnemar_midp']:.4g} | {s['baseline']} → {s['ablated']} |")
    md += [
        "",
        f"Prompt-level sign test, full Q5c (seeds collapsed within prompt): "
        f"{pl['changed_to_clean']} prompts went construction→clean, "
        f"{pl['changed_to_hit']} went clean→construction, "
        f"{pl['hit_in_both']} hit in both; mid-p = {pl['sign_test_midp']:.4g} "
        f"over {pl['n_prompts']} prompts.",
        "",
        "Q5d (the two-feature eval) has no held-out slice: all 40 of its",
        "prefixes are inside the selection set. It should be read as an",
        "in-sample demo number, not a confirmation.",
    ]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"→ {OUT_MD}")


if __name__ == "__main__":
    main()
