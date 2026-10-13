"""Tier 0a — run classifier on the blind set and compare to hand labels.

Reads data/classifier_blind_set.jsonl and data/classifier_blind_labels.jsonl
(both committed BEFORE this script ran — see git history). Runs the strict
classifier, computes confusion against hand labels, reports per-variant
P/R and the Tier 0a kill verdict.

Kill threshold (from pre_registration.yaml):
  precision_on_c1_c2_c3_min: 0.70
  recall_on_c1_c2_c3_min:    0.70
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from classifier import detect_construction

REPO = Path(__file__).resolve().parent.parent
SET = REPO / "data" / "classifier_blind_set.jsonl"
LABELS = REPO / "data" / "classifier_blind_labels.jsonl"
OUT = REPO / "reports" / "tier_0a_classifier_blind_eval.md"
KILL_P = 0.70
KILL_R = 0.70


def load_jsonl_skip_meta(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith('{"_meta'):
            continue
        out.append(json.loads(line))
    return out


def main() -> None:
    sentences = {r["id"]: r for r in load_jsonl_skip_meta(SET)}
    labels = {r["id"]: r for r in load_jsonl_skip_meta(LABELS)}
    assert set(sentences) == set(labels), "id mismatch between set and labels"

    rows = []
    for sid in sorted(sentences):
        text = sentences[sid]["text"]
        true = labels[sid]["label"]
        hits = detect_construction(text, strict=True)
        predicted = sorted({h.variant.value for h in hits})
        rows.append({
            "id": sid,
            "source": sentences[sid]["source"],
            "text": text,
            "true": true,
            "predicted": predicted,
            "note": labels[sid].get("note", ""),
        })

    # per-variant P/R
    per_variant = {}
    for v in ("C1", "C2", "C3", "C4"):
        tp = sum(1 for r in rows if r["true"] == v and v in r["predicted"])
        fp = sum(1 for r in rows if r["true"] != v and v in r["predicted"])
        fn = sum(1 for r in rows if r["true"] == v and v not in r["predicted"])
        tn = sum(1 for r in rows if r["true"] != v and v not in r["predicted"])
        precision = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
        recall = tp / (tp + fn) if (tp + fn) else 1.0  # no positives → vacuously full recall
        per_variant[v] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                          "precision": precision, "recall": recall,
                          "n_true_positives": tp + fn}

    # any_core (C1 ∪ C2 ∪ C3 — what the kill check evaluates)
    core = {"C1", "C2", "C3"}
    tp = sum(1 for r in rows if r["true"] in core and (set(r["predicted"]) & core))
    fp = sum(1 for r in rows if r["true"] not in core and (set(r["predicted"]) & core))
    fn = sum(1 for r in rows if r["true"] in core and not (set(r["predicted"]) & core))
    tn = sum(1 for r in rows if r["true"] not in core and not (set(r["predicted"]) & core))
    any_core_p = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
    any_core_r = tp / (tp + fn) if (tp + fn) else 1.0

    # Per-variant kill check: only evaluate variants that have ≥1 true positive
    # in this set. For variants with 0 positives, P/R isn't meaningful and we
    # report it as "no positives in set" rather than gaming via vacuous recall.
    kill_per_variant = {}
    for v in ("C1", "C2", "C3"):
        s = per_variant[v]
        if s["n_true_positives"] == 0:
            kill_per_variant[v] = {"status": "no positives in set",
                                    "p_pass": s["precision"] >= KILL_P,
                                    "r_pass": None}
        else:
            kill_per_variant[v] = {"status": "tested",
                                    "p_pass": s["precision"] >= KILL_P,
                                    "r_pass": s["recall"] >= KILL_R}

    # Aggregate kill: any_core P AND R must clear the threshold AND every
    # tested variant must clear it individually.
    any_core_pass = (any_core_p >= KILL_P) and (any_core_r >= KILL_R)
    per_variant_pass = all(
        (v["p_pass"] is True or v["p_pass"] is None) and
        (v["r_pass"] is True or v["r_pass"] is None)
        for v in kill_per_variant.values()
    )
    tier_passes = any_core_pass and per_variant_pass

    # Mismatch dump
    mismatches = [r for r in rows
                  if (r["true"] == "none" and r["predicted"]) or
                     (r["true"] != "none" and r["true"] not in r["predicted"])]

    md = [
        "# Tier 0a — Classifier blind validation",
        "",
        f"**Set:** {len(rows)} sentences (30 human D3 + 30 AI Gemma-2-2b-it).  ",
        f"**Pre-registered kill threshold:** P/R ≥ {KILL_P:.2f} on C1-C3 (any_core).",
        "",
        "## Per-variant",
        "",
        "| Variant | n true positives | TP | FP | FN | TN | Precision | Recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for v in ("C1", "C2", "C3", "C4"):
        s = per_variant[v]
        md.append(f"| **{v}** | {s['n_true_positives']} | {s['tp']} | {s['fp']} | "
                  f"{s['fn']} | {s['tn']} | {s['precision']:.3f} | {s['recall']:.3f} |")
    md.append(f"| **any_core (C1∪C2∪C3)** | {tp + fn} | {tp} | {fp} | {fn} | {tn} | "
              f"{any_core_p:.3f} | {any_core_r:.3f} |")
    md.append("")
    md.append("## Kill check\n")
    md.append(f"- any_core precision {any_core_p:.3f} ≥ {KILL_P:.2f}: "
              f"{'PASS' if any_core_p >= KILL_P else 'FAIL'}")
    md.append(f"- any_core recall    {any_core_r:.3f} ≥ {KILL_R:.2f}: "
              f"{'PASS' if any_core_r >= KILL_R else 'FAIL'}")
    md.append("")
    md.append("**Per-variant kill verdicts (only variants with ≥1 true positive are gated):**")
    for v, k in kill_per_variant.items():
        if k["status"] == "no positives in set":
            md.append(f"- {v}: no positives in this set (not gated). "
                      f"Precision {per_variant[v]['precision']:.3f}.")
        else:
            md.append(f"- {v}: precision = {per_variant[v]['precision']:.3f}, "
                      f"recall = {per_variant[v]['recall']:.3f} → "
                      f"{'PASS' if k['p_pass'] and k['r_pass'] else 'FAIL'}")
    md.append("")
    md.append(f"### Verdict: {'PASS — proceed to Tier 0b' if tier_passes else 'KILL — STOP'}")
    md.append("")

    if mismatches:
        md.append("## Mismatches (classifier disagrees with hand label)")
        md.append("")
        for r in mismatches:
            md.append(f"- `{r['id']}` ({r['source']})  true=**{r['true']}**  "
                      f"pred=**{','.join(r['predicted']) or '∅'}**  "
                      f"text: `{r['text'][:120].strip()}`")
        md.append("")
    else:
        md.append("## Mismatches\n\nNone.")
        md.append("")

    md.append("## Caveat — small positive set\n")
    md.append("This blind set contains only 1 true positive across 60 sentences "
              "(consistent with the natural ~1.8% any_core rate in Phase 2). With "
              "n=1 positive, recall is binary (the classifier finds it or doesn't) "
              "and precision depends entirely on the false-positive count. The "
              "kill threshold here is informative but not statistically tight; if "
              "this passes, Tier 0a is followed by a positive-enriched test "
              "drawn from a wider Phase 2 pool to thicken the recall measurement.")

    OUT.write_text("\n".join(md))

    print(f"any_core P = {any_core_p:.3f}, R = {any_core_r:.3f}")
    print(f"VERDICT: {'PASS' if tier_passes else 'KILL'}")
    print(f"\n→ {OUT}")
    return 0 if tier_passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
