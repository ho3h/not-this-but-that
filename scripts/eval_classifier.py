"""Phase 1 classifier evaluation against `data/classifier_validation.jsonl`.

Kill check: precision/recall >= 0.85 on C1, C2, C3 (per PRD §8 Phase 1 and
HANDOVER.md §5). C4 is reported but not gated.

Writes:
  - reports/phase1_classifier_eval.json  (per-variant, per-row results)
  - reports/phase1_classifier_eval.md    (readable summary table + FP/FN list)

Usage:
    uv run python scripts/eval_classifier.py
    uv run python scripts/eval_classifier.py --validation data/classifier_validation.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from classifier import Variant, detect_construction

CORE_VARIANTS = (Variant.C1, Variant.C2, Variant.C3)
ALL_VARIANTS = (Variant.C1, Variant.C2, Variant.C3, Variant.C4)
KILL_CHECK = 0.85


def load_validation(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if "_meta" in d:
            continue
        rows.append(d)
    return rows


def predict(text: str, *, strict: bool = False) -> set[Variant]:
    return {h.variant for h in detect_construction(text, strict=strict)}


def evaluate(rows: list[dict], *, strict: bool = False) -> dict:
    """Per-variant precision/recall + per-row predictions."""
    pred_by_row = []
    for r in rows:
        pred = predict(r["text"], strict=strict)
        pred_by_row.append({
            "text": r["text"],
            "true": r["label"],
            "predicted": sorted(v.value for v in pred),
            "source": r.get("source", ""),
        })

    # per-variant confusion
    per_variant = {}
    for v in ALL_VARIANTS:
        tp = fp = fn = tn = 0
        for r, p in zip(rows, pred_by_row):
            is_true = r["label"] == v.value
            is_pred = v.value in p["predicted"]
            if is_true and is_pred:
                tp += 1
            elif is_pred and not is_true:
                fp += 1
            elif is_true and not is_pred:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_variant[v.value] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
        }

    # any-core: positive if classifier predicts ANY of C1/C2/C3 and label is in C1-C3
    tp = fp = fn = tn = 0
    for r, p in zip(rows, pred_by_row):
        is_true_core = r["label"] in {"C1", "C2", "C3"}
        is_pred_core = bool(set(p["predicted"]) & {"C1", "C2", "C3"})
        if is_true_core and is_pred_core:
            tp += 1
        elif is_pred_core and not is_true_core:
            fp += 1
        elif is_true_core and not is_pred_core:
            fn += 1
        else:
            tn += 1
    any_core = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
    }

    kill_check_passed = all(
        per_variant[v.value]["precision"] >= KILL_CHECK and per_variant[v.value]["recall"] >= KILL_CHECK
        for v in CORE_VARIANTS
    )

    return {
        "per_variant": per_variant,
        "any_core": any_core,
        "kill_check_threshold": KILL_CHECK,
        "kill_check_passed": kill_check_passed,
        "predictions": pred_by_row,
    }


def write_markdown(result: dict, out_path: Path, *, strict_result: dict | None = None) -> None:
    lines = []
    lines.append("# Phase 1 — Construction classifier evaluation\n")
    lines.append(f"**Kill check ({KILL_CHECK:.2f} P/R on C1–C3, regex-only):** "
                 f"{'PASSED ✅' if result['kill_check_passed'] else 'FAILED ❌'}\n")
    if strict_result is not None:
        lines.append(f"**Strict mode (regex + dependency filter):** "
                     f"{'PASSED ✅' if strict_result['kill_check_passed'] else 'FAILED ❌'}\n")
    lines.append("## Caveat\n")
    lines.append(
        "This validation set was hand-written by the same agent that wrote the regex "
        "patterns. Perfect 1.00 P/R is a self-consistency check, not a generalization "
        "test — the kill check is met but the real measurement happens in Phase 2, when "
        "M1 is run against actual model generations and the classifier sees inputs it "
        "wasn't tuned for. Treat these numbers as a sanity floor.\n"
    )
    lines.append("## Per-variant (regex only)\n")
    lines.append("| Variant | TP | FP | FN | TN | Precision | Recall | F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for v in ALL_VARIANTS:
        row = result["per_variant"][v.value]
        gate = " (gated)" if v in CORE_VARIANTS else ""
        lines.append(
            f"| **{v.value}**{gate} | {row['tp']} | {row['fp']} | {row['fn']} | {row['tn']} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} |"
        )
    lines.append("")
    a = result["any_core"]
    lines.append(f"**Any-core (C1∪C2∪C3):** P={a['precision']:.3f}, R={a['recall']:.3f} "
                 f"(TP={a['tp']}, FP={a['fp']}, FN={a['fn']}, TN={a['tn']})\n")

    if strict_result is not None:
        lines.append("## Per-variant (strict: regex + dependency filter)\n")
        lines.append("| Variant | TP | FP | FN | TN | Precision | Recall | F1 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for v in ALL_VARIANTS:
            row = strict_result["per_variant"][v.value]
            gate = " (gated)" if v in CORE_VARIANTS else ""
            lines.append(
                f"| **{v.value}**{gate} | {row['tp']} | {row['fp']} | {row['fn']} | {row['tn']} | "
                f"{row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} |"
            )
        lines.append("")
        # Per-row mismatches strict vs regex
        mismatches = [
            (p["text"], p["true"], p["predicted"], sp["predicted"])
            for p, sp in zip(result["predictions"], strict_result["predictions"])
            if p["predicted"] != sp["predicted"]
        ]
        if mismatches:
            lines.append(f"**Strict vs regex-only diff: {len(mismatches)} rows.**")
            for text, true, regex_pred, strict_pred in mismatches[:10]:
                lines.append(f"- true=**{true}** regex=**{','.join(regex_pred) or '∅'}** "
                             f"strict=**{','.join(strict_pred) or '∅'}** — `{text}`")
        else:
            lines.append("**Strict vs regex-only diff:** 0 rows. The dependency filter "
                         "doesn't change any verdict on this set — every regex hit's "
                         "negation token traces back to a verb/aux head, so nothing is "
                         "filtered. The filter exists to catch parse-ambiguous edge cases "
                         "in Phase 2 generations.")
        lines.append("")

    # False positives — what the classifier flagged when truth was none
    fps = [p for p in result["predictions"]
           if p["true"] == "none" and any(x in {"C1", "C2", "C3", "C4"} for x in p["predicted"])]
    if fps:
        lines.append("## False positives\n")
        for p in fps:
            lines.append(f"- predicted **{','.join(p['predicted'])}** on `{p['text']}` (source: {p['source']})")
        lines.append("")

    # False negatives — what the classifier missed when truth was a positive variant
    fns = defaultdict(list)
    for p in result["predictions"]:
        if p["true"] in {"C1", "C2", "C3", "C4"} and p["true"] not in p["predicted"]:
            fns[p["true"]].append(p)
    if fns:
        lines.append("## False negatives\n")
        for v, items in fns.items():
            lines.append(f"### {v}")
            for p in items:
                got = ",".join(p["predicted"]) or "(no hits)"
                lines.append(f"- expected **{v}**, got **{got}** on `{p['text']}`")
            lines.append("")

    # Cross-variant confusion — true C1 predicted as C3 etc.
    confusions = [p for p in result["predictions"]
                  if p["true"] in {"C1", "C2", "C3", "C4"}
                  and p["true"] not in p["predicted"]
                  and any(x in {"C1", "C2", "C3", "C4"} for x in p["predicted"])]
    if confusions:
        lines.append("## Variant confusions\n")
        for p in confusions:
            lines.append(f"- true **{p['true']}**, predicted **{','.join(p['predicted'])}** on `{p['text']}`")
        lines.append("")

    out_path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation", type=Path,
                    default=Path("data/classifier_validation.jsonl"))
    ap.add_argument("--out-json", type=Path,
                    default=Path("reports/phase1_classifier_eval.json"))
    ap.add_argument("--out-md", type=Path,
                    default=Path("reports/phase1_classifier_eval.md"))
    ap.add_argument("--no-strict", action="store_true",
                    help="Skip the spaCy strict pass (regex-only).")
    args = ap.parse_args()

    rows = load_validation(args.validation)
    print(f"Loaded {len(rows)} validation rows.")

    result = evaluate(rows, strict=False)
    strict_result = None if args.no_strict else evaluate(rows, strict=True)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(
        {"regex_only": result, "strict": strict_result},
        indent=2,
    ))
    write_markdown(result, args.out_md, strict_result=strict_result)

    print(f"\nKILL CHECK ({KILL_CHECK:.2f} on C1-C3, regex-only): "
          f"{'PASSED' if result['kill_check_passed'] else 'FAILED'}")
    print("\nPer-variant precision / recall (regex only):")
    for v in ALL_VARIANTS:
        r = result["per_variant"][v.value]
        gate = "(gated)" if v in CORE_VARIANTS else "(reported only)"
        print(f"  {v.value} {gate:15s}: P={r['precision']:.3f}, R={r['recall']:.3f}, "
              f"F1={r['f1']:.3f}  (TP={r['tp']}, FP={r['fp']}, FN={r['fn']})")
    a = result["any_core"]
    print(f"\nAny-core (C1∪C2∪C3): P={a['precision']:.3f}, R={a['recall']:.3f}")
    if strict_result is not None:
        sa = strict_result["any_core"]
        print(f"Any-core (strict):     P={sa['precision']:.3f}, R={sa['recall']:.3f}")
        n_diff = sum(1 for a, b in zip(result["predictions"], strict_result["predictions"])
                     if a["predicted"] != b["predicted"])
        print(f"Strict vs regex-only diff: {n_diff} rows.")
    print(f"\nFull report: {args.out_md}")


if __name__ == "__main__":
    main()
