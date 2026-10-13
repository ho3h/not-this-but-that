"""Re-score Q5b / Q5c / Q5d across every detector tier. The post's receipt.

Tiers:
  strict        v1 (regex hinges + spaCy), blind-validated P=0.80/R=1.00 —
                single-sentence negated forms.
  permissive    detect_v2 permissive layer (2026-06-09 negation-mandatory
                revision; FP audit at reports/permissive_fix_audit.md) —
                looser syntax + cross-sentence, still negation-anchored.
  union         strict ∪ permissive = the CONSTRUCTION FAMILY as defined by
                the pre-registration (negation-anchored C1-C4). Headline tier.
  family+cousin union ∪ the affirmative "more than just X(.;,) it's/they're Y"
                minimizer — a rhetorical cousin OUTSIDE the registered family
                (it denies nothing), detected symmetrically in comma and
                period forms. Reported because the ablated model reroutes
                into it heavily; hiding it would overstate the kill.

Primed evals (Q5b/Q5d) additionally get PREFIX-INCLUSIVE rows: the prefix
("It's not a tool") prepended to the continuation before scoring, so a
continuation that completes the construction (", but a partner") counts.
Completion-only scoring (the default rows) measures re-occurrence inside the
continuation instead; both are reported, they answer different questions.

Writes reports/m1_rescore_union.json.
"""
from __future__ import annotations
import json, pathlib

from classifier import detect_construction
from classifier.detect_v2 import detect_more_than_just, detect_permissive


def perm(text: str) -> bool:
    return bool(detect_permissive(text))


def strict(text: str) -> bool:
    hits = detect_construction(text, strict=False)
    return any(h.variant.value in ("C1", "C2", "C3") for h in hits)


def union(text: str) -> bool:
    return strict(text) or perm(text)


def cousin(text: str) -> bool:
    return bool(detect_more_than_just(text))


def family_plus_cousin(text: str) -> bool:
    return union(text) or cousin(text)


TIERS = [("strict", strict), ("permissive", perm), ("union", union),
         ("mtj_cousin_only", cousin), ("family_plus_cousin", family_plus_cousin)]


def score_run(rows: list, prefix_key: str | None = None) -> dict:
    n = len(rows)
    def text(r, field):
        t = r[field]
        if prefix_key:
            t = (r.get(prefix_key) or "") + t
        return t
    out = {"n": n, "prefix_inclusive": bool(prefix_key)}
    for name, fn in TIERS:
        b = sum(1 for r in rows if fn(text(r, "baseline")))
        a = sum(1 for r in rows if fn(text(r, "ablated")))
        out[name] = {"baseline_hits": b, "ablated_hits": a,
                     "baseline_rate": b / n, "ablated_rate": a / n,
                     "rel_drop": (b - a) / b if b else None}
    return out


def main():
    repo = pathlib.Path(__file__).resolve().parent.parent
    out_path = repo / "reports" / "m1_rescore_union.json"

    targets = [
        ("Q5b primed (top-25, 100 prefixes × 3 seeds)",
         "reports/q5b_d1_continuation.json", "prefix"),
        ("Q5c neutral (top-25, 102 prompts × 3 seeds)",
         "reports/q5c_d2_high_power.json", None),
        ("Q5d minimal-core (3223+9909, 40 prefixes × 3 seeds)",
         "reports/q5d_minimal_set_d1_n120.json", "prefix"),
    ]

    results = {}
    for label, path, prefix_key in targets:
        rows = json.loads((repo / path).read_text())["rows"]
        if prefix_key and prefix_key not in rows[0]:
            prefix_key = "prompt" if "prompt" in rows[0] else None
        results[label] = {"completion_only": score_run(rows)}
        if prefix_key:
            results[label]["prefix_inclusive"] = score_run(rows, prefix_key)
        for mode, s in results[label].items():
            print(f"=== {label} [{mode}] (n={s['n']}) ===")
            for name, _ in TIERS:
                v = s[name]
                rd = f"{v['rel_drop']:6.1%}" if v["rel_drop"] is not None else "    — "
                print(f"  {name:>18}: base {v['baseline_hits']:>3} = {v['baseline_rate']:6.2%}, "
                      f"abl {v['ablated_hits']:>3} = {v['ablated_rate']:6.2%}, rel drop {rd}")
            print()

    out = {
        "method_note": (
            "Tiers: strict = v1 blind-validated; union = strict ∪ permissive "
            "= the negation-anchored construction FAMILY (the registered "
            "definition; headline tier); family_plus_cousin adds the "
            "affirmative 'more than just' minimizer (outside the family — "
            "it denies nothing) which the ablated model reroutes into. "
            "Primed evals also scored prefix-inclusive (prefix prepended), "
            "where a continuation completing the construction counts."
        ),
        "results": results,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
