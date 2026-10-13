"""Re-score every committed generation set with detector v3 (sensitivity analysis).

The pre-registered verdicts stay as registered (scored with the frozen v2
detector). This script reports how the main numbers move under v3, which fixes
the edge cases found in the October 2026 audit (see src/classifier/detect_v3.py),
and adds the `reversed` and `beyond` tiers.

Writes reports/detector_v3.{json,md}. No GPU needed.
"""

from __future__ import annotations

import json
import pathlib
import sys
from math import comb

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from classifier.detect_v2 import detect_construction_v2, detect_more_than_just  # noqa: E402
from classifier.detect_v3 import detect_v3, normalise  # noqa: E402


def midp(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    pmf = lambda i: comb(n, i) / 2 ** n  # noqa: E731
    return float(min(1.0, 2.0 * (sum(pmf(i) for i in range(k)) + 0.5 * pmf(k))))


def tiers_v3(text: str) -> dict:
    r = detect_v3(text)
    fam, cou = bool(r["family"]), bool(r["cousin"])
    return {"family": fam, "cousin": cou, "family_or_cousin": fam or cou,
            "reversed": bool(r["reversed"]), "beyond": bool(r["beyond"])}


def tiers_v2(text: str) -> dict:
    t = normalise(text)
    fam, cou = bool(detect_construction_v2(t)), bool(detect_more_than_just(t))
    return {"family": fam, "cousin": cou, "family_or_cousin": fam or cou}


def paired(pairs, fn, key):
    b = sum(1 for x, y in pairs if fn(x)[key] and not fn(y)[key])
    c = sum(1 for x, y in pairs if not fn(x)[key] and fn(y)[key])
    base = sum(fn(x)[key] for x, _ in pairs)
    abl = sum(fn(y)[key] for _, y in pairs)
    return {"n": len(pairs), "baseline": base, "ablated": abl,
            "rel_change": (abl - base) / base if base else None, "mid_p": midp(b, c)}


def load_sets():
    sets = {}
    rows = [json.loads(line) for line in (REPO / "reports/confirmation_v2_generations.jsonl").read_text().splitlines()]
    sets["Confirmation v2, meaning prompts (pre-registered, chat)"] = [(r["baseline"], r["coalition"]) for r in rows if r["type"] == "M"]
    sets["Confirmation v2, process/scene prompts"] = [(r["baseline"], r["coalition"]) for r in rows if r["type"] == "P"]
    for name, f in (("First round, everyday prompts (Q5c)", "q5c_d2_high_power.json"),
                    ("First round, primed, completion only (Q5b)", "q5b_d1_continuation.json"),
                    ("Confirmation 1, concrete prompts (D2b)", "confirmation_run.json")):
        sets[name] = [(r["baseline"], r["ablated"]) for r in json.loads((REPO / "reports" / f).read_text())["rows"]]
    return sets


def main() -> None:
    sets = load_sets()
    out = {}
    for name, pairs in sets.items():
        out[name] = {
            "v2": {k: paired(pairs, tiers_v2, k) for k in ("family", "cousin", "family_or_cousin")},
            "v3": {k: paired(pairs, tiers_v3, k) for k in ("family", "cousin", "family_or_cousin", "reversed", "beyond")},
        }
    rows = [json.loads(line) for line in (REPO / "reports/confirmation_v2_generations.jsonl").read_text().splitlines()]
    h2 = {}
    for det, fn in (("v2", tiers_v2), ("v3", tiers_v3)):
        m = [fn(r["baseline"])["family_or_cousin"] for r in rows if r["type"] == "M"]
        p = [fn(r["baseline"])["family_or_cousin"] for r in rows if r["type"] == "P"]
        h2[det] = {"M": f"{sum(m)}/{len(m)}", "P": f"{sum(p)}/{len(p)}"}
    out["H2_prompt_type_normal_answers"] = h2
    (REPO / "reports/detector_v3.json").write_text(json.dumps(out, indent=2))

    fmt = lambda t: (f"{t['baseline']} → {t['ablated']}"  # noqa: E731
                     + (f" ({100 * t['rel_change']:+.0f}%)" if t["rel_change"] is not None else "")
                     + f", p {t['mid_p']:.2g}")
    md = ["# Detector v3: sensitivity re-score", "",
          "Pre-registered verdicts stand as registered (frozen v2 detector). This re-scores every set with v3,",
          "which fixes the edge cases in `src/classifier/detect_v3.py`. Counts are answers containing the form;",
          "\"normal → switched off\", McNemar mid-p on the pairs.", ""]
    for name, d in out.items():
        if name.startswith("H2"):
            continue
        md += [f"## {name} ({d['v3']['family']['n']} pairs)", "", "| tier | v2 (registered) | v3 |", "|---|---|---|"]
        for k in ("family", "cousin", "family_or_cousin"):
            md.append(f"| {k.replace('_', ' ')} | {fmt(d['v2'][k])} | {fmt(d['v3'][k])} |")
        for k in ("reversed", "beyond"):
            md.append(f"| {k} (separate tier) | n/a | {fmt(d['v3'][k])} |")
        md.append("")
    md += ["## Prompt type (normal answers, family or cousin)", "",
           f"- v2: meaning {h2['v2']['M']}, process/scene {h2['v2']['P']}",
           f"- v3: meaning {h2['v3']['M']}, process/scene {h2['v3']['P']}", ""]
    md += ["## Precision audit", "",
           "A sample of the hits v3 adds over v2 (and of the v1 hits it drops) was labelled by an annotator",
           "who saw only the text excerpts, never the condition (`data/detector_v3_audit_items.json`,",
           "`data/detector_v3_audit_labels.tsv`, summary in `detector_v3_audit.json`):", "",
           "- added cousin hits: 40 of 40 confirmed;",
           "- added family hits: 10 of 17 confirmed in the first draft. All 7 misses came from two patterns",
           "  (a staccato second sentence opening with This/These, and a loose \"no longer\" form), which were",
           "  removed; the remaining additions in the sample are 4 of 4 confirmed. A narrow replacement",
           "  (\"This wasn't just X. This was Y.\", same demonstrative in both sentences) was added afterwards",
           "  and is not separately audited;",
           "- dropped v1 hits (epistemic hedges such as \"I'm not sure what you mean, but\"): 4 of 4 were false positives.", ""]
    (REPO / "reports/detector_v3.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
