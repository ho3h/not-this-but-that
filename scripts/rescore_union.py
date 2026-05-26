"""Re-score Q5b / Q5c / Q5d with both detectors and their UNION.

The strict Python classifier (src/classifier/detect.py) misses F2 staccato
("isn't just X. It's Y") because it requires the hinge and pivot in the same
sentence. The JS-frontend permissive regex catches F2 but misses some forms
the dependency-checked strict classifier finds. Honest reporting needs both,
or the union.

Writes reports/m1_rescore_union.json so the Medium post can cite a single
reproducible artifact.
"""
from __future__ import annotations
import json, pathlib, re
from classifier import detect_construction

# Identical to web/demo/playground.js — the F2 staccato is the important one.
PATTERNS_PERMISSIVE = [
    re.compile(r"\b(is|are|isn'?t|aren'?t|was|were|wasn'?t|weren'?t|don'?t|doesn'?t|don)\s+(?:not\s+)?(?:just\s+)?[^.,;:!?\n]{1,80}[,;—–\-]\s*(?:it'?s?|they'?re?|they|he'?s?|she'?s?|we'?re?|but\s+|but\b)", re.I),
    re.compile(r"\b(?:is|are|isn'?t|aren'?t|was|were|wasn'?t|weren'?t|don'?t|doesn'?t)\s+(?:not|just)\s+(?:just\s+)?[^.!?\n]{1,80}[.!?]\s*(?:It'?s?|They'?re?|He'?s?|She'?s?|We'?re?|But\s+|Rather|Instead)"),
    re.compile(r"(?:\bless\b\s+[^.,;:!?\n]{1,40}\s*[,;—–\-]\s*more\b|\bnot\s+about\b\s+[^.,;:!?\n]{1,40}\s*[,;—–\-]\s*(?:it'?s?\s+about|about))", re.I),
]


def perm(text: str) -> bool:
    return any(p.search(text) for p in PATTERNS_PERMISSIVE)


def strict(text: str) -> bool:
    hits = detect_construction(text, strict=False)
    return any(h.variant.value in ("C1", "C2", "C3") for h in hits)


def union(text: str) -> bool:
    return strict(text) or perm(text)


def score_run(rows: list) -> dict:
    n = len(rows)
    def cnt(field, fn):
        return sum(1 for r in rows if fn(r[field]))
    sb, sa = cnt("baseline", strict), cnt("ablated", strict)
    pb, pa = cnt("baseline", perm),   cnt("ablated", perm)
    ub, ua = cnt("baseline", union),  cnt("ablated", union)
    def rel(b, a):
        return (b - a) / b if b else 0.0
    return {
        "n": n,
        "strict": {"baseline_rate": sb/n, "ablated_rate": sa/n,
                    "baseline_hits": sb, "ablated_hits": sa, "rel_drop": rel(sb, sa)},
        "permissive": {"baseline_rate": pb/n, "ablated_rate": pa/n,
                        "baseline_hits": pb, "ablated_hits": pa, "rel_drop": rel(pb, pa)},
        "union": {"baseline_rate": ub/n, "ablated_rate": ua/n,
                   "baseline_hits": ub, "ablated_hits": ua, "rel_drop": rel(ub, ua)},
    }


def main():
    repo = pathlib.Path(__file__).resolve().parent.parent
    out_path = repo / "reports" / "m1_rescore_union.json"

    targets = [
        ("Q5b primed (D1 continuation, top-25, IT model, 40×2)",
         "reports/q5b_d1_continuation.json"),
        ("Q5c neutral (D2, top-25, IT model, 40×3)",
         "reports/q5c_d2_high_power.json"),
        ("Q5d minimal-core (3223+9909, D1, IT model, 8×2)",
         "reports/q5d_minimal_set_d1.json"),
    ]

    results = {}
    for label, path in targets:
        d = json.loads((repo / path).read_text())
        rows = d["rows"]
        results[label] = score_run(rows)
        s = results[label]
        print(f"=== {label} ===")
        for k in ("strict", "permissive", "union"):
            v = s[k]
            print(f"  {k:>10}: base {v['baseline_hits']:>3}/{s['n']:>3} = {v['baseline_rate']:6.2%}, "
                  f"abl {v['ablated_hits']:>3}/{s['n']:>3} = {v['ablated_rate']:6.2%}, "
                  f"rel drop {v['rel_drop']:6.2%}")
        print()

    out = {
        "method_note": (
            "The strict classifier (src/classifier/detect.py) uses regex hinges + "
            "spaCy dependency parsing; it catches single-sentence constructions "
            "(F1/F3 etc) but misses F2 staccato (\"isn't just X. It's Y\" across "
            "two sentences). The permissive regex catches F2 staccato but misses "
            "some dependency-parsed forms. The UNION reports a hit if either "
            "detector fires. The honest 'we caught everything we can see' number "
            "is the union; the strict-only number overstates the drop by counting "
            "the model's F2-staccato rerouting as a clean kill."
        ),
        "results": results,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
