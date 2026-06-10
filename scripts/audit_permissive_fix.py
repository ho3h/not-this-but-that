"""Before/after audit of the 2026-06-09 permissive-regex fix.

The original permissive layer (see git history of src/classifier/detect_v2.py)
made negation optional in the same-sentence pattern and had unboundaried
pronoun tails in the staccato pattern. This script re-scores the committed
Q5b/Q5c generations under the OLD and NEW permissive patterns and writes a
side-by-side audit so the change is inspectable: every hit the fix dropped
(expected: concessive "X, but Y" false positives) and every hit it added
(expected: first-person "I'm not X, I'm Y" forms) is listed with context.

Writes reports/permissive_fix_audit.{md,json}.
"""
from __future__ import annotations

import json
import pathlib
import re

from classifier import has_construction  # the deployed union (v1 ∪ NEW permissive)
from classifier.detect import detect_construction
from classifier.detect_v2 import detect_permissive as detect_permissive_new

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT_MD = REPO / "reports" / "permissive_fix_audit.md"
OUT_JSON = REPO / "reports" / "permissive_fix_audit.json"

# --- the OLD permissive patterns, verbatim from git (pre-fix) -------------------
_OLD = [
    re.compile(
        r"\b(is|are|isn'?t|aren'?t|was|were|wasn'?t|weren'?t|don'?t|doesn'?t|don)"
        r"\s+(?:not\s+)?(?:just\s+)?[^.,;:!?\n]{1,80}"
        r"[,;—–\-]\s*(?:it'?s?|they'?re?|they|he'?s?|she'?s?|we'?re?|but\s+|but\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:is|are|isn'?t|aren'?t|was|were|wasn'?t|weren'?t|don'?t|doesn'?t)"
        r"\s+(?:not|just)\s+(?:just\s+)?[^.!?\n]{1,80}"
        r"[.!?]\s*(?:It'?s?|They'?re?|He'?s?|She'?s?|We'?re?|But\s+|Rather|Instead)",
    ),
    re.compile(
        r"(?:\bless\b\s+[^.,;:!?\n]{1,40}\s*[,;—–\-]\s*more\b"
        r"|\bnot\s+about\b\s+[^.,;:!?\n]{1,40}\s*[,;—–\-]\s*"
        r"(?:it'?s?\s+about|about))",
        re.IGNORECASE,
    ),
]


def old_permissive_hit(text: str):
    for pat in _OLD:
        m = pat.search(text)
        if m:
            return m
    return None


def new_permissive_hit(text: str):
    hits = detect_permissive_new(text)
    return hits[0] if hits else None


def v1_hit(text: str) -> bool:
    return bool(detect_construction(text, strict=False))


def ctx(text: str, span: tuple[int, int], pad: int = 60) -> str:
    s, e = span
    lo, hi = max(0, s - pad), min(len(text), e + pad)
    pre = text[lo:s].replace("\n", " ")
    mark = text[s:e].replace("\n", " ")
    post = text[e:hi].replace("\n", " ")
    return f"…{pre}**{mark}**{post}…"


def audit_file(path: str, name: str) -> dict:
    rows = json.loads((REPO / path).read_text())["rows"]
    res = {"name": name, "n": len(rows), "conditions": {}, "dropped": [], "added": []}
    for cond in ("baseline", "ablated"):
        v1 = old_only = new_only = union_old = union_new = 0
        for r in rows:
            t = r[cond]
            h_v1 = v1_hit(t)
            m_old = old_permissive_hit(t)
            m_new = new_permissive_hit(t)
            v1 += h_v1
            old_only += (m_old is not None) and not h_v1
            new_only += (m_new is not None) and not h_v1
            union_old += h_v1 or (m_old is not None)
            union_new += h_v1 or (m_new is not None)
            # status changes among permissive-only rows
            if not h_v1 and (m_old is not None) and m_new is None:
                res["dropped"].append({
                    "file": name, "condition": cond,
                    "prompt": (r.get("prompt") or r.get("prefix") or "")[:80],
                    "match": ctx(t, m_old.span()),
                })
            if not h_v1 and m_old is None and (m_new is not None):
                res["added"].append({
                    "file": name, "condition": cond,
                    "prompt": (r.get("prompt") or r.get("prefix") or "")[:80],
                    "match": ctx(t, m_new.span),
                })
        res["conditions"][cond] = {
            "v1_nonstrict": v1,
            "permissive_only_old": old_only,
            "permissive_only_new": new_only,
            "union_old": union_old,
            "union_new": union_new,
        }
    return res


def main():
    files = [
        ("reports/q5b_d1_continuation.json", "Q5b primed (n=300)"),
        ("reports/q5c_d2_high_power.json", "Q5c neutral (n=306)"),
        ("reports/q5d_minimal_set_d1_n120.json", "Q5d two-feature (n=120)"),
    ]
    out = []
    for path, name in files:
        out.append(audit_file(path, name))

    OUT_JSON.write_text(json.dumps(out, indent=2))

    md = [
        "# Permissive-regex fix — before/after audit (2026-06-09)",
        "",
        "The pre-fix permissive layer made negation optional in its same-sentence",
        "pattern, so ordinary concessives (\"X is great, but…\") counted as the",
        "construction; its pronoun tails also lacked `\\b` (\"He\" prefix-matched",
        "\"Here's\"). The fix makes negation mandatory and boundaries the tails,",
        "and adds first-person forms (\"I'm not X, I'm Y\") both layers missed.",
        "Counts below are per-generation hits on the committed eval JSONs;",
        "`union` = v1 (non-strict) ∪ permissive — the post's headline detector.",
        "",
        "| eval | condition | v1 | perm-only OLD | perm-only NEW | union OLD | union NEW |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in out:
        for cond, c in r["conditions"].items():
            md.append(
                f"| {r['name']} | {cond} | {c['v1_nonstrict']} | "
                f"{c['permissive_only_old']} | {c['permissive_only_new']} | "
                f"{c['union_old']} | {c['union_new']} |"
            )
    md += ["", "## Hits dropped by the fix (expected: concessive false positives)", ""]
    for d in [d for r in out for d in r["dropped"]]:
        md.append(f"- **{d['file']} / {d['condition']}** ({d['prompt']}…): {d['match']}")
    md += ["", "## Hits added by the fix (expected: first-person forms)", ""]
    added = [a for r in out for a in r["added"]]
    if not added:
        md.append("(none)")
    for a in added:
        md.append(f"- **{a['file']} / {a['condition']}** ({a['prompt']}…): {a['match']}")
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"→ {OUT_MD}")
    print(f"→ {OUT_JSON}")

    for r in out:
        print(f"\n{r['name']}")
        for cond, c in r["conditions"].items():
            print(f"  {cond:9s} union {c['union_old']:>3} → {c['union_new']:>3}  "
                  f"(perm-only {c['permissive_only_old']} → {c['permissive_only_new']}, v1 {c['v1_nonstrict']})")


if __name__ == "__main__":
    main()
