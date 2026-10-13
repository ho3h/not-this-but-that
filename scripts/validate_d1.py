"""Validate D1 contrast pairs.

For every pair {with, without, variant}:
  - `detect_construction(with)` should include the labelled variant.
  - `detect_construction(without)` should be empty (no C1-C4 hit).

Any failure means the pair is either mislabelled or the "without" still has
the construction. Phase 3's differential-activation signal lives on these
pairs, so any failure poisons the signal and must be fixed by hand.

Writes:
  - reports/d1_validation.md (summary + any failures)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from classifier import detect_construction

REPO_ROOT = Path(__file__).resolve().parent.parent
D1_PATH = REPO_ROOT / "data" / "D1_contrast_pairs.jsonl"
OUT = REPO_ROOT / "reports" / "d1_validation.md"


def main() -> None:
    rows = []
    for line in D1_PATH.read_text().splitlines():
        if not line.strip() or line.startswith('{"_meta'):
            continue
        rows.append(json.loads(line))

    variant_counts = Counter(r["variant"] for r in rows)
    print(f"total pairs: {len(rows)}")
    print(f"per variant: {dict(variant_counts)}")

    with_failures = []  # WITH side doesn't include labelled variant
    without_failures = []  # WITHOUT side has any construction hit
    for r in rows:
        with_hits = {h.variant.value for h in detect_construction(r["with"])}
        without_hits = {h.variant.value for h in detect_construction(r["without"])}
        if r["variant"] not in with_hits:
            with_failures.append({"pair": r, "got_with": sorted(with_hits)})
        if without_hits:
            without_failures.append({"pair": r, "got_without": sorted(without_hits)})

    lines = [
        "# D1 contrast pairs — validation",
        "",
        f"**Total pairs:** {len(rows)}  ",
        "**Per variant:** " + ", ".join(f"{v}={c}" for v, c in variant_counts.items()),
        "",
        f"**WITH-side failures** (regex/strict missed the labelled variant): {len(with_failures)}",
        f"**WITHOUT-side failures** (paraphrase still triggers a construction): {len(without_failures)}",
        "",
    ]

    if with_failures:
        lines.append("## WITH-side failures\n")
        for f in with_failures:
            r = f["pair"]
            lines.append(f"- expected **{r['variant']}**, got **{','.join(f['got_with']) or '∅'}** "
                         f"on `{r['with']}`")
        lines.append("")
    if without_failures:
        lines.append("## WITHOUT-side failures\n")
        for f in without_failures:
            r = f["pair"]
            lines.append(f"- got **{','.join(f['got_without'])}** on `{r['without']}` "
                         f"(pair labelled {r['variant']})")
        lines.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"WITH-side failures: {len(with_failures)}")
    print(f"WITHOUT-side failures: {len(without_failures)}")
    print(f"→ {OUT}")
    return 0 if not (with_failures or without_failures) else 1


if __name__ == "__main__":
    raise SystemExit(main())
