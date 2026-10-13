"""Which forms absorbed the reroute in confirmation v2 (exploratory, post hoc).

Run after the one-shot pre-registered analysis (`confirmation_v2_run.py
--analyze`). Breaks the family and cousin counts down by surface form for
normal vs coalition-off answers, and lists example pairs where the
same-sentence textbook form became a two-sentence form or the cousin. None of
this was registered in advance.

Writes reports/confirmation_v2_forms.{json,md}.
"""

from __future__ import annotations

import collections
import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import v2_engine as E  # noqa: E402

sys.path.insert(0, str(E.REPO / "src"))
from classifier.detect import detect_construction  # noqa: E402
from classifier.detect_v2 import detect_more_than_just, detect_permissive  # noqa: E402

GENS = E.REPO / "reports" / "confirmation_v2_generations.jsonl"
LABELS = {
    "v1": "textbook form (v1 regex: isn't X, it's Y / not just X, it's Y / not X, but Y / not only X, but Y)",
    "same": "permissive same-sentence (isn't X, it's Y with comma, semicolon or dash)",
    "cross": "permissive cross-sentence (isn't X. It's Y.)",
    "lessmore": "permissive less X, more Y / not about X, about Y",
    "cousin": "cousin (more than just X. It's Y)",
}


def forms(text: str) -> set[str]:
    t = E.normalise(text)
    out = set()
    if detect_construction(t, strict=False):
        out.add("v1")
    for h in detect_permissive(t):
        out.add({"same-sentence comma/dash": "same", "cross-sentence staccato": "cross"}.get(h.hinge, "lessmore"))
    if detect_more_than_just(t):
        out.add("cousin")
    return out


def main() -> None:
    rows = [json.loads(line) for line in GENS.read_text().splitlines() if line.strip()]
    result, examples = {}, []
    for typ in ("M", "P"):
        sub = [r for r in rows if r["type"] == typ]
        b, a = collections.Counter(), collections.Counter()
        for r in sub:
            fb, fa = forms(r["baseline"]), forms(r["coalition"])
            b.update(fb)
            a.update(fa)
            if typ == "M" and "v1" in fb and "v1" not in fa and ({"cross", "cousin"} & fa):
                examples.append({"prompt": r["prompt"], "normal": E.normalise(r["baseline"])[:240],
                                 "coalition_off": E.normalise(r["coalition"])[:240]})
        result[typ] = {"pairs": len(sub), "normal": dict(b), "coalition_off": dict(a)}
    result["textbook_to_cross_or_cousin_pairs"] = len(examples)
    result["examples"] = examples[:8]
    (E.REPO / "reports" / "confirmation_v2_forms.json").write_text(json.dumps(result, indent=2))

    md = ["# Confirmation v2: which forms absorbed the reroute (exploratory)", "",
          "Post hoc breakdown of `confirmation_v2_generations.jsonl`, run after the one-shot",
          "pre-registered analysis. Counts are answers containing each form (one answer can contain several).", "",
          "| form | M normal | M coalition off | P normal | P coalition off |", "|---|---:|---:|---:|---:|"]
    for k, label in LABELS.items():
        md.append(f"| {label} | {result['M']['normal'].get(k, 0)} | {result['M']['coalition_off'].get(k, 0)} | "
                  f"{result['P']['normal'].get(k, 0)} | {result['P']['coalition_off'].get(k, 0)} |")
    md += ["", f"M pairs where the textbook form disappeared and a two-sentence form or the cousin appeared: "
               f"{len(examples)} of {result['M']['pairs']}.", "", "## Examples", ""]
    for ex in examples[:6]:
        md += [f"**{ex['prompt']}**", "", f"- normal: {ex['normal']}", f"- coalition off: {ex['coalition_off']}", ""]
    (E.REPO / "reports" / "confirmation_v2_forms.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:14]))


if __name__ == "__main__":
    main()
