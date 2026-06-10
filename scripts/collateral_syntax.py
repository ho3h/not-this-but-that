"""Collateral-syntax table: what ELSE the coalition ablation changes.

The coalition was selected by causal attribution to P(pivot), where the pivot
token set includes the comma, " but", " it", and dash tokens. A fair reader
will ask: did the intervention remove the construction specifically, or did
it suppress comma/but syntax wholesale and the construction died as
collateral? This script measures the syntactic collateral directly on the
committed Q5b/Q5c generations so the post can own the answer instead of
being ambushed by it.

Per condition, per eval: share of generations containing "but" / "however" /
"instead"; mean commas, em-dashes, "not"s, sentences, and words per
generation; ", but"-style pivot bigrams per generation.

Writes reports/collateral_syntax.{md,json}.
"""
from __future__ import annotations

import json
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT_MD = REPO / "reports" / "collateral_syntax.md"
OUT_JSON = REPO / "reports" / "collateral_syntax.json"

WORD_BUT = re.compile(r"\bbut\b", re.IGNORECASE)
WORD_HOWEVER = re.compile(r"\bhowever\b", re.IGNORECASE)
WORD_INSTEAD = re.compile(r"\binstead\b", re.IGNORECASE)
WORD_NOT = re.compile(r"\bnot\b|n't\b", re.IGNORECASE)
PIVOT_BIGRAM = re.compile(r"[,;]\s*(?:but|it'?s|they'?re|he'?s|she'?s)\b", re.IGNORECASE)
SENT_SPLIT = re.compile(r"[.!?]+")


def stats(texts: list[str]) -> dict:
    n = len(texts)
    def share(pat):
        return sum(1 for t in texts if pat.search(t)) / n
    def mean(fn):
        return sum(fn(t) for t in texts) / n
    return {
        "n": n,
        "share_with_but": share(WORD_BUT),
        "share_with_however": share(WORD_HOWEVER),
        "share_with_instead": share(WORD_INSTEAD),
        "mean_buts": mean(lambda t: len(WORD_BUT.findall(t))),
        "mean_commas": mean(lambda t: t.count(",")),
        "mean_emdashes": mean(lambda t: t.count("—") + t.count("–")),
        "mean_nots": mean(lambda t: len(WORD_NOT.findall(t))),
        "mean_pivot_bigrams": mean(lambda t: len(PIVOT_BIGRAM.findall(t))),
        "mean_words": mean(lambda t: len(t.split())),
        "mean_sentences": mean(lambda t: max(1, len([s for s in SENT_SPLIT.split(t) if s.strip()]))),
    }


def main():
    files = [
        ("Q5b primed (n=300)", "reports/q5b_d1_continuation.json"),
        ("Q5c neutral (n=306)", "reports/q5c_d2_high_power.json"),
    ]
    out = {}
    for name, path in files:
        rows = json.loads((REPO / path).read_text())["rows"]
        out[name] = {
            "baseline": stats([r["baseline"] for r in rows]),
            "ablated": stats([r["ablated"] for r in rows]),
        }

    OUT_JSON.write_text(json.dumps(out, indent=2))

    md = [
        "# Collateral syntax under top-25 coalition ablation",
        "",
        "The coalition was selected for its causal effect on P(pivot), whose",
        "token set includes the comma, \" but\", \" it\" and dash tokens. This",
        "table reports what the ablation does to contrastive/clausal syntax",
        "*overall* — not just to the construction — on the committed Q5b/Q5c",
        "generations. Per-generation means; share = fraction of generations",
        "containing the word at least once.",
        "",
    ]
    keys = [
        ("share_with_but", "share w/ \"but\"", "{:.1%}"),
        ("share_with_however", "share w/ \"however\"", "{:.1%}"),
        ("share_with_instead", "share w/ \"instead\"", "{:.1%}"),
        ("mean_buts", "\"but\" per gen", "{:.2f}"),
        ("mean_commas", "commas per gen", "{:.2f}"),
        ("mean_emdashes", "em/en-dashes per gen", "{:.2f}"),
        ("mean_nots", "negations per gen", "{:.2f}"),
        ("mean_pivot_bigrams", "\",-pivot\" bigrams per gen", "{:.2f}"),
        ("mean_words", "words per gen", "{:.0f}"),
        ("mean_sentences", "sentences per gen", "{:.1f}"),
    ]
    for name, conds in out.items():
        md += [f"## {name}", "", "| measure | baseline | ablated | change |", "|---|---:|---:|---:|"]
        for k, label, fmt in keys:
            b, a = conds["baseline"][k], conds["ablated"][k]
            chg = (a - b) / b if b else 0.0
            md.append(f"| {label} | {fmt.format(b)} | {fmt.format(a)} | {chg:+.0%} |")
        md.append("")
    OUT_MD.write_text("\n".join(md))
    print(f"→ {OUT_MD}")
    for name, conds in out.items():
        b, a = conds["baseline"], conds["ablated"]
        print(f"{name}: but-share {b['share_with_but']:.1%} → {a['share_with_but']:.1%}; "
              f"commas {b['mean_commas']:.2f} → {a['mean_commas']:.2f}; "
              f"words {b['mean_words']:.0f} → {a['mean_words']:.0f}")


if __name__ == "__main__":
    main()
