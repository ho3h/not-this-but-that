"""Gallery figure: "not X but Y" on major brands' own sites (reports/figures/d5_brands.png).

Every quote is from 2026 and verbatim from a page checked live on 2026-10-04 (screenshots kept
outside the repo). The figure makes no claim about how the text was written.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PAPER, INK, MUTED, RULE, ACCENT, CARD = "#faf8f4", "#1f1d1a", "#5d5850", "#e4ded2", "#b4611c", "#ffffff"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Georgia", "DejaVu Serif"], "text.color": INK,
                     "figure.facecolor": PAPER, "savefig.facecolor": PAPER})
SANS = {"family": ["DejaVu Sans"]}

ITEMS = [
    ("Google", "Opening line of a Google.org announcement · News from Google, Feb 2026",
     "Government AI adoption isn't just a technical upgrade, it's a way to solve real world challenges…"),
    ("Salesforce", "Salesforce 360 Blog, Aug 2026 (the second half is in bold on the page)",
     "…giving back isn’t just something we do – it’s who we are."),
    ("NVIDIA", "GTC write-up · NVIDIA Blog, Mar 2026",
     "The bottleneck isn’t just data — it’s the entire data factory."),
    ("McKinsey & Company", "Standfirst of “Technology M&A: AI enters its industrial phase” · Feb 2026",
     "AI investment isn’t just fueling innovation; it’s reshaping competitive dynamics…"),
    ("Amazon", "Michael Wright, global head of MGM+ · Amazon press release, Sep 2026",
     "What makes this bundle so compelling isn't just the incredible value—it's the curation."),
    ("Spotify", "Co-CEOs Alex Norström and Gustav Söderström · Spotify newsroom, Jan 2026",
     "AI isn’t about automation or taking your hands off the wheel; it’s about agency."),
]


def main() -> None:
    fig = plt.figure(figsize=(10, 7.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100); ax.set_ylim(0, 72); ax.set_axis_off()
    ax.text(4, 69, "Who else says “not X but Y”?", fontsize=19, weight="bold", va="top")
    ax.text(4, 64.2, "Six of the world’s biggest companies, on their own websites, all in 2026. Quoted word for word.",
            fontsize=10.5, color=MUTED, va="top", **SANS)
    for i, (brand, who, quote) in enumerate(ITEMS):
        col, row = i % 2, i // 2
        x, y = 4 + col * 47, 42 - row * 19.5
        ax.add_patch(FancyBboxPatch((x, y), 45, 17.5, boxstyle="round,pad=0,rounding_size=1.2", fc=CARD, ec=RULE, lw=1))
        ax.add_patch(FancyBboxPatch((x, y), 0.9, 17.5, boxstyle="square,pad=0", fc=ACCENT, ec="none"))
        ax.text(x + 3, y + 15, brand, fontsize=13.5, weight="bold", va="top")
        ax.text(x + 3, y + 11.4, "\n".join(textwrap.wrap(who, 62)), fontsize=7.6, color=MUTED, va="top", **SANS)
        ax.text(x + 3, y + 6.6, "“" + "\n".join(textwrap.wrap(quote, 52)) + "”", fontsize=10.5, style="italic",
                va="top", color=INK, linespacing=1.35)
    out = REPO / "reports/figures/d5_brands.png"
    fig.savefig(out, dpi=200)
    print("wrote", out.name)


if __name__ == "__main__":
    main()
