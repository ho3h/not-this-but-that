"""Explainer diagrams for the v3 post (reports/figures/d*.png), in the story page's palette."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "reports/figures"
PAPER, INK, MUTED, FAINT, RULE, ACCENT, SOFT, CARD = "#faf8f4", "#1f1d1a", "#5d5850", "#8d877c", "#e4ded2", "#b4611c", "#e2b48a", "#ffffff"
GREEN = "#3f7a52"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Georgia", "DejaVu Serif"], "text.color": INK,
                     "figure.facecolor": PAPER, "savefig.facecolor": PAPER})
SANS = {"family": ["DejaVu Sans"]}


def canvas(w=9, h=5):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w * 10); ax.set_ylim(0, h * 10); ax.set_axis_off()
    return fig, ax


def box(ax, x, y, w, h, fc=CARD, ec=RULE, lw=1.0, r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))


def arrow(ax, a, b, color=MUTED, rad=0.0, lw=1.6):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}"))


def title(ax, x, y, main, sub=None):
    ax.text(x, y, main, fontsize=17, weight="bold", va="top")
    if sub:
        ax.text(x, y - 4.6, sub, fontsize=10.5, color=MUTED, va="top", **SANS)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("wrote", name)


def d1_anatomy():
    fig, ax = canvas(9, 5.2)
    title(ax, 4, 49, 'Anatomy of "not X but Y"')
    runs = [("“This isn’t just ", INK, "normal"), ("a problem", FAINT, "normal"), (", it’s ", INK, "normal"),
            ("an epidemic", ACCENT, "bold"), (".”", INK, "normal")]
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    widths = []
    for text, color, weight in runs:
        t = ax.text(0, 0, text, fontsize=19, weight=weight)
        bb = t.get_window_extent(renderer=rend)
        widths.append(inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0])
        t.remove()
    x = 45 - sum(widths) / 2
    for (text, color, weight), wd in zip(runs, widths):
        ax.text(x, 37, text, fontsize=19, color=color, weight=weight, va="center")
        if text == "a problem":
            ax.plot([x, x + wd], [37.3, 37.3], color=ACCENT, lw=2.2)
        x += wd
    steps = [("1", "Set up a plain idea", "a problem"), ("2", "Knock it down", "isn’t just…"),
             ("3", "Swap in a grander one", "an epidemic")]
    for i, (n, head, eg) in enumerate(steps):
        x = 4 + i * 28.5
        box(ax, x, 9, 25.5, 17)
        ax.add_patch(Circle((x + 3.6, 22), 2.2, color=ACCENT))
        ax.text(x + 3.6, 22, n, ha="center", va="center", color="white", fontsize=11, weight="bold", **SANS)
        ax.text(x + 7.2, 22, head, va="center", fontsize=10, weight="bold", **SANS)
        ax.text(x + 3, 14, eg, va="center", fontsize=13, style="italic", color=ACCENT if n == "3" else MUTED)
        if i < 2:
            arrow(ax, (x + 25.8, 17.5), (x + 28.2, 17.5))
    ax.text(4, 4, 'Costumes it wears: "It\'s not X, it\'s Y"  ·  "This isn\'t just X. It\'s Y"  ·  "X is more than just Y"',
            fontsize=10, color=MUTED, **SANS)
    save(fig, "d1_anatomy.png")


def d2_fork():
    data = json.loads((OUT / "shortlist_coffee.json").read_text())
    norm = [(" is", 0.422), (" acts", 0.199), (" can", 0.199), (" isn't", 0.083)]
    abl = [(" can", 0.316), (" acts", 0.246), (" profoundly", 0.035), (" deeply", 0.026), (" is", 0.021), (" isn't", 0.0)]
    assert abs(dict(data["normal"])[" is"] - 0.422) < 1e-6  # numbers come from the model run
    fig, ax = canvas(9, 6.0)
    title(ax, 4, 58, "Catching Gemma at the fork", "The next word after “A city's coffee culture…”, with Gemma's odds for each.")
    ax.text(4, 45, "“A city's coffee culture …”", fontsize=15, style="italic", va="center")
    for col, (head, rows, x0) in enumerate((("Normal", norm, 4), ("25 features ablated", abl, 48))):
        ax.text(x0, 38, head, fontsize=12.5, weight="bold", **SANS)
        for i, (w, p) in enumerate(rows):
            y = 33 - i * 5
            hot = w.strip() in ("is", "isn't")
            ax.text(x0 + 11, y, w.strip(), ha="right", va="center", fontsize=12, color=ACCENT if hot else INK,
                    weight="bold" if hot else "normal", **SANS)
            ax.add_patch(FancyBboxPatch((x0 + 12.5, y - 1.5), 24, 3, boxstyle="round,pad=0,rounding_size=0.8", fc="#efe9de", ec="none"))
            if p > 0:
                ax.add_patch(FancyBboxPatch((x0 + 12.5, y - 1.5), max(0.6, 24 * p / 0.45), 3,
                                            boxstyle="round,pad=0,rounding_size=0.8", fc=ACCENT if hot else SOFT, ec="none"))
            ax.text(x0 + 37.5, y, f"{100 * p:.0f}%", va="center", fontsize=11, color=ACCENT if hot else MUTED, **SANS)
    ax.text(4, 11, 'Orange: the roads into "not X but Y"\n("isn\'t just…" and "is more than just…").', fontsize=10, color=MUTED, va="top", **SANS)
    save(fig, "d2_fork.png")


def d3_dial():
    fig, ax = canvas(9, 5.4)
    title(ax, 4, 52, "Ablate, amplify, clamp",
          "As the model writes, each feature adds a small push to what comes next.\nResearchers can change the size of that push.")
    cards = [("Ablate", "push × 0", 0.0, "Silenced.", "The feature has\nno say at all."),
             ("Normal", "push × 1", 1.0, "As trained.", "Business as usual."),
             ("Amplify", "push × 3", 3.0, "Shouting.", "Its effect counts\nthree times over."),
             ("Clamp", "pinned on, every word", 4.0, "Obsessed.", "“Not-So-Not-So Simple\nScrambled Eggs”")]
    for i, (name, how, push, big, small) in enumerate(cards):
        x = 4 + i * 21.5
        box(ax, x, 6, 19.5, 31, ec=ACCENT if name in ("Amplify", "Clamp") else RULE)
        ax.text(x + 1.8, 33, name, fontsize=14, weight="bold")
        ax.text(x + 1.8, 29, how, fontsize=9.5, color=MUTED, **SANS)
        ax.add_patch(FancyBboxPatch((x + 1.8, 22.5), 15.9, 2.6, boxstyle="round,pad=0,rounding_size=0.8", fc="#efe9de", ec="none"))
        if push:
            ax.add_patch(FancyBboxPatch((x + 1.8, 22.5), 15.9 * min(1, push / 4), 2.6, boxstyle="round,pad=0,rounding_size=0.8",
                                        fc=ACCENT if push >= 3 else SOFT, ec="none"))
        ax.text(x + 1.8, 15.5, big, fontsize=17, weight="bold", color=ACCENT if push >= 3 else INK)
        ax.text(x + 1.8, 12, small, fontsize=10, va="top", color=ACCENT if name == "Clamp" else MUTED,
                style="italic" if name == "Clamp" else "normal", **({} if name == "Clamp" else SANS))
    save(fig, "d3_dial.png")


def d4_double_duty():
    fig, ax = canvas(9, 5.6)
    title(ax, 4, 54, "One set of features, two jobs", "Ablating the 25 removes both.")
    ax.add_patch(Circle((13, 26), 7.5, color=ACCENT))
    ax.text(13, 27.5, "the 25", ha="center", va="center", color="white", fontsize=13, weight="bold", **SANS)
    ax.text(13, 24, "features", ha="center", va="center", color="white", fontsize=10, **SANS)
    rows = [(36, '"not X but Y"', "“…isn’t just about the hot beverages; it’s a reflection of its character…”",
             "“…coffee culture deeply shapes its character…”", True),
            (15, 'saying "not" when it matters', "“Never connect the positive (red) cable to the negative (black) battery terminal.”",
             "“Connect the positive and negative cables to the wrong terminals.”", False)]
    for y, job, normal, ablated, good in rows:
        arrow(ax, (21, 26 + (y - 25) * 0.25), (25, y + 2), color=ACCENT, rad=-0.15 if y > 25 else 0.15)
        ax.text(26, y + 8.6, job, fontsize=11.5, weight="bold", color=ACCENT, **SANS)
        box(ax, 26, y - 3.4, 60, 10.6)
        ax.text(28, y + 4.4, "Normal", fontsize=8.5, color=FAINT, weight="bold", **SANS)
        ax.text(36, y + 4.4, normal, fontsize=9.6, va="center", style="italic", wrap=True)
        ax.text(28, y - 0.4, "Ablated", fontsize=8.5, color=FAINT, weight="bold", **SANS)
        ax.text(36, y - 0.4, ablated, fontsize=9.6, va="center", style="italic", color=GREEN if good else "#a8382a")
        ax.text(84.5, y - 0.4, "✓" if good else "✗", fontsize=13, ha="right", va="center", color=GREEN if good else "#a8382a", **SANS)
    save(fig, "d4_double_duty.png")


if __name__ == "__main__":
    d1_anatomy(); d2_fork(); d3_dial(); d4_double_duty()
