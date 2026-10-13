"""Summary figure for the "why" experiments (reports/figures/fig7_why.png), in the story page's palette.

Reads reports/why_stages.json, why_arena.json, why_prefdata.json, why_rewardmodel.json and priming.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
R = REPO / "reports"
PAPER, INK, MUTED, FAINT, RULE, ACCENT, SOFT = "#faf8f4", "#1f1d1a", "#5d5850", "#8d877c", "#e4ded2", "#b4611c", "#e2b48a"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Georgia", "DejaVu Serif"], "text.color": INK,
                     "axes.edgecolor": RULE, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER})
SANS = {"family": "sans-serif"}


def load(name: str) -> dict:
    return json.loads((R / name).read_text())


def panel_title(fig, ax, text: str, sub: str) -> None:
    top = ax.get_position().y1
    fig.text(0.04, top + 0.058, text, fontsize=12.5, color=INK, weight="bold", va="bottom", **SANS)
    fig.text(0.04, top + 0.012, sub, fontsize=9.5, color=MUTED, va="bottom", linespacing=1.4, **SANS)


def bars(ax, labels, values, colors, fmt, xmax, ref=None):
    ys = list(range(len(labels)))[::-1]
    for y, lab, v, c in zip(ys, labels, values, colors):
        ax.barh(y, xmax, color="#efe9de", height=0.6, zorder=1)
        ax.barh(y, v, color=c, height=0.6, zorder=2)
        ax.text(-xmax * 0.02, y, lab, ha="right", va="center", fontsize=9.5, color=INK, **SANS)
        ax.text(v + xmax * 0.015, y, fmt(v), ha="left", va="center", fontsize=9.5, color=INK, zorder=3, **SANS)
    if ref is not None:
        ax.axvline(ref, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=3)
    ax.set_xlim(0, xmax); ax.set_ylim(-0.6, len(labels) - 0.4); ax.set_axis_off()


def main() -> None:
    st, ar, pdat, rm, pr = (load(n) for n in ("why_stages.json", "why_arena.json", "why_prefdata.json",
                                              "why_rewardmodel.json", "priming.json"))
    fig = plt.figure(figsize=(9, 10.4))
    fig.text(0.04, 0.975, "Why chatbots keep saying it", fontsize=17, weight="bold", va="top")
    fig.text(0.04, 0.945, "Three experiments, three clues. All exploratory.", fontsize=10.5, color=MUTED, va="top", **SANS)

    ax1 = fig.add_axes([0.33, 0.64, 0.57, 0.18])
    m = st["M"]
    labels = ["Pretrained only", "+ instruction tuning", "+ preference tuning", "+ final RL stage"]
    vals = [m[s]["per_1000_words"] for s in ("base", "SFT", "DPO", "Instruct")]
    bars(ax1, labels, vals, [RULE, SOFT, SOFT, ACCENT], lambda v: f"{v:.2f}", max(vals) * 1.35)
    panel_title(fig, ax1, "1. Training on preferences turns it up",
                "OLMo 2 1B at each public training stage. Uses per 1,000 words, answering\n93 questions about meaning.")

    ax2 = fig.add_axes([0.33, 0.36, 0.57, 0.18])
    same_len = ar["by_length_ratio_bin"]["-0.25..+0.25"]["win_rate"]
    labels = ["People (LMArena)", "GPT-4 (UltraFeedback)", "AI judge (OLMo 2 data)", "Reward model"]
    vals = [100 * same_len, 100 * pdat["UltraFeedback (GPT-4 ratings)"]["construction_chosen_equal_length"],
            100 * pdat["OLMo 2 preference mix"]["construction_chosen_equal_length"],
            100 * rm["original_vs_also"]["original_preferred"]]
    bars(ax2, labels, vals, [RULE, RULE, RULE, ACCENT], lambda v: f"{v:.0f}%", 100, ref=50)
    panel_title(fig, ax2, "2. People shrug. The reward model doesn't.",
                "How often the answer using it wins, at equal length. The reward model compares the\nsame idea "
                "written with and without the contrast. Dashed line: 50%, a coin flip.")

    ax3 = fig.add_axes([0.33, 0.07, 0.57, 0.16])
    mp = pr["M"]
    labels = ["\"X isn't just A; it's B.\"", "\"X is A, and it's B.\"", "\"X is B.\""]
    vals = [100 * mp[c]["rate"] for c in ("construction", "both", "plain")]
    bars(ax3, labels, vals, [RULE, SOFT, ACCENT], lambda v: f"{v:.0f}%", 50)
    panel_title(fig, ax3, "3. A grand claim calls for its foil",
                "Gemma 2. We wrote the first sentence (left); the share of answers that went on\nto use "
                "\"not X but Y\" anyway. 20 questions about meaning, 8 answers each.")
    fig.savefig(R / "figures/fig7_why.png", dpi=200)
    print("wrote fig7_why.png")


if __name__ == "__main__":
    main()
