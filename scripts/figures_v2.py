"""Static figures for the v2 post (reports/figures/fig*.png), in the story page's palette."""

from __future__ import annotations

import colorsys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "reports/figures"
OUT.mkdir(exist_ok=True)
PAPER, INK, MUTED, FAINT, RULE, ACCENT = "#faf8f4", "#1f1d1a", "#5d5850", "#8d877c", "#e4ded2", "#b4611c"
SOFT = "#e2b48a"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Georgia", "DejaVu Serif"], "text.color": INK,
                     "axes.edgecolor": RULE, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER})


def title(fig, main, sub=None):
    fig.text(0.04, 0.95, main, fontsize=17, weight="bold", va="top")
    if sub:
        fig.text(0.04, 0.885, sub, fontsize=10.5, color=MUTED, va="top", family="sans-serif")


def save(fig, name):
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("wrote", name)


def fig1_map():
    layout = json.loads((REPO / "web/demo/layout.json").read_text())["features"]
    chat = json.loads((REPO / "reports/confirmation_v3_prereg.json").read_text())["chat_features"]
    co = json.loads((REPO / "reports/switch_coactivation.json").read_text())
    xy = {f["idx"]: (f["x"], f["y"]) for f in layout}
    cids = sorted({f["cid"] for f in layout})
    hue = {c: (i * 0.618) % 1 for i, c in enumerate(cids)}
    cols = [colorsys.hls_to_rgb(hue[f["cid"]], 0.62, 0.42) for f in layout]
    fig = plt.figure(figsize=(9, 7.6))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.84])
    ax.scatter([f["x"] for f in layout], [f["y"] for f in layout], s=2.6, c=cols, alpha=0.6, linewidths=0)
    edges = [p for p in co["pairs"] if p["lift"] >= 6][:40]
    for e in edges:
        (x1, y1), (x2, y2) = xy[e["a"]], xy[e["b"]]
        ax.plot([x1, x2], [y1, y2], color=ACCENT, alpha=min(0.85, 0.18 + e["lift"] / 60), lw=0.6 + min(2.2, e["lift"] / 25), zorder=2)
    ax.scatter([xy[f][0] for f in chat], [xy[f][1] for f in chat], s=70, c=ACCENT, edgecolors="white", linewidths=1.2, zorder=3)
    labels = {15646: ("conditions or exceptions", (110, -40)), 13565: ("comparison or enhancement", (-210, 40)),
              9775: ('the verb "to be"', (120, 10)), 11313: ('the word "this"', (60, 25)),
              10655: ("legal processes", (-150, -40)), 10252: ("present-tense verbs", (20, 8))}
    for f, (text, off) in labels.items():
        x, y = xy[f]
        ax.annotate(text, (x, y), xytext=off, textcoords="offset points", fontsize=10, color=INK, family="sans-serif",
                    bbox=dict(boxstyle="round,pad=0.3", fc=PAPER, ec=RULE, lw=0.6, alpha=0.95), zorder=4,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=0, shrinkB=5))
    ax.set_axis_off()
    title(fig, "25 features on a map of 16,384",
          "Every dot is a feature inside Gemma 2 2B, coloured by cluster. Orange: the 25 features behind \"not X but Y\".\n"
          "Lines join the pairs that most often fire together in Gemma's answers.")
    save(fig, "fig1_feature_map.png")


def hbars(name, main, sub, rows, xmax=100, fmt="−{:.0f}%"):
    fig = plt.figure(figsize=(9, 0.62 * len(rows) + 2.0))
    ax = fig.add_axes([0.24, 0.08, 0.66, 0.70 - 0.02 * (8 - len(rows))])
    ys = np.arange(len(rows))[::-1]
    for y, (lab, val, style) in zip(ys, rows):
        color = {"hot": ACCENT, "warm": SOFT, "dim": RULE}[style]
        ax.barh(y, xmax, color="#efe9de", height=0.62, zorder=1)
        ax.barh(y, max(val, xmax * 0.004), color=color, height=0.62, zorder=2)
        ax.text(-xmax * 0.02, y, lab, ha="right", va="center", fontsize=11.5, family="sans-serif",
                color=INK if style != "dim" else FAINT, weight="bold" if style == "hot" else "normal")
        ax.text(xmax * 1.02, y, (fmt.format(val) if round(val) else (f"{val:.1f}%" if val >= 0.05 else "0%")), ha="left", va="center", fontsize=12, family="sans-serif",
                color=INK if style != "dim" else FAINT, weight="bold" if style == "hot" else "normal")
    ax.set_xlim(0, xmax); ax.set_ylim(-0.6, len(rows) - 0.4); ax.set_axis_off()
    title(fig, main, sub)
    save(fig, name)


def fig2_ladder():
    d = json.loads((REPO / "reports/decision_point_ladder.json").read_text())["kinds"]["family"]
    rows = [(f"{k} feature{'s' if k != '1' else ''} ablated", 100 * v, "hot" if k == "25" else "warm") for k, v in d["ladder"].items()]
    rows += [("25 decoys, random", max(0, 100 * d["random_25"]["mean"]), "dim"), ("25 decoys, matched", max(0, 100 * d["matched_25"]["mean"]), "dim")]
    hbars("fig2_ladder.png", "A handful of features does most of the work",
          "Drop in Gemma's odds of heading for \"not X but Y\" when the top features are ablated,\non sentences they were not chosen from.",
          rows)


def fig3_dial():
    r = json.loads((REPO / "reports/turn_up.json").read_text())["all"]
    rows = [("25 ablated", 100 * r["25 off"]["habit"] / r["25 off"]["answers"], "dim"),
            ("normal", 100 * r["normal"]["habit"] / r["normal"]["answers"], "warm"),
            ("25 amplified ×2", 100 * r["25 x2"]["habit"] / r["25 x2"]["answers"], "warm"),
            ("25 amplified ×3", 100 * r["25 x3"]["habit"] / r["25 x3"]["answers"], "hot"),
            ("25 decoys amplified ×3", 100 * r["matched 25 x3"]["habit"] / r["matched 25 x3"]["answers"], "dim")]
    hbars("fig3_dial.png", "Ablate it, amplify it: the 25 features work as a dial",
          "Share of answers using \"not X but Y\", on 152 questions that played no part in choosing the features.",
          rows, xmax=60, fmt="{:.0f}%")


def fig4_tradeoff():
    fig = plt.figure(figsize=(8.5, 6.6))
    ax = fig.add_axes([0.12, 0.11, 0.80, 0.68])
    pts = [("Gemma 2, original 25", 91, 60, ACCENT), ("Gemma 2, cleaner 25", 72, 95, ACCENT), ("Gemma 3, 24 switches", 83, 39, MUTED)]
    ax.add_patch(plt.Rectangle((90, 90), 10, 10, color="#efe9de", zorder=0))
    ax.text(95, 97.5, "the dream", ha="center", va="center", fontsize=10, color=FAINT, family="sans-serif", style="italic")
    for lab, x, y, c in pts:
        ax.scatter(x, y, s=180, color=c, edgecolors="white", linewidths=1.5, zorder=3)
        ax.annotate(lab, (x, y), xytext=(-12, 0), textcoords="offset points", ha="right", va="center", fontsize=11, family="sans-serif")
    ax.annotate("", xy=(72, 95), xytext=(91, 60), arrowprops=dict(arrowstyle="->", color=SOFT, lw=1.5, connectionstyle="arc3,rad=-0.2"))
    ax.set_xlim(50, 100); ax.set_ylim(20, 100)
    ax.set_xlabel("% of the sentence removed", family="sans-serif", fontsize=11)
    ax.set_ylabel('% of ordinary "not"s kept', family="sans-serif", fontsize=11)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color=RULE, lw=0.6, zorder=0)
    title(fig, "Removing the habit costs some negation",
          "Each dot is a set of switches, tested on the same 140 fresh questions. Top right would be ideal.")
    save(fig, "fig4_tradeoff.png")


def fig5_questions():
    fig = plt.figure(figsize=(9, 5.2))
    ax = fig.add_axes([0.08, 0.12, 0.88, 0.62])
    groups = [("Gemma 2 2B", 100 * 234 / 800, 100 * 2 / 400), ("Gemma 3 4B", 100 * 294 / 400, 100 * 26 / 200)]
    x = np.arange(len(groups)); wbar = 0.34
    for i, (lab, why, how) in enumerate(groups):
        ax.bar(i - wbar / 2, why, wbar, color=ACCENT); ax.bar(i + wbar / 2, how, wbar, color=SOFT)
        ax.text(i - wbar / 2, why + 2, f"{why:.0f}%", ha="center", fontsize=12, family="sans-serif", weight="bold")
        ax.text(i + wbar / 2, how + 2, f"{how:.1f}%" if how < 1 else f"{how:.0f}%", ha="center", fontsize=12, family="sans-serif")
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=12, family="sans-serif")
    ax.set_ylim(0, 85); ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.bar(0, 0, color=ACCENT, label='"why does this matter?" questions'); ax.bar(0, 0, color=SOFT, label='"how does this work?" questions')
    ax.legend(frameon=False, loc="upper left", prop={"family": ["sans-serif"], "size": 10.5})
    title(fig, "Ask about meaning, get \"not X but Y\"", "Share of answers using \"not X but Y\", by kind of question.")
    save(fig, "fig5_questions.png")


if __name__ == "__main__":
    fig1_map(); fig2_ladder(); fig3_dial(); fig4_tradeoff(); fig5_questions()
