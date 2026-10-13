"""Pared-back figures for the v7 post (reports/figures/v4/): no titles in the image, minimal text.

Captions in the post carry the explanation. Six figures: tell (the sentence), map (the 16,384
features), fork (next-word odds), dial (amplify/ablate), trend (books + news), judges (who prefers it).
"""

from __future__ import annotations

import colorsys
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
R = REPO / "reports"
OUT = R / "figures/v4"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "scripts"))
import trend_analysis as T  # noqa: E402

PAPER, INK, MUTED, FAINT, RULE, ACCENT, SOFT, TRACK = ("#faf8f4", "#1f1d1a", "#5d5850", "#8d877c", "#e4ded2",
                                                       "#b4611c", "#e2b48a", "#efe9de")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Georgia", "DejaVu Serif"], "text.color": INK,
                     "axes.edgecolor": RULE, "xtick.color": FAINT, "ytick.color": FAINT,
                     "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER})
SANS = {"family": "sans-serif"}


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=220)
    plt.close(fig)
    print("wrote", name)


def text_width(ax, fig, s, **kw) -> float:
    t = ax.text(0, 0, s, **kw)
    fig.canvas.draw()
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    inv = ax.transData.inverted()
    w = inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0]
    t.remove()
    return w


def tell() -> None:
    fig = plt.figure(figsize=(9, 1.25))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 19); ax.set_axis_off()
    runs = [("“This isn’t just ", INK, "normal"), ("a problem", FAINT, "normal"), (", it’s ", INK, "normal"),
            ("an epidemic", ACCENT, "bold"), (".”", INK, "normal")]
    kw = dict(fontsize=24)
    widths = [text_width(ax, fig, t, weight=w, **kw) for t, _, w in runs]
    x = 50 - sum(widths) / 2
    for (t, c, w), wd in zip(runs, widths):
        ax.text(x, 9.5, t, color=c, weight=w, va="center", **kw)
        if t == "a problem":
            ax.plot([x, x + wd], [9.9, 9.9], color=ACCENT, lw=2.4)
        x += wd
    save(fig, "tell.png")


def feature_map() -> None:
    layout = json.loads((REPO / "web/demo/layout.json").read_text())["features"]
    chat = json.loads((R / "confirmation_v3_prereg.json").read_text())["chat_features"]
    co = json.loads((R / "switch_coactivation.json").read_text())
    xy = {f["idx"]: (f["x"], f["y"]) for f in layout}
    cids = sorted({f["cid"] for f in layout})
    hue = {c: (i * 0.618) % 1 for i, c in enumerate(cids)}
    cols = [colorsys.hls_to_rgb(hue[f["cid"]], 0.64, 0.38) for f in layout]
    fig = plt.figure(figsize=(9, 7.0))
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    ax.scatter([f["x"] for f in layout], [f["y"] for f in layout], s=2.2, c=cols, alpha=0.55, linewidths=0)
    for e in [p for p in co["pairs"] if p["lift"] >= 6][:40]:
        (x1, y1), (x2, y2) = xy[e["a"]], xy[e["b"]]
        ax.plot([x1, x2], [y1, y2], color=ACCENT, alpha=min(0.85, 0.18 + e["lift"] / 60),
                lw=0.6 + min(2.2, e["lift"] / 25), zorder=2)
    ax.scatter([xy[f][0] for f in chat], [xy[f][1] for f in chat], s=64, c=ACCENT, edgecolors=PAPER,
               linewidths=1.2, zorder=3)
    for f, text, off in ((15646, "conditions or exceptions", (16, -22)), (13565, "comparison or enhancement", (-18, 26))):
        x, y = xy[f]
        ax.annotate(text, (x, y), xytext=off, textcoords="offset points", fontsize=10, color=MUTED, style="italic",
                    ha="left" if off[0] > 0 else "right", zorder=4,
                    arrowprops=dict(arrowstyle="-", color=FAINT, lw=0.6, shrinkA=2, shrinkB=5))
    ax.set_axis_off()
    save(fig, "map.png")


def fork() -> None:
    data = json.loads((R / "figures/shortlist_coffee.json").read_text())
    norm = [(" is", 0.422), (" acts", 0.199), (" can", 0.199), (" isn't", 0.083)]
    abl = [(" can", 0.316), (" acts", 0.246), (" profoundly", 0.035), (" deeply", 0.026), (" is", 0.021), (" isn't", 0.0)]
    assert abs(dict(data["normal"])[" is"] - 0.422) < 1e-6  # numbers come from the model run
    fig = plt.figure(figsize=(9, 4.1))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 90); ax.set_ylim(0, 41); ax.set_axis_off()
    for head, rows, x0 in (("Normal", norm, 3), ("25 features ablated", abl, 48)):
        ax.text(x0 + 11, 37, head, fontsize=11, color=MUTED, ha="right" if False else "left", **SANS)
        for i, (w, p) in enumerate(rows):
            y = 31 - i * 5.2
            hot = w.strip() in ("is", "isn't")
            ax.text(x0 + 10, y, w.strip(), ha="right", va="center", fontsize=13, color=ACCENT if hot else INK,
                    weight="bold" if hot else "normal")
            ax.add_patch(plt.Rectangle((x0 + 11, y - 1.4), 24, 2.8, fc=TRACK, ec="none"))
            if p > 0:
                ax.add_patch(plt.Rectangle((x0 + 11, y - 1.4), max(0.4, 24 * p / 0.45), 2.8,
                                           fc=ACCENT if hot else SOFT, ec="none"))
            ax.text(x0 + 36, y, f"{100 * p:.0f}%", va="center", fontsize=10.5, color=ACCENT if hot else FAINT, **SANS)
    save(fig, "fork.png")


def dial() -> None:
    r = json.loads((R / "turn_up.json").read_text())["all"]
    rate = lambda k: 100 * r[k]["habit"] / r[k]["answers"]  # noqa: E731
    pts = [("off", rate("25 off"), RULE), ("normal", rate("normal"), SOFT), ("×2", rate("25 x2"), "#cf8a4f"),
           ("×3", rate("25 x3"), ACCENT)]
    fig = plt.figure(figsize=(9, 2.2))
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.92]); ax.set_xlim(-3, 63); ax.set_ylim(-5, 6); ax.set_axis_off()
    ax.plot([0, 60], [0, 0], color=TRACK, lw=14, solid_capstyle="round", zorder=1)
    for lab, v, c in pts:
        ax.scatter([v], [0], s=260, color=c, edgecolors=PAPER, linewidths=2, zorder=3)
        ax.text(v, 2.4, lab, ha="center", fontsize=12, color=INK if lab != "off" else MUTED)
        ax.text(v, 4.3, f"{v:.0f}%" if v >= 1 else f"{v:.1f}%", ha="center", fontsize=9.5, color=MUTED, **SANS)
    dv = rate("matched 25 x3")
    ax.scatter([dv], [0], s=150, facecolors="none", edgecolors=FAINT, linewidths=1.6, zorder=3)
    ax.text(dv, -2.6, f"decoys ×3: {dv:.0f}%", ha="center", va="top", fontsize=9.5, color=FAINT, **SANS)
    save(fig, "dial.png")


def trend() -> None:
    base = T.monthly("according_to")
    fig = plt.figure(figsize=(9, 6.4))
    ax1 = fig.add_axes([0.07, 0.62, 0.80, 0.32])
    years = np.arange(1900, 2023)
    b = T.ngram("is_not_just")
    ax1.plot(years, b, color=ACCENT, lw=2)
    ax1.text(1903, b.max() * 0.95, "books", fontsize=11, color=MUTED, va="top", **SANS)
    ax1.set_xlim(1900, 2027); ax1.set_ylim(0, b.max() * 1.15)
    ax1.set_xticks([1900, 1950, 2000]); ax1.set_yticks([])
    ax2 = fig.add_axes([0.07, 0.08, 0.80, 0.44])
    for name, color, lw in (("not_only", RULE, 1.6), ("isnt_just", ACCENT, 2.4)):
        s = T.per_control(T.monthly(name), base)
        ms = sorted(s)
        pre = T.window_mean(s, *T.PRE)
        idx = np.array([100 * s[m] / pre for m in ms])
        sm = np.convolve(idx, np.ones(3) / 3, mode="same"); sm[0], sm[-1] = idx[0], idx[-1]
        x = np.array([int(m[:4]) + (int(m[5:]) - 0.5) / 12 for m in ms])
        ax2.plot(x, sm, color=color, lw=lw)
        ax2.text(x[-1] + 0.12, sm[-1], "“isn’t just”" if name == "isnt_just" else "“not only”", fontsize=10,
                 color=INK if name == "isnt_just" else FAINT, va="center", **SANS)
        if name == "isnt_just":
            k = int(np.argmax(sm))
            ax2.text(x[k] - 0.15, sm[k], f"{sm[k] / 100:.1f}×", ha="right", va="center", fontsize=10, color=ACCENT, **SANS)
    ax2.axvline(2022 + 11 / 12, color=FAINT, lw=0.8, ls=(0, (2, 3)))
    ax2.text(2022 + 11 / 12 + 0.1, 0.04, "ChatGPT", transform=ax2.get_xaxis_transform(), fontsize=9, color=FAINT, **SANS)
    ax2.text(2017.1, 0.95, "news", transform=ax2.get_xaxis_transform(), fontsize=11, color=MUTED, va="top", **SANS)
    ax2.set_xlim(2017, 2027.6); ax2.set_yticks([])
    ax2.set_xticks([2018, 2020, 2022, 2024, 2026])
    for ax in (ax1, ax2):
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=9, length=0)
    save(fig, "trend.png")


def judges() -> None:
    ar = json.loads((R / "why_arena.json").read_text())
    pdat = json.loads((R / "why_prefdata.json").read_text())
    rm = json.loads((R / "why_rewardmodel.json").read_text())
    rows = [("People", 100 * ar["by_length_ratio_bin"]["-0.25..+0.25"]["win_rate"], FAINT),
            ("GPT-4", 100 * pdat["UltraFeedback (GPT-4 ratings)"]["construction_chosen_equal_length"], FAINT),
            ("AI judge", 100 * pdat["OLMo 2 preference mix"]["construction_chosen_equal_length"], FAINT),
            ("Reward model", 100 * rm["original_vs_also"]["original_preferred"], ACCENT)]
    fig = plt.figure(figsize=(9, 2.9))
    ax = fig.add_axes([0.2, 0.04, 0.74, 0.92]); ax.set_xlim(35, 90); ax.set_ylim(-1.0, len(rows) - 0.5)
    ax.plot([50, 50], [-0.45, len(rows) - 0.6], color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(50, -0.6, "coin flip", ha="center", va="top", fontsize=9, color=MUTED, **SANS)
    for i, (lab, v, c) in enumerate(rows[::-1]):
        ax.plot([35, 90], [i, i], color=TRACK, lw=1, zorder=1)
        ax.scatter([v], [i], s=170 if c == ACCENT else 120, color=c, edgecolors=PAPER, linewidths=1.5, zorder=3)
        ax.text(33.5, i, lab, ha="right", va="center", fontsize=12, color=INK if c == ACCENT else MUTED)
        ax.text(v + 1.6, i, f"{v:.0f}%", va="center", fontsize=10, color=c if c == ACCENT else MUTED, **SANS)
    ax.set_axis_off()
    save(fig, "judges.png")


if __name__ == "__main__":
    tell(); feature_map(); fork(); dial(); trend(); judges()
