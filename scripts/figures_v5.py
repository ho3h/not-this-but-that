"""Explainer visuals for the v8 post (reports/figures/v5/). One idea per image, readable in two seconds.

Shared code: one mark = one answer; orange = it used "not X but Y". No titles in images.
  waffle_dial.png   every answer to 152 held-out questions at four dial settings (turn_up)
  waffle_questions  meaning vs process questions, Gemma 2 (confirmation v5 counts)
  ribbons.png       OLMo 2 answers at four training stages: length and where the construction lands
  tug.png           who picks the "not X but Y" version, at equal length
  fork_tree.png     where Gemma goes after "A city's coffee culture" (needs vis_data.py)
  lightup.gif/.png  the 25 features lighting up word by word (needs vis_data.py)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
R = REPO / "reports"
OUT = R / "figures/v5"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "src"))
from classifier.detect_v3 import detect_v3  # noqa: E402

PAPER, INK, MUTED, FAINT, RULE, ACCENT, SOFT, EMPTY = ("#faf8f4", "#1f1d1a", "#5d5850", "#8d877c", "#e4ded2",
                                                       "#b4611c", "#e2b48a", "#e8e2d6")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Georgia", "DejaVu Serif"], "text.color": INK,
                     "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER})
SANS = {"family": "sans-serif"}


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=220)
    plt.close(fig)
    print("wrote", name)


def uses(text: str) -> bool:
    d = detect_v3(text)
    return bool(d["family"] or d["cousin"])


def waffle(ax, n: int, k: int, cols: int, cell: float = 1.0, gap: float = 0.18) -> None:
    """n squares, the first k orange, filled from the bottom row up, left to right."""
    rows = int(np.ceil(n / cols))
    for i in range(n):
        r, c = divmod(i, cols)
        ax.add_patch(plt.Rectangle((c * cell, r * cell), cell - gap, cell - gap,
                                   fc=ACCENT if i < k else EMPTY, ec="none"))
    ax.set_xlim(-0.5, cols * cell + 0.3); ax.set_ylim(-0.5, rows * cell + 0.3); ax.set_aspect("equal"); ax.set_axis_off()


def waffle_dial() -> None:
    rows = [json.loads(l) for l in (R / "turn_up_generations.jsonl").read_text().splitlines() if l.strip()]
    conds = [("25 off", "ablated"), ("normal", "normal"), ("25 x2", "amplified ×2"), ("25 x3", "amplified ×3")]
    fig = plt.figure(figsize=(9, 3.6))
    for j, (c, lab) in enumerate(conds):
        texts = [r["text"] for r in rows if r["condition"] == c]
        k = sum(uses(t) for t in texts)
        ax = fig.add_axes([0.02 + j * 0.245, 0.2, 0.22, 0.75])
        waffle(ax, len(texts), k, cols=16)
        fig.text(0.02 + j * 0.245 + 0.11, 0.11, lab, ha="center", fontsize=12, color=INK)
        fig.text(0.02 + j * 0.245 + 0.11, 0.035, f"{100 * k / len(texts):.0f}%" if k / len(texts) >= 0.01 else f"{k} of {len(texts)}",
                 ha="center", fontsize=10.5, color=ACCENT if k else FAINT, **SANS)
        print(f"  dial {lab}: {k}/{len(texts)}")
    save(fig, "waffle_dial.png")


def waffle_questions() -> None:
    fig = plt.figure(figsize=(9, 3.5))
    for j, (lab, n, k, rows_) in enumerate((("why does it matter?", 800, 234, 20), ("how does it work?", 400, 2, 10))):
        ax = fig.add_axes([0.03 + j * 0.5, 0.2, 0.44, 0.75])
        waffle(ax, n, k, cols=40)
        ax.set_ylim(-0.5, 20.3)
        fig.text(0.03 + j * 0.5 + 0.22, 0.11, f"“{lab}”", ha="center", fontsize=12.5, color=INK, style="italic")
        fig.text(0.03 + j * 0.5 + 0.22, 0.035, f"{k} of {n} answers", ha="center", fontsize=10.5,
                 color=ACCENT if k > 10 else FAINT, **SANS)
    save(fig, "waffle_questions.png")


def ribbons() -> None:
    rows = [json.loads(l) for l in (R / "why_stages_generations.jsonl").read_text().splitlines() if l.strip()]
    cut = re.compile(r"<\|[^|<>]{1,30}\|>")
    stages = [("base", "pretrained"), ("SFT", "+ instructions"), ("DPO", "+ preferences"), ("Instruct", "+ final RL")]
    ids = sorted({r["id"] for r in rows if r["type"] == "M"})
    fig = plt.figure(figsize=(9, 6.4))
    for j, (st, lab) in enumerate(stages):
        ax = fig.add_axes([0.02 + j * 0.245, 0.09, 0.225, 0.89])
        by = {r["id"]: r for r in rows if r["stage"] == st and r["sample"] == 0}
        dots = 0
        for y, i in enumerate(ids):
            text = cut.split(by[i]["text"])[0].strip()
            words = len(text.split())
            ax.plot([0, min(words, 210)], [y, y], color=RULE, lw=1.1, solid_capstyle="round")
            d = detect_v3(text)
            for h in d["family"] + d["cousin"]:
                pos = len(text[:h.span[0]].split())
                ax.scatter([min(pos, 210)], [y], s=16, color=ACCENT, zorder=3, linewidths=0)
                dots += 1
        ax.set_xlim(-4, 215); ax.set_ylim(len(ids), -1); ax.set_axis_off()
        fig.text(0.02 + j * 0.245 + 0.005, 0.045, lab, fontsize=11.5, color=INK)
        fig.text(0.02 + j * 0.245 + 0.005, 0.012, f"{dots} use{'s' if dots != 1 else ''}", fontsize=9.5,
                 color=ACCENT if dots else FAINT, **SANS)
        print(f"  ribbons {lab}: {dots} uses in {len(ids)} answers")
    save(fig, "ribbons.png")


def tug() -> None:
    ar = json.loads((R / "why_arena.json").read_text())
    pdat = json.loads((R / "why_prefdata.json").read_text())
    rm = json.loads((R / "why_rewardmodel.json").read_text())
    rows = [("People", 100 * ar["by_length_ratio_bin"]["-0.25..+0.25"]["win_rate"]),
            ("GPT-4 as judge", 100 * pdat["UltraFeedback (GPT-4 ratings)"]["construction_chosen_equal_length"]),
            ("AI judge", 100 * pdat["OLMo 2 preference mix"]["construction_chosen_equal_length"]),
            ("Reward model", 100 * rm["original_vs_also"]["original_preferred"])]
    fig = plt.figure(figsize=(9, 3.2))
    ax = fig.add_axes([0.2, 0.06, 0.76, 0.8]); ax.set_xlim(0, 100); ax.set_ylim(-0.6, len(rows) - 0.2); ax.set_axis_off()
    fig.text(0.2, 0.92, "picked “not X but Y”", fontsize=10, color=ACCENT, **SANS)
    fig.text(0.96, 0.92, "picked it plain", fontsize=10, color=FAINT, ha="right", **SANS)
    for i, (lab, v) in enumerate(rows[::-1]):
        hot = lab == "Reward model"
        ax.barh(i, v, height=0.56, color=ACCENT if hot else SOFT, edgecolor=PAPER, linewidth=2)
        ax.barh(i, 100 - v, left=v, height=0.56, color=EMPTY, edgecolor=PAPER, linewidth=2)
        ax.text(-1.5, i, lab, ha="right", va="center", fontsize=12, color=INK if hot else MUTED)
        ax.text(v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=10.5, color="white" if hot else INK, **SANS)
    ax.plot([50, 50], [-0.45, len(rows) - 0.55], color=INK, lw=1.2, zorder=4)
    save(fig, "tug.png")


def _band(ax, x0, y0a, y0b, x1, y1a, y1b, color, alpha=1.0):
    """A ribbon from the vertical segment [y0a, y0b] at x0 to [y1a, y1b] at x1."""
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch
    xm = (x0 + x1) / 2
    verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a), (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4, MPath.CURVE4,
             MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), fc=color, ec="none", alpha=alpha))


def _mix(share: float):
    a, b = np.array(matplotlib.colors.to_rgb(EMPTY)), np.array(matplotlib.colors.to_rgb(ACCENT))
    return tuple(a + (b - a) * share)


def fork_tree() -> None:
    d = json.loads((OUT / "fork_samples.json").read_text())
    fig = plt.figure(figsize=(9, 7.6))
    for j, cond in enumerate(("normal", "25 ablated")):
        rows = [r for r in d["rows"] if r["condition"] == cond]
        ax = fig.add_axes([0.0, 0.51 - j * 0.49, 1.0, 0.39]); ax.set_xlim(-0.62, 3.05); ax.set_ylim(-8, 312); ax.set_axis_off()
        w1 = lambda r: (r["text"].split() or [""])[0]  # noqa: E731
        w2 = lambda r: " ".join(r["text"].split()[:2])  # noqa: E731
        firsts = [k for k, _ in __import__("collections").Counter(map(w1, rows)).most_common(5)]
        groups = {}
        for r in rows:
            a = w1(r) if w1(r) in firsts else "…"
            groups.setdefault(a, []).append(r)
        order = sorted(groups, key=lambda a: (a == "…", -len(groups[a])))
        gap, y_root = 6, 0.0
        total = len(rows) + gap * (len(order) - 1)
        scale = 300 / total
        y1 = 300.0
        ax.add_patch(plt.Rectangle((0, 0), 0.03, 300, fc=INK, ec="none"))
        ax.text(-0.04, 150, "A city’s\ncoffee\nculture…", ha="right", va="center", fontsize=11.5, style="italic", color=INK,
                linespacing=1.25)
        root_y = 300.0
        for a in order:
            rs = groups[a]
            h = len(rs) * scale
            share = np.mean([r["construction"] for r in rs])
            _band(ax, 0.03, root_y, root_y - len(rs), 1.0, y1, y1 - h, _mix(share), 0.9)
            root_y -= len(rs)
            seconds = __import__("collections").Counter(map(w2, rs)).most_common(3)
            keep = [k for k, n in seconds if n >= 6][:2]
            sub = {}
            for r in rs:
                sub.setdefault(w2(r) if w2(r) in keep else "…", []).append(r)
            sorder = sorted(sub, key=lambda b: (b == "…", -len(sub[b])))
            yy, y2 = y1, y1
            for b in sorder:
                srs = sub[b]
                hh = len(srs) * scale
                sh = np.mean([r["construction"] for r in srs])
                _band(ax, 1.0, yy, yy - hh, 2.0, y2, y2 - hh, _mix(sh), 0.9)
                if b != "…" and hh >= 7:
                    ax.text(2.04, y2 - hh / 2, b + "…", va="center", fontsize=11 if sh < 0.5 else 11.5,
                            color=ACCENT if sh >= 0.5 else MUTED, weight="bold" if sh >= 0.5 else "normal")
                yy -= hh; y2 -= hh
            y1 -= h + gap * scale
        uses = sum(r["construction"] for r in rows)
        fig.text(0.02, 0.965 - j * 0.49, "normal" if cond == "normal" else "25 features ablated", ha="left",
                 fontsize=11.5, color=INK, **SANS)
        fig.text(0.02, 0.935 - j * 0.49, f"{uses} of {len(rows)} answers use “not X but Y”", ha="left", fontsize=10,
                 color=ACCENT if uses else FAINT, **SANS)
    save(fig, "fork_tree.png")


def leaderboard() -> None:
    d = json.loads((R / "reroute_attribution.json").read_text())["family"]["ranked"]
    chat = set(json.loads((R / "confirmation_v3_prereg.json").read_text())["chat_features"])
    sc = np.array([r["score"] for r in d]); top = [r["feature_idx"] in chat for r in d]
    fig = plt.figure(figsize=(9, 3.6))
    ax = fig.add_axes([0.03, 0.14, 0.94, 0.82]); ax.set_xlim(-3, len(d) + 3); ax.set_ylim(0, sc.max() * 1.18); ax.set_axis_off()
    ax.bar(np.arange(len(d)), sc, width=0.75, color=[ACCENT if t else RULE for t in top], linewidth=0)
    ax.annotate("conditions or exceptions", (0, sc[0]), xytext=(18, -2), textcoords="offset points", fontsize=10.5,
                style="italic", color=INK, va="center", arrowprops=dict(arrowstyle="-", color=FAINT, lw=0.7))
    ax.plot([0, 24], [sc[24] * -0.0 - sc.max() * 0.04] * 2, color=ACCENT, lw=2, clip_on=False)
    ax.text(0, -sc.max() * 0.09, "the 25 I ablated", ha="left", va="top", fontsize=10, color=ACCENT, **SANS)
    ax.text(len(d), -sc.max() * 0.09, f"{len(d)} features, ranked", ha="right", va="top", fontsize=10, color=FAINT, **SANS)
    save(fig, "leaderboard.png")


def lightup() -> None:
    from PIL import Image

    d = json.loads((OUT / "lightup.json").read_text())
    layout = json.loads((REPO / "web/demo/layout.json").read_text())["features"]
    toks, A, tm, xy = d["tokens"], np.array(d["acts"]), np.array(d["typical_max"]), np.array(d["xy"])
    rel = (A / tm[None, :]).clip(0, 1.5)
    bx, by = np.array([f["x"] for f in layout]), np.array([f["y"] for f in layout])
    frames, durations = [], []
    peak = int(np.argmax(rel.sum(1)))
    for i in range(len(toks) + 1):
        fig = plt.figure(figsize=(7.2, 6.2), dpi=110)
        axt = fig.add_axes([0.04, 0.84, 0.92, 0.14]); axt.set_xlim(0, 1); axt.set_ylim(0, 1); axt.set_axis_off()
        shown = "".join(toks[:i])
        lines, cur = [], ""
        for word in shown.split(" "):
            if len(cur) + len(word) > 46:
                lines.append(cur); cur = word
            else:
                cur = (cur + " " + word).strip()
        lines.append(cur)
        last = toks[i - 1].strip() if i > 0 else ""
        shown_lines = lines[-2:]
        for k, line in enumerate(shown_lines):
            y = 0.72 - k * 0.42
            if k == len(shown_lines) - 1 and last and line.endswith(last):
                head = axt.text(0.0, y, line[: len(line) - len(last)], fontsize=15, va="center", color=INK)
                fig.canvas.draw()
                bb = head.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(axt.transAxes.inverted())
                axt.text(bb.x1, y, last, fontsize=15, va="center", color=ACCENT, weight="bold")
            else:
                axt.text(0.0, y, line, fontsize=15, va="center", color=INK)
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.8]); ax.set_axis_off()
        ax.scatter(bx, by, s=1.2, color=RULE, alpha=0.7, linewidths=0)
        g = rel[i - 1] if i > 0 else np.zeros(len(xy))
        ax.scatter(xy[:, 0], xy[:, 1], s=26, color=SOFT, alpha=0.55, linewidths=0, zorder=2)
        on = g > 0
        ax.scatter(xy[on, 0], xy[on, 1], s=60 + 900 * g[on] ** 1.3, color=ACCENT, alpha=0.18, linewidths=0, zorder=3)
        ax.scatter(xy[on, 0], xy[on, 1], s=30 + 160 * g[on], color=ACCENT, alpha=0.95, linewidths=0, zorder=4)
        fig.canvas.draw()
        img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
        plt.close(fig)
        frames.append(img)
        durations.append(2600 if i == len(toks) else (1500 if i - 1 == peak else 380))
        if i - 1 == peak:
            img.save(OUT / "lightup_still.png")
    pal = [f.quantize(colors=96, method=Image.Quantize.MEDIANCUT) for f in frames]
    pal[0].save(OUT / "lightup.gif", save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True)
    print("wrote lightup.gif", f"({(OUT / 'lightup.gif').stat().st_size / 1e6:.1f} MB), peak at", repr(toks[peak]))


if __name__ == "__main__":
    which = sys.argv[1:] or ["waffle_dial", "waffle_questions", "ribbons", "tug"]
    for name in which:
        globals()[name]()
