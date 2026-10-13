"""Trend in "isn't just" across printed books (to 2022) and English-language news (2017 to 2026).

Inputs come from scripts/trend_gdelt.py (data/trend/gdelt/) and the Google Books Ngram series
cached in data/trend/ngram/. Missing GDELT series are skipped, so the script can run while the
fetch is still going.

News rates are per 1,000 articles that contain the neutral phrase "according to" in the same
month, which cancels changes in how many sources GDELT monitors. Each series is also indexed
to its own 2019-01..2022-10 average (= 100), the window before ChatGPT's launch on 2022-11-30.
The current, partial month is dropped. Writes reports/trend.{md,json} and
reports/figures/fig6_trend.png.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
GD, NG = REPO / "data/trend/gdelt", REPO / "data/trend/ngram"
PAPER, INK, MUTED, FAINT, RULE, ACCENT, SOFT = "#faf8f4", "#1f1d1a", "#5d5850", "#8d877c", "#e4ded2", "#b4611c", "#e2b48a"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Georgia", "DejaVu Serif"], "text.color": INK,
                     "axes.edgecolor": RULE, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER})
PRE = ("2019-01", "2022-10")
POST = ("2025-01", "2026-09")
LAST_FULL = "2026-09"
LABELS = {"isnt_just": "\"isn't just\"", "not_just": "\"not just\"", "more_than_just": "\"more than just\"",
          "is_not_just": "\"is not just\"", "its_not_just": "\"it's not just\"",
          "not_only": "\"not only\" (control)", "isnt": "\"isn't\" (control)"}


def monthly(name: str) -> dict[str, float] | None:
    path = GD / f"{name}.json"
    if not path.exists() or not path.read_text().lstrip().startswith("{"):
        return None
    out: dict[str, float] = defaultdict(float)
    for d in json.loads(path.read_text())["timeline"][0]["data"]:
        m = f"{d['date'][:4]}-{d['date'][4:6]}"
        if m <= LAST_FULL:
            out[m] += d["value"]
    return dict(out)


def window_mean(series: dict[str, float], lo: str, hi: str) -> float:
    vals = [v for m, v in series.items() if lo <= m <= hi]
    return float(np.mean(vals)) if vals else float("nan")


def per_control(num: dict[str, float], den: dict[str, float]) -> dict[str, float]:
    return {m: 1000 * num[m] / den[m] for m in sorted(num) if den.get(m)}


def yearly(num: dict[str, float], den: dict[str, float]) -> dict[str, float]:
    ny, dy = defaultdict(float), defaultdict(float)
    for m, v in num.items():
        ny[m[:4]] += v
    for m, v in den.items():
        dy[m[:4]] += v
    return {y: 1000 * ny[y] / dy[y] for y in sorted(ny) if dy.get(y)}


def ngram(name: str) -> np.ndarray | None:
    path = NG / f"{name}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return np.array(d[0]["timeseries"]) * 1e6 if d else None


def main() -> None:
    res: dict = {"news": {}, "wires": {}, "books": {}}
    base = monthly("according_to")
    news = {}
    if base:
        for name in LABELS:
            s = monthly(name)
            if s:
                r = per_control(s, base)
                pre, post = window_mean(r, *PRE), window_mean(r, *POST)
                news[name] = r
                res["news"][name] = {"per_1000_pre": pre, "per_1000_post": post, "post_over_pre": post / pre,
                                     "yearly_per_1000": yearly(s, base)}
        isnt, ij = monthly("isnt"), monthly("isnt_just")
        if isnt and ij:
            share = {m: 100 * ij[m] / isnt[m] for m in ij if isnt.get(m)}
            res["news"]["isnt_just_share_of_isnt"] = {"pct_pre": window_mean(share, *PRE),
                                                      "pct_post": window_mean(share, *POST)}
    for tag, dom in (("prn", "PR Newswire"), ("bw", "Business Wire"), ("gnw", "GlobeNewswire")):
        num, den = monthly(f"isnt_just__{tag}"), monthly(f"according_to__{tag}")
        if num and den:
            res["wires"][dom] = yearly(num, den)
    years = np.arange(1900, 2023)
    for name in ("is_not_just", "not_only", "according_to"):
        s = ngram(name)
        if s is not None:
            res["books"][name] = {str(y): float(v) for y, v in zip(years, s)}
    (REPO / "reports/trend.json").write_text(json.dumps(res, indent=2))
    figure(res, news, years)
    report(res)


def figure(res: dict, news: dict, years: np.ndarray) -> None:
    fig = plt.figure(figsize=(9, 8.2))
    fig.text(0.04, 0.965, "A century of slow growth, then a two-year spike", fontsize=16, weight="bold", va="top")
    fig.text(0.04, 0.925, "Top: printed books, uses per million words. Bottom: English-language news, indexed so the\n"
             "2019–2022 average is 100 (3-month average). Grey lines are control phrases.",
             fontsize=10, color=MUTED, va="top", family="sans-serif")
    ax1 = fig.add_axes([0.08, 0.56, 0.86, 0.27])
    books = res["books"]
    if "is_not_just" in books:
        v = np.array(list(books["is_not_just"].values()))
        ax1.plot(years, v, color=ACCENT, lw=2)
        ax1.text(2022.5, v[-1], "\"is not just\"", color=INK, fontsize=10, va="center", family="sans-serif")
        ax1.set_ylim(0, max(v) * 1.15)
    ax1.set_xlim(1900, 2030)
    ax1.set_title("Books, 1900–2022 (Google Books Ngram)", loc="left", fontsize=11, color=INK, family="sans-serif")
    ax1.set_ylabel("per million words", family="sans-serif", fontsize=9.5)
    ax2 = fig.add_axes([0.08, 0.08, 0.86, 0.38])
    for name, color, lw in (("not_only", FAINT, 1.4), ("isnt", RULE, 1.4), ("not_just", SOFT, 1.8),
                            ("isnt_just", ACCENT, 2.2)):
        if name not in news:
            continue
        r = news[name]
        ms = sorted(r)
        base = window_mean(r, *PRE)
        idx = np.array([100 * r[m] / base for m in ms])
        smooth = np.convolve(idx, np.ones(3) / 3, mode="same")
        smooth[0], smooth[-1] = idx[0], idx[-1]
        x = np.array([int(m[:4]) + (int(m[5:]) - 0.5) / 12 for m in ms])
        ax2.plot(x, smooth, color=color, lw=lw)
        if name == "isnt_just":
            k = int(np.argmax(smooth))
            month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][int(ms[k][5:]) - 1]
            ax2.annotate(f"Peak, {month} {ms[k][:4]}: {smooth[k] / 100:.1f}× the 2019–22 level", (x[k], smooth[k]),
                         xytext=(-14, 0), textcoords="offset points", ha="right", va="center", fontsize=9,
                         color=INK, family="sans-serif", bbox=dict(fc=PAPER, ec="none", pad=1.5), zorder=5)
        ax2.text(x[-1] + 0.08, smooth[-1], LABELS[name], color=INK if name == "isnt_just" else MUTED,
                 fontsize=9.5, va="center", family="sans-serif", weight="bold" if name == "isnt_just" else "normal")
    ax2.axvline(2022 + 11 / 12, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax2.text(2022 + 11 / 12 - 0.08, 0.62, "ChatGPT\nlaunches", ha="right", va="top", transform=ax2.get_xaxis_transform(),
             fontsize=9, color=MUTED, family="sans-serif")
    ax2.axhline(100, color=RULE, lw=0.8)
    ax2.set_xlim(2017, 2027.9)
    ax2.set_title("News, 2017–2026 (GDELT)", loc="left", fontsize=11, color=INK, family="sans-serif")
    ax2.set_ylabel("index, 2019–2022 = 100", family="sans-serif", fontsize=9.5)
    for ax in (ax1, ax2):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color=RULE, lw=0.6)
        ax.tick_params(labelsize=9)
    out = REPO / "reports/figures/fig6_trend.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out.name)


def report(res: dict) -> None:
    md = ["# \"Isn't just\" in books and the news (exploratory)", "",
          "News: GDELT DOC 2.0, English-language sources, articles per 1,000 articles containing \"according to\" "
          f"in the same month. Before = {PRE[0]} to {PRE[1]}; after = {POST[0]} to {POST[1]}.", "",
          "| phrase | before (per 1,000) | after (per 1,000) | after ÷ before |", "|---|---:|---:|---:|"]
    for name, v in res["news"].items():
        if name in LABELS:
            md.append(f"| {LABELS[name]} | {v['per_1000_pre']:.1f} | {v['per_1000_post']:.1f} | **{v['post_over_pre']:.2f}×** |")
    sh = res["news"].get("isnt_just_share_of_isnt")
    if sh:
        md += ["", f"Share of all \"isn't\" articles that contain \"isn't just\": {sh['pct_pre']:.1f}% before, "
               f"{sh['pct_post']:.1f}% after."]
    if "isnt_just" in res["news"]:
        md += ["", "By year (\"isn't just\" per 1,000):", "",
               " · ".join(f"{y}: {v:.1f}" for y, v in res["news"]["isnt_just"]["yearly_per_1000"].items())]
    if res["wires"]:
        md += ["", "Press-release wires (\"isn't just\" per 1,000 releases containing \"according to\"):", ""]
        for dom, ys in res["wires"].items():
            md.append(f"- {dom}: " + " · ".join(f"{y}: {v:.1f}" for y, v in ys.items()))
    b = res["books"]
    if "is_not_just" in b:
        bj = b["is_not_just"]
        md += ["", "Books (Google Books Ngram, English, per million words; contractions are not indexed reliably, so "
               "the uncontracted form is used): \"is not just\" " + ", ".join(f"{y}: {bj[y]:.2f}" for y in
                                                                             ("1920", "1950", "1980", "2000", "2019"))
               + "."]
        if "not_only" in b:
            no = b["not_only"]
            md.append(f"Control \"not only\": 1920: {no['1920']:.1f}, 2019: {no['2019']:.1f}.")
    md += ["", "Caveats: GDELT's source list changes over time and its matching is by article, not by sentence; "
           "the control-phrase ratio corrects for volume but not for a changing mix of outlets. Ngram ends in 2022."]
    (REPO / "reports/trend.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
