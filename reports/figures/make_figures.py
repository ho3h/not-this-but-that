#!/usr/bin/env python
"""Generate the three Medium-post figures for the not-X-but-Y coalition finding.

Outputs (written next to this script, reports/figures/):
  ladder.png             - ablation ladder: relative drop in P(pivot) vs coalition size
  atlas.png              - 16,384-feature SAE map with the 25-feature coalition in red
  cards.html             - styled before/after generation cards (source for the PNG)
  before_after_cards.png - screenshot of cards.html (rendered via Playwright / a browser)

Run:  .venv/bin/python reports/figures/make_figures.py

Data sources (read-only):
  reports/joint_ablation.json          (n=80: k = 1, 2, 5, 10)
  reports/asymptote_ladder.json        (n=80: k = 25, 50, 75, 100 + random controls)
  reports/asymptote_ladder_n200.json   (n=200: k = 5..100 + random controls)
  reports/q5c_d2_high_power.json       (canonical evaluated top-25 coalition)
  reports/pivot_attribution_n40.json   (rank check for the coalition)
  web/demo/layout.json                 (16,384 feature positions + community ids)
  web/demo/examples.json               (curated baseline-vs-ablated generation pairs)
  reports/demo_gallery.json            (gallery baseline-vs-ablated generation pairs)
"""

from pathlib import Path
import colorsys
import html as html_mod
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

# ---- shared style ----------------------------------------------------------
BG = "#0d1117"      # GitHub-dark canvas
PANEL = "#161b22"   # card background
FG = "#e6edf3"      # light text
MUTED = "#8b949e"   # secondary text
GRID = "#21262d"    # hairlines
SPINE = "#30363d"
ACCENT = "#ff4d4d"  # the coalition red (single accent)
ACCENT_SOFT = "#a06168"  # desaturated companion for the secondary series

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": FG,
    "axes.edgecolor": SPINE,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
})


def load(relpath):
    with open(ROOT / relpath) as f:
        return json.load(f)


# ============================================================================
# FIGURE 1 - the ablation ladder
# ============================================================================

def fig_ladder():
    joint = load("reports/joint_ablation.json")
    asym80 = load("reports/asymptote_ladder.json")["result"]
    asym200 = load("reports/asymptote_ladder_n200.json")["result"]

    # n=80 blended series: k=1,2,5,10 from joint_ablation; k=25,50,75,100 from
    # asymptote_ladder (same prompt set, same baseline P(pivot)=0.2778).
    assert abs(joint["baseline_mean_p_pivot"] - asym80["baseline_mean_p_pivot"]) < 1e-9
    n80 = [
        (1, joint["by_condition"]["single_3223"]["rel_drop"]),
        (2, joint["by_condition"]["attrib_top2"]["rel_drop"]),
        (5, joint["by_condition"]["attrib_top5"]["rel_drop"]),
        (10, joint["by_condition"]["attrib_top10"]["rel_drop"]),
        (25, asym80["by_condition"]["attrib_top25"]["rel_drop"]),
        (50, asym80["by_condition"]["attrib_top50"]["rel_drop"]),
        (75, asym80["by_condition"]["attrib_top75"]["rel_drop"]),
        (100, asym80["by_condition"]["attrib_top100"]["rel_drop"]),
    ]
    # n=200 rerun: k=1 and k=2 rungs were not run at n=200, so this series
    # starts at k=5 (which is why the blended n=80 series is the main line).
    n200 = [(k, asym200["by_condition"][f"attrib_top{k}"]["rel_drop"])
            for k in (5, 10, 25, 50, 75, 100)]

    x80, y80 = zip(*[(k, 100 * r) for k, r in n80])
    x200, y200 = zip(*[(k, 100 * r) for k, r in n200])

    # 100-random-features control (n=200 run): rel drops are all ~0
    ctrl_drops = [100 * asym200["by_condition"][f"random_k100_d{i}"]["rel_drop"]
                  for i in range(3)]
    ctrl = float(np.mean(ctrl_drops))  # ~ -0.09 %
    assert all(abs(d) < 0.2 for d in ctrl_drops)

    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=200)
    fig.subplots_adjust(left=0.085, right=0.965, top=0.94, bottom=0.115)

    ax.set_xscale("log")
    ax.grid(axis="y", color=GRID, lw=0.9, alpha=0.9, zorder=0)

    # asymptote band ~79-82 %
    ax.axhspan(79, 82, color=MUTED, alpha=0.10, lw=0, zorder=1)
    ax.text(0.95, 80.5, "asymptote ≈ 79–82%", fontsize=9, color=MUTED,
            ha="left", va="center")

    # random control near 0 %
    ax.axhline(ctrl, ls=(0, (5, 4)), color=MUTED, lw=1.1, alpha=0.85, zorder=2)
    ax.text(1.0, ctrl + 2.2, "100 random features (control)  ≈ 0% drop",
            fontsize=9, color=MUTED, ha="left", va="bottom")

    # secondary series: n=200 rerun (desaturated, behind)
    ax.plot(x200, y200, "-", color=ACCENT_SOFT, lw=1.5, alpha=0.95, zorder=3,
            marker="s", ms=4.2, mfc=ACCENT_SOFT, mec=BG, mew=0.6,
            label="n = 200 rerun (k ≥ 5)")

    # main series: blended n=80 ladder
    ax.plot(x80, y80, "-", color=ACCENT, lw=2.4, zorder=4,
            marker="o", ms=6.0, mfc=ACCENT, mec=BG, mew=0.9,
            label="cumulative top-k ablation (n = 80)")

    # callout: the 2-feature core
    ax.annotate("2 features (3223 + 9909)\nalready −37%",
                xy=(2, y80[1]), xytext=(1.02, 59),
                fontsize=9.5, color=FG, ha="left", va="top", linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9,
                                shrinkA=4, shrinkB=5,
                                connectionstyle="arc3,rad=-0.18"))

    # callout: top-25 coalition
    ax.annotate("top-25 coalition: −72% (n=80) · −76% (n=200)",
                xy=(25, y200[2]), xytext=(5.2, 88),
                fontsize=9.5, color=FG, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9,
                                shrinkA=6, shrinkB=6,
                                connectionstyle="arc3,rad=0.16"))

    ax.set_xlim(0.88, 122)
    ax.set_ylim(-7, 94)
    ax.set_xticks([1, 2, 5, 10, 25, 50, 100])
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(
        ["1", "2", "5", "10", "25", "50", "100"]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.tick_params(labelsize=10, length=3, color=SPINE)

    ax.set_xlabel("features ablated (cumulative top-k by attribution, log scale)",
                  fontsize=10.5, labelpad=7)
    ax.set_ylabel("relative drop in P(pivot)  (%)", fontsize=10.5, labelpad=7)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[1], handles[0]], [labels[1], labels[0]],
              loc="lower right", bbox_to_anchor=(0.99, 0.085),
              frameon=False, fontsize=9.5, handlelength=2.2, labelcolor=FG)
    fig.savefig(OUT / "ladder.png")
    plt.close(fig)
    print(f"wrote {OUT / 'ladder.png'}")


# ============================================================================
# FIGURE 2 - the feature atlas with the coalition in red
# ============================================================================

def fig_atlas():
    feats = load("web/demo/layout.json")["features"]
    top25 = load("reports/q5c_d2_high_power.json")["top25"]
    # sanity: canonical coalition == top-25 of the n=40 attribution, same order
    pa = load("reports/pivot_attribution_n40.json")["top_promotes_pivot"][:25]
    assert top25 == [r["feature_idx"] for r in pa]

    xs = np.array([f["x"] for f in feats])
    ys = np.array([f["y"] for f in feats])
    cids = np.array([f["cid"] for f in feats])
    by_idx = {f["idx"]: f for f in feats}

    # muted desaturated categorical palette, one hue per community.
    # red hues are excluded so the accent stays unique.
    uniq = sorted(set(cids.tolist()))
    palette = {}
    for i, cid in enumerate(uniq):
        h = 0.07 + 0.80 * i / max(1, len(uniq) - 1)   # hue 0.07..0.87
        l = 0.60 if i % 2 == 0 else 0.70              # alternate lightness
        s = 0.42
        palette[cid] = colorsys.hls_to_rgb(h, l, s)
    colors = np.array([palette[c] for c in cids])

    fig = plt.figure(figsize=(8.0, 8.0), dpi=200)
    ax = fig.add_axes([0.025, 0.025, 0.95, 0.95])
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.14, 1.14)
    ax.set_ylim(-1.14, 1.14)

    # the nebula
    ax.scatter(xs, ys, s=3.4, c=colors, alpha=0.50, linewidths=0, zorder=1,
               rasterized=False)

    # the coalition: soft glow = concentric layers behind each red dot
    cx = np.array([by_idx[i]["x"] for i in top25])
    cy = np.array([by_idx[i]["y"] for i in top25])
    for s_glow, a_glow in [(560, 0.05), (300, 0.10), (150, 0.20)]:
        ax.scatter(cx, cy, s=s_glow, c=ACCENT, alpha=a_glow, linewidths=0,
                   zorder=3)
    ax.scatter(cx, cy, s=60, c=ACCENT, alpha=1.0, zorder=4,
               edgecolors="#ffb4b4", linewidths=0.7)

    # annotate the two core features with thin leader lines; a white ring
    # disambiguates which red dot each label points to.
    def callout(fid, text, xytext, ha):
        f = by_idx[fid]
        ax.scatter([f["x"]], [f["y"]], s=185, facecolors="none",
                   edgecolors="#f5f7fa", linewidths=1.1, alpha=0.95, zorder=5)
        ax.annotate(text, xy=(f["x"], f["y"]), xytext=xytext,
                    fontsize=11.5, color=FG, ha=ha, va="center",
                    zorder=5,
                    arrowprops=dict(arrowstyle="-", color="#9da7b3", lw=0.8,
                                    alpha=0.9, shrinkA=6, shrinkB=13,
                                    connectionstyle="arc3,rad=0.12"))

    callout(3223, '3223  “exceptions & negations”', (-0.96, 0.74), "left")
    callout(9909, '9909  “digital technology”', (0.56, -0.86), "left")

    fig.savefig(OUT / "atlas.png")
    plt.close(fig)
    print(f"wrote {OUT / 'atlas.png'}")


# ============================================================================
# FIGURE 3 - before/after generation cards (HTML -> screenshot)
# ============================================================================

def _clean(text):
    """Whitespace/markup cleanup only - the words stay verbatim."""
    t = text.replace("**", "")
    t = re.sub(r"\s*\n\s*", " ", t)
    t = re.sub(r" {2,}", " ", t)
    return t.strip()


def _prep(raw, marker=None):
    """Clean a raw completion, optionally truncate after `marker`, and keep
    the glue character so the text joins the prompt the way the model wrote
    it (', it's...' attaches directly; ' that...' needs its space)."""
    glue = "" if raw[:1] in ",.;:!?" else " "
    t = _clean(raw)
    if marker is not None:
        t = _truncate_after(t, marker)
    return glue + t


def _truncate_after(text, marker):
    i = text.find(marker)
    assert i >= 0, f"marker not found: {marker!r}"
    return text[: i + len(marker)] + " …"


def _mark(text, spans, cls="hl"):
    """HTML-escape, then wrap each highlight span."""
    out = html_mod.escape(text, quote=False)
    for sp in spans:
        esc = html_mod.escape(sp, quote=False)
        assert esc in out, f"highlight span not found: {sp!r}"
        out = out.replace(esc, f'<span class="{cls}">{esc}</span>', 1)
    return out


def build_cards_html():
    """Two verified-clean kills (every detector tier silent on the ablated
    side, eyeball-checked for reroutes) plus the springboard REROUTE,
    labelled as such — the kill and the mutation side by side. Pairs 1-2 are
    instruction prompts from the out-of-sample Q5c eval; pair 3 is the primed
    flagship from the baked demo examples."""
    examples = {e["id"]: e for e in load("web/demo/examples.json")["examples"]}
    q5c = load("reports/q5c_d2_high_power.json")["rows"]

    # --- pair 1: clean kill on a neutral prompt (q5c row 92, seed 2) ---
    rt = q5c[92]
    assert rt["prompt"].startswith("Reflect on what makes a long road trip")
    pair1 = dict(
        mode="answer",
        prompt=rt["prompt"],
        base=_mark(_prep(rt["baseline"], "it's about the journey itself."), [
            "It's not just about reaching the destination; it's about the "
            "journey itself.",
        ]),
        abl=html_mod.escape(
            _prep(rt["ablated"], "make a trip memorable."), quote=False),
    )

    # --- pair 2: clean kill on a neutral prompt (q5c row 62, seed 2) ---
    bg = q5c[62]
    assert bg["prompt"].startswith("Discuss why some readers fall in love")
    base_raw = bg["baseline"].split("**Answer:**", 1)[1]
    abl_raw = bg["ablated"].split("**Answer:**", 1)[1]
    pair2 = dict(
        mode="answer",
        prompt=bg["prompt"],
        base=_mark(_prep(base_raw, "into their hearts, struggles,"), [
            "It's not just a story about someone's life; it's a journey "
            "into their hearts",
        ]),
        abl=html_mod.escape(
            _prep(abl_raw, "worlds apart from our own."), quote=False),
    )

    # --- pair 3: the REROUTE — springboard (web/demo/examples.json) ---
    sb = examples["setback"]
    pair3 = dict(
        mode="continuation",
        prompt=sb["prompt"],
        base=_mark(_prep(sb["runs"]["baseline"]["completion"]), [
            "it's a springboard",
            "not the end of the journey, but rather a chance to adjust, "
            "adapt, and move forward",
        ]),
        abl=_mark(_prep(sb["runs"]["top25"]["completion"],
                        "make the best decision."), [
            "It's a opportunity",
        ], cls="hl2"),
        abl_tag="COALITION SILENCED · THE REROUTE",
        abl_class="abl-reroute",
        note="same-sentence form dead — the period-form takes its place",
    )

    def pair_html(p):
        prompt_esc = html_mod.escape(p["prompt"], quote=False)
        # continuation mode shows the prefix inline (the completion attaches
        # to it); answer mode lets the completion stand alone.
        inline = (f'<span class="ptext">{prompt_esc}</span>'
                  if p.get("mode") == "continuation" else "")
        abl_tag = p.get("abl_tag", "COALITION SILENCED · 25 FEATURES")
        abl_cls = p.get("abl_class", "")
        tag_cls = "tag-reroute" if abl_cls else "tag-abl"
        note = (f'<div class="note">{html_mod.escape(p["note"], quote=False)}</div>'
                if p.get("note") else "")
        return f"""
    <section class="pair">
      <div class="prompt-row"><span class="chip">PROMPT</span>
        <span class="prompt">“{prompt_esc}”</span></div>
      <div class="cols">
        <div class="card base">
          <div class="tag tag-base">BASELINE</div>
          <p>{inline}{p["base"]}</p>
        </div>
        <div class="card abl {abl_cls}">
          <div class="tag {tag_cls}">{abl_tag}</div>
          <p>{inline}{p["abl"]}</p>
          {note}
        </div>
      </div>
    </section>"""

    page = f"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    width: 1600px; background: {BG}; color: {FG};
    font-family: -apple-system, "Helvetica Neue", "Segoe UI", Arial, sans-serif;
    padding: 56px 64px 40px;
  }}
  .pair {{ margin-bottom: 52px; }}
  .pair:last-of-type {{ margin-bottom: 28px; }}
  .prompt-row {{ display: flex; align-items: baseline; gap: 16px; margin: 0 4px 18px; }}
  .chip {{
    font-size: 15px; letter-spacing: 2.5px; font-weight: 700; color: {MUTED};
    border: 1px solid {SPINE}; border-radius: 999px; padding: 5px 14px 4px;
  }}
  .prompt {{ font-size: 26px; font-weight: 600; color: {FG}; }}
  .cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 28px; }}
  .card {{
    background: {PANEL}; border-radius: 16px; padding: 26px 30px 30px;
    border: 1px solid {SPINE};
  }}
  .card.abl {{ border: 1px solid rgba(63, 185, 80, 0.45); }}
  .tag {{
    font-size: 14.5px; font-weight: 700; letter-spacing: 2.2px;
    margin-bottom: 16px;
  }}
  .tag-base {{ color: #f08a8a; }}
  .tag-abl {{ color: #3fb950; }}
  .card p {{ font-size: 26px; line-height: 1.58; color: {FG}; }}
  .ptext {{ color: {MUTED}; }}
  .hl {{
    background: rgba(255, 77, 77, 0.16);
    border-bottom: 2.5px solid {ACCENT};
    border-radius: 3px; padding: 0 2px;
  }}
  .hl2 {{
    background: rgba(227, 179, 65, 0.16);
    border-bottom: 2.5px solid #e3b341;
    border-radius: 3px; padding: 0 2px;
  }}
  .card.abl-reroute {{ border: 1px solid rgba(227, 179, 65, 0.45); }}
  .tag-reroute {{ color: #e3b341; }}
  .note {{ margin-top: 14px; font-size: 17px; color: {MUTED}; font-style: italic; }}
  .foot {{
    margin-top: 10px; font-size: 16px; color: {MUTED}; text-align: center;
    letter-spacing: 0.3px;
  }}
</style></head>
<body>
{pair_html(pair1)}
{pair_html(pair2)}
{pair_html(pair3)}
  <div class="foot">gemma-2-2b-it &nbsp;·&nbsp; identical prompt, seed &amp; sampling
  &nbsp;·&nbsp; only difference: the 25-feature coalition zeroed during generation</div>
</body></html>"""

    out = OUT / "cards.html"
    out.write_text(page)
    print(f"wrote {out}")
    return out


def screenshot_cards(html_path):
    """Best-effort local render of cards.html -> before_after_cards.png.

    The canonical path uses Playwright (any flavour). If the Python package
    is not installed, fall back to headless Chrome; otherwise print manual
    instructions.
    """
    png = OUT / "before_after_cards.png"
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        if chrome.exists():
            import subprocess
            subprocess.run(
                [str(chrome), "--headless", "--disable-gpu",
                 "--window-size=1600,3200", "--hide-scrollbars",
                 f"--screenshot={png}", html_path.as_uri()],
                check=True, capture_output=True)
            # trim the empty canvas below the content (equivalent of a
            # Playwright full-page screenshot)
            from PIL import Image
            img = Image.open(png)
            px = img.convert("RGB")
            bg = px.getpixel((2, px.height - 2))
            bottom = px.height
            while bottom > 1 and all(
                    px.getpixel((x, bottom - 1)) == bg
                    for x in range(0, px.width, 40)):
                bottom -= 1
            img.crop((0, 0, img.width, min(px.height, bottom + 40))).save(png)
            print(f"wrote {png} (headless Chrome, trimmed to content)")
        else:
            print("playwright not installed and Chrome not found - "
                  f"open {html_path} at 1600px wide and screenshot it to {png}")
        return
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1200})
        page.goto(html_path.as_uri())
        page.screenshot(path=str(png), full_page=True)
        browser.close()
    print(f"wrote {png}")


if __name__ == "__main__":
    fig_ladder()
    fig_atlas()
    cards = build_cards_html()
    screenshot_cards(cards)
