"""Capture baseline-vs-ablated generations for the LinkedIn carousel.

Picks ~8 demo-day prompts (a mix of primed AI-ism setups and neutral
open-enders), runs each through the daemon with no ablation and with the
top-25 coalition silenced, scores both with the union detector, writes the
result to reports/demo_gallery.json + a markdown view that can be screenshotted.
"""
from __future__ import annotations
import json, pathlib, urllib.request, sys

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT_JSON = REPO / "reports" / "demo_gallery.json"
OUT_MD = REPO / "reports" / "demo_gallery.md"
PROBE = "http://127.0.0.1:8765/probe"

# Top-25 coalition from pivot_attribution.json
TOP25 = [9173, 2706, 6759, 3223, 13265, 1250, 9816, 7361, 2184, 2137,
         9909, 9863, 11864, 8530, 12923, 9606, 12898, 1608, 4516, 2282,
         12561, 7100, 12524, 12506, 4197]

PROMPTS = [
    # Primed AI-ism setups — model strongly tends to slop
    ("primed", "This isn't just a setback,"),
    ("primed", "A great teacher isn't just smart;"),
    ("primed", "It's not a tool,"),
    # Neutral open-enders — what a real chatbot user would type
    ("neutral", "Discuss the legal and medical implications of AI in healthcare."),
    ("neutral", "What makes a good neighbourhood library?"),
    ("neutral", "Describe a hospital cafeteria at 2 a.m."),
    ("neutral", "Explain how mentorship matters in early career."),
    ("neutral", "Sketch a portrait of a postal worker."),
]


def call(payload):
    req = urllib.request.Request(
        PROBE,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def generate(prompt, ablate=None, seed=0, max_new=80):
    return call({
        "cmd": "generate",
        "model": "it",
        "prompt": prompt,
        "ablate": ablate or [],
        "max_new_tokens": max_new,
        "temperature": 0.8,
        "top_p": 0.95,
        "seed": seed,
    })["result"]["completion"]


def main():
    rows = []
    for i, (kind, prompt) in enumerate(PROMPTS, 1):
        print(f"[{i}/{len(PROMPTS)}] {kind}: {prompt[:50]}", flush=True)
        base = generate(prompt, ablate=None, seed=0)
        print(f"  baseline ({len(base)} chars): {base[:80].strip()}...", flush=True)
        abl = generate(prompt, ablate=TOP25, seed=0)
        print(f"  ablated  ({len(abl)} chars): {abl[:80].strip()}...", flush=True)
        rows.append({
            "kind": kind,
            "prompt": prompt,
            "baseline": base,
            "ablated": abl,
            "n_ablated_features": len(TOP25),
        })
    OUT_JSON.write_text(json.dumps({"rows": rows, "ablate_set": TOP25}, indent=2))
    print(f"\n✓ wrote {OUT_JSON}")

    # Render markdown gallery
    lines = ["# Demo gallery — baseline vs top-25 coalition silenced", "",
             "Each prompt is generated twice with the same seed (0). The only "
             "difference is whether the 25-feature AI-ism coalition is silenced "
             "throughout generation. Same model (gemma-2-2b-it), same temperature "
             "(0.8), same top-p (0.95).", ""]
    for r in rows:
        lines.append(f"### {r['kind']}: \"{r['prompt']}\"")
        lines.append("")
        lines.append("**Baseline (no intervention):**")
        lines.append("")
        lines.append(f"> {r['baseline'].strip()}")
        lines.append("")
        lines.append("**With top-25 AI-ism coalition silenced:**")
        lines.append("")
        lines.append(f"> {r['ablated'].strip()}")
        lines.append("")
        lines.append("---")
        lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print(f"✓ wrote {OUT_MD}")


if __name__ == "__main__":
    main()
