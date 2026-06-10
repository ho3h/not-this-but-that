"""Bake the playground's attribution presets for static (daemon-less) hosting.

The playground fetches its two preset feature sets from the probe daemon:

    {"cmd": "attribution", "top_n": 25, "kind": "promote"}   -> TOP25
    {"cmd": "attribution", "top_n": 10, "kind": "suppress"}  -> SUPPRESSORS10

On a static host (Vercel) POST /probe 404s, so this script pre-computes the
exact `result` payloads cmd_attribution would return (see
scripts/probe_daemon.py::ProbeEngine.cmd_attribution) and writes them to
web/demo/attribution_static.json as:

    {"promote_top25": <result>, "suppress_top10": <result>}

playground.js falls back to this file when the probe call fails.

Usage:  .venv/bin/python scripts/bake_attribution_static.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ATTRIB_PATH = REPO_ROOT / "reports" / "pivot_attribution.json"
LABELS_PATH = REPO_ROOT / "data" / "labels_cache.json"
OUT_PATH = REPO_ROOT / "web" / "demo" / "attribution_static.json"


def load_labels() -> dict[int, str]:
    if not LABELS_PATH.exists():
        return {}
    raw = json.loads(LABELS_PATH.read_text())
    return {int(k): v.get("text", "") for k, v in raw.items()}


def attribution_result(data: dict, labels: dict[int, str],
                       top_n: int, kind: str) -> dict:
    """Mirror ProbeEngine.cmd_attribution exactly (no slice arg needed:
    the playground only ever sends top_n + kind)."""
    if "full_ranked_by_score" in data:
        full = list(data["full_ranked_by_score"])
        if kind == "suppress":
            full = sorted(full, key=lambda r: r["scored"])   # most-negative first
        else:
            full = sorted(full, key=lambda r: -r["scored"])  # most-positive first
    else:
        key = "top_promotes_pivot" if kind == "promote" else "top_suppresses_pivot"
        full = data[key]
    rows = full[:top_n]
    return {
        "features": [int(r["feature_idx"]) for r in rows],
        "details": [
            {"index": int(r["feature_idx"]),
             "mean_drop": float(r["mean_attribution_drop"]),
             "n_active": int(r["n_prompts_active"]),
             "score": float(r["scored"]),
             "label": labels.get(int(r["feature_idx"]), r.get("label", ""))}
            for r in rows
        ],
        "n_with_signal": int(data.get("n_features_with_signal", len(full))),
    }


def main() -> None:
    data = json.loads(ATTRIB_PATH.read_text())
    labels = load_labels()
    out = {
        # Keys mirror the two queries playground.js makes on init.
        "promote_top25": attribution_result(data, labels, top_n=25, kind="promote"),
        "suppress_top10": attribution_result(data, labels, top_n=10, kind="suppress"),
    }
    OUT_PATH.write_text(json.dumps(out, indent=1) + "\n")
    p = out["promote_top25"]["features"]
    s = out["suppress_top10"]["features"]
    print(f"wrote {OUT_PATH}")
    print(f"  promote_top25: {len(p)} features, head {p[:5]}")
    print(f"  suppress_top10: {len(s)} features, head {s[:5]}")


if __name__ == "__main__":
    main()
