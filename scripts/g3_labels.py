"""Fetch Neuronpedia auto-interp labels for Gemma 3 4B-it Gemma Scope 2 features.

Caches to data/g3_labels_cache.json as {"<layer>": {"<idx>": "label"}}. Labels the
top 60 of each ranking in reports/g3_attribution_L{layer}.json.

  .venv/bin/python scripts/g3_labels.py 22 29 17
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "data/g3_labels_cache.json"
URL = "https://www.neuronpedia.org/api/feature/gemma-3-4b-it/{layer}-gemmascope-2-res-16k/{idx}"


def fetch(layer: int, idx: int) -> str:
    with urllib.request.urlopen(URL.format(layer=layer, idx=idx), timeout=30) as r:
        d = json.load(r)
    ex = d.get("explanations") or []
    if ex and ex[0].get("description"):
        return ex[0]["description"]
    pos = [t.strip() for t in (d.get("pos_str") or [])[:6] if t.strip()]
    return ("[no label; promotes: " + ", ".join(pos) + "]") if pos else ""


def main() -> None:
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    for layer in map(int, sys.argv[1:]):
        path = REPO / f"reports/g3_attribution_L{layer}.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        feats = {x["feature_idx"] for k in ("family", "cousin") for x in r[k]["ranked"][:60]}
        lc = cache.setdefault(str(layer), {})
        for f in sorted(feats):
            if lc.get(str(f)):
                continue
            try:
                lc[str(f)] = fetch(layer, f)
            except Exception as e:  # keep going; a missing label is not fatal
                lc[str(f)] = ""
                print(layer, f, "error", e, flush=True)
            time.sleep(0.15)
        CACHE.write_text(json.dumps(cache, indent=1))
        print(layer, len(lc), "labels", flush=True)


if __name__ == "__main__":
    main()
