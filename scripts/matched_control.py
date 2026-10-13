"""Firing-rate-matched control set for a feature set (same procedure as decision_point_ladder.py).

For each feature in the set, in order, pick at random (random.Random(7)) one of the
10 non-set features nearest to it in firing rate. Densities: web/demo/layout.json
(Gemma 2 layer 20) or reports/g3_densities.json (Gemma 3, per layer).

  .venv/bin/python scripts/matched_control.py gemma2 reports/specific_selection_gemma2.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def matched(features: list[int], density: dict[int, float], n_total: int = 16384, seed: int = 7) -> list[int]:
    rng = random.Random(seed)
    pool = [f for f in range(n_total) if f not in set(features)]
    by_density = sorted(pool, key=lambda f: density.get(f, 0.0))
    dens = np.array([density.get(f, 0.0) for f in by_density])
    out: list[int] = []
    for f in features:
        j = int(np.searchsorted(dens, density.get(f, 0.0)))
        cands = [by_density[k] for k in range(max(0, j - 5), min(len(by_density), j + 5)) if by_density[k] not in out]
        out.append(rng.choice(cands))
    return sorted(out)


def densities(which: str, layer: int) -> dict[int, float]:
    if which == "gemma2":
        return {int(f["idx"]): float(f.get("density", 0.0)) for f in json.loads((REPO / "web/demo/layout.json").read_text())["features"]}
    return dict(enumerate(json.loads((REPO / "reports/g3_densities.json").read_text())[str(layer)]))


if __name__ == "__main__":
    which, path = sys.argv[1], sys.argv[2]
    sel = json.loads((REPO / path).read_text())
    print(json.dumps(matched(sel["specific_set"], densities(which, sel["layer"]))))
