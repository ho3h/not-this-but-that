"""Precompute the 2D feature-cloud layout for the demo frontend.

UMAP the L20 SAE decoder vectors (16384 features × 2304 dims) to 2D. Pulls
each feature's community id and label from Neo4j so the frontend can colour
and label. Writes `web/demo/layout.json` and `web/demo/communities.json`.

Run once; cheap (~1 min on CPU). Re-run only when you change the SAE.
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import umap
from sae_lens import SAE

from neograph.config import SAE as GS
from neograph.cypher import NeographClient

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "web" / "demo"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("loading SAE on CPU…")
    sae = SAE.from_pretrained(release=GS.release, sae_id=GS.sae_id, device="cpu")
    if isinstance(sae, tuple):
        sae = sae[0]
    decoder = sae.W_dec.detach().numpy()  # (16384, 2304)
    print(f"  decoder shape: {decoder.shape}")

    print("pulling per-feature metadata from Neo4j…")
    with NeographClient() as c:
        rows = c.run(
            """
            MATCH (f:SAEFeature) WHERE f.sae_id CONTAINS 'L20/16k'
            OPTIONAL MATCH (f)-[:LABELED_AS {primary: true}]->(l:AutoInterpLabel)
            RETURN f.index AS idx, f.communityId AS cid,
                   f.activation_density AS density, l.text AS label
            ORDER BY f.index
            """
        )
    by_idx = {int(r["idx"]): r for r in rows}
    print(f"  pulled {len(by_idx)} features")

    print("running UMAP (this is the slow bit — ~30-60s)…")
    t0 = time.perf_counter()
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.05,
                         metric="cosine", random_state=42, low_memory=True)
    coords = reducer.fit_transform(decoder)
    print(f"  done in {time.perf_counter()-t0:.0f}s")

    # Normalise to [-1, 1] for convenient frontend rendering
    xs = coords[:, 0]; ys = coords[:, 1]
    xs = 2 * (xs - xs.min()) / (xs.max() - xs.min()) - 1
    ys = 2 * (ys - ys.min()) / (ys.max() - ys.min()) - 1

    # Per-feature record
    features = []
    for i in range(decoder.shape[0]):
        meta = by_idx.get(i, {})
        features.append({
            "idx": i,
            "x": float(xs[i]),
            "y": float(ys[i]),
            "cid": int(meta.get("cid", -1)) if meta.get("cid") is not None else -1,
            "density": float(meta.get("density", 0.0) or 0.0),
            "label": (meta.get("label") or "")[:80],
        })

    # Community centroids (for region rendering)
    by_cid: dict[int, list] = {}
    for f in features:
        by_cid.setdefault(f["cid"], []).append(f)
    communities = []
    for cid, feats in sorted(by_cid.items()):
        cx = float(np.mean([f["x"] for f in feats]))
        cy = float(np.mean([f["y"] for f in feats]))
        communities.append({"cid": cid, "count": len(feats), "cx": cx, "cy": cy})

    layout_path = OUT_DIR / "layout.json"
    comm_path = OUT_DIR / "communities.json"
    layout_path.write_text(json.dumps({"features": features}))
    comm_path.write_text(json.dumps({"communities": communities}))
    print(f"\n→ {layout_path}  ({layout_path.stat().st_size // 1024} KB)")
    print(f"→ {comm_path}    ({comm_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
