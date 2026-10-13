"""Generate human-readable names for each Leiden community.

For each of the ~18 communities in the L20/16k SAE, look at the top features
by activation density, pick a representative label, and write a short name.

Writes web/demo/community_names.json: {cid: {name, size, exemplar_labels}}
"""
from __future__ import annotations
import json
import pathlib
from collections import Counter

from neograph.cypher import NeographClient

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "web" / "demo" / "community_names.json"


def main():
    with NeographClient() as c:
        rows = c.run(
            """
            MATCH (f:SAEFeature) WHERE f.sae_id CONTAINS 'L20/16k'
              AND f.communityId IS NOT NULL
            OPTIONAL MATCH (f)-[:LABELED_AS {primary: true}]->(l:AutoInterpLabel)
            RETURN f.index AS idx, f.communityId AS cid,
                   f.activation_density AS density, l.text AS label
            """
        )
    by_cid: dict[int, list] = {}
    for r in rows:
        by_cid.setdefault(int(r["cid"]), []).append(r)

    out = {}
    for cid in sorted(by_cid):
        feats = by_cid[cid]
        feats.sort(key=lambda r: -(r["density"] or 0))
        top_labels = [(r["label"] or "(no label)") for r in feats[:8] if r["label"]]
        # Quick heuristic: pick the top-density feature's label as the name,
        # truncated to a short phrase.
        name = top_labels[0] if top_labels else f"Community {cid}"
        # Trim to a snappier form
        if "," in name:
            name = name.split(",")[0]
        if " in " in name:
            name = name.split(" in ")[0]
        if " related to " in name:
            name = name.split(" related to ")[1]
        if " of " in name and len(name) > 30:
            name = name.split(" of ")[0]
        name = name.strip().rstrip(".").lower()
        # Truncate long phrases
        if len(name) > 60:
            name = name[:60].rsplit(" ", 1)[0] + "…"
        out[str(cid)] = {
            "cid": cid,
            "size": len(feats),
            "name": name,
            "exemplar_labels": top_labels[:5],
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"→ {OUT}")
    print(f"  {len(out)} communities named")
    for cid, info in out.items():
        print(f"  cid {cid:>3} ({info['size']:>5} feats): {info['name']}")


if __name__ == "__main__":
    main()
