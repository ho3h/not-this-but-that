"""Seed `:Behaviour` nodes + INCLUDES edges into Neo4j.

A Behaviour is a named coalition of features that, jointly ablated, suppress
a particular construction or chatbot tic. Currently:
  - "ai-ism" : the top-25 attribution coalition from this project's main finding

Schema:
  (:Behaviour {name, description, anchor_feature, discovered_via, created_at})
  (:Behaviour)-[:INCLUDES {weight, rank, cost_when_removed}]->(:SAEFeature)

The weight = the per-feature attribution drop (mean P(pivot) drop from
ablating just this feature). Rank = position in attribution top-N.
cost_when_removed = leave-one-out cost (how much the coalition loses without
this feature).

Re-run safely: MERGEs the Behaviour, replaces its INCLUDES edges. Idempotent.
"""
from __future__ import annotations
import json, pathlib
from neograph.cypher import NeographClient

REPO = pathlib.Path(__file__).resolve().parent.parent
ATTRIB_PATH = REPO / "reports" / "pivot_attribution.json"
LOO_PATH = REPO / "reports" / "q3_leave_one_out.json"


def seed_aiism():
    d = json.loads(ATTRIB_PATH.read_text())
    top25 = d["top_promotes_pivot"][:25]
    loo = json.loads(LOO_PATH.read_text())
    cost_by_feat = {r["removed"]: r["cost_of_removal"]
                    for r in (loo["leave_one_out"] or []) if r}

    features_rows = []
    for rank, r in enumerate(top25, start=1):
        idx = int(r["feature_idx"])
        features_rows.append({
            "idx": idx,
            "weight": float(r["mean_attribution_drop"]),
            "rank": rank,
            "cost_when_removed": float(cost_by_feat.get(idx, 0.0)),
            "label": r.get("label", ""),
        })

    description = (
        "The 25 SAE features that, ablated jointly, kill the 'not X but Y' "
        "construction (AI-ism) by 72% relative at the pivot decision and 80% "
        "in actual generation. Top-2 (3223, 9909) are individually "
        "indispensable; the rest are substitutable supporters."
    )
    with NeographClient() as c:
        c.run(
            """
            MERGE (b:Behaviour {name: $name})
              SET b.description = $description,
                  b.anchor_feature = $anchor,
                  b.discovered_via = 'per-feature causal attribution to P(pivot) at truncated D1 prefixes',
                  b.coalition_size = $size,
                  b.created_at = datetime()
            """,
            name="ai-ism", description=description, anchor=3223,
            size=len(features_rows),
        )
        # Replace edges (clean slate so re-runs don't accumulate)
        c.run(
            "MATCH (b:Behaviour {name: $name})-[r:INCLUDES]->() DELETE r",
            name="ai-ism",
        )
        c.run(
            """
            UNWIND $rows AS row
            MATCH (b:Behaviour {name: $name})
            MATCH (f:SAEFeature {index: row.idx})
              WHERE f.sae_id CONTAINS 'L20/16k'
            MERGE (b)-[r:INCLUDES]->(f)
              SET r.weight = row.weight,
                  r.rank = row.rank,
                  r.cost_when_removed = row.cost_when_removed
            """,
            name="ai-ism", rows=features_rows,
        )
        # Verify
        n = c.run(
            "MATCH (:Behaviour {name: $name})-[r:INCLUDES]->() RETURN count(r) AS n",
            name="ai-ism",
        )[0]["n"]
        print(f"✓ Seeded Behaviour 'ai-ism' with {n} INCLUDES edges")
        # Show a few
        rows = c.run(
            """
            MATCH (b:Behaviour {name: $name})-[r:INCLUDES]->(f:SAEFeature)
            OPTIONAL MATCH (f)-[:LABELED_AS {primary: true}]->(l:AutoInterpLabel)
            RETURN f.index AS idx, r.rank AS rank, r.weight AS w,
                   r.cost_when_removed AS cost, l.text AS label
            ORDER BY r.rank LIMIT 6
            """,
            name="ai-ism",
        )
        print("  top 6 by rank:")
        for r in rows:
            print(f"    #{r['rank']:>2}  feat {r['idx']:>5}  weight={r['w']:+.5f}  cost={r['cost']:+.5f}  {(r['label'] or '')[:50]}")


def seed_concept_behaviour(name: str, description: str, query: str,
                            k: int = 25):
    """Seed a Behaviour via concept retrieval (label-embedding similarity).

    Less rigorous than the AI-ism coalition (which was discovered via causal
    attribution), but useful for the Demo 2 mixer where users want a handful
    of named-coalition dials to play with. Each is "the top-K features whose
    labels match the query."
    """
    import urllib.request, json as _json
    PROBE = "http://127.0.0.1:8765/probe"
    req = urllib.request.Request(PROBE,
        data=_json.dumps({"cmd": "concept_retrieve", "prompt": query, "k": k}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = _json.loads(r.read())
    if not resp.get("ok"):
        print(f"  ! concept_retrieve failed: {resp.get('error')}")
        return
    matches = resp["result"]["matches"]
    rows = [{"idx": int(m["idx"]), "weight": float(m["score"]),
             "rank": rank, "cost_when_removed": 0.0, "label": m["label"]}
            for rank, m in enumerate(matches, start=1)]
    with NeographClient() as c:
        c.run(
            """
            MERGE (b:Behaviour {name: $name})
              SET b.description = $description,
                  b.discovered_via = 'concept retrieval (label-embedding cosine match)',
                  b.seed_query = $seed_q,
                  b.coalition_size = $size,
                  b.created_at = datetime()
            """,
            name=name, description=description, seed_q=query, size=len(rows),
        )
        c.run(
            "MATCH (b:Behaviour {name: $name})-[r:INCLUDES]->() DELETE r",
            name=name,
        )
        c.run(
            """
            UNWIND $rows AS row
            MATCH (b:Behaviour {name: $name})
            MATCH (f:SAEFeature {index: row.idx})
              WHERE f.sae_id CONTAINS 'L20/16k'
            MERGE (b)-[r:INCLUDES]->(f)
              SET r.weight = row.weight, r.rank = row.rank
            """,
            name=name, rows=rows,
        )
        print(f"✓ Seeded Behaviour '{name}' with {len(rows)} INCLUDES edges (concept-retrieved)")


if __name__ == "__main__":
    seed_aiism()
    print()
    # Three more behaviours seeded via concept retrieval (less rigorous than
    # the AI-ism coalition's causal attribution, but useful as named knobs
    # for the Demo 2 style mixer)
    seed_concept_behaviour(
        "bullets",
        "Features semantically related to bullet lists, enumeration, and structured list output",
        "bullet list enumeration structured items separated dash asterisk",
        k=25,
    )
    seed_concept_behaviour(
        "hedging",
        "Features semantically related to caveats, qualifiers, uncertainty, and hedging language",
        "perhaps maybe might could possibly uncertain caveat qualifier",
        k=25,
    )
    seed_concept_behaviour(
        "formal_register",
        "Features semantically related to formal, academic, register-elevated language",
        "furthermore moreover additionally therefore consequently herein",
        k=25,
    )
