"""Seed `:Behaviour` nodes + INCLUDES edges into Neo4j.

A Behaviour is a named coalition of features that, jointly ablated, suppress
a particular construction or chatbot tic. Currently:
  - "ai-ism" : the 25 chat-model features confirmed in the pre-registered
               confirmation v3 (reports/confirmation_v3_prereg.json)

Schema:
  (:Behaviour {name, description, anchor_feature, discovered_via, created_at})
  (:Behaviour)-[:INCLUDES {weight, rank, cost_when_removed}]->(:SAEFeature)

The weight = the per-feature attribution drop at the decision point (mean drop
in P(commit) from switching off just this feature, family or cousin, whichever
is larger; reports/reroute_attribution.json). Rank = position in the
registered list. cost_when_removed = leave-one-out cost on held-out decision
points (reports/decision_point_ladder.json, family).

Re-run safely: MERGEs the Behaviour, replaces its INCLUDES edges. Idempotent.
"""
from __future__ import annotations
import json, pathlib
from neograph.cypher import NeographClient

REPO = pathlib.Path(__file__).resolve().parent.parent
PREREG_PATH = REPO / "reports" / "confirmation_v3_prereg.json"
ATTRIB_PATH = REPO / "reports" / "reroute_attribution.json"
LOO_PATH = REPO / "reports" / "decision_point_ladder.json"
LABELS_PATH = REPO / "data" / "labels_cache.json"


def seed_aiism():
    chat25 = json.loads(PREREG_PATH.read_text())["chat_features"]
    attrib = json.loads(ATTRIB_PATH.read_text())
    drop = {}
    for kind in ("family", "cousin"):
        for r in attrib[kind]["ranked"]:
            i = int(r["feature_idx"])
            drop[i] = max(drop.get(i, 0.0), float(r["mean_drop"]))
    loo = json.loads(LOO_PATH.read_text())["kinds"]["family"]["leave_one_out_cost"]
    cost_by_feat = {int(f): float(c) for f, c in loo}
    labels = json.loads(LABELS_PATH.read_text())

    features_rows = []
    for rank, idx in enumerate(chat25, start=1):
        lab = labels.get(str(idx), "")
        features_rows.append({
            "idx": int(idx),
            "weight": drop.get(int(idx), 0.0),
            "rank": rank,
            "cost_when_removed": cost_by_feat.get(int(idx), 0.0),
            "label": lab.get("text", "") if isinstance(lab, dict) else lab,
        })

    description = (
        "The 25 layer-20 features chosen in gemma-2-2b-it at the token where it "
        "commits to 'isn't just' / 'more than just'. Switched off on 102 held-out "
        "prompts (pre-registered confirmation v3), the construction or its cousin "
        "fell from 88 answers to 3 (-97%) with no measurable cost."
    )
    with NeographClient() as c:
        c.run(
            """
            MERGE (b:Behaviour {name: $name})
              SET b.description = $description,
                  b.anchor_feature = $anchor,
                  b.discovered_via = 'per-feature causal attribution to P(commit) at decision points in chat answers',
                  b.coalition_size = $size,
                  b.created_at = datetime()
            """,
            name="ai-ism", description=description, anchor=chat25[0],
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
