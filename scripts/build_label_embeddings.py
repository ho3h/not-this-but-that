"""Embed all 16,384 SAE feature labels with sentence-transformers, write to
Neo4j as `:AutoInterpLabel.embedding` (Float[]) with a vector index.

This is the index the runtime concept_retrieve command searches against.

One-shot. Re-run only if labels change or model changes. ~2 min on CPU.
"""
from __future__ import annotations
import time
import pathlib

import numpy as np
from sentence_transformers import SentenceTransformer

from neograph.config import EMBEDDING_MODEL, EMBEDDING_DIM
from neograph.cypher import NeographClient

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT_NPY = REPO / "reports" / "label_embeddings.npy"
OUT_IDX = REPO / "reports" / "label_embedding_idx.json"


def main():
    print(f"loading sentence-transformer: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("pulling all autointerp labels from Neo4j…")
    with NeographClient() as c:
        rows = c.run(
            """
            MATCH (f:SAEFeature) WHERE f.sae_id CONTAINS 'L20/16k'
            MATCH (f)-[:LABELED_AS {primary: true}]->(l:AutoInterpLabel)
            RETURN f.index AS idx, l.text AS text, id(l) AS label_node_id
            ORDER BY f.index
            """
        )
    print(f"  {len(rows)} (feature, label) pairs")

    texts = [r["text"] for r in rows]
    print("encoding…")
    t0 = time.perf_counter()
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True,
                         batch_size=128)
    print(f"  done in {time.perf_counter()-t0:.0f}s, shape={embs.shape}")

    # Save as numpy for fast in-memory load (no graph query needed at runtime)
    OUT_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_NPY, embs.astype(np.float32))
    print(f"→ {OUT_NPY} ({OUT_NPY.stat().st_size // 1024} KB)")

    # Sidecar: feature index alignment (row i of npy = feature texts[i] with idx rows[i].idx)
    import json
    OUT_IDX.write_text(json.dumps({
        "feature_indices": [int(r["idx"]) for r in rows],
        "label_node_ids":  [int(r["label_node_id"]) for r in rows],
        "labels":          texts,
        "model":           EMBEDDING_MODEL,
        "dim":             int(embs.shape[1]),
    }))
    print(f"→ {OUT_IDX}")

    # Also write into Neo4j for graph-side queries (vector index works alongside)
    print("writing embeddings into :AutoInterpLabel nodes…")
    with NeographClient() as c:
        # Vector index (Neo4j 5.x)
        c.run(
            "CREATE VECTOR INDEX label_embedding_idx IF NOT EXISTS "
            "FOR (l:AutoInterpLabel) ON (l.embedding) "
            "OPTIONS {indexConfig: {`vector.dimensions`: $dim, `vector.similarity_function`: 'cosine'}}",
            dim=int(embs.shape[1]),
        )
        # Batch write embeddings
        batch_size = 500
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            chunk_embs = embs[start:start + batch_size].tolist()
            params = [
                {"node_id": int(r["label_node_id"]), "emb": chunk_embs[i]}
                for i, r in enumerate(chunk)
            ]
            c.run(
                "UNWIND $rows AS row "
                "MATCH (l:AutoInterpLabel) WHERE id(l) = row.node_id "
                "SET l.embedding = row.emb",
                rows=params,
            )
            print(f"  wrote {start + len(chunk)}/{len(rows)}")

    print("\nDone. Index `label_embedding_idx` available for db.index.vector.queryNodes queries.")


if __name__ == "__main__":
    main()
