"""Meaning preservation via sentence-embedding cosine.

The embedding model is the same MiniLM the neograph substrate uses
(see neograph.config.EMBEDDING_MODEL) — reused so the embedding-dim
constraint stays consistent across the project.
"""

from __future__ import annotations


def semantic_similarity(a: str, b: str) -> float:
    """Cosine similarity between MiniLM embeddings of `a` and `b`. Phase 5 stub."""
    raise NotImplementedError("Phase 5.")
