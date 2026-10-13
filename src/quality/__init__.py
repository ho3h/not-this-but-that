"""Quality preservation evaluation (M3).

Three checks gate any intervention's product claim:
  - fluency   : held-out perplexity under the intervened model
  - coherence : LLM-judge 1-5 rating of intervened generations
  - meaning   : embedding-cosine between original and de-slopped output

The mechanism claim survives an M3 failure; the de-slop product claim does not.
"""

from __future__ import annotations

from quality.fluency import perplexity
from quality.meaning import semantic_similarity

__all__ = ["perplexity", "semantic_similarity"]
