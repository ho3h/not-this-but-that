"""LLM-judge coherence rating.

A small jury (Anthropic Claude via the existing `anthropic` dep, plus
optionally one OSS judge for triangulation) rates intervened generations on a
1-5 coherence scale. The prompt template lives in `data/judge_template.txt`
(written in Phase 5).
"""

from __future__ import annotations


def coherence_rating(generation: str) -> float:
    """Return a 1-5 LLM-judge coherence score for `generation`. Phase 5 stub."""
    raise NotImplementedError("Phase 5.")
