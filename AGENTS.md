# Agent Instructions

This project treats the graph as the interpretability substrate, not as a decorative overlay on a 2D map.

## Product and Research Stance

- Do not frame isolated SAE atoms as complete concepts. Prefer language like "feature atoms," "feature groups," "activation-linked sets," and "behavior manifolds."
- Treat the 2D UMAP as an atlas for navigation only. It is a lossy projection of high-dimensional decoder vectors and should not be presented as the source of truth for semantic relatedness.
- Prefer graph relationships grounded in observed activations over decoder-vector similarity when explaining why features belong together. `CO_ACTIVATES_WITH` is the primary interactive relationship; `DECODER_SIMILAR` is a secondary diagnostic or fallback.
- Product copy should make relationship provenance explicit: co-activation, behavior membership, community membership, prompt-label retrieval, or decoder similarity.
- Goodfire/manifold-inspired direction: meaningful behaviors may be tiled across many localized or redundant SAE atoms. Build interfaces around coherent groups and relationship structure, not single "concept neurons."
- Edge or path vectors are allowed as an experimental phase-two idea. Keep them local to validated behavior groups/manifolds, test their causal effect, and avoid global path-vector infrastructure until there is evidence it improves steering or explanation.

## Implementation Bias

- Keep the production playground simple: presets first, one advanced dropdown, graph substrate visible only when it clarifies the intervention.
- When adding graph modes, favor a small local relationship view or explicit link overlay over drawing many global edges.
- If a relationship source is unavailable, degrade gracefully and label the fallback honestly.
