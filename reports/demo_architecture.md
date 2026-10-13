# Live feature-intervention demo — architecture sketch

*Companion to the Medium post. The demo is the experiment, not a separate
deliverable. It's how the redundancy thesis becomes legible.*

## The one design decision

The model and the graph are two organs. Don't conflate them.

```
┌──────────────────┐   ablation set   ┌─────────────┐   stream     ┌──────────┐
│  Cloud           │ ───────────────▶ │  Graph      │ ────────────▶│  Model   │
│  (Three.js)      │   (Cypher write) │  (Neo4j)    │   (poll)     │  service │
│                  │                  │             │              │  (Py)    │
│                  │ ◀──────────────  │             │ ◀────────────│          │
└──────────────────┘   layout + state └─────────────┘   activations└──────────┘
```

**Single contract that keeps the system coherent:** *the graph holds the
current ablation set. The model service reads it on every forward pass.* Any
client that wants to intervene writes to the graph, never to the model.

That contract buys three things for free:

1. **Replay.** Every intervention is a Cypher snapshot. Share an intervention
   as a link, restore it as a Cypher merge.
2. **Composability.** Click, voice, batch script, and pre-registered
   experiments all hit the same write path. No drift between "the demo" and
   "the science."
3. **Auditability.** Every claim in the Medium post is reproducible by
   restoring a graph state and re-running the model. The honesty contract
   from PRD §7 becomes operational.

## Three tiers — each proves something specific

### Tier 1 — **Watch** (weekend)

SAE post-encode activations stream per token to the frontend; nodes light up
as features fire. No toggling. No Neo4j strictly required for v0 (a flat
manifest of feature indices + UMAP positions is enough).

**What it proves:** nothing causal. It's the hook. People see Gemma's
attention shift across a feature manifold as it writes, and they care about
the next bit. Ship this even if Tier 2 never lands.

**Cost:** wrap `deslop_demo.py`'s sample loop, add a WebSocket that emits
`{token_idx, top_k_active_features, magnitudes}`. Frontend: a UMAP-projected
point cloud, opacity = recent activation, color by community.

**Anti-feature:** don't render all 16,384 features. UMAP to 2D, draw
communities as Voronoi regions à la Cartographer, only the few hundred
active-in-this-generation features animate.

### Tier 2 — **Toggle** (the actual experiment, ~1 week)

Click a node or a community region → graph writes `(:SAEFeature)-[:ABLATED_IN]
->(:Session {id})` → model service polls the session's ablation set → re-runs
generation with the cluster zeroed at the SAE post-encode hook → streams the
new text plus new firing pattern back to the cloud.

**What it proves:** the redundancy thesis. The user runs the pre-registered
ablation ladder live and watches where the construction dies and where it
doesn't. The Medium post becomes a screen recording of that ladder being
executed.

**Pre-registered ladder to run inside the UI:**

| Step | Set | Hypothesis |
|---|---|---|
| 1 | single 3223 | partial F1 kill, no F3 effect (Phase 4 baseline) |
| 2 | decoder-cosine neighborhood of 3223 (top-9) | structural prior — should add little if causal-coalition ≠ decoder-similar |
| 3 | co-activation top partners of 3223 (top-9) | structural prior — same test from the other angle |
| 4 | pivot-attribution top-5 | first causal escalation |
| 5 | pivot-attribution top-10 | does the drop scale linearly? |
| 6 | pivot-attribution top-25 | localizable constellation or true Hydra? |
| 7 | random size-matched control (same N as the winning set) | rules out "any mass ablation suppresses P(pivot)" |

The user (or the screen recording) plays this in sequence. The graph's state
after each toggle is a saved snapshot the post can link to.

**Cost:** add `(:SAEFeature)-[:ABLATED_IN]->(:Session)` schema; model service
queries it before each generation step; frontend adds toggle + ladder-runner
UI.

### Tier 3 — **Ask** (only if Tier 2 lands well)

text2cypher over a small library of hypothesis templates:

- "ablate the decoder neighborhood of *3223*" → `MATCH (a:SAEFeature {index:
  $idx})-[r:DECODER_SIMILAR]-(b) WHERE r.cosine > $thresh RETURN b`
- "ablate the community containing the *negation* feature" → label-embedding
  similarity to find the anchor, then community lookup
- "show me what fires alongside *3223*" → highlight, don't ablate

**What it proves:** natural-language navigation of the 16k-feature search
space — which is what GraphRAG over neural-feature data actually is. Not a
parlor trick. The honest framing: NL gets you to the right
*neighborhood*, then you refine by clicking.

**Cost:** ~50 lines of LLM-call wrapper around a template library +
Cypher generation; share the same write path as Tier 2.

**Skip voice for v1.** It's on-brand for Theo, but adds an ASR failure mode
to a demo whose whole appeal is "watch this happen reliably."

## What the graph genuinely buys

Not visualization. **Hypothesis generation.** The graph turns a flat
power-set search over 16,384 features into a constrained traversal over
structural priors:

- decoder-cosine neighborhood (single Cypher query)
- co-activation cluster (single Cypher query)
- Leiden community membership (already computed, single property lookup)
- intersection of (community of X) ∩ (community of Y) (one query)
- pivot-attribution top-N (load from JSON, but expressible as a Cypher view)

The joint-ablation experiment already finds that **decoder-cosine and
co-activation are not predictive** of causal-coalition membership for the
pivot decision (see [joint_ablation.md](joint_ablation.md)). That's a finding
*the graph generated* — the graph let us ask "is structural similarity the
right prior?" cheaply enough to discover the answer is no. A flat
attribution-ranked list could never have asked that question structurally.

## Honest scope

- **One model, 2B, one SAE, one layer.** Don't extrapolate.
- **Latency is not the issue.** 2B on MPS regenerates in a second or two —
  fine for live. The real gotchas are keeping graph-as-source-of-truth for
  ablation state (push to model, don't let them drift) and not rendering the
  full 16k node set.
- **The cloud is the analytical layer.** The model service is dumb (read
  ablation set, install hooks, generate). All policy lives in the graph
  layer.
- **Don't promise the construction dies.** Promise the instrument finds the
  answer. The pre-registered ladder either localizes the construction to a
  knowable set or demonstrates true Hydra — both are publishable.

## Minimum viable path

1. Write `scripts/streaming_generate.py` — wraps the deslop sample loop,
   emits per-token `{tokens, active_feature_indices, magnitudes}` over
   stdout/WebSocket.
2. Stand up a thin FastAPI service in front of it: `/generate` (POST prompt,
   stream tokens+activations), `/state` (read current ablation set from
   graph).
3. Add the `(:Session)-[:ABLATED_IN]-(:SAEFeature)` schema migration to
   `cypher/`.
4. Frontend: minimal Three.js scene, UMAP of 16k features pre-computed and
   served as a static JSON, animation on per-token messages.
5. Add a `/ablate {features: [...]}` endpoint that writes to the graph; the
   next `/generate` call picks it up.
6. Hard-code the seven-step ladder as a "Play the experiment" button.

Steps 1–4 are Tier 1. Steps 5–6 are Tier 2. Together they ARE the experiment
that's worth writing about — the rest of the post is commentary on what
happens when you run the ladder.
