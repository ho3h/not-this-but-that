# LinkedIn post — drafts

LinkedIn's hard limit is ~3,000 chars. Below: a punchy version (~1,800 chars,
recommended) and a longer storytelling version (~2,800 chars). Pick one.

---

## Draft A — Punchy (1,800 chars) · recommended

I spent a month inside a chatbot's head trying to find the neuron that makes AI write like AI.

You know the sentence I mean. "This isn't a setback — it's a springboard." "It's not a tool, it's a partner." Every chatbot, every prompt, the same shape.

The published mech-interp playbook says behaviours like this live in *one feature*. Refusal does. Sycophancy does. So I went looking for the AI-ism feature in Gemma 2 2B.

I found it on day one. Feature 3223 — labelled "phrases conveying exceptions or negations." Beautiful. I zeroed it.

The model still produced the construction. Differently. It rerouted.

It turns out the AI-ism doesn't live in one feature. It lives in a coalition of twenty-five. Two of them are doing the heavy lifting (the negation feature + a "digital tech" feature, of all things). The other twenty-three are redundant supporters — substitutable, but together they cover every escape route.

Ablate all twenty-five together: the construction rate on 306 neutral chatbot-style prompts drops from 12% to 2%. That's an 80% relative drop, p < 10⁻⁶, bootstrap CI [+67%, +92%]. The model loses the tic. Perplexity moves 1.08× — small.

Then I built three demos that I think only work because everything is in a graph:

1️⃣ **Surgical de-slop** — embed your prompt, intersect with the coalition via Cypher, silence only the overlap. RAG-for-activations.

2️⃣ **Mix your own chatbot** — sliders for ai-ism / bullets / hedging / formal-register. Each is a named :Behaviour subgraph. The composer is one Cypher union.

3️⃣ **Audit trail** — every silenced feature has a graph-traceable lineage. *Why did the model say that?* is one MATCH query.

Mech-interp + a knowledge graph = composable, auditable model steering. Open source.

Full writeup + reproducible code: [github.com/theohopkinson/not-this-but-that]

#MechanisticInterpretability #AI #Neo4j #SAE

---

## Draft B — Storytelling (2,800 chars)

There's a sentence I want you to read out loud:

"This isn't a setback, it's a springboard."

You've seen that sentence five hundred times this year. You couldn't pick the chatbot that wrote it out of a lineup, because every chatbot writes it. Different topic, same shape. *It isn't X, it's Y.* The AI-ism. The deep tell.

I wanted to know if I could turn it off — not by prompting, but by reaching inside the model.

The published mech-interp playbook says behaviours like this live in a *single feature*. Refusal does. Sycophancy does. So I scanned all 16,384 SAE features in Gemma 2 2B for the one that drives the construction.

I found it on day one. Feature 3223 — *"phrases conveying exceptions or negations."* Literal negation. Beautiful. I projected it out of every layer of the residual stream — the same trick that erased refusal from a dozen chat models last year.

The first generation under the kill, on a prompt about Antarctica:

> "Imagine a world, **not of green meadows and warm breezes, but of endless white**, the sun a distant memory…"

The intervention designed to make the construction impossible *opened* with the construction.

So I did the unfashionable thing: I tried 25 features at once.

The AI-ism doesn't live in one feature. It lives in a coalition. Two cores (negation + digital-tech, of all things) doing about a third of the work each, plus twenty-three redundant supporters. Ablate just the two cores: 78% drop. Ablate all twenty-five: 80% drop on neutral chatbot prompts (12% → 2%, n=306, p < 10⁻⁶, bootstrap CI [+67%, +92%]). The other twenty-three are mostly cleanup. The model loses the tic without losing fluency (perplexity 1.08×).

Behaviours have *coalition addresses*, not switch addresses. And the prior you need to find the coalition is causal, not structural — the graph's decoder-neighbours and Leiden-community guesses both failed clean.

Then I built three demos on top of the graph:

1️⃣ **Surgical de-slop** — embed the prompt, intersect with the coalition via Cypher set algebra, silence only the overlap. RAG-for-activations: retrieval into the model's internal concept space instead of into a document corpus.

2️⃣ **Mix your own chatbot** — sliders for ai-ism / bullets / hedging / formal register. Each is a 25-feature :Behaviour subgraph in Neo4j. The composer is one weighted Cypher UNION.

3️⃣ **Audit trail** — every silenced feature gets a graph-traceable lineage node. *Why did the model say that?* is one MATCH query weeks later.

Mech-interp + knowledge graphs = composable, auditable, glass-box model steering. The graph isn't the visualisation; it's the substrate.

Open source. Full writeup + reproducible code:
github.com/theohopkinson/not-this-but-that

#MechanisticInterpretability #AI #Neo4j #SAE #LLM

---

## Posting checklist

- [ ] Replace `[github.com/theohopkinson/not-this-but-that]` with the real public URL
- [ ] If repo is private, add a public link to the Medium post instead
- [ ] Decide between Draft A (punchy) and Draft B (storytelling)
- [ ] Attach demo screenshot or GIF (capture from playground UI)
- [ ] Consider posting Wednesday-Thursday 9-11am for max reach
- [ ] Cross-post a shortened version to X/Twitter with a thread

## Suggested image attachments

1. Screenshot of the playground showing the 16,384-dot UMAP map with the 25 coalition features highlighted in red
2. Screenshot of the Demo 1 surgical-deslop output ("2 AI-ism features silenced — the overlap of 80 prompt-concepts ∩ 25 coalition features")
3. Screenshot of the Mixer with sliders
4. Side-by-side baseline vs ablated for one of the cleaner kills (e.g. "This isn't a setback" → "it's a turning point" rather than "it's a springboard")
