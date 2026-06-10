# LinkedIn post — drafts

LinkedIn's hard limit is ~3,000 chars. Below: a punchy version (~2,100 chars,
recommended) and a longer storytelling version (~2,900 chars). Pick one.
Numbers revised 2026-06-09 after the permissive-detector fix
(see `reports/permissive_fix_audit.md`) — these match the Medium post.

---

## Draft A — Punchy (~2,100 chars) · recommended

I spent a month inside a chatbot's head trying to find the neuron that makes AI write like AI.

You know the sentence I mean. "This isn't a setback — it's a springboard." "It's not a tool, it's a partner." Every chatbot, every prompt, the same shape.

The mech-interp playbook says behaviours like this live in *one feature*. Refusal famously does. So I went looking for the AI-ism feature in Gemma 2 2B.

I found it on day one. Feature 3223 — labelled "phrases conveying exceptions or negations." Beautiful. I zeroed it.

The model still produced the construction. It rerouted.

It turns out the AI-ism doesn't live in one feature. It lives in a coalition of twenty-five. Two do the heavy lifting (the negation feature + a "digital tech" feature, of all things); twenty-three keep the kill killed while the model writes. Silence all twenty-five at once and, on 306 neutral chatbot-style prompts:

→ the textbook form ("isn't X, it's Y") drops 93%
→ the whole negated family drops 44% (odds that's luck: ~1 in 85)
→ and the model's favourite escape — the affirmative cousin "it's more than just X. It's Y" — actually goes UP. Count everything that smells like the tic: −25%.
→ fluency intact: same length, same perplexity. Primed with "It's not a tool", the model completes the construction 89% of the time at baseline, 40% silenced.

Honesty note, because it's the best part: an earlier draft claimed an 80% drop. I audited my own detector twice, and each fix revealed another escape route the model was using. The number fell; the finding got better. You can't delete a habit by deleting its grammar — the model re-expresses the intention through whatever grammar is left. Every dropped and added detection is hand-auditable in the repo.

Then I built three demos that only work because everything lives in a Neo4j graph:

1️⃣ Surgical de-slop — embed your prompt, intersect with the coalition via Cypher, silence only the overlap. RAG-for-activations.

2️⃣ Mix your own chatbot — sliders for ai-ism / bullets / hedging / formal register. Each is a named :Behaviour subgraph; the composer is one Cypher UNION.

3️⃣ Audit trail — every silenced feature has graph-traceable lineage. "Why did the model say that?" is one MATCH query.

Mech-interp + a knowledge graph = composable, auditable model steering. Open source — writeup, receipts, and a paste-your-own-prose slop-o-meter:

github.com/ho3h/not-this-but-that

#MechanisticInterpretability #AI #Neo4j #SAE

---

## Draft B — Storytelling (~2,900 chars)

There's a sentence I want you to read out loud:

"This isn't a setback, it's a springboard."

You've seen that sentence five hundred times this year. You couldn't pick the chatbot that wrote it out of a lineup, because every chatbot writes it. Different topic, same shape. *It isn't X, it's Y.* The AI-ism. The deep tell.

I wanted to know if I could turn it off — not by prompting, but by reaching inside the model.

The mech-interp playbook says behaviours like this live in a *single feature*. Refusal famously does. So I scanned all 16,384 SAE features in Gemma 2 2B for the one that drives the construction.

I found it on day one. Feature 3223 — *"phrases conveying exceptions or negations."* Literal negation. I projected it out of every layer of the residual stream — the same trick that erased refusal from a dozen chat models.

The first generation under the kill, on a prompt about Antarctica:

"Imagine a world, NOT of green meadows and warm breezes, BUT of endless white…"

The intervention designed to make the construction impossible *opened* with the construction. Ninety generations later it had removed almost exactly nothing.

So I did the unfashionable thing: I went looking for the rest of the coalition.

The AI-ism lives in twenty-five features. Two cores (negation + digital-tech, of all things) doing a third of the work each, plus twenty-three supporters that keep the kill killed while the model writes. Silence all twenty-five on 306 neutral prompts: the textbook form drops 93% (14 hits → 1); the whole negated family drops 44% (~1 in 85 by luck; nine prompts changed status, eight toward clean). The model stays fluent — it just stops pivoting every claim against a strawman.

And then the twist: the model reroutes. Its favourite escape is an affirmative cousin — "it's more than just X. It's Y" — which *rises* under the kill. Count everything that smells like the tic and the net drop is 25%. You can't delete a habit by deleting its grammar; the model re-expresses the intention through whatever grammar remains. What those features really control is the model's *contrast machinery* — "but" falls from 13% of generations to 1% — and the AI-ism is just that machinery's loudest output. The graph's structural guesses at coalition membership (decoder neighbours, Leiden communities) all failed; only direct causal attribution found it. Behaviours have *coalition addresses*, and you need causal priors to find them.

Then I built three demos on the graph: surgical de-slop (prompt ∩ coalition via Cypher), a mix-your-own-chatbot slider board (each behaviour a :Behaviour subgraph), and an audit trail where "why did the model say that?" is one MATCH query.

Open source — full writeup, every receipt, and a slop-o-meter you can paste your own prose into:

github.com/ho3h/not-this-but-that

#MechanisticInterpretability #AI #Neo4j #SAE #LLM

---

## Posting checklist

- [ ] Decide between Draft A (punchy) and Draft B (storytelling)
- [ ] Attach the 30–60s playground screen recording (autoplay video outperforms links), or the figure pack below
- [ ] First comment: link the Medium post + the live demo URL (links in first comment, not the post body, preserves reach)
- [ ] Cross-post the X/Twitter thread (Draft C)

## Suggested image attachments

1. `reports/figures/atlas.png` — the 16,384-dot map with the 25 coalition features in red
2. `reports/figures/before_after_cards.png` — baseline vs silenced, same prompt and seed
3. `reports/figures/ladder.png` — the ablation ladder with the random-control flatline
4. Screenshot of Demo 1 surgical-deslop ("2 AI-ism features silenced — the overlap of prompt-concepts ∩ coalition")

---

## Draft C — X/Twitter thread (7 tweets, ≤280 chars each)

Cross-post target. Each `(n/7)` is one tweet. Thread together.

### Tweet 1 (hook)
> Spent a month inside a chatbot's head looking for the neuron that makes AI write like AI.
>
> You know the sentence: "This isn't a setback, it's a springboard." "It's not a tool, it's a partner." Same shape, every chatbot.
>
> I found the feature. It didn't fix the problem. 🧵

### Tweet 2 (the naive attack)
> The mech-interp playbook says behaviours like this live in one feature. Refusal does. So I scanned all 16,384 SAE features in Gemma 2 2B.
>
> Day one: feature 3223. Label: "phrases conveying exceptions or negations." Cinematic. I projected it out of every layer.

### Tweet 3 (the surprise)
> First generation under the kill, prompt about Antarctica:
>
> "Imagine a world, NOT of green meadows and warm breezes, BUT of endless white..."
>
> The intervention designed to make the construction impossible opened with the construction. 90 gens later: nothing removed.

### Tweet 4 (the coalition)
> So I tried 25 features at once.
>
> The AI-ism lives in a coalition: 2 cores (negation + digital tech) + 23 supporters that keep the kill killed.
>
> Ablate all 25 on 306 neutral prompts: textbook form −93%, the whole negated family −44% (~1 in 85 by luck). Fluency intact.

### Tweet 5 (the honest twist)
> An earlier draft said −80%. I audited my own detector twice; each fix revealed another escape route. Final: −44%.
>
> The model's favourite escape? Drop the negation: "It's more than just X. It's Y." That cousin went UP under the kill.
>
> The Hydra changes heads. Receipts in repo.

### Tweet 6 (the structural lesson + demos)
> The graph guessed wrong: decoder neighbours, co-activators, Leiden community — all failed. Only causal attribution found the coalition.
>
> Then 3 demos, each one Cypher query:
> 1️⃣ surgical de-slop (prompt ∩ behaviour)
> 2️⃣ mix-your-own-chatbot sliders
> 3️⃣ audit-trail MATCH

### Tweet 7 (CTA)
> Mech-interp + knowledge graphs = composable, auditable, glass-box model steering.
>
> Open source: 4,000-word writeup, every receipt, live demo + a slop-o-meter for your own prose:
>
> github.com/ho3h/not-this-but-that
>
> #MechanisticInterpretability #AI #Neo4j
