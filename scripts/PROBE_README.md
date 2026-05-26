# Probe daemon — iterative ablation experiments

A persistent process holding `gemma-2-2b` + Gemma Scope L20/16k SAE in
memory, with a JSON HTTP API at `http://localhost:8765/probe`. Lets you
run arbitrarily many ablation experiments without paying the 15–20 s
model-load cost each time.

Use it whenever you want to riff on a new ablation hypothesis without
writing a new script.

## Start / stop

```bash
# Foreground (Ctrl-C to stop):
HF_TOKEN=$(grep ^HF_TOKEN= .env | cut -d= -f2-) .venv/bin/python scripts/probe_daemon.py

# Background (logs to /tmp/probe_daemon.log):
HF_TOKEN=$(...) nohup .venv/bin/python scripts/probe_daemon.py > /tmp/probe_daemon.log 2>&1 &

# Graceful shutdown:
.venv/bin/python scripts/probe.py '{"cmd": "stop"}'
```

The daemon prints `probe daemon listening on http://127.0.0.1:8765/probe`
when it's ready. Don't send compute-heavy commands before that line.

## Client

`scripts/probe.py` is a 30-line stdlib HTTP client. Pipe via `jq` to
pretty-print:

```bash
.venv/bin/python scripts/probe.py '{"cmd": "ping"}' | jq .
.venv/bin/python scripts/probe.py @body.json | jq .   # @-prefix reads from file
```

Plain `curl` works too:

```bash
curl -s -X POST -H 'Content-Type: application/json' \
     -d '{"cmd": "ping"}' http://localhost:8765/probe | jq .
```

## Commands

Every response is wrapped: `{"ok": bool, "result": ..., "elapsed_s": float}`
on success, `{"ok": false, "error": str}` on failure.

### `ping`

Health check. Returns device, d_sae, model name.

```bash
probe.py '{"cmd": "ping"}'
```

### `labels` — Neuronpedia labels

```bash
probe.py '{"cmd": "labels", "features": [3223, 9909]}'
```

### `attribution` — top-N from pivot_attribution.json

```bash
probe.py '{"cmd": "attribution", "top_n": 10, "kind": "promote"}'
probe.py '{"cmd": "attribution", "top_n": 10, "kind": "suppress"}'
```

### `graph` — structural priors from Neo4j

```bash
# Top-10 decoder-cosine neighbors of 3223
probe.py '{"cmd": "graph", "query": "decoder_neighbors", "anchor": 3223, "k": 10}'

# Top-10 co-activating partners by Jaccard (use "pmi" for PMI rank)
probe.py '{"cmd": "graph", "query": "coact_partners", "anchor": 3223, "k": 10, "rank_by": "jaccard"}'

# Top-50 community-12 members by activation density
probe.py '{"cmd": "graph", "query": "community", "cid": 12, "limit": 50}'
```

### `graph_cypher` — raw Cypher passthrough

```bash
probe.py '{"cmd": "graph_cypher",
           "query": "MATCH (f:SAEFeature) WHERE f.communityId = $cid RETURN count(*) AS n",
           "params": {"cid": 12}}'
```

### `measure_pivot` — M2 single-condition

Runs joint ablation on truncated D1 prompts; returns mean P(pivot)
under ablation vs baseline.

```bash
probe.py '{"cmd": "measure_pivot",
           "ablate": [3223, 9909, 12898],
           "variants": ["C1", "C2", "C3"],
           "max_samples": 80}'
```

### `ladder` — M2 across many conditions in one pass

Runs many named conditions plus random size-matched controls. This is
the workhorse for "is set X different from a same-size random set?"
questions.

```bash
probe.py '{"cmd": "ladder",
           "conditions": {
             "single_3223": [3223],
             "attrib_top5": [3223, 9909, 12898, 4197, 6759],
             "decoder_neighbors_10": [3223, 11406, 5759, 9816, 1250, 4956, 5798, 9870, 2830, 8266]
           },
           "n_random_per_size": 3,
           "variants": ["C1", "C2", "C3"],
           "max_samples": 80,
           "seed": 11}'
```

Cost: roughly `(named + random) * max_samples * 0.3 s`. At default
`max_samples=80` and `n_random_per_size=3`, an 8-condition ladder is
~9 minutes.

### `generate` — single completion, sustained ablation

Generate a completion with the ablation set zeroed at *every* position,
not just at the pivot. `model` selects between base (`"base"`, default,
for completion-style prompts) and instruction-tuned (`"it"`, for
chat-style prompts).

```bash
probe.py '{"cmd": "generate",
           "model": "it",
           "prompt": "Describe what a busy hospital cafeteria sounds like at 2 a.m.",
           "ablate": [3223, 9909, 12898, 4197, 6759],
           "max_new_tokens": 120,
           "temperature": 0.8,
           "top_p": 0.95,
           "seed": 0}'
```

Cost: roughly `max_new_tokens * 0.05 s` per generation.

### `m1_eval` — sustained ablation across D2 prompts, M1 scored

Runs the construction classifier on baseline vs ablated generations
across the first N D2 prompts × S seeds. This is the right test for
"does sustained joint ablation actually lower construction rate in
open-ended generation?" — i.e. the question Phase 7 left open after
single-feature ablation gave a null.

```bash
probe.py '{"cmd": "m1_eval",
           "model": "it",
           "ablate": [3223, 9909, 12898, 4197, 6759],
           "n_prompts": 12,
           "seeds": 3,
           "max_new_tokens": 120}'
```

Returns baseline + ablated construction rates, absolute and relative
drop, the first 6 (prompt, baseline, ablated) triples for eyeballing,
and the full per-row data.

Cost: `n_prompts * seeds * 2 * max_new_tokens * 0.05 s`. At defaults
(12 × 3 × 2 × 120 × 0.05) ≈ 7 min.

### `stop`

Graceful shutdown.

## Composing experiments

Most ablation hypotheses can be built up like this:

1. **Resolve a feature set from a structural prior** (`graph` or
   `attribution` or `graph_cypher`).
2. **Send it to `ladder` as a named condition**, alongside any other
   sets you want to compare and a random size-matched null.
3. **Optionally**: take the strongest set, send it to `generate` with
   a D2-style neutral prompt to see if sustained ablation moves M1.

Example: testing whether ablating the top-50 attribution features beats
the top-25.

```bash
TOP25=$(probe.py '{"cmd": "attribution", "top_n": 25, "kind": "promote"}' | jq -c .result.features)
TOP50=$(probe.py '{"cmd": "attribution", "top_n": 50, "kind": "promote"}' | jq -c .result.features)
probe.py "{\"cmd\": \"ladder\",
           \"conditions\": {\"top25\": $TOP25, \"top50\": $TOP50},
           \"n_random_per_size\": 3,
           \"max_samples\": 80}"
```

## Concurrency

The daemon serializes all requests through a single coarse lock. Sending
multiple compute-heavy requests in parallel won't speed anything up; just
queue them sequentially.

## When *not* to use the daemon

- For experiments that need new code paths (e.g. a different model, a
  different hook site, a new measurement metric), write a new script.
  The daemon is for ablation experiments at the *existing* L20 SAE hook.
- For experiments where you'd rather have a permanent reproducible
  artefact, write a script. The daemon is for interactive iteration; once
  a result matters, freeze it as a script that's checked in.
