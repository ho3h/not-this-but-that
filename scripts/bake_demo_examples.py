"""Pre-bake the canonical demo runs as a static JSON the frontend can play
back instantly, with no daemon required.

Each entry is one (prompt, ablation_set) generation, captured with the
top-K-features-per-token playback record.

Output: web/demo/examples.json
"""

from __future__ import annotations

import json
import pathlib
import time
import urllib.request

PROBE = "http://127.0.0.1:8765/probe"
REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "web" / "demo" / "examples.json"


def call(body, timeout=600):
    req = urllib.request.Request(
        PROBE, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


CANONICAL = [
    # Each: (id, label, prompt, max_new, seed).
    # Seeds chosen so the baseline visibly produces the construction; ablated
    # generations should reroute. Picked from D1-style prefixes plus a couple
    # of open prompts that reliably trip the model into F2 staccato.
    {
        "id": "library",
        "label": "What makes a good library?",
        "prompt": "Tell me what makes a good neighbourhood library.",
        "model": "it",
        "max_new_tokens": 70,
        "seed": 0,
    },
    {
        "id": "tool",
        "label": "It's not a tool…",
        "prompt": "It's not a tool",
        "model": "it",
        "max_new_tokens": 55,
        "seed": 0,
    },
    {
        "id": "setback",
        "label": "This isn't a setback…",
        "prompt": "This isn't a setback",
        "model": "it",
        "max_new_tokens": 55,
        "seed": 0,
    },
    {
        "id": "hardware",
        "label": "Small-town hardware store",
        "prompt": "Talk about what makes a small-town hardware store feel timeless.",
        "model": "it",
        "max_new_tokens": 70,
        "seed": 0,
    },
]


def bake_one(spec, ablate, ablate_label):
    print(f"  baking [{spec['id']}] x [{ablate_label}]…")
    t0 = time.perf_counter()
    resp = call({
        "cmd": "generate_with_activations",
        "model": spec["model"],
        "prompt": spec["prompt"],
        "ablate": ablate,
        "max_new_tokens": spec["max_new_tokens"],
        "seed": spec["seed"],
        "top_k_features": 10,
    })
    elapsed = time.perf_counter() - t0
    if not resp.get("ok"):
        raise RuntimeError(f"failed: {resp.get('error')}")
    r = resp["result"]
    print(f"    {elapsed:.1f}s, {len(r['records'])} tokens, {len(r['completion'])} chars")
    return {
        "ablate_label": ablate_label,
        "ablate_set": ablate,
        "completion": r["completion"],
        "records": r["records"],
    }


def main():
    # Pull canonical ablation sets from the daemon (which reads from
    # pivot_attribution.json on disk — single source of truth).
    top25 = call({"cmd": "attribution", "top_n": 25, "kind": "promote"})["result"]["features"]
    minimal = [3223, 9909]
    print(f"top-25 from attribution: {top25[:5]}… (total {len(top25)})")

    examples = {
        "top25_set": top25,
        "minimal_set": minimal,
        "examples": [],
    }
    for spec in CANONICAL:
        print(f"\n--- {spec['id']} ---")
        baseline = bake_one(spec, None, "no intervention")
        top25_run = bake_one(spec, top25, "top-25 coalition ablated")
        minimal_run = bake_one(spec, minimal, "minimal core (3223 + 9909) ablated")
        examples["examples"].append({
            **{k: v for k, v in spec.items()},
            "runs": {
                "baseline": baseline,
                "top25": top25_run,
                "minimal": minimal_run,
            },
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(examples, separators=(",", ":")))
    print(f"\n→ {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
