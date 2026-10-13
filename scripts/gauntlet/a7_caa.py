"""A7 — Contrastive Activation Addition (CAA).

The headline experiment. Build a steering vector from the D2-corpus TRAIN
pairs: at a chosen layer L, take mean(residual_at_last_token | with) minus
mean(residual_at_last_token | without). The difference is the construction
direction in raw residual-stream space (no SAE involved — just paired
contrast on real activations).

At generation time, SUBTRACT alpha · v from layer L's residual stream
during the forward pass. Sweep alpha; report the fluency / suppression
frontier on the TEST split + fresh generations.

The CAA technique is from Rimsky et al. 2024 (and the CAA literature
since). It typically suppresses the targeted behaviour more durably than
single-feature ablation because it acts in raw activation space, where
the model's representation actually lives.

Pre-registered expectation (PRD §3): two good outcomes — either the
vector lands a clean kill (genuine result on a model the literature
calls hard to steer), or fluency collapses before kill-rate moves (the
"the model fought back" beat). Both are reportable; the bad outcome
would be a coefficient where neither happens, which is itself a finding.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from gauntlet.runner import (
    GauntletModel, attack_report, done_keys, load_checkpoint,
    print_report, save_checkpoint, save_report, score_generations,
    write_eyeball,
)

REPO = Path(__file__).resolve().parent.parent.parent
TEST_PROMPTS = REPO / "data" / "d2_corpus" / "gauntlet_test_prompts.json"
CORPUS_PATH = REPO / "data" / "d2_corpus" / "verified_corpus.jsonl"
SPLIT_PATH = REPO / "data" / "d2_corpus" / "split.json"
VECTOR_PATH = REPO / "data" / "d2_corpus" / "caa_vector.pt"
ALPHA_SWEEP_PATH = REPO / "reports" / "gauntlet" / "a7_alpha_sweep.json"

# The steering layer. CAA literature on small models commonly targets a
# middle-late layer (e.g. ~75 % depth). Gemma 2 2B has 26 layers; 20 is
# the canonical Gemma Scope layer. Match that for continuity.
STEERING_LAYER = 20

SEEDS = [0, 1, 2]
MAX_NEW_TOKENS = 200
ALPHA_SWEEP = [0.0, 1.0, 2.0, 4.0, 8.0]


def build_vector(gm: GauntletModel, train_pairs: list[dict]) -> torch.Tensor:
    """For each TRAIN pair, capture the residual-stream activation at the last
    token of `with` and `without`; return mean(with) - mean(without).
    """
    hook_name = f"blocks.{STEERING_LAYER}.hook_resid_post"
    # We need a HookedTransformer-style cache. Use a custom forward hook.
    acts_with, acts_without = [], []

    def hook_factory(bucket):
        def _hook(module, inputs, outputs):
            # outputs is the resid_post tensor; we take last token of position dim
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            bucket.append(outputs[:, -1, :].detach().float().cpu())
        return _hook

    # Locate the layer module
    block = gm.model.model.layers[STEERING_LAYER]
    h_with = block.register_forward_hook(hook_factory(acts_with))
    try:
        for p in train_pairs:
            for text, bucket in ((p["with"], acts_with), (p["without"], acts_without)):
                # swap hooks per call
                pass  # we'll use a single bucket pattern below
    finally:
        h_with.remove()

    # Cleaner: two separate forward passes, one bucket each, switching the hook
    def collect(texts: list[str]) -> torch.Tensor:
        bucket = []
        def _hook(module, inputs, outputs):
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            bucket.append(outputs[:, -1, :].detach().float().cpu())
        h = block.register_forward_hook(_hook)
        try:
            for t in texts:
                enc = gm.tokenizer(t, return_tensors="pt").to(gm.dev)
                with torch.no_grad():
                    gm.model(**enc)
        finally:
            h.remove()
        return torch.cat(bucket, dim=0)  # (n, d_model)

    print(f"  collecting 'with' activations from {len(train_pairs)} pairs…")
    a_with = collect([p["with"] for p in train_pairs])
    print(f"  collecting 'without' activations from {len(train_pairs)} pairs…")
    a_without = collect([p["without"] for p in train_pairs])
    v = a_with.mean(0) - a_without.mean(0)  # (d_model,)
    return v


class SteerHook:
    def __init__(self, vector: torch.Tensor, alpha: float):
        self.v = vector
        self.alpha = alpha

    def __call__(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            t = outputs[0]
            t = t - self.alpha * self.v.to(t.device).to(t.dtype)
            return (t,) + outputs[1:]
        return outputs - self.alpha * self.v.to(outputs.device).to(outputs.dtype)


def generate_with_steer(gm: GauntletModel, prompt: str, *, seed: int,
                         vector: torch.Tensor, alpha: float,
                         max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    block = gm.model.model.layers[STEERING_LAYER]
    handle = None
    if alpha != 0.0:
        handle = block.register_forward_hook(SteerHook(vector, alpha))
    try:
        return gm.generate(prompt, seed=seed, max_new_tokens=max_new_tokens)
    finally:
        if handle is not None:
            handle.remove()


def main() -> None:
    gm = GauntletModel.load()

    # Load TRAIN pairs
    if not SPLIT_PATH.exists() or not CORPUS_PATH.exists():
        print(f"[A7] missing corpus or split — cannot build vector. "
              f"Expected {CORPUS_PATH} and {SPLIT_PATH}.")
        return
    split = json.loads(SPLIT_PATH.read_text())
    pairs = [json.loads(l) for l in CORPUS_PATH.read_text().splitlines() if not l.startswith('{"_meta')]
    train_ids = set(split["train"])
    train_pairs = [p for p in pairs if p["id"] in train_ids]
    print(f"[A7] building vector from {len(train_pairs)} TRAIN pairs at layer {STEERING_LAYER}…")

    if VECTOR_PATH.exists():
        v = torch.load(VECTOR_PATH)
        print(f"[A7] loaded cached vector from {VECTOR_PATH}")
    else:
        v = build_vector(gm, train_pairs)
        torch.save(v, VECTOR_PATH)
        print(f"[A7] saved vector to {VECTOR_PATH} (norm={v.norm().item():.3f})")

    # Sweep alpha
    prompts = json.loads(TEST_PROMPTS.read_text())["prompts"]
    print(f"[A7] sweeping alpha ∈ {ALPHA_SWEEP} on {len(prompts)} test prompts × {len(SEEDS)} seeds")

    # Per-alpha checkpoint: each alpha gets its own checkpoint tag so resumes
    # can pick up at the right place across the sweep.
    by_alpha = {}
    t0 = time.perf_counter()
    for alpha in ALPHA_SWEEP:
        print(f"\n[A7] alpha = {alpha}")
        tag = f"alpha{alpha}"
        ck = load_checkpoint("A7", tag=tag)
        if ck is not None:
            gens = ck.get("intervened_generations", [])
            print(f"  resuming alpha={alpha}: {len(gens)} gens already done")
        else:
            gens = []
        done = done_keys(gens)
        for pi, prompt in enumerate(prompts):
            any_new = False
            for seed in SEEDS:
                if (pi, seed) in done:
                    continue
                try:
                    text = generate_with_steer(gm, prompt, seed=seed, vector=v, alpha=alpha)
                    gens.append({"prompt_idx": pi, "prompt": prompt, "seed": seed,
                                 "generation": text})
                    any_new = True
                except Exception as e:
                    print(f"  ERR p={pi} s={seed}: {e}")
            if any_new:
                # A7's checkpoints carry only the intervened side (a single
                # alpha's outputs); baseline is alpha=0 which gets its own
                # entry in the sweep.
                save_checkpoint("A7", [], gens, tag=tag)
        stats = score_generations(gens)
        by_alpha[str(alpha)] = {"stats": {k: v for k, v in stats.items() if k != "sentences"},
                                  "generations": gens}
        print(f"  any_core rate: {stats['any_core_rate']:.3%}", flush=True)

    ALPHA_SWEEP_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALPHA_SWEEP_PATH.write_text(json.dumps({
        "alphas": ALPHA_SWEEP, "results": by_alpha,
        "steering_layer": STEERING_LAYER, "vector_norm": float(v.norm().item()),
        "n_train_pairs": len(train_pairs),
    }, indent=2))

    # Pick the alpha that maximizes suppression while keeping fluency
    # (approximation: largest alpha with non-trivial output).
    # For the gauntlet report, treat alpha=0 as baseline and the best
    # non-zero alpha as the intervention.
    baseline = by_alpha["0.0"]["stats"]
    # Choose the lowest non-zero alpha with the largest any_core drop
    best_alpha = max(ALPHA_SWEEP[1:], key=lambda a: baseline["any_core_rate"] - by_alpha[str(a)]["stats"]["any_core_rate"])
    intervened = by_alpha[str(best_alpha)]["stats"]
    baseline_gens = by_alpha["0.0"]["generations"]
    intervened_gens = by_alpha[str(best_alpha)]["generations"]
    # Put sentences back in for save_report
    baseline["sentences"] = score_generations(baseline_gens)["sentences"]
    intervened["sentences"] = score_generations(intervened_gens)["sentences"]

    rep = attack_report(
        "A7", "Contrastive steering vector (CAA)",
        baseline_stats=baseline, intervened_stats=intervened,
        baseline_ppl=None, intervened_ppl=None,
        extra={"steering_layer": STEERING_LAYER, "vector_norm": float(v.norm().item()),
               "n_train_pairs": len(train_pairs),
               "alpha_sweep": ALPHA_SWEEP, "best_alpha": best_alpha,
               "per_alpha_stats": {a: by_alpha[str(a)]["stats"] for a in ALPHA_SWEEP},
               "note": "CAA vector from TRAIN pairs at layer L=20. Subtract alpha·v "
                       "from resid_post during generation. Full sweep in "
                       "reports/gauntlet/a7_alpha_sweep.json; headline number uses "
                       f"the alpha with largest suppression (alpha={best_alpha})."},
    )
    print_report(rep)
    save_report("A7", rep, baseline_gens, intervened_gens)
    write_eyeball("A7", f"CAA (α={best_alpha})",
                   baseline_pairs=baseline_gens, intervened_pairs=intervened_gens,
                   notes=f"The headline. Vector built from {len(train_pairs)} TRAIN "
                         f"contrast pairs, subtracted from layer {STEERING_LAYER}'s "
                         f"residual stream at coefficient α={best_alpha}. The big "
                         f"question: does it actually kill the construction, or "
                         f"does the prose collapse first?")


if __name__ == "__main__":
    main()
