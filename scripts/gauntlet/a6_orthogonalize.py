"""A6 — Orthogonalize the direction (Arditi-style directional ablation).

For the construction-recruiting direction (taken from feature 3223's SAE
decoder column — a unit-norm vector in residual-stream space), at every
transformer block's resid_post hook, project the activation onto the
hyperplane orthogonal to that direction:

    act' = act - (act · d) * d        where d is unit-norm

Applied at all layers and all positions. The model's downstream layers
see a residual stream that NEVER has any component in direction d. If
the construction lives in d, the model can't represent it anywhere.

This is the heavier intervention compared to A4/A5 (which only act at
one SAE-encoder layer) and is the standard refusal-direction-ablation
recipe from Arditi et al. 2024.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from sae_lens import HookedSAETransformer, SAE

from gauntlet.runner import (
    attack_report, print_report, save_report, score_generations, write_eyeball,
)

REPO = Path(__file__).resolve().parent.parent.parent
TEST_PROMPTS = REPO / "data" / "d2_corpus" / "gauntlet_test_prompts.json"

HF_NAME = "google/gemma-2-2b-it"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_20/width_16k/canonical"
TARGET_FEATURE = 3223

SEEDS = [0, 1, 2]
MAX_NEW_TOKENS = 200
N_LAYERS = 26  # Gemma 2 2B has 26 transformer blocks


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def get_direction(sae) -> torch.Tensor:
    """Unit-normalize feature 3223's decoder column."""
    d = sae.W_dec[TARGET_FEATURE].detach().clone()  # shape (d_model,)
    d = d / d.norm()
    return d


class OrthogonalizeHook:
    def __init__(self, direction: torch.Tensor):
        self.d = direction.to(dtype=torch.float32)

    def __call__(self, acts: torch.Tensor, **kwargs) -> torch.Tensor:
        # acts: (batch, seq, d_model)
        d = self.d.to(acts.device).to(acts.dtype)
        proj = (acts * d).sum(dim=-1, keepdim=True) * d  # (batch, seq, d_model)
        return acts - proj


def generate_chat(model, tokenizer, user_prompt: str, *, seed: int,
                   max_new_tokens: int, hook_pairs: list[tuple] | None = None) -> str:
    torch.manual_seed(seed)
    if model.cfg.device == "mps":
        torch.mps.manual_seed(seed)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        add_generation_prompt=True, tokenize=False,
    )
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    input_ids = input_ids.to(model.cfg.device)
    out_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            if hook_pairs:
                logits = model.run_with_hooks(out_ids, fwd_hooks=hook_pairs)
            else:
                logits = model(out_ids)
        next_logits = logits[0, -1, :] / 0.8
        probs = torch.softmax(next_logits, dim=-1)
        sorted_p, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_p, dim=-1)
        keep = cum <= 0.95
        keep[0] = True
        kept_p = sorted_p[keep]
        kept_i = sorted_idx[keep]
        kept_p = kept_p / kept_p.sum()
        choice = torch.multinomial(kept_p, num_samples=1)
        next_id = kept_i[choice]
        out_ids = torch.cat([out_ids, next_id.unsqueeze(0)], dim=1)
        if next_id.item() == tokenizer.eos_token_id:
            break
    input_len = input_ids.shape[1]
    return tokenizer.decode(out_ids[0, input_len:], skip_special_tokens=True)


def main() -> None:
    dev = device()
    print(f"[A6] loading {HF_NAME} + SAE on {dev}…")
    model = HookedSAETransformer.from_pretrained(HF_NAME, device=dev)
    model.eval()
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=dev)
    if isinstance(sae, tuple):
        sae = sae[0]
    sae.eval()
    tokenizer = model.tokenizer

    direction = get_direction(sae)
    print(f"[A6] direction shape {direction.shape}, norm {direction.norm().item():.4f}")

    # Install orthogonalize hooks at every block's hook_resid_post
    hook_pairs = [(f"blocks.{li}.hook_resid_post", OrthogonalizeHook(direction))
                  for li in range(N_LAYERS)]

    prompts = json.loads(TEST_PROMPTS.read_text())["prompts"]
    print(f"[A6] {len(prompts)} prompts × {len(SEEDS)} seeds × 2 conditions, "
          f"orthogonalize at {N_LAYERS} layers")

    baseline_gens, intervened_gens = [], []
    t0 = time.perf_counter()
    for pi, prompt in enumerate(prompts):
        for seed in SEEDS:
            try:
                b = generate_chat(model, tokenizer, prompt, seed=seed,
                                    max_new_tokens=MAX_NEW_TOKENS, hook_pairs=None)
                i = generate_chat(model, tokenizer, prompt, seed=seed,
                                    max_new_tokens=MAX_NEW_TOKENS, hook_pairs=hook_pairs)
                baseline_gens.append({"prompt_idx": pi, "prompt": prompt,
                                        "seed": seed, "generation": b})
                intervened_gens.append({"prompt_idx": pi, "prompt": prompt,
                                          "seed": seed, "generation": i})
            except Exception as e:
                print(f"  ERR p={pi} s={seed}: {e}")
        if (pi + 1) % 3 == 0:
            done = (pi + 1) * len(SEEDS) * 2
            rate = done / max(time.perf_counter() - t0, 1e-6)
            print(f"  {done}/{len(prompts)*len(SEEDS)*2}  ({rate:.2f}/s)")

    b_stats = score_generations(baseline_gens)
    i_stats = score_generations(intervened_gens)

    rep = attack_report(
        "A6", "Orthogonalize it out",
        baseline_stats=b_stats, intervened_stats=i_stats,
        baseline_ppl=None, intervened_ppl=None,
        extra={"target_feature": TARGET_FEATURE,
               "n_layers": N_LAYERS,
               "model": HF_NAME,
               "sae_release": SAE_RELEASE, "sae_id": SAE_ID,
               "note": "Arditi-style directional ablation: project feature 3223's "
                       "decoder direction OUT of the residual stream at every "
                       "block, at every position. The model literally cannot "
                       "represent the construction direction anywhere."},
    )
    print_report(rep)
    save_report("A6", rep, baseline_gens, intervened_gens)
    write_eyeball("A6", "Orthogonalize",
                   baseline_pairs=baseline_gens, intervened_pairs=intervened_gens,
                   notes="The whole direction, projected out of every layer. "
                         "If A5 left the feature asleep, A6 removes the very "
                         "axis the model would have used.")


if __name__ == "__main__":
    main()
