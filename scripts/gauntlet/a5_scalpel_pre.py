"""A5 — The scalpel, pre-emptive.

Unconditional zero-ablation of feature 3223 throughout the entire
generation. Identical to Phase 7's de-slop attempt. Pre-registered
expectation per PRD §3: NO-OP, because the feature is dormant on
neutral prompts. Same intervention as A4 minus the "only-when-firing"
condition.

This is the narrative hinge of the post: the surgical-looking attack
that does nothing, because the surgery target isn't there yet when
you cut.
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


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


class PreemptiveZeroHook:
    """Unconditionally zero feature TARGET_FEATURE at all positions."""

    def __call__(self, acts: torch.Tensor, **kwargs) -> torch.Tensor:
        acts = acts.clone()
        acts[..., TARGET_FEATURE] = 0.0
        return acts


def generate_chat(model, sae, tokenizer, user_prompt: str, *, seed: int,
                   max_new_tokens: int, install_hook: bool) -> str:
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

    fwd_hooks = []
    if install_hook:
        hook_name = f"{sae.cfg.metadata.hook_name}.hook_sae_acts_post"
        fwd_hooks = [(hook_name, PreemptiveZeroHook())]

    for _ in range(max_new_tokens):
        with torch.no_grad():
            if install_hook:
                logits = model.run_with_hooks_with_saes(
                    out_ids, saes=[sae], fwd_hooks=fwd_hooks,
                )
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
    print(f"[A5] loading {HF_NAME} + SAE on {dev}…")
    model = HookedSAETransformer.from_pretrained(HF_NAME, device=dev)
    model.eval()
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=dev)
    if isinstance(sae, tuple):
        sae = sae[0]
    sae.eval()
    tokenizer = model.tokenizer

    prompts = json.loads(TEST_PROMPTS.read_text())["prompts"]
    print(f"[A5] {len(prompts)} prompts × {len(SEEDS)} seeds × 2 conditions")

    baseline_gens, intervened_gens = [], []
    t0 = time.perf_counter()
    for pi, prompt in enumerate(prompts):
        for seed in SEEDS:
            try:
                b = generate_chat(model, sae, tokenizer, prompt, seed=seed,
                                    max_new_tokens=MAX_NEW_TOKENS, install_hook=False)
                i = generate_chat(model, sae, tokenizer, prompt, seed=seed,
                                    max_new_tokens=MAX_NEW_TOKENS, install_hook=True)
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
        "A5", "Scalpel pre-emptive",
        baseline_stats=b_stats, intervened_stats=i_stats,
        baseline_ppl=None, intervened_ppl=None,
        extra={"target_feature": TARGET_FEATURE,
               "sae_release": SAE_RELEASE, "sae_id": SAE_ID,
               "model": HF_NAME,
               "pre_registered_expectation": "no-op (feature dormant on neutral prompts)",
               "note": "Phase 7 replication. Feature 3223 zeroed at every position. "
                       "If Phase 7 was right, this changes nothing."},
    )
    print_report(rep)
    save_report("A5", rep, baseline_gens, intervened_gens)
    write_eyeball("A5", "Scalpel pre-emptive",
                   baseline_pairs=baseline_gens, intervened_pairs=intervened_gens,
                   notes="The narrative hinge: we sliced out the feature before "
                         "the model had decided to use it. Spot the difference.")


if __name__ == "__main__":
    main()
