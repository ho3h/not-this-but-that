"""A4 — The scalpel, mid-act.

Conditional zero-ablation of feature 3223 at the SAE encoder layer (L20),
firing only when the feature is already active above a small threshold.
This catches the model mid-construction: the feature is dormant on neutral
prose so the hook is a no-op until the model has decided to enter the
construction, at which point its contribution is zeroed out.

Caveats (carried in the writeup):
  - The SAE is Gemma Scope, trained on gemma-2-2b base; we apply it to
    gemma-2-2b-it. The transfer is plausible (same architecture) but
    independent VE certification is not available in this codebase.
  - feature 3223's Neuronpedia label is "phrases conveying exceptions or
    negations" — the field's autointerp label, not ours.

Expected: real construction-rate drop with fluency preserved (the
elegant beat). If A4 catches the swing, that's a genuine result. If it
sleeps through everything, that's the surprising twist.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
from sae_lens import HookedSAETransformer, SAE

from gauntlet.runner import (
    EYEBALL_DIR, attack_report, done_keys, load_checkpoint,
    perplexity_on_d3, print_report, save_checkpoint, save_report,
    score_generations, split_sentences, write_eyeball,
)

REPO = Path(__file__).resolve().parent.parent.parent
TEST_PROMPTS = REPO / "data" / "d2_corpus" / "gauntlet_test_prompts.json"

HF_NAME = "google/gemma-2-2b-it"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_20/width_16k/canonical"
TARGET_FEATURE = 3223  # the necessity candidate from Phase 4
ACTIVATION_THRESHOLD = 1e-3  # ablate only if act > this (mid-act condition)

SEEDS = [0, 1, 2]
MAX_NEW_TOKENS = 200


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


class ScalpelMidActHook:
    """SAE hook that zeros feature TARGET_FEATURE at any position where the
    feature is already firing above ACTIVATION_THRESHOLD. Hook signature
    matches sae_lens 6.x's hook_sae_acts_post hook point."""

    def __call__(self, acts: torch.Tensor, **kwargs) -> torch.Tensor:
        acts = acts.clone()
        col = acts[..., TARGET_FEATURE]
        mask = (col > ACTIVATION_THRESHOLD)
        acts[..., TARGET_FEATURE] = torch.where(mask, torch.zeros_like(col), col)
        return acts


def build_model_and_sae():
    dev = device()
    print(f"[A4] loading {HF_NAME} on {dev}…")
    model = HookedSAETransformer.from_pretrained(HF_NAME, device=dev)
    model.eval()
    print(f"[A4] loading SAE {SAE_RELEASE} / {SAE_ID}")
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=dev)
    if isinstance(sae, tuple):
        sae = sae[0]
    sae.eval()
    hook_name = f"{sae.cfg.metadata.hook_name}.hook_sae_acts_post"
    return model, sae, hook_name, dev


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
        fwd_hooks = [(f"{sae.cfg.metadata.hook_name}.hook_sae_acts_post",
                      ScalpelMidActHook())]

    for _ in range(max_new_tokens):
        with torch.no_grad():
            if install_hook:
                logits = model.run_with_hooks_with_saes(
                    out_ids, saes=[sae], fwd_hooks=fwd_hooks,
                )
            else:
                logits = model(out_ids)
        next_logits = logits[0, -1, :] / 0.8  # temperature
        probs = torch.softmax(next_logits, dim=-1)
        # top-p sampling
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
    model, sae, hook_name, dev = build_model_and_sae()
    tokenizer = model.tokenizer

    prompts = json.loads(TEST_PROMPTS.read_text())["prompts"]
    print(f"[A4] {len(prompts)} prompts × {len(SEEDS)} seeds × 2 conditions")

    # Resume from checkpoint if present.
    ck = load_checkpoint("A4")
    if ck is not None:
        baseline_gens = ck.get("baseline_generations", [])
        intervened_gens = ck.get("intervened_generations", [])
        print(f"[A4] resuming from checkpoint: {len(baseline_gens)} baseline + "
              f"{len(intervened_gens)} intervened already done")
    else:
        baseline_gens = []
        intervened_gens = []
    done_b = done_keys(baseline_gens)
    done_i = done_keys(intervened_gens)

    t0 = time.perf_counter()
    for pi, prompt in enumerate(prompts):
        any_new = False
        for seed in SEEDS:
            try:
                if (pi, seed) not in done_b:
                    b = generate_chat(model, sae, tokenizer, prompt, seed=seed,
                                        max_new_tokens=MAX_NEW_TOKENS, install_hook=False)
                    baseline_gens.append({"prompt_idx": pi, "prompt": prompt,
                                            "seed": seed, "generation": b})
                    any_new = True
                if (pi, seed) not in done_i:
                    i = generate_chat(model, sae, tokenizer, prompt, seed=seed,
                                        max_new_tokens=MAX_NEW_TOKENS, install_hook=True)
                    intervened_gens.append({"prompt_idx": pi, "prompt": prompt,
                                              "seed": seed, "generation": i})
                    any_new = True
            except Exception as e:
                print(f"  ERR p={pi} s={seed}: {e}")
        # Checkpoint after every prompt (atomic write).
        if any_new:
            save_checkpoint("A4", baseline_gens, intervened_gens)
        if (pi + 1) % 3 == 0:
            done = len(baseline_gens) + len(intervened_gens)
            total = len(prompts) * len(SEEDS) * 2
            rate = done / max(time.perf_counter() - t0, 1e-6)
            print(f"  {done}/{total}  ({rate:.2f}/s)", flush=True)

    print("[A4] scoring + measuring perplexity…")
    b_stats = score_generations(baseline_gens)
    i_stats = score_generations(intervened_gens)

    # Fluency: with SAE-hook installed for the intervened forward
    # Skip the standalone PPL measurement here — HookedSAETransformer's loss
    # surface differs from regular HF model's, and we don't want to compare
    # incommensurable numbers. EYEBALL is the fluency check for A4.
    b_ppl = i_ppl = None

    rep = attack_report(
        "A4", "Scalpel mid-act",
        baseline_stats=b_stats, intervened_stats=i_stats,
        baseline_ppl=b_ppl, intervened_ppl=i_ppl,
        extra={"target_feature": TARGET_FEATURE,
               "activation_threshold": ACTIVATION_THRESHOLD,
               "sae_release": SAE_RELEASE, "sae_id": SAE_ID,
               "model": HF_NAME,
               "note": "Mid-act conditional ablation: hook only fires at positions "
                       "where feature 3223 is already active above threshold. "
                       "Captures the model mid-construction; sleeps through "
                       "neutral prose."},
    )
    print_report(rep)
    save_report("A4", rep, baseline_gens, intervened_gens)
    write_eyeball("A4", "Scalpel mid-act",
                   baseline_pairs=baseline_gens, intervened_pairs=intervened_gens,
                   notes="Ablation conditioned on feature 3223 already firing. "
                         "Watch whether the model swerves mid-sentence or just "
                         "carries on unchanged.")


if __name__ == "__main__":
    main()
