"""Cross-model replication — does the register finding hold up on instruct
models from other families?

Replicates two things across new models:

  (a) Variant-composition signature: any_core C3-share. Gemma-2-2b-it sat at
      94% C3 of its construction usage; if Qwen-it / Llama-it / Gemma-9b-it
      also produce constructions and those are also predominantly C3, the
      'instruct installs a C3-shaped register' claim is cross-family.

  (b) H17 position effect: do constructions cluster early in the generation?
      In Gemma-2-2b-it we saw median relative position 0.10 (positive) vs
      0.50 (negative) — strong opener effect. Does it replicate?

The full D2 corpus (102 prompts) is fair game on these new models — the
Discovery/Confirmation firewall was scoped to Gemma-2-2b-it. New models =
new data.

Usage:
    uv run python scripts/replicate_register_finding.py \
        --model qwen2.5_7b_it --seeds 3 --max-new-tokens 150
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

from classifier import detect_construction

REPO = Path(__file__).resolve().parent.parent
D2_PATH = REPO / "data" / "D2_neutral_prompts.json"
OUT_DIR = REPO / "reports" / "replication"

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“])")


@dataclass(frozen=True)
class ModelSpec:
    nickname: str
    hf_name: str
    chat_template: bool
    dtype: torch.dtype = torch.float16


MODELS: dict[str, ModelSpec] = {
    "qwen2.5_7b_it": ModelSpec("qwen2.5_7b_it", "Qwen/Qwen2.5-7B-Instruct",
                                chat_template=True, dtype=torch.float16),
    "qwen3_1.7b_it": ModelSpec("qwen3_1.7b_it", "Qwen/Qwen3-1.7B",
                                chat_template=True, dtype=torch.float16),
    "gemma_9b_it": ModelSpec("gemma_9b_it", "google/gemma-2-9b-it",
                              chat_template=True, dtype=torch.float16),
}


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(text.strip()) if len(s.strip()) > 30]


def make_input(prompt: str, tokenizer, chat_template: bool) -> dict:
    if chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=False,
        )
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    else:
        enc = tokenizer(prompt, return_tensors="pt")
    return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}


def generate_one(model, tokenizer, prompt: str, *, chat_template: bool,
                 seed: int, max_new_tokens: int, temperature: float,
                 top_p: float, dev: str) -> str:
    torch.manual_seed(seed)
    if dev == "mps":
        torch.mps.manual_seed(seed)
    enc = make_input(prompt, tokenizer, chat_template)
    enc = {k: v.to(dev) for k, v in enc.items()}
    input_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0, input_len:], skip_special_tokens=True)
    return text


def run(model_key: str, n_prompts: int, seeds: int, max_new_tokens: int) -> dict:
    spec = MODELS[model_key]
    dev = device()

    print(f"[{spec.nickname}] loading {spec.hf_name} on {dev} ({spec.dtype})…")
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(spec.hf_name, dtype=spec.dtype).to(dev)
    model.eval()

    prompts = json.loads(D2_PATH.read_text())["prompts"][:n_prompts]
    print(f"[{spec.nickname}] generating {len(prompts) * seeds} continuations…")

    generations = []
    t0 = time.perf_counter()
    for pi, prompt in enumerate(prompts):
        for seed in range(seeds):
            try:
                text = generate_one(
                    model, tokenizer, prompt, chat_template=spec.chat_template,
                    seed=seed, max_new_tokens=max_new_tokens,
                    temperature=0.8, top_p=0.95, dev=dev,
                )
            except Exception as e:
                print(f"  generation failed for prompt {pi} seed {seed}: {e}")
                continue
            generations.append({"prompt_idx": pi, "prompt": prompt, "seed": seed,
                                 "generation": text})
        if (pi + 1) % 5 == 0 or pi == len(prompts) - 1:
            rate = (pi + 1) * seeds / (time.perf_counter() - t0)
            eta = (len(prompts) - pi - 1) * seeds / max(rate, 1e-6)
            print(f"  {(pi + 1) * seeds}/{len(prompts) * seeds}  "
                  f"({rate:.2f}/s, ETA {eta/60:.1f} min)")

    elapsed = time.perf_counter() - t0
    print(f"[{spec.nickname}] {len(generations)} generations in {elapsed/60:.1f} min")

    # Free model
    del model
    if dev == "mps":
        torch.mps.empty_cache()
    elif dev == "cuda":
        torch.cuda.empty_cache()

    # Score each sentence
    print(f"[{spec.nickname}] scoring with classifier (strict)…")
    sent_rows = []
    for g in generations:
        sents = split_sentences(g["generation"])
        for i, s in enumerate(sents):
            hits = detect_construction(s, strict=True)
            variants_hit = {h.variant.value for h in hits}
            sent_rows.append({
                "prompt_idx": g["prompt_idx"], "seed": g["seed"],
                "sent_idx": i, "n_sents_in_gen": len(sents),
                "sentence": s,
                "C1": "C1" in variants_hit, "C2": "C2" in variants_hit,
                "C3": "C3" in variants_hit, "C4": "C4" in variants_hit,
                "any_core": bool(variants_hit & {"C1", "C2", "C3"}),
                "rel_position": i / max(len(sents) - 1, 1),
            })

    n_total = len(sent_rows)
    n_pos = sum(1 for r in sent_rows if r["any_core"])
    n_c1 = sum(1 for r in sent_rows if r["C1"])
    n_c2 = sum(1 for r in sent_rows if r["C2"])
    n_c3 = sum(1 for r in sent_rows if r["C3"])
    any_core_rate = n_pos / max(n_total, 1)
    c3_share = n_c3 / max(n_pos, 1)

    # H17 position effect via Mann-Whitney
    pos_when_construction = [r["rel_position"] for r in sent_rows if r["any_core"]]
    pos_when_not = [r["rel_position"] for r in sent_rows if not r["any_core"]]
    if len(pos_when_construction) >= 2 and len(pos_when_not) >= 2:
        u, p_two = stats.mannwhitneyu(pos_when_construction, pos_when_not,
                                       alternative="two-sided")
        median_pos = float(np.median(pos_when_construction))
        median_neg = float(np.median(pos_when_not))
    else:
        u, p_two, median_pos, median_neg = float("nan"), 1.0, float("nan"), float("nan")

    return {
        "model": spec.nickname,
        "hf_name": spec.hf_name,
        "n_prompts": len(prompts),
        "seeds": seeds,
        "n_generations": len(generations),
        "n_sentences": n_total,
        "n_construction_positive": n_pos,
        "any_core_rate": any_core_rate,
        "c1_count": n_c1, "c2_count": n_c2, "c3_count": n_c3,
        "c3_share_of_any_core": c3_share,
        "h17_median_pos_construction": median_pos,
        "h17_median_pos_no_construction": median_neg,
        "h17_mannwhitney_p": float(p_two),
        "h17_direction": ("opener" if median_pos < median_neg else
                          "closer" if median_pos > median_neg else "diffuse"),
        "generations": generations,
        "sentences": sent_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--n-prompts", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=150)
    args = ap.parse_args()

    result = run(args.model, n_prompts=args.n_prompts, seeds=args.seeds,
                  max_new_tokens=args.max_new_tokens)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / f"{result['model']}.json"
    out_json.write_text(json.dumps(result, indent=2))

    print("\n=== replication on", result["model"], "===")
    print(f"  any_core rate:       {result['any_core_rate']:.3%}")
    print(f"  C3 share of any_core: {result['c3_share_of_any_core']:.1%}")
    print(f"  H17 median rel-pos w/ construction:    {result['h17_median_pos_construction']:.3f}")
    print(f"  H17 median rel-pos w/o construction:   {result['h17_median_pos_no_construction']:.3f}")
    print(f"  H17 Mann-Whitney p:  {result['h17_mannwhitney_p']:.4f}")
    print(f"  H17 direction:       {result['h17_direction']}")
    print(f"\n→ {out_json}")


if __name__ == "__main__":
    main()
