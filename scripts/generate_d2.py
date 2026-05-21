"""Phase 2 — generate continuations on D2 for the 4-model baseline.

Models (PRD §8 Phase 2):
  - gemma-2-2b           (base)
  - gemma-2-2b-it        (instruct; chat template applied)
  - gpt2                 (base)
  - pythia-70m-deduped   (base)

Output: JSONL at reports/phase2_generations_<model>.jsonl with one line per
(prompt, seed): {model, prompt_idx, seed, prompt, generation, gen_tokens, ...}.

Sampling: temperature=0.8, top_p=0.95, ~150 new tokens. The construction rate
is a property of the sampling distribution, not the argmax — temperature=0 is
the wrong measurement here.

Usage:
    uv run python scripts/generate_d2.py --models pythia_70m gpt2 gemma_2b gemma_2b_it
    uv run python scripts/generate_d2.py --models gemma_2b_it --seeds 5 --max-new-tokens 150
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neograph.util import get_logger

log = get_logger("phase2.generate")

REPO_ROOT = Path(__file__).resolve().parent.parent
D2_PATH = REPO_ROOT / "data" / "D2_neutral_prompts.json"
OUT_DIR = REPO_ROOT / "reports"


@dataclass(frozen=True)
class ModelSpec:
    nickname: str
    hf_name: str
    chat_template: bool
    dtype: torch.dtype = torch.float16


MODELS: dict[str, ModelSpec] = {
    "pythia_70m": ModelSpec("pythia_70m", "EleutherAI/pythia-70m-deduped",
                            chat_template=False, dtype=torch.float32),
    "gpt2":        ModelSpec("gpt2", "gpt2", chat_template=False, dtype=torch.float32),
    "gemma_2b":    ModelSpec("gemma_2b", "google/gemma-2-2b",
                             chat_template=False, dtype=torch.float16),
    "gemma_2b_it": ModelSpec("gemma_2b_it", "google/gemma-2-2b-it",
                             chat_template=True, dtype=torch.float16),
}


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def make_input(prompt: str, tokenizer, chat_template: bool) -> dict:
    if chat_template:
        # Modern HF returns the templated string; tokenize it ourselves so we
        # get input_ids + attention_mask together.
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
                 top_p: float, dev: str) -> tuple[str, int]:
    torch.manual_seed(seed)
    if dev == "mps":
        torch.mps.manual_seed(seed)
    elif dev == "cuda":
        torch.cuda.manual_seed_all(seed)

    enc = make_input(prompt, tokenizer, chat_template)
    enc = {k: v.to(dev) for k, v in enc.items()}
    input_len = enc["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_tokens = out[0, input_len:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return text, gen_tokens.numel()


def run_model(spec: ModelSpec, prompts: list[str], *, seeds: int,
              max_new_tokens: int, temperature: float, top_p: float,
              out_path: Path) -> None:
    dev = device()
    log.info(f"[{spec.nickname}] loading {spec.hf_name} on {dev} ({spec.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_name, dtype=spec.dtype
    ).to(dev)
    model.eval()

    n_total = len(prompts) * seeds
    log.info(f"[{spec.nickname}] generating {n_total} continuations "
             f"({len(prompts)} prompts × {seeds} seeds, "
             f"~{max_new_tokens} tokens each)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with out_path.open("w") as f:
        for prompt_idx, prompt in enumerate(prompts):
            for seed in range(seeds):
                try:
                    text, n_tok = generate_one(
                        model, tokenizer, prompt,
                        chat_template=spec.chat_template,
                        seed=seed, max_new_tokens=max_new_tokens,
                        temperature=temperature, top_p=top_p, dev=dev,
                    )
                except Exception as e:
                    log.error(f"[{spec.nickname}] prompt {prompt_idx} seed {seed}: {e}")
                    continue
                row = {
                    "model": spec.nickname,
                    "hf_name": spec.hf_name,
                    "prompt_idx": prompt_idx,
                    "seed": seed,
                    "prompt": prompt,
                    "generation": text,
                    "gen_tokens": n_tok,
                    "chat_template": spec.chat_template,
                    "temperature": temperature,
                    "top_p": top_p,
                }
                f.write(json.dumps(row) + "\n")
            done = (prompt_idx + 1) * seeds
            if (prompt_idx + 1) % 10 == 0 or prompt_idx == len(prompts) - 1:
                elapsed = time.perf_counter() - t0
                rate = done / elapsed
                eta = (n_total - done) / rate if rate > 0 else float("inf")
                log.info(f"[{spec.nickname}] {done}/{n_total}  "
                         f"({rate:.2f}/s, ETA {eta/60:.1f} min)")

    # free
    del model
    if dev == "mps":
        torch.mps.empty_cache()
    elif dev == "cuda":
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0
    log.info(f"[{spec.nickname}] done in {elapsed/60:.1f} min → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                    choices=list(MODELS.keys()))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--prompts-file", type=Path, default=D2_PATH)
    ap.add_argument("--limit-prompts", type=int, default=0,
                    help="Cap prompts for quick smoke tests; 0 = all.")
    args = ap.parse_args()

    d2 = json.loads(args.prompts_file.read_text())
    prompts = d2["prompts"]
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]
    log.info(f"loaded {len(prompts)} prompts from {args.prompts_file}")

    for key in args.models:
        spec = MODELS[key]
        out_path = OUT_DIR / f"phase2_generations_{spec.nickname}.jsonl"
        run_model(spec, prompts, seeds=args.seeds,
                  max_new_tokens=args.max_new_tokens,
                  temperature=args.temperature, top_p=args.top_p,
                  out_path=out_path)


if __name__ == "__main__":
    main()
