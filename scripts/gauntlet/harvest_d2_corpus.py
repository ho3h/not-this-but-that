"""G3 — Harvest the D2-corpus for the kill-the-AI-ism gauntlet.

Generates ~2 280 continuations from gemma-2-2b-it across 152 register-diverse
prompts × 15 seeds × 300 tokens. JSONL output. Downstream:
  G6 (mine + hand-verify)  → src/gauntlet/harvest_detector.py
  G7 (author negatives)    → manual labour
  G8 (commit train/test)   → seeded 70/30 split

Time budget: ~3 hours on M5 Max MPS (gemma-2-2b-it fp16 ≈ 0.3 prompt/s at
300 tokens). Runs in background; checkpoint every 100 generations so a
kill doesn't lose the lot.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent.parent
PROMPTS_PATH = REPO / "data" / "d2_corpus" / "harvest_prompts.json"
OUT_PATH = REPO / "data" / "d2_corpus" / "harvest_generations.jsonl"
CHECKPOINT_EVERY = 50

HF_NAME = "google/gemma-2-2b-it"
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 300
DEFAULT_SEEDS = 15


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def make_input(prompt: str, tokenizer) -> dict:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False,
    )
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--limit-prompts", type=int, default=None)
    args = ap.parse_args()

    dev = device()
    print(f"[harvest] loading {HF_NAME} on {dev}…")
    tokenizer = AutoTokenizer.from_pretrained(HF_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(HF_NAME, dtype=torch.float16).to(dev)
    model.eval()

    prompts = json.loads(PROMPTS_PATH.read_text())["prompts"]
    if args.limit_prompts:
        prompts = prompts[:args.limit_prompts]
    total = len(prompts) * args.seeds
    print(f"[harvest] {len(prompts)} prompts × {args.seeds} seeds = {total} generations")

    # Resume support: read existing JSONL, skip (prompt_idx, seed) pairs already done
    done = set()
    if OUT_PATH.exists():
        for line in OUT_PATH.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            done.add((row["prompt_idx"], row["seed"]))
        print(f"[harvest] resuming with {len(done)} generations already on disk")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    f_out = OUT_PATH.open("a")

    t0 = time.perf_counter()
    n_done = len(done)
    for pi, prompt in enumerate(prompts):
        for seed in range(args.seeds):
            if (pi, seed) in done:
                continue
            try:
                torch.manual_seed(seed)
                if dev == "mps":
                    torch.mps.manual_seed(seed)
                enc = make_input(prompt, tokenizer)
                enc = {k: v.to(dev) for k, v in enc.items()}
                input_len = enc["input_ids"].shape[1]
                with torch.no_grad():
                    out = model.generate(
                        **enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                        temperature=TEMPERATURE, top_p=TOP_P,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                text = tokenizer.decode(out[0, input_len:], skip_special_tokens=True)
            except Exception as e:
                print(f"  ERR prompt={pi} seed={seed}: {e}")
                continue

            row = {"prompt_idx": pi, "prompt": prompt, "seed": seed, "generation": text}
            f_out.write(json.dumps(row) + "\n")
            f_out.flush()
            n_done += 1

            if n_done % 50 == 0 or n_done == total:
                elapsed = time.perf_counter() - t0
                rate = max(n_done - len(done), 1) / max(elapsed, 1)
                eta = (total - n_done) / max(rate, 1e-6)
                print(f"[harvest] {n_done}/{total}  ({rate:.2f}/s, ETA {eta/60:.1f} min)")

    f_out.close()
    print(f"[harvest] complete: {n_done}/{total} → {OUT_PATH}")


if __name__ == "__main__":
    main()
