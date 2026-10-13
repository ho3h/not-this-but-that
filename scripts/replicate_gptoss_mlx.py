"""Full Path B replication on gpt-oss-20b (MLX 8-bit) — matched config + harmony.

Deviations from the prior runs, surfaced honestly in the writeup:

  1. Framework: MLX instead of transformers. Reason: transformers' MXFP4 →
     bf16 dequant path on MPS hits a missing torch.ldexp kernel
     (PyTorch 2.11). MLX is the only path on Apple Silicon for this model.
     Reproducibility — the PRD's actual constraint — is preserved via
     mx.random.seed.

  2. Quantization: 8-bit MLX weights vs fp16 for Gemma/Qwen. Sentence-level
     construction statistics should be robust to this; flagged.

  3. max_new_tokens: 600 (sanctioned by user 2026-05-21). Reason: gpt-oss
     at reasoning:medium spends ~200-250 tokens on the analysis channel
     before emitting any `final` content. Matching the Gemma/Qwen 150
     budget literally would leave gpt-oss with empty `final` channels in
     >80% of cases (verified by the §2 gate at 150 tokens). Raising to
     600 (~CoT + 150-token final) gives gpt-oss the same FINAL-output
     space the other models had. All other generation params unchanged:
     temperature 0.8, top_p 0.95, reasoning: medium, 30 D2 prompts ×
     3 seeds = 90 generations.

  4. The §2 gate already verified clean channel separation on 5 samples
     at 600 tokens. The full run only writes the extracted `final`
     channel to the generations JSONL; analysis is discarded but its
     length is recorded per-generation for the comparison table.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
from openai_harmony import (
    Conversation, HarmonyEncodingName, Message, ReasoningEffort,
    Role, SystemContent, load_harmony_encoding,
)
from scipy import stats

from classifier import detect_construction

REPO = Path(__file__).resolve().parent.parent
D2_PATH = REPO / "data" / "D2_neutral_prompts.json"
OUT_PATH = REPO / "reports" / "replication" / "gpt_oss_20b.json"
MODEL_NAME = "lmstudio-community/gpt-oss-20b-MLX-8bit"

TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 600
N_PROMPTS = 30
N_SEEDS = 3

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(text.strip()) if len(s.strip()) > 30]


def build_prompt(encoding, user_text: str) -> list[int]:
    conversation = Conversation.from_messages([
        Message.from_role_and_content(
            Role.SYSTEM,
            SystemContent.new().with_reasoning_effort(ReasoningEffort.MEDIUM),
        ),
        Message.from_role_and_content(Role.USER, user_text),
    ])
    return encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)


def generate_token_ids(model, prompt_ids: list[int], stop_token_ids: set[int],
                        max_new_tokens: int, sampler) -> list[int]:
    prompt_arr = mx.array(prompt_ids)
    out_tokens = []
    for step, (tok, _logprobs) in enumerate(generate_step(
        prompt=prompt_arr, model=model, max_tokens=max_new_tokens, sampler=sampler,
    )):
        tid = int(tok.item()) if hasattr(tok, "item") else int(tok)
        if tid in stop_token_ids:
            break
        out_tokens.append(tid)
        if step + 1 >= max_new_tokens:
            break
    return out_tokens


def extract_final(encoding, completion_token_ids: list[int]) -> tuple[str, int, int, str | None]:
    """Returns (final_text, analysis_chars, final_chars, parse_error_or_None).

    The harmony parser is strict about message boundaries; when generation
    hits max_new_tokens mid-header it can raise "Unexpected EOS while
    waiting for message header to complete." We retry by progressively
    trimming trailing tokens — usually the truncation is a single half-
    emitted header. If that fails too, return parse_error and treat the
    generation as empty-final (logged in the metadata).
    """
    last_error = None
    for trim in range(0, 20):  # try original, then strip last 1, 2, ..., 19 tokens
        try_ids = completion_token_ids[:len(completion_token_ids) - trim] if trim else completion_token_ids
        if len(try_ids) < 4:
            break
        try:
            msgs = encoding.parse_messages_from_completion_tokens(try_ids, Role.ASSISTANT)
            analysis_chars = 0
            final_chunks = []
            for m in msgs:
                as_dict = m.to_dict()
                channel = as_dict.get("channel")
                contents = as_dict.get("content", [])
                if isinstance(contents, list):
                    text = "".join(c.get("text", "") if isinstance(c, dict) else "" for c in contents)
                else:
                    text = contents if isinstance(contents, str) else ""
                if channel == "final":
                    final_chunks.append(text)
                elif channel == "analysis":
                    analysis_chars += len(text)
            final_text = "".join(final_chunks).strip()
            return final_text, analysis_chars, len(final_text), None
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            continue
    return "", 0, 0, last_error


def main() -> None:
    print(f"loading {MODEL_NAME}…")
    t0 = time.perf_counter()
    model, _tokenizer = load(MODEL_NAME)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    stop_token_ids = set(encoding.stop_tokens_for_assistant_actions())
    sampler = make_sampler(temp=TEMPERATURE, top_p=TOP_P)

    prompts = json.loads(D2_PATH.read_text())["prompts"][:N_PROMPTS]
    total = N_PROMPTS * N_SEEDS
    print(f"generating {total} continuations…\n")

    generations = []
    t0 = time.perf_counter()
    for pi, prompt in enumerate(prompts):
        prompt_ids = build_prompt(encoding, prompt)
        for seed in range(N_SEEDS):
            mx.random.seed(seed)
            completion = generate_token_ids(
                model, prompt_ids, stop_token_ids, MAX_NEW_TOKENS, sampler,
            )
            final_text, analysis_chars, final_chars, parse_error = extract_final(encoding, completion)
            generations.append({
                "prompt_idx": pi, "prompt": prompt, "seed": seed,
                "generation": final_text,
                "analysis_chars": analysis_chars,
                "final_chars": final_chars,
                "n_completion_tokens": len(completion),
                "parse_error": parse_error,
            })
        if (pi + 1) % 3 == 0:
            done = (pi + 1) * N_SEEDS
            rate = done / (time.perf_counter() - t0)
            eta = (total - done) / max(rate, 1e-6)
            print(f"  {done}/{total}  ({rate:.2f}/s, ETA {eta/60:.1f} min)")

    elapsed = time.perf_counter() - t0
    print(f"\nall {len(generations)} done in {elapsed/60:.1f} min")

    # ---- Score ----
    print("scoring with strict classifier…")
    sent_rows = []
    for g in generations:
        sents = split_sentences(g["generation"])
        for i, s in enumerate(sents):
            hits = detect_construction(s, strict=True)
            vs = {h.variant.value for h in hits}
            sent_rows.append({
                "prompt_idx": g["prompt_idx"], "seed": g["seed"],
                "sent_idx": i, "n_sents_in_gen": len(sents),
                "sentence": s,
                "C1": "C1" in vs, "C2": "C2" in vs, "C3": "C3" in vs, "C4": "C4" in vs,
                "any_core": bool(vs & {"C1", "C2", "C3"}),
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
    pos_w = [r["rel_position"] for r in sent_rows if r["any_core"]]
    pos_n = [r["rel_position"] for r in sent_rows if not r["any_core"]]
    if len(pos_w) >= 2 and len(pos_n) >= 2:
        u, p_two = stats.mannwhitneyu(pos_w, pos_n, alternative="two-sided")
        median_pos = float(np.median(pos_w))
        median_neg = float(np.median(pos_n))
    else:
        u, p_two, median_pos, median_neg = float("nan"), 1.0, float("nan"), float("nan")

    mean_analysis_chars = float(np.mean([g["analysis_chars"] for g in generations]))
    mean_final_chars = float(np.mean([g["final_chars"] for g in generations]))
    n_parse_failures = sum(1 for g in generations if g.get("parse_error"))
    print(f"  parse failures (treated as empty-final): {n_parse_failures}/{len(generations)}")

    result = {
        "model": "gpt_oss_20b",
        "hf_name": MODEL_NAME,
        "framework": "mlx",
        "quantization": "8-bit MLX",
        "reasoning_effort": "medium",
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "n_prompts": N_PROMPTS,
        "n_seeds": N_SEEDS,
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
        "mean_analysis_chars": mean_analysis_chars,
        "mean_final_chars": mean_final_chars,
        "generations": generations,
        "sentences": sent_rows,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))

    print("\n=== gpt_oss_20b replication ===")
    print(f"  generations: {len(generations)}, sentences: {n_total}")
    print(f"  mean analysis chars: {mean_analysis_chars:.0f}")
    print(f"  mean final chars:    {mean_final_chars:.0f}")
    print(f"  any_core rate:       {any_core_rate:.4%}  ({n_pos}/{n_total})")
    print(f"  C1: {n_c1}, C2: {n_c2}, C3: {n_c3}")
    print(f"  C3 share of any_core: {c3_share:.1%}")
    print(f"  H17 median rel-pos w/ construction:    {median_pos:.3f}")
    print(f"  H17 median rel-pos w/o construction:   {median_neg:.3f}")
    print(f"  H17 Mann-Whitney p:  {p_two:.4f}")
    print(f"  H17 direction:       {result['h17_direction']}")
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
