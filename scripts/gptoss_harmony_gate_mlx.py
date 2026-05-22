"""§2 gate, MLX edition — Harmony channel extraction sanity check on Apple Silicon.

The transformers MXFP4 → bf16 path on MPS hits a missing `torch.ldexp` kernel
(PyTorch 2.11). The Apple-Silicon-native path is MLX. We use the community
8-bit MLX quantization of gpt-oss-20b — same architecture and tokenizer,
just 8-bit-quantized weights instead of bf16/MXFP4.

Methodological note (committed before the run, surfaced in the writeup):
  - PRD §1 specified "transformers or vLLM, not Ollama" because Ollama's
    seed control is weaker. MLX has proper deterministic seeding via
    `mx.random.seed`. The PRD's reproducibility constraint is satisfied;
    only the framework choice differs.
  - Quantization: this run uses 8-bit weights; the Gemma and Qwen runs
    used fp16. For sentence-level construction-rate statistics this is
    almost certainly fine, but the caveat is recorded.

The §2 gate itself is unchanged: generate 5 samples, parse via
openai-harmony, manually inspect for analysis/CoT leakage into the
"final" channel string.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler
from openai_harmony import (
    Conversation, HarmonyEncodingName, Message, ReasoningEffort,
    Role, SystemContent, load_harmony_encoding,
)

REPO = Path(__file__).resolve().parent.parent
D2_PATH = REPO / "data" / "D2_neutral_prompts.json"
MODEL_NAME = "lmstudio-community/gpt-oss-20b-MLX-8bit"

# Matched Qwen config (kept identical across all three models for cross-model
# comparison validity). Reasoning effort fixed at medium per PRD §1.
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 600
# CoT-aware budget adjustment, sanctioned by user 2026-05-21:
# Gemma/Qwen got 150 tokens of *final* output (no separate CoT). gpt-oss at
# reasoning:medium spends ~200-250 tokens on analysis before any final emits.
# To give it the same final-output space the other models had, total budget
# needs to be ~analysis + 150. 600 is generous; we'll measure mean final
# length in the run metadata for honest comparison.


def build_prompt(encoding, user_text: str) -> list[int]:
    """Render a harmony Conversation into completion-prompt tokens."""
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
    """Stream tokens out of mlx-lm's generate_step until a stop token or max-new."""
    prompt_arr = mx.array(prompt_ids)
    out_tokens = []
    for step, (tok, _logprobs) in enumerate(generate_step(
        prompt=prompt_arr, model=model, max_tokens=max_new_tokens, sampler=sampler,
    )):
        # mlx-lm 0.31 yields token as Python int (or 0-dim mx.array on other paths)
        tid = int(tok.item()) if hasattr(tok, "item") else int(tok)
        if tid in stop_token_ids:
            break
        out_tokens.append(tid)
        if step + 1 >= max_new_tokens:
            break
    return out_tokens


def extract_final(encoding, completion_token_ids: list[int]) -> tuple[str, list[dict]]:
    """Parse completion tokens into harmony Messages; concatenate `final`-channel text."""
    msgs = encoding.parse_messages_from_completion_tokens(completion_token_ids, Role.ASSISTANT)
    meta = []
    final_chunks = []
    for m in msgs:
        as_dict = m.to_dict()
        channel = as_dict.get("channel")
        contents = as_dict.get("content", [])
        if isinstance(contents, list):
            text = "".join(c.get("text", "") if isinstance(c, dict) else "" for c in contents)
        elif isinstance(contents, str):
            text = contents
        else:
            text = ""
        meta.append({"channel": channel, "recipient": as_dict.get("recipient"),
                     "text_preview": text[:200], "text_len": len(text)})
        if channel == "final":
            final_chunks.append(text)
    return "".join(final_chunks).strip(), meta


def main() -> None:
    print(f"loading {MODEL_NAME} via mlx-lm…")
    t0 = time.perf_counter()
    model, _tokenizer = load(MODEL_NAME)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    stop_token_ids = set(encoding.stop_tokens_for_assistant_actions())
    sampler = make_sampler(temp=TEMPERATURE, top_p=TOP_P)

    prompts = json.loads(D2_PATH.read_text())["prompts"][:5]
    print(f"running gate on {len(prompts)} prompts (reasoning: medium)…\n")

    for pi, prompt in enumerate(prompts):
        prompt_ids = build_prompt(encoding, prompt)
        mx.random.seed(0)  # deterministic per-call seed
        t0 = time.perf_counter()
        completion = generate_token_ids(
            model, prompt_ids, stop_token_ids, MAX_NEW_TOKENS, sampler,
        )
        elapsed = time.perf_counter() - t0
        final_text, meta = extract_final(encoding, completion)

        print(f"{'='*72}")
        print(f"PROMPT {pi}: {prompt}")
        print(f"  generated {len(completion)} tokens in {elapsed:.1f}s "
              f"({len(completion)/max(elapsed,1e-3):.1f} tok/s)")
        print(f"{'='*72}")
        print(f"parsed messages ({len(meta)}):")
        for j, m in enumerate(meta):
            print(f"  [{j}] channel={m['channel']!r}  recipient={m['recipient']!r}  "
                  f"len={m['text_len']}  preview={m['text_preview'][:140]!r}")
        print()
        print(f"EXTRACTED FINAL ({len(final_text)} chars):")
        print(final_text[:700] if final_text else "(EMPTY)")
        print()


if __name__ == "__main__":
    main()
