"""§2 gate — Harmony channel extraction sanity check.

Generates 5 sample outputs from openai/gpt-oss-20b on D2 prompts, parses
each via the openai-harmony StreamableParser, extracts the `final`
channel only, and PRINTS BOTH the raw decoded output and the extracted
final string for manual inspection. Kill condition (per PRD §2): if
any of the 5 cannot be cleanly isolated to a final-only string, STOP
and surface — do not run the full replication.

We use the same generation config the prior runs will use, applied
end-to-end here, so the gate exercises the actual production pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from openai_harmony import (
    Conversation, DeveloperContent, HarmonyEncodingName, Message,
    ReasoningEffort, Role, SystemContent, load_harmony_encoding,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
D2_PATH = REPO / "data" / "D2_neutral_prompts.json"
MODEL_NAME = "openai/gpt-oss-20b"

# Matched Qwen config: temperature=0.8, top_p=0.95, max_new_tokens=150.
# Reasoning effort fixed at medium per §1.
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 150  # NB: with a reasoning model this caps total output, including CoT


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


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


def extract_final(encoding, completion_token_ids: list[int]) -> tuple[str, list[dict]]:
    """Parse completion tokens into harmony Messages; return (final_text, all_messages_meta).

    final_text concatenates the `text` of every Message whose channel == 'final'.
    all_messages_meta is a small dict per message so the gate inspector can see
    that no analysis/commentary slipped through.
    """
    stop_tokens = set(encoding.stop_tokens_for_assistant_actions())
    completion_token_ids = [int(t) for t in completion_token_ids if int(t) not in stop_tokens]
    msgs = encoding.parse_messages_from_completion_tokens(completion_token_ids, Role.ASSISTANT)
    meta = []
    final_chunks = []
    for m in msgs:
        as_dict = m.to_dict()
        channel = as_dict.get("channel")
        contents = as_dict.get("content", [])
        # contents is a list of {type, text} dicts (or similar)
        if isinstance(contents, list):
            text = "".join(
                c.get("text", "") if isinstance(c, dict) else ""
                for c in contents
            )
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
    dev = device()
    print(f"loading {MODEL_NAME} on {dev}…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype="auto", device_map=dev,
    )
    model.eval()

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()

    prompts = json.loads(D2_PATH.read_text())["prompts"][:5]
    print(f"running gate on {len(prompts)} prompts (reasoning: medium)…")

    for pi, prompt in enumerate(prompts):
        completion_prefix = build_prompt(encoding, prompt)
        input_ids = torch.tensor([completion_prefix], device=dev)
        attention_mask = torch.ones_like(input_ids)
        torch.manual_seed(0)
        if dev == "mps":
            torch.mps.manual_seed(0)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=stop_token_ids,
            )
        completion = out[0, input_ids.shape[1]:].tolist()
        raw_decoded = tokenizer.decode(completion, skip_special_tokens=False)
        final_text, meta = extract_final(encoding, completion)

        print(f"\n{'='*72}")
        print(f"PROMPT {pi}: {prompt}")
        print(f"{'='*72}")
        print(f"raw decoded (first 600 chars, SPECIAL TOKENS VISIBLE):")
        print(repr(raw_decoded[:600]))
        print()
        print(f"parsed messages ({len(meta)}):")
        for j, m in enumerate(meta):
            print(f"  [{j}] channel={m['channel']!r}  recipient={m['recipient']!r}  "
                  f"len={m['text_len']}  preview={m['text_preview'][:140]!r}")
        print()
        print(f"EXTRACTED FINAL ({len(final_text)} chars):")
        print(final_text[:600] if final_text else "(EMPTY)")


if __name__ == "__main__":
    main()
