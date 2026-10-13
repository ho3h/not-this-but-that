"""Chat model vs base model, re-scored with detector v3 (exploratory, no GPU).

Uses the phase 2 generations (50 D2 prompts x 5 seeds per model; the base model
continues the raw prompt, the chat model answers it). Counts answers with the
negated family or the "more than just" cousin and bootstraps the ratio over
prompts. Writes reports/base_vs_chat_v3.md.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from classifier.detect_v3 import detect_v3  # noqa: E402


def hits(model: str) -> dict[int, list[bool]]:
    by: dict[int, list[bool]] = {}
    for line in (REPO / f"reports/phase2_generations_{model}.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            d = detect_v3(r["generation"])
            by.setdefault(r["prompt_idx"], []).append(bool(d["family"] or d["cousin"]))
    return by


def main() -> None:
    base, chat = hits("gemma_2b"), hits("gemma_2b_it")
    prompts = sorted(set(base) & set(chat))
    nb = sum(sum(base[p]) for p in prompts); nc = sum(sum(chat[p]) for p in prompts)
    na = sum(len(chat[p]) for p in prompts)
    rng = random.Random(0); ratios = []
    for _ in range(5000):
        s = [rng.choice(prompts) for _ in prompts]
        b = sum(sum(base[p]) for p in s)
        if b:
            ratios.append(sum(sum(chat[p]) for p in s) / b)
    ratios.sort()
    lo, hi = ratios[int(0.025 * len(ratios))], ratios[int(0.975 * len(ratios))]
    md = ["# Chat vs base model, detector v3 (exploratory)", "",
          f"Phase 2 generations, {len(prompts)} D2 prompts, {na} answers per model.", "",
          "| model | answers with family or cousin |", "|---|---:|",
          f"| gemma-2-2b (base) | {nb} / {na} |", f"| gemma-2-2b-it (chat) | {nc} / {na} |", "",
          f"Ratio chat/base: **{nc / nb:.1f}×** (95% prompt-clustered bootstrap CI {lo:.1f}× to {hi:.1f}×).", "",
          "The earlier 12× figure (`phase2_baseline.md`) counts only the strict C3 form per sentence and rests",
          "on very few base-model events; this is the broader and steadier count."]
    (REPO / "reports/base_vs_chat_v3.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
