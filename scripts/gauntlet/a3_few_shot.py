"""A3 — Show the cure.

Few-shot anti-examples in the chat history. We prepend a brief "system"-style
preamble containing 4 demonstrations of de-slopped writing (the same content
the construction would have expressed, but stated declaratively). Tests
whether in-context style-transfer suppresses the construction.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from gauntlet.runner import (
    GauntletModel, attack_report, perplexity_on_d3, print_report,
    save_report, score_generations, write_eyeball,
)

REPO = Path(__file__).resolve().parent.parent.parent
TEST_PROMPTS = REPO / "data" / "d2_corpus" / "gauntlet_test_prompts.json"

SEEDS = [0, 1, 2]
MAX_NEW_TOKENS = 200

FEW_SHOT_PREAMBLE = """Write your answer in plain declarative prose. Avoid the negative-parallelism patterns LLMs over-use ("not X, it's Y"; "It's not just X — it's Y"; "less X, more Y"; etc.). Here are four examples of the same idea written normally:

Example 1
Style to avoid: "A neighbourhood library isn't a building full of books — it's a living community hub."
Plain rewrite: "A neighbourhood library is a community hub built around a collection of books."

Example 2
Style to avoid: "Not only does public transit reduce traffic, but it also improves air quality."
Plain rewrite: "Public transit reduces traffic and improves air quality."

Example 3
Style to avoid: "It's less about speed and more about consistency."
Plain rewrite: "Consistency matters more than speed in this case."

Example 4
Style to avoid: "Rather than complain about the rain, she packed a coat."
Plain rewrite: "She packed a coat instead of complaining about the rain."

Now answer the following in plain declarative prose:

"""


def main() -> None:
    gm = GauntletModel.load()
    prompts = json.loads(TEST_PROMPTS.read_text())["prompts"]
    print(f"[A3] {len(prompts)} test prompts × {len(SEEDS)} seeds = {len(prompts) * len(SEEDS)} generations × 2 conditions")

    baseline_gens = []
    intervened_gens = []
    t0 = time.perf_counter()
    for pi, prompt in enumerate(prompts):
        for seed in SEEDS:
            b = gm.generate(prompt, seed=seed, max_new_tokens=MAX_NEW_TOKENS)
            i = gm.generate(FEW_SHOT_PREAMBLE + prompt, seed=seed,
                             max_new_tokens=MAX_NEW_TOKENS)
            baseline_gens.append({"prompt_idx": pi, "prompt": prompt, "seed": seed,
                                   "generation": b})
            intervened_gens.append({"prompt_idx": pi, "prompt": prompt, "seed": seed,
                                     "generation": i})
        if (pi + 1) % 5 == 0:
            done = (pi + 1) * len(SEEDS) * 2
            total = len(prompts) * len(SEEDS) * 2
            rate = done / max(time.perf_counter() - t0, 1e-6)
            print(f"  {done}/{total}  ({rate:.2f}/s)")

    print("[A3] scoring with referee…")
    b_stats = score_generations(baseline_gens)
    i_stats = score_generations(intervened_gens)
    b_ppl = perplexity_on_d3(gm)
    i_ppl = b_ppl  # same caveat as A1 — in-context steering doesn't apply to D3 forward

    rep = attack_report(
        "A3", "Show the cure",
        baseline_stats=b_stats, intervened_stats=i_stats,
        baseline_ppl=b_ppl, intervened_ppl=i_ppl,
        extra={"preamble_chars": len(FEW_SHOT_PREAMBLE),
               "note": "Fluency assessed via EYEBALL; PPL ratio not applicable to "
                       "in-context interventions (intervention is in the prompt, "
                       "not the model)."},
    )
    print_report(rep)
    save_report("A3", rep, baseline_gens, intervened_gens)
    write_eyeball("A3", "Show the cure",
                   baseline_pairs=baseline_gens, intervened_pairs=intervened_gens,
                   notes="Four de-slopped examples in the preamble. Does in-context "
                         "style transfer beat the model's habit?")


if __name__ == "__main__":
    main()
