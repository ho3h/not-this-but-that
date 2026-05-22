"""A1 — Ask nicely.

System-prompt instruction asking the model not to use the construction.
Expected: partial compliance + relapse, or workarounds. The baseline
weapon. If asking-politely worked we wouldn't have a gauntlet.

Runs the same prompt set twice: once with no system instruction (baseline),
once with the system instruction (intervened). Both at the same seed so
the only difference is the prompt prefix.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from gauntlet.runner import (
    EYEBALL_DIR, GauntletModel, attack_report, perplexity_on_d3,
    print_report, save_report, score_generations, write_eyeball,
)

REPO = Path(__file__).resolve().parent.parent.parent
TEST_PROMPTS = REPO / "data" / "d2_corpus" / "gauntlet_test_prompts.json"

SEEDS = [0, 1, 2]
MAX_NEW_TOKENS = 200

SYSTEM_INSTRUCTION = (
    "Important style rule for this response: do NOT use the rhetorical pattern "
    "\"not X, but Y\" or any of its variants — including \"It's not just X, it's Y\", "
    "\"Not only X, but Y\", \"Less X, more Y\", \"No X. No Y. Just Z.\", "
    "\"Rather than X, Y\", or \"Far from X, Y\". Write your answer normally, "
    "in your own words, without negative-parallelism constructions of any kind."
)


def main() -> None:
    gm = GauntletModel.load()
    prompts = json.loads(TEST_PROMPTS.read_text())["prompts"]
    print(f"[A1] {len(prompts)} test prompts × {len(SEEDS)} seeds = {len(prompts) * len(SEEDS)} generations × 2 conditions")

    baseline_gens = []
    intervened_gens = []
    t0 = time.perf_counter()
    for pi, prompt in enumerate(prompts):
        for seed in SEEDS:
            b = gm.generate(prompt, seed=seed, max_new_tokens=MAX_NEW_TOKENS)
            i = gm.generate(prompt, seed=seed, max_new_tokens=MAX_NEW_TOKENS,
                            system_prompt=SYSTEM_INSTRUCTION)
            baseline_gens.append({"prompt_idx": pi, "prompt": prompt, "seed": seed,
                                   "generation": b})
            intervened_gens.append({"prompt_idx": pi, "prompt": prompt, "seed": seed,
                                     "generation": i})
        if (pi + 1) % 5 == 0:
            done = (pi + 1) * len(SEEDS) * 2
            total = len(prompts) * len(SEEDS) * 2
            rate = done / max(time.perf_counter() - t0, 1e-6)
            print(f"  {done}/{total}  ({rate:.2f}/s)")

    print("[A1] scoring with referee…")
    b_stats = score_generations(baseline_gens)
    i_stats = score_generations(intervened_gens)
    b_ppl = perplexity_on_d3(gm)  # no hook
    # Fluency under a system-prompt intervention is approximated by the same
    # baseline perplexity (the system instruction doesn't apply during the D3
    # forward pass since D3 is plain text, not a chat input).
    # We measure conversational fluency through coherence-of-output instead;
    # mark perplexity ratio as 1.000 to be honest about the limitation.
    i_ppl = b_ppl

    rep = attack_report(
        "A1", "Ask nicely",
        baseline_stats=b_stats, intervened_stats=i_stats,
        baseline_ppl=b_ppl, intervened_ppl=i_ppl,
        extra={"note": "Perplexity ratio not measured for A1 — the system-prompt "
               "intervention only manifests inside chat-formatted inputs; we score "
               "construction kill-rate on chat generations and report baseline PPL "
               "as a reference. Fluency for A1 is best read off the EYEBALL pairs."},
    )
    print_report(rep)
    save_report("A1", rep, baseline_gens, intervened_gens)
    write_eyeball("A1", "Ask nicely",
                   baseline_pairs=baseline_gens, intervened_pairs=intervened_gens,
                   notes="The polite ask. Did Gemma comply? Compare the same "
                         "(prompt, seed) under no-instruction vs system-instruction.")


if __name__ == "__main__":
    main()
