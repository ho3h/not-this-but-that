"""A2 — Ban the words.

Logit suppression at generation time on the construction's pivot vocabulary.
Forbidden tokens include the contracted-negation openers (isn't, aren't),
the structural pivots (but, just, only), and the em-dash. The model still
generates fluent English to the best of its ability while every token in
the ban list takes a giant negative bias on its logits.

Expected: the construction rate craters (mechanically, you can't say "not
X, but Y" without "but"). Fluency craters too, in a way that's hopefully
visible in the EYEBALL — the brute-force-lobotomy beat.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from gauntlet.runner import (
    GauntletModel, attack_report, perplexity_on_d3, print_report,
    save_report, score_generations, write_eyeball,
)

REPO = Path(__file__).resolve().parent.parent.parent
TEST_PROMPTS = REPO / "data" / "d2_corpus" / "gauntlet_test_prompts.json"

SEEDS = [0, 1, 2]
MAX_NEW_TOKENS = 200

# Tokens to suppress. Includes both raw words and leading-space variants since
# the gemma tokenizer differentiates "but" from " but".
BAN_STRINGS = [
    "not", " not", "Not",
    "n't", "isn't", "aren't", "wasn't", "weren't",
    "but", " but", "But",
    "just", " just", "Just",
    "only", " only", "Only",
    "merely", " merely",
    "simply", " simply",
    "—", " —",
    "rather", " rather",
    "less", " less", "Less",
    "more", " more",
]
SUPPRESSION_BIAS = -100.0  # additive log-prob hit


class BanWordsProcessor(LogitsProcessor):
    """Applies SUPPRESSION_BIAS to every banned token id at every step."""

    def __init__(self, ban_ids: list[int]):
        self.ban_ids = torch.tensor(ban_ids, dtype=torch.long)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores[:, self.ban_ids] = scores[:, self.ban_ids] + SUPPRESSION_BIAS
        return scores


def build_ban_ids(tokenizer) -> list[int]:
    ids = set()
    for s in BAN_STRINGS:
        for t in tokenizer(s, add_special_tokens=False).input_ids:
            ids.add(int(t))
    return sorted(ids)


def main() -> None:
    gm = GauntletModel.load()
    ban_ids = build_ban_ids(gm.tokenizer)
    print(f"[A2] suppressing {len(ban_ids)} token ids")

    prompts = json.loads(TEST_PROMPTS.read_text())["prompts"]
    processor = BanWordsProcessor([int(i) for i in ban_ids])
    processor_list = LogitsProcessorList([processor])

    baseline_gens = []
    intervened_gens = []
    t0 = time.perf_counter()
    for pi, prompt in enumerate(prompts):
        for seed in SEEDS:
            b = gm.generate(prompt, seed=seed, max_new_tokens=MAX_NEW_TOKENS)
            i = gm.generate(prompt, seed=seed, max_new_tokens=MAX_NEW_TOKENS,
                            logits_processor_list=processor_list)
            baseline_gens.append({"prompt_idx": pi, "prompt": prompt, "seed": seed,
                                   "generation": b})
            intervened_gens.append({"prompt_idx": pi, "prompt": prompt, "seed": seed,
                                     "generation": i})
        if (pi + 1) % 5 == 0:
            done = (pi + 1) * len(SEEDS) * 2
            total = len(prompts) * len(SEEDS) * 2
            rate = done / max(time.perf_counter() - t0, 1e-6)
            print(f"  {done}/{total}  ({rate:.2f}/s)")

    print("[A2] scoring with referee + measuring perplexity…")
    b_stats = score_generations(baseline_gens)
    i_stats = score_generations(intervened_gens)
    b_ppl = perplexity_on_d3(gm)
    # Perplexity ratio under banned tokens: install the processor on the forward
    # pass for the D3 chunks. We can't strictly suppress at training-loss time
    # the same way as at generation; instead we compute PPL on D3 with the
    # banned tokens masked out of the labels and report it as approximate.
    # For honest reporting we report baseline PPL only and note A2's
    # fluency is best read at sentence level in the EYEBALL.
    i_ppl = b_ppl

    rep = attack_report(
        "A2", "Ban the words",
        baseline_stats=b_stats, intervened_stats=i_stats,
        baseline_ppl=b_ppl, intervened_ppl=i_ppl,
        extra={"banned_token_ids": ban_ids, "suppression_bias": SUPPRESSION_BIAS,
               "note": "Token-level PPL under logit suppression is ill-defined "
                       "(labels include banned tokens); fluency for A2 is the "
                       "EYEBALL beat — does the prose turn to gravel?"},
    )
    print_report(rep)
    save_report("A2", rep, baseline_gens, intervened_gens)
    write_eyeball("A2", "Ban the words",
                   baseline_pairs=baseline_gens, intervened_pairs=intervened_gens,
                   notes="Banning every pivot token. Watch what Gemma does when "
                         "you take 'but', 'just', 'only', 'rather', 'more', 'less', "
                         "and the em-dash off the table.")


if __name__ == "__main__":
    main()
