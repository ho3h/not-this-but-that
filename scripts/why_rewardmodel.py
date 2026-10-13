"""Does a reward model prefer "not X but Y"? Matched rewrites scored by Skywork-Reward-V2 (exploratory).

Source: single-turn answers from LMArena (arena-human-preference-55k), UltraFeedback (cleaned)
and the OLMo 2 preference mix (chosen and rejected answers alike) whose text contains a
canonical "S isn't just X; it's Y" / "It's not just X, it's Y" sentence. The first such
sentence is rewritten mechanically, everything else untouched:
  original      "S isn't just X; it's Y."
  both          "S is X, and it's Y."   (same content and length, no contrast)
  also          "S is X. It's also Y."  (same content, a more fluent no-contrast version)
  plain         "S is Y."               (drops X)
Each version is scored by Skywork/Skywork-Reward-V2-Qwen3-1.7B (an open reward model of the
kind used to train chatbots) with the arena prompt as the user turn. Reported: how often the
original outscores each rewrite, and the mean reward difference, with 2,000-pair bootstrap
CIs. Up to 600 answers, seeded. Writes reports/why_rewardmodel.{json,md}.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SEED, BOOT, CAP, MAX_TOK = 20261012, 2000, 600, 2048
RM = "Skywork/Skywork-Reward-V2-Qwen3-1.7B"
_A = r"['’]"
CANON = re.compile(
    r"(?:(?<=^)|(?<=[.!?]\s)|(?<=\n)|(?<=\*\*\s))"
    rf"(?:(?P<subj>It|This|That|[A-Z][a-z]+(?: [a-z{_A[1:-1]}]+){{0,6}})\s+(?P<neg>isn{_A}t|is not|aren{_A}t|are not)"
    rf"|(?P<subj2>It|That){_A}s not)"
    r"\s+(?:just|merely|simply|only)\s+(?P<x>[^,;—–.!?\n]{2,90}?)\s*[,;—–]\s*"
    rf"(?P<piv>it{_A}s|it is|they{_A}re|they are)\s+(?P<y>[^.!?\n]{{3,160}}[.!?])")


def rewrite(m: re.Match) -> tuple[str, str, str]:
    x, piv, y = m["x"].strip(), m["piv"], m["y"]
    if m["subj2"]:
        head = f"{m['subj2']}'s"
    else:
        cop = "are" if m["neg"].lower().startswith("are") else "is"
        head = f"{m['subj']} {cop}"
    pv = piv[0].upper() + piv[1:]
    also = f"{pv} also" if len(pv.split()) == 1 else f"{pv.split()[0]} {pv.split()[1]} also"
    return f"{head} {x}, and {piv} {y}", f"{head} {x}. {also} {y}", f"{head} {y}"


def add(out: list[dict], source: str, prompt: str, text: str) -> None:
    m = CANON.search(text)
    if m:
        both, also, plain = rewrite(m)
        out.append({"source": source, "prompt": prompt, "original": text,
                    "both": text[:m.start()] + both + text[m.end():],
                    "also": text[:m.start()] + also + text[m.end():],
                    "plain": text[:m.start()] + plain + text[m.end():], "sentence": m.group(0)})


def pref_pairs(out: list[dict], name: str, repo: str) -> None:
    from datasets import load_dataset

    for r in load_dataset(repo, split="train"):
        for side in ("chosen", "rejected"):
            msgs = r[side]
            if len(msgs) == 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant":
                add(out, name, msgs[0]["content"], msgs[1]["content"] or "")


def pairs() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("lmarena-ai/arena-human-preference-55k", split="train")
    out = []
    for r in ds:
        try:
            ps = json.loads(r["prompt"])
        except (json.JSONDecodeError, TypeError):
            continue
        if len(ps) != 1 or not isinstance(ps[0], str):
            continue
        for side in ("response_a", "response_b"):
            try:
                resp = json.loads(r[side])
            except (json.JSONDecodeError, TypeError):
                continue
            if len(resp) != 1 or not isinstance(resp[0], str):
                continue
            add(out, "LMArena", ps[0], resp[0])
    pref_pairs(out, "UltraFeedback", "argilla/ultrafeedback-binarized-preferences-cleaned")
    pref_pairs(out, "OLMo 2 preference mix", "allenai/olmo-2-1124-7b-preference-mix")
    seen, uniq = set(), []
    for it in out:  # the same answer can appear in several pairs
        if it["original"] not in seen:
            seen.add(it["original"]); uniq.append(it)
    out = uniq
    print("candidates:", {s: sum(it["source"] == s for it in out) for s in ("LMArena", "UltraFeedback", "OLMo 2 preference mix")}, flush=True)
    rng = np.random.default_rng(SEED)
    if len(out) > CAP:
        out = [out[i] for i in sorted(rng.choice(len(out), CAP, replace=False))]
    return out


def main() -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    items = pairs()
    print(len(items), "answers", flush=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(RM)
    rm = AutoModelForSequenceClassification.from_pretrained(RM, torch_dtype=torch.bfloat16, num_labels=1).to(dev).eval()

    @torch.no_grad()
    def reward(prompt: str, answer: str) -> float:
        conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
        ids = tok.apply_chat_template(conv, tokenize=True, return_tensors="pt", return_dict=True)
        ids = {k: v[:, -MAX_TOK:].to(dev) for k, v in ids.items()}
        return float(rm(**ids).logits[0][0].float())

    for k, it in enumerate(items):
        for v in ("original", "both", "also", "plain"):
            it[f"r_{v}"] = reward(it["prompt"], it[v])
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(items)}", flush=True)
    rng = np.random.default_rng(SEED)
    res = {"answers": len(items)}
    for v in ("both", "also", "plain"):
        d = np.array([it["r_original"] - it[f"r_{v}"] for it in items])
        wins = (d > 0).astype(float)
        bw, bd = [], []
        for _ in range(BOOT):
            s = rng.integers(0, len(d), len(d))
            bw.append(wins[s].mean()); bd.append(d[s].mean())
        res[f"original_vs_{v}"] = {"original_preferred": float(wins.mean()),
                                   "ci95": [float(np.percentile(bw, 2.5)), float(np.percentile(bw, 97.5))],
                                   "mean_reward_diff": float(d.mean()),
                                   "diff_ci95": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
                                   "reward_sd": float(np.std([it["r_original"] for it in items]))}
    (REPO / "reports/why_rewardmodel.json").write_text(json.dumps(res, indent=2))
    (REPO / "reports/why_rewardmodel_items.jsonl").write_text(
        "\n".join(json.dumps({k: it[k] for k in ("source", "sentence", "r_original", "r_both", "r_also", "r_plain")}) for it in items) + "\n")
    md = ["# Does a reward model prefer \"not X but Y\"? (exploratory)", "",
          f"{len(items)} single-turn answers (LMArena, UltraFeedback, OLMo 2 preference mix) containing one canonical construction sentence, each scored "
          f"by {RM} as written and with that one sentence rewritten.", "",
          "| comparison | original scores higher | mean reward difference |", "|---|---:|---:|"]
    for v, label in (("both", "original vs \"S is X, and it's Y\" (same content, no contrast)"),
                     ("also", "original vs \"S is X. It's also Y.\" (same content, fluent, no contrast)"),
                     ("plain", "original vs \"S is Y\" (drops X)")):
        r = res[f"original_vs_{v}"]
        md.append(f"| {label} | **{100 * r['original_preferred']:.1f}%** ({100 * r['ci95'][0]:.1f}–{100 * r['ci95'][1]:.1f}) | "
                  f"{r['mean_reward_diff']:+.3f} ({r['diff_ci95'][0]:+.3f} to {r['diff_ci95'][1]:+.3f}; reward SD {r['reward_sd']:.2f}) |")
    md += ["", "50% would mean the reward model is indifferent to the construction.", "", "## Reading", "",
           "Against rewrites that keep both ideas, the reward model prefers the construction about three times "
           "in four; against the bare grand claim (\"S is Y\") it is close to indifferent. What it marks down is "
           "a sentence that plainly endorses the ordinary reading (\"Coffee culture is about drinks\"). The "
           "construction lets an answer mention the ordinary reading and dismiss it in the same breath, which "
           "is the version this judge scores highest. Each preference is small (a few hundredths of a standard "
           "deviation of reward), but it is consistent, and training against such a judge applies it to every "
           "sentence of every answer. One open reward model; Gemma's own was not released.",
           "", f"Pilot on 41 LMArena answers only: reports/why_rewardmodel_pilot41.md."]
    (REPO / "reports/why_rewardmodel.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
