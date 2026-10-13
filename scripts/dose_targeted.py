"""Targeted follow-up to dose_response.py: switch off the escape routes too.

Exploratory. Uses the features reroute_attribution.py found behind the
negated family ("isn't just X; it's Y") and the cousin ("more than just X.
It's Y") in gemma-2-2b-it's own chat answers, and measures the same habit and
cost numbers as dose_response.py on the same 40 prompts and random numbers.

  normal
  L20 coalition (25)              the original group, chosen on the base model
  chat family top 25              chosen in the chat model at the substitution
  chat family + cousin top 25 each
  chat family + cousin top 100 each
  chat routes top 100 + all 500   plus every original promoter at layer 20

Writes reports/dose_targeted_generations.jsonl, reports/dose_targeted.{json,md}.
"""

from __future__ import annotations

import json
import re
import sys
import time

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import dose_response as DR  # noqa: E402
import v2_engine as E  # noqa: E402
import v2_mlx as M  # noqa: E402
import v2_mlx_multi as MM  # noqa: E402

sys.path.insert(0, str(E.REPO / "src"))
from classifier.detect_v3 import detect_v3  # noqa: E402

REPO = E.REPO
GENS = REPO / "reports" / "dose_targeted_generations.jsonl"


def build_doses():
    ra = json.loads((REPO / "reports/reroute_attribution.json").read_text())
    fam = [r["feature_idx"] for r in ra["family"]["ranked"] if r["mean_drop"] > 0]
    cou = [r["feature_idx"] for r in ra["cousin"]["ranked"] if r["mean_drop"] > 0]
    all500 = DR.ranked_promoters(20)
    doses = [
        ("normal", {}),
        ("L20 coalition (25)", {20: set(E.COALITION_25)}),
        ("chat family top 25", {20: set(fam[:25])}),
        ("chat family + cousin top 25", {20: set(fam[:25]) | set(cou[:25])}),
        ("chat family + cousin top 100", {20: set(fam[:100]) | set(cou[:100])}),
        ("chat routes top 100 + all 500", {20: set(fam[:100]) | set(cou[:100]) | set(all500)}),
    ]
    union = {20: sorted(set().union(*(d.get(20, set()) for _, d in doses)))}
    return doses, union


def main() -> None:
    doses, union = build_doses()
    model, tok, wraps = MM.load_multi("google/gemma-2-2b-it", union)
    prompts = [p for p in json.loads((REPO / "data/D4_confirmation_v2_prompts.json").read_text())["prompts"]
               if p["type"] == "M" and p["id"] % 2 == 1]
    D = len(doses)
    done = set()
    if GENS.exists():
        done = {json.loads(l)["id"] for l in GENS.read_text().splitlines() if l.strip()}
    MM.set_row_masks(wraps, [d for _, d in doses], DR.N_SAMPLES)
    t0 = time.time()
    for i, p in enumerate(prompts):
        if p["id"] in done:
            continue
        U = np.random.default_rng(30001 + p["id"]).random((DR.N_SAMPLES, DR.MAX_NEW)).astype(np.float32)
        outs = MM.sample_rows(model, tok, M.chat_tokens(tok, p["prompt"]), mx.array(np.tile(U, (D, 1))),
                              max_new_tokens=DR.MAX_NEW, temperature=DR.TEMP, top_p=DR.TOP_P)
        with GENS.open("a") as f:
            for d, (name, _) in enumerate(doses):
                for k in range(DR.N_SAMPLES):
                    f.write(json.dumps({"id": p["id"], "prompt": p["prompt"], "dose": name, "sample": k,
                                        "text": outs[d * DR.N_SAMPLES + k]}) + "\n")
        print(f"prompt {i + 1}/{len(prompts)} · {(time.time() - t0) / 60:.1f} min", flush=True)

    ppl = DR.perplexities(model, tok, wraps, doses)
    acc = DR.quiz(model, tok, wraps, doses)
    rows = [json.loads(l) for l in GENS.read_text().splitlines() if l.strip()]
    result = {"doses": []}
    for d, (name, dose) in enumerate(doses):
        texts = [r["text"] for r in rows if r["dose"] == name]
        tiers = [detect_v3(t) for t in texts]
        result["doses"].append({
            "name": name, "features_off": sum(len(v) for v in dose.values()), "answers": len(texts),
            "family": sum(bool(x["family"]) for x in tiers) / len(texts),
            "cousin": sum(bool(x["cousin"]) for x in tiers) / len(texts),
            "family_or_cousin": sum(bool(x["family"] or x["cousin"]) for x in tiers) / len(texts),
            "any_incl_beyond": sum(bool(x["family"] or x["cousin"] or x["beyond"]) for x in tiers) / len(texts),
            "but_share": sum(bool(re.search(r"\bbut\b", t, re.I)) for t in texts) / len(texts),
            "words": sum(len(t.split()) for t in texts) / len(texts),
            "degenerate": sum(DR.degenerate(t) for t in texts) / len(texts),
            "perplexity_ratio": ppl[d] / ppl[0], "quiz_accuracy": acc[d],
        })
    (REPO / "reports/dose_targeted.json").write_text(json.dumps(result, indent=2))
    md = ["# Targeted switch-off: the escape routes too (exploratory)", "",
          "| dose | features off | family | cousin | family or cousin | + goes beyond | \"but\" | degenerate | perplexity | quiz |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in result["doses"]:
        md.append(f"| {x['name']} | {x['features_off']} | {100 * x['family']:.0f}% | {100 * x['cousin']:.0f}% | "
                  f"{100 * x['family_or_cousin']:.0f}% | {100 * x['any_incl_beyond']:.0f}% | {100 * x['but_share']:.0f}% | "
                  f"{100 * x['degenerate']:.0f}% | {x['perplexity_ratio']:.2f}× | {100 * x['quiz_accuracy']:.0f}% |")
    (REPO / "reports/dose_targeted.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
