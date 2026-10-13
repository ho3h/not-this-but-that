"""How much has to be switched off before the habit disappears, and what it costs.

Exploratory. Escalating switch-offs (error-term-preserving SAE ablation, every
position except BOS) on gemma-2-2b-it in chat format, all decoded in one batch
per prompt with common random numbers, so every dose's answer matches the
normal one until that dose changes something:

  normal            nothing switched off
  L20 top 25        the coalition
  L20 top 100
  L20 all 500       every layer-20 feature whose removal lowered P(pivot)
  3 layers top 100  top 100 at layers 12, 20 and 25
  3 layers all      every pivot-promoting feature at layers 12, 20, 25 (1,894)

Habit: detector v3 tiers (family, cousin, "goes beyond", reversed) on 40
meaning/value prompts from D4 x 5 samples. Cost: perplexity on held-out human
prose (data/D3_fluency.txt), answer length, "but" share, degeneration (a
4-word phrase repeated 3+ times), and 20 short general-knowledge questions
answered greedily.

  .venv/bin/python scripts/dose_response.py
Writes reports/dose_response_generations.jsonl, reports/dose_response.{json,md}.
"""

from __future__ import annotations

import collections
import json
import re
import sys
import time

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import v2_engine as E  # noqa: E402
import v2_mlx as M  # noqa: E402
import v2_mlx_multi as MM  # noqa: E402

sys.path.insert(0, str(E.REPO / "src"))
from classifier.detect_v3 import detect_v3  # noqa: E402

REPO = E.REPO
GENS = REPO / "reports" / "dose_response_generations.jsonl"
N_SAMPLES, MAX_NEW, TEMP, TOP_P = 5, 150, 0.8, 0.95

QUIZ = [
    ("What is the capital of France?", r"paris"), ("What is 7 times 8?", r"\b56\b"),
    ("What is the chemical symbol for gold?", r"\bau\b"), ("Who wrote Romeo and Juliet?", r"shakespeare"),
    ("How many days are in a leap year?", r"\b366\b"), ("Which planet is known as the Red Planet?", r"mars"),
    ("At what temperature in Celsius does water boil at sea level?", r"\b100\b"),
    ("How do you say 'thank you' in French?", r"merci"), ("What is the largest ocean on Earth?", r"pacific"),
    ("What is 15 plus 27?", r"\b42\b"), ("In which year did the Second World War end?", r"\b1945\b"),
    ("Which gas do plants take in for photosynthesis?", r"carbon dioxide|co2|co₂"),
    ("What is the square root of 81?", r"\b9\b|nine"), ("Which language is mainly spoken in Brazil?", r"portuguese"),
    ("What is the opposite of 'ancient'?", r"modern|new|young|recent"), ("How many continents are there?", r"\b7\b|seven"),
    ("At what temperature in Fahrenheit does water freeze?", r"\b32\b"), ("Who painted the Mona Lisa?", r"leonardo|da vinci"),
    ("What is the capital of Japan?", r"tokyo"), ("Spell the word 'cat' backwards.", r"\btac\b"),
]


def ranked_promoters(layer: int) -> list[int]:
    f = "pivot_attribution.json" if layer == 20 else f"pivot_attribution_L{layer}.json"
    d = json.loads((REPO / "reports" / f).read_text())
    return [int(r["feature_idx"]) for r in d["full_ranked_by_score"] if r["mean_attribution_drop"] > 0]


def build_doses():
    p = {L: ranked_promoters(L) for L in (12, 20, 25)}
    assert set(p[20][:25]) == set(E.COALITION_25), "top 25 of the ranking must be the coalition"
    doses = [
        ("normal", {}),
        ("L20 top 25", {20: set(p[20][:25])}),
        ("L20 top 100", {20: set(p[20][:100])}),
        ("L20 all 500", {20: set(p[20])}),
        ("3 layers top 100", {L: set(p[L][:100]) for L in (12, 20, 25)}),
        ("3 layers all", {L: set(p[L]) for L in (12, 20, 25)}),
    ]
    union = {L: sorted(set().union(*(d.get(L, set()) for _, d in doses))) for L in (12, 20, 25)}
    return doses, union


def degenerate(text: str) -> bool:
    w = re.findall(r"\w+", text.lower())
    grams = collections.Counter(tuple(w[i:i + 4]) for i in range(len(w) - 3))
    return bool(grams) and max(grams.values()) >= 3


def perplexities(model, tok, wraps, doses) -> list[float]:
    text = (REPO / "data" / "D3_fluency.txt").read_text()
    ids = tok.encode(text)
    chunks = [ids[i:i + 256] for i in range(0, len(ids) - 32, 256)]
    MM.set_row_masks(wraps, [d for _, d in doses], 1)
    nll = np.zeros(len(doses)); count = 0
    for c in chunks:
        x = mx.array([[tok.bos_token_id] + c] * len(doses))
        logits = model(x)[:, :-1, :].astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tgt = mx.array([c] * len(doses))
        nll += -np.array(mx.take_along_axis(lp, tgt[..., None], axis=-1).squeeze(-1).sum(axis=-1))
        count += len(c)
    return list(np.exp(nll / count))


def quiz(model, tok, wraps, doses) -> list[float]:
    MM.set_row_masks(wraps, [d for _, d in doses], 1)
    correct = np.zeros(len(doses))
    U = mx.zeros((len(doses), 24))
    for q, pat in QUIZ:
        pt = M.chat_tokens(tok, q + " Answer in one word or number.")
        outs = MM.sample_rows(model, tok, pt, U, max_new_tokens=24, temperature=1e-6, top_p=1.0)
        correct += [bool(re.search(pat, o.lower())) for o in outs]
    return list(correct / len(QUIZ))


def main() -> None:
    doses, union = build_doses()
    model, tok, wraps = MM.load_multi("google/gemma-2-2b-it", union)
    prompts = [p for p in json.loads((REPO / "data/D4_confirmation_v2_prompts.json").read_text())["prompts"]
               if p["type"] == "M" and p["id"] % 2 == 1]
    D = len(doses)

    # Sanity: with nothing switched off in any row, all doses decode identically.
    MM.set_row_masks(wraps, [{} for _ in doses], N_SAMPLES)
    U0 = np.random.default_rng(1).random((N_SAMPLES, 40)).astype(np.float32)
    outs = MM.sample_rows(model, tok, M.chat_tokens(tok, prompts[0]["prompt"]), mx.array(np.tile(U0, (D, 1))),
                          max_new_tokens=40, temperature=TEMP, top_p=TOP_P)
    assert all(outs[d * N_SAMPLES + k] == outs[k] for d in range(D) for k in range(N_SAMPLES)), "CRN sanity failed"
    print("sanity ok: identical rows with nothing switched off", flush=True)

    done = set()
    if GENS.exists():
        done = {json.loads(l)["id"] for l in GENS.read_text().splitlines() if l.strip()}
    MM.set_row_masks(wraps, [d for _, d in doses], N_SAMPLES)
    t0 = time.time()
    for i, p in enumerate(prompts):
        if p["id"] in done:
            continue
        U = np.random.default_rng(30001 + p["id"]).random((N_SAMPLES, MAX_NEW)).astype(np.float32)
        outs = MM.sample_rows(model, tok, M.chat_tokens(tok, p["prompt"]), mx.array(np.tile(U, (D, 1))),
                              max_new_tokens=MAX_NEW, temperature=TEMP, top_p=TOP_P)
        with GENS.open("a") as f:
            for d, (name, _) in enumerate(doses):
                for k in range(N_SAMPLES):
                    f.write(json.dumps({"id": p["id"], "prompt": p["prompt"], "dose": name, "sample": k,
                                        "text": outs[d * N_SAMPLES + k]}) + "\n")
        print(f"prompt {i + 1}/{len(prompts)} · {(time.time() - t0) / 60:.1f} min", flush=True)

    ppl = perplexities(model, tok, wraps, doses)
    acc = quiz(model, tok, wraps, doses)
    rows = [json.loads(l) for l in GENS.read_text().splitlines() if l.strip()]
    result = {"doses": []}
    for d, (name, dose) in enumerate(doses):
        texts = [r["text"] for r in rows if r["dose"] == name]
        tiers = [detect_v3(t) for t in texts]
        rate = lambda k: sum(bool(x[k]) for x in tiers) / len(texts)  # noqa: E731
        result["doses"].append({
            "name": name, "features_off": sum(len(v) for v in dose.values()), "answers": len(texts),
            "family": rate("family"), "cousin": rate("cousin"),
            "family_or_cousin": sum(bool(x["family"] or x["cousin"]) for x in tiers) / len(texts),
            "beyond": rate("beyond"), "reversed": rate("reversed"),
            "any_incl_beyond": sum(bool(x["family"] or x["cousin"] or x["beyond"]) for x in tiers) / len(texts),
            "but_share": sum(bool(re.search(r"\bbut\b", t, re.I)) for t in texts) / len(texts),
            "words": sum(len(t.split()) for t in texts) / len(texts),
            "degenerate": sum(degenerate(t) for t in texts) / len(texts),
            "perplexity": ppl[d], "perplexity_ratio": ppl[d] / ppl[0], "quiz_accuracy": acc[d],
        })
    (REPO / "reports/dose_response.json").write_text(json.dumps(result, indent=2))
    md = ["# Dose response: switching off more and more (exploratory)", "",
          f"40 meaning/value prompts from D4 × {N_SAMPLES} samples per dose, chat format, detector v3.", "",
          "| dose | features off | family or cousin | + goes beyond | \"but\" | words | degenerate | perplexity on human prose | quiz |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in result["doses"]:
        md.append(f"| {x['name']} | {x['features_off']} | {100 * x['family_or_cousin']:.0f}% | {100 * x['any_incl_beyond']:.0f}% | "
                  f"{100 * x['but_share']:.0f}% | {x['words']:.0f} | {100 * x['degenerate']:.0f}% | "
                  f"{x['perplexity_ratio']:.2f}× | {100 * x['quiz_accuracy']:.0f}% |")
    (REPO / "reports/dose_response.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
