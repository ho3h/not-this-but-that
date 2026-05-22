"""G5 — Build the hand-labeled referee holdout.

100 sentences, mixed AI + human + adversarial:
  - 30 AI sentences sampled from existing Phase 2 generations (random)
  - 20 AI sentences sampled from existing Phase 2 generations that contain
    construction-suspicious lexicon (over-sampling positives so the referee's
    recall is testable)
  - 30 human sentences from D3
  - 20 adversarial human-written sentences that LOOK like they might trip
    each form (the FP test — does the referee correctly reject)

The script SAMPLES the sentences and writes them to a JSONL without labels.
Hand labels go in a separate file (committed before referee is scored).
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
PHASE2_FILES = [
    REPO / "reports" / "phase2_generations_gemma_2b_it.jsonl",
    REPO / "reports" / "phase2_generations_gemma_2b.jsonl",
    REPO / "reports" / "phase2_generations_pythia_70m.jsonl",
    REPO / "reports" / "phase2_generations_gpt2.jsonl",
]
D3_PATH = REPO / "data" / "D3_fluency.txt"
OUT_PATH = REPO / "data" / "d2_corpus" / "referee_holdout_sentences.jsonl"

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“])")

SUSPICIOUS = re.compile(
    r"\b(not|isn'?t|aren'?t|wasn'?t|weren'?t|less\s+\w+\s+more|"
    r"no\s+\w+\.\s+no\s+\w+\.|far\s+from|rather\s+than|not\s+only|not\s+just)\b",
    re.IGNORECASE,
)


# 20 adversarial human-written sentences. Hand-authored to trip the referee's
# surface lexicon in non-construction ways — testing precision specifically.
ADVERSARIAL = [
    # "Not"-containing but not the construction
    "She did not arrive on time, which threw the whole afternoon off schedule.",
    "Whether or not we should proceed remains the central question of the meeting.",
    "The committee voted not to amend the proposal until further notice.",
    "He's not particularly fond of jazz, though he tolerates it at parties.",
    "I'm not entirely convinced by the argument she made yesterday.",
    "It is not unusual to feel anxious before a performance.",
    "She is not, by any reasonable measure, an unkind person.",
    # "Less"-containing but not F5
    "She earned less than he did despite having more experience.",
    "There were fewer people there than expected, though no less enthusiastic.",
    # "No"-containing but not F6
    "There is no easy answer to the question of why this kept happening.",
    "She had no idea what time it was when she finally looked up.",
    # "Far from" used as a comparative
    "The shop was far from the station, which made the trip awkward.",
    # "Rather" used adverbially
    "I'd rather walk than drive in this weather.",
    "The meeting was rather long but ended on a productive note.",
    # "But" in normal coordination, not the construction
    "She likes coffee but prefers tea in the afternoon.",
    "He arrived early but had to wait outside.",
    # Real constructions in human prose (positive controls)
    "It is not what we say but what we do that matters most.",  # real F1
    "Less government and more freedom — that was the platform.",  # real F5
    "Rather than complain about the weather, she packed a coat.",  # real F7
    "This isn't the season for tomatoes. It's the season for squash.",  # real F2
]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(text.strip()) if 40 <= len(s.strip()) < 400]


def collect_ai_pool() -> list[dict]:
    pool = []
    for p in PHASE2_FILES:
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for s in split_sentences(row["generation"]):
                if s.startswith(("*", "#", "-")):
                    continue
                pool.append({"text": s, "source": p.stem})
    return pool


def collect_human_pool() -> list[dict]:
    text = D3_PATH.read_text()
    return [{"text": s, "source": "D3"} for s in split_sentences(text)]


def main() -> None:
    rng = random.Random(41)
    ai_pool = collect_ai_pool()
    human_pool = collect_human_pool()
    print(f"AI pool: {len(ai_pool)} sentences")
    print(f"Human pool: {len(human_pool)} sentences")

    # Split AI pool into suspicious / not, sample 20 / 30
    ai_suspicious = [s for s in ai_pool if SUSPICIOUS.search(s["text"])]
    ai_normal = [s for s in ai_pool if not SUSPICIOUS.search(s["text"])]
    rng.shuffle(ai_suspicious)
    rng.shuffle(ai_normal)
    ai_sample = ai_suspicious[:20] + ai_normal[:30]

    # Human: take 30
    rng.shuffle(human_pool)
    human_sample = human_pool[:30]

    # Adversarial: 20 (deterministic; hand-authored)
    adv_sample = [{"text": t, "source": "adversarial"} for t in ADVERSARIAL]

    all_rows = ai_sample + human_sample + adv_sample
    rng.shuffle(all_rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        f.write(json.dumps({
            "_meta": "Referee holdout (G5). 100 sentences: 50 AI from Phase 2 gens (30 random + 20 suspicious-lexicon-over-sampled) + 30 D3 human + 20 adversarial. Hand-labels go in referee_holdout_labels.jsonl, committed BEFORE the referee runs on this set.",
            "_n": len(all_rows),
        }) + "\n")
        for i, r in enumerate(all_rows):
            f.write(json.dumps({
                "id": f"H{i:03d}",
                "text": r["text"],
                "source": r["source"],
            }) + "\n")
    print(f"wrote {len(all_rows)} sentences to {OUT_PATH}")


if __name__ == "__main__":
    main()
