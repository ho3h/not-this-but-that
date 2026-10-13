"""Capability checks for a switched-off model: ARC-Easy accuracy and WikiText-2 perplexity.

Both run every condition as one batch row per item, through the same
masked-ablation wrappers as generation (v2_mlx_multi), so the comparison is
paired and the only difference is the switch-off.

- ARC-Easy: 300 questions from the test split (data/arc_easy_300.json), shown in
  Gemma's chat template with lettered options and "Answer with the letter
  only."; the prediction is whichever of A/B/C/D gets the highest next-token
  probability.
- WikiText-2: the first 120,000 characters of the test split
  (data/wikitext2_test_excerpt.txt), scored in 512-token windows.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import v2_engine as E  # noqa: E402
import v2_mlx as M  # noqa: E402
import v2_mlx_multi as MM  # noqa: E402

REPO = E.REPO


def arc_accuracy(model, tok, wraps, doses: list[dict]) -> list[float]:
    qs = json.loads((REPO / "data/arc_easy_300.json").read_text())
    MM.set_row_masks(wraps, doses, 1)
    letters = ["A", "B", "C", "D"]
    letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in letters]
    correct = np.zeros(len(doses))
    for q in qs:
        body = q["q"] + "\n" + "\n".join(f"{l}. {c}" for l, c in zip(letters, q["choices"])) + "\nAnswer with the letter only."
        ids = M.chat_tokens(tok, body)
        lg = model(mx.array([ids] * len(doses)))[:, -1, :].astype(mx.float32)
        pick = np.array(mx.argmax(lg[:, letter_ids], axis=-1))
        correct += (pick == letters.index(q["answer"]))
    return list(correct / len(qs))


def wikitext_perplexity(model, tok, wraps, doses: list[dict], window: int = 512) -> list[float]:
    text = (REPO / "data/wikitext2_test_excerpt.txt").read_text()
    ids = tok.encode(text, add_special_tokens=False)
    MM.set_row_masks(wraps, doses, 1)
    nll = np.zeros(len(doses)); count = 0
    for i in range(0, len(ids) - 64, window):
        c = ids[i:i + window]
        x = mx.array([[tok.bos_token_id] + c] * len(doses))
        logits = model(x)[:, :-1, :].astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tgt = mx.array([c] * len(doses))
        nll += -np.array(mx.take_along_axis(lp, tgt[..., None], axis=-1).squeeze(-1).sum(axis=-1))
        count += len(c)
    return list(np.exp(nll / count))
