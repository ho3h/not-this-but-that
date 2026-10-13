"""Find the layer-20 features behind the escape routes, in the chat model itself.

Exploratory. The original coalition was selected on the base model at the
comma before the turn. Here selection runs on gemma-2-2b-it, in chat format,
inside real answers, at the point where the substitution commits:

  "...isn't just a pond;" -> P(next token starts "it's / they're / it ...")
  "...is more than just a waterhole for feathered friends." -> P(" It's" ...)

Prefixes come from normal (nothing switched off) answers in
reports/confirmation_v2_generations.jsonl that contain the family or the
cousin (detector v3), cut right after the pivot punctuation. For each prefix,
every layer-20 feature active anywhere in the answer is switched off (one per
batch row, at every position except BOS, as in the generation tests) and the drop in P(substitution) is
recorded. Features are ranked by mean drop x sqrt(prefixes active), as in
pivot_attribution.py.

Writes reports/reroute_attribution.json.
"""

from __future__ import annotations

import json
import math
import re
import sys

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import v2_engine as E  # noqa: E402
import v2_mlx as M  # noqa: E402
import v2_mlx_multi as MM  # noqa: E402

sys.path.insert(0, str(E.REPO / "src"))
from classifier.detect_v3 import detect_v3, normalise  # noqa: E402

REPO = E.REPO
# Decision points. Family: the token that starts the negation (" isn" of
# "isn't", " not" after "is"). Cousin: the token that starts the minimiser
# (" more" / " much" / " so" / " far" / " way" before "more than just").
TARGETS = {"family": [" isn", " aren", " wasn", " weren", " doesn", " not"],
           "cousin": [" more", " much", " so", " far", " way"]}
CUT = {"family": re.compile(r"\s(?:isn't|aren't|wasn't|weren't|doesn't|not)\b", re.I),
       "cousin": re.compile(r"\s(?:so much |much |far |way )?more than\b", re.I)}
WINDOW, MAX_CANDIDATES = 12, 256


def prefixes(max_per_kind: int = 40):
    """Normal answers containing the form, cut just before it commits."""
    rows = [json.loads(l) for l in (REPO / "reports/confirmation_v2_generations.jsonl").read_text().splitlines() if l.strip()]
    out = {"family": [], "cousin": []}
    seen = set()
    for r in rows:
        if r["type"] != "M":
            continue
        t = normalise(r["baseline"])
        hits = detect_v3(t)
        for kind in ("family", "cousin"):
            if len(out[kind]) >= max_per_kind:
                continue
            for h in hits[kind]:
                m = CUT[kind].search(t, max(0, h.span[0] - 1), h.span[1])
                if not m or m.start() < 5:
                    continue
                cut = t[:m.start()]
                if (r["prompt"], cut) not in seen:
                    seen.add((r["prompt"], cut))
                    out[kind].append({"prompt": r["prompt"], "answer_prefix": cut})
                break
    return out


def main() -> None:
    sae = E.load_sae()
    W_enc, b_enc, thr, W_dec = (np.array(t.detach().float().cpu().numpy()) for t in (sae.W_enc, sae.b_enc, sae.threshold, sae.W_dec))
    model, tok, wraps = MM.load_multi("google/gemma-2-2b-it", {20: list(range(16384))})
    w = wraps[20]
    target_ids = {k: sorted({tok.encode(x, add_special_tokens=False)[0] for x in v}) for k, v in TARGETS.items()}
    sets = prefixes()
    print({k: len(v) for k, v in sets.items()}, flush=True)

    # capture layer-20 activations to find active features
    captured = {}
    orig = type(w).__call__

    def spy(self, x, mask=None, cache=None):
        h = self.layer(x, mask, cache)
        captured["h"] = np.array(h.astype(mx.float32))
        return h

    stats = {}
    results = {}
    for kind, items in sets.items():
        drops = {}
        for it in items:
            pt = M.chat_tokens(tok, it["prompt"])
            ans = tok.encode(it["answer_prefix"], add_special_tokens=False)
            ids = pt + ans
            type(w).__call__ = spy
            w.row_mask = None
            logits = model(mx.array([ids]))[:, -1, :].astype(mx.float32)
            type(w).__call__ = orig
            p0 = float(mx.sum(mx.softmax(logits, axis=-1)[0, target_ids[kind]]))
            h = captured["h"][0, -WINDOW:].astype(np.float64)  # the words leading up to the turn
            pre = h @ W_enc.astype(np.float64) + b_enc
            acts = np.where(pre > thr, pre, 0.0)
            total = acts.sum(axis=0)
            active = np.argsort(-total)[:MAX_CANDIDATES]
            active = active[total[active] > 0]
            # one feature per row, switched off at every answer position
            B = 64
            for s in range(0, len(active), B):
                feats = active[s:s + B]
                m = np.zeros((len(feats), 16384), dtype=np.float32)
                m[np.arange(len(feats)), feats] = 1.0
                w.row_mask = mx.array(m)
                lg = model(mx.array([ids] * len(feats)))[:, -1, :].astype(mx.float32)
                p = np.array(mx.sum(mx.softmax(lg, axis=-1)[:, target_ids[kind]], axis=-1))
                for f, pf in zip(feats, p):
                    drops.setdefault(int(f), []).append(p0 - float(pf))
            w.row_mask = None
            stats.setdefault(kind, []).append(p0)
            print(f"{kind} prefix {len(stats[kind])}/{len(items)} · P(commit) {p0:.2f} · {len(active)} candidates", flush=True)
        ranked = sorted(((f, float(np.mean(d)), len(d), float(np.mean(d)) * math.sqrt(len(d))) for f, d in drops.items()),
                        key=lambda x: -x[3])
        results[kind] = {"n_prefixes": len(items), "mean_p_commit": float(np.mean(stats[kind])),
                         "ranked": [{"feature_idx": f, "mean_drop": md, "n_active": n, "score": sc} for f, md, n, sc in ranked[:500]]}
        print(kind, "baseline P(commit)", round(float(np.mean(stats[kind])), 3), "top:", [r[0] for r in ranked[:10]], flush=True)
    results["coalition_overlap_top25"] = {k: len(set(r["feature_idx"] for r in results[k]["ranked"][:25]) & set(E.COALITION_25))
                                          for k in ("family", "cousin")}
    (REPO / "reports/reroute_attribution.json").write_text(json.dumps(results, indent=2))
    print(results["coalition_overlap_top25"])


if __name__ == "__main__":
    main()
