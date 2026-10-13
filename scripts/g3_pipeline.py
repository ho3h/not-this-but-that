"""Gemma 3 replication of the decision-point procedure (exploratory stages).

Mirrors the Gemma 2 2B work step for step, with gemma-3-4b-it and Gemma Scope 2
instruct residual SAEs (16k, l0_medium):

  smoke      batched decoding matches batch-size-1 decoding; SAE fidelity
             (variance explained and loss recovered, excluding BOS) at each layer
  baseline   normal answers to D4 (selection) and D2 (held-out) prompts,
             5 samples each; habit rate by prompt type; per-feature firing rates
  attribute  at each layer: cut normal D4 answers just before they commit to the
             negation / the minimiser, switch off each active feature one at a
             time, rank by mean drop in P(commit) x sqrt(prefixes active)
  ladder     held-out decision points from D2 answers: top 1/2/5/10/25 of the
             family ranking, 20 random 25s, 20 firing-rate-matched 25s, singles,
             leave-one-out

  .venv/bin/python scripts/g3_pipeline.py smoke
  .venv/bin/python scripts/g3_pipeline.py baseline
  .venv/bin/python scripts/g3_pipeline.py attribute --layer 22
  .venv/bin/python scripts/g3_pipeline.py ladder --layer 22

Outputs go to reports/g3_*.json / .md and reports/g3_baseline_generations.jsonl.
Selection uses D4 answers only; D2 answers are held out for the ladder. A
pre-registered generation test follows as a separate, frozen runner.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import g3_mlx as G  # noqa: E402
import v2_engine as E  # noqa: E402
import v2_mlx as M  # noqa: E402
import v2_mlx_multi as MM  # noqa: E402

sys.path.insert(0, str(E.REPO / "src"))
from classifier.detect_v3 import detect_v3, normalise  # noqa: E402

REPO = E.REPO
MODEL = "google/gemma-3-4b-it"
LAYERS = (17, 22, 29)
N_FEAT = 16384
GENS = REPO / "reports/g3_baseline_generations.jsonl"
N, MAX_NEW, TEMP, TOP_P, SEED = 5, 300, 0.8, 0.95, 20261020
# First token of each word that starts the commitment (full words, so this works
# whatever the tokenizer splits off).
TARGET_WORDS = {"family": [" isn't", " aren't", " wasn't", " weren't", " doesn't", " not"],
                "cousin": [" more", " much", " so", " far", " way"]}
CUT = {"family": re.compile(r"\s(?:isn't|aren't|wasn't|weren't|doesn't|not)\b", re.I),
       "cousin": re.compile(r"\s(?:so much |much |far |way )?more than\b", re.I)}
WINDOW, MAX_CANDIDATES, N_DRAWS = 12, 256, 20


def target_ids(tok) -> dict[str, list[int]]:
    return {k: sorted({tok.encode(w, add_special_tokens=False)[0] for w in ws}) for k, ws in TARGET_WORDS.items()}


def prompt_sets() -> dict[str, list[dict]]:
    d4 = json.loads((REPO / "data/D4_confirmation_v2_prompts.json").read_text())["prompts"]
    d2 = json.loads((REPO / "data/D2_neutral_prompts.json").read_text())["prompts"]
    return {"D4": [{"set": "D4", "id": i, "prompt": p["prompt"], "type": p["type"]} for i, p in enumerate(d4)],
            "D2": [{"set": "D2", "id": i, "prompt": p} for i, p in enumerate(d2)]}


# ─── smoke ──────────────────────────────────────────────────────────────────
def smoke() -> None:
    import torch

    model, tok, _ = G.load_multi(MODEL, {})
    ids = M.chat_tokens(tok, "Explain how a city's coffee culture shapes its character.")
    U = np.random.default_rng(0).random((1, 60)).astype(np.float32)
    rows = MM.sample_rows(model, tok, ids, mx.array(np.tile(U, (4, 1))), max_new_tokens=60, temperature=0.8, top_p=0.95)
    single = MM.sample_rows(model, tok, ids, mx.array(U), max_new_tokens=60, temperature=0.8, top_p=0.95)[0]
    print("batched rows identical to single:", all(r == single for r in rows))
    print("sample:", single[:200].replace("\n", " / "))

    texts = [single] + MM.sample_rows(model, tok, M.chat_tokens(tok, "Why do people keep diaries?"),
                                      mx.array(np.random.default_rng(1).random((3, 120)).astype(np.float32)),
                                      max_new_tokens=120, temperature=0.8, top_p=0.95)
    out = {"batched_rows_identical_to_single": bool(all(r == single for r in rows)),
           "batched_rows_identical_to_each_other": len(set(rows)) == 1}
    prompt = M.chat_tokens(tok, "Why do people keep diaries?")
    for L in LAYERS:
        sae = G.load_sae(MODEL, L)
        cap = G.capture(model, L)
        ve_num = ve_den = 0.0
        nll = {"clean": 0.0, "splice": 0.0, "zero": 0.0}
        for t in texts:
            seq = prompt + tok.encode(t, add_special_tokens=False)
            x = mx.array([seq])
            tgt = np.array(seq[1:])
            logits = model(x).astype(mx.float32)
            h = np.array(cap.h.astype(mx.float32))[0, 1:]  # exclude BOS
            with torch.no_grad():
                rec = sae.decode(sae.encode(torch.tensor(h))).numpy()
            ve_num += float(((h - rec) ** 2).sum()); ve_den += float(((h - h.mean(0)) ** 2).sum())
            lp = np.array(logits[0, :-1] - mx.logsumexp(logits[0, :-1], axis=-1, keepdims=True))
            nll["clean"] += float(-lp[np.arange(len(tgt)), tgt].sum())
            inner = cap.layer
            for mode, r in (("splice", rec), ("zero", np.zeros_like(rec))):
                class Patch(nn.Module):
                    def __call__(self, xx, mask=None, cache=None, _r=r):
                        hh = inner(xx, mask, cache)
                        hf = np.array(hh.astype(mx.float32))
                        hf[0, 1:] = _r
                        return mx.array(hf).astype(hh.dtype)
                model.language_model.model.layers[L] = Patch()
                lg = model(x).astype(mx.float32)
                lpp = np.array(lg[0, :-1] - mx.logsumexp(lg[0, :-1], axis=-1, keepdims=True))
                nll[mode] += float(-lpp[np.arange(len(tgt)), tgt].sum())
                model.language_model.model.layers[L] = cap
        G.release_capture(model, L)
        out[L] = {"variance_explained_excl_bos": 1 - ve_num / ve_den,
                  "loss_recovered": (nll["zero"] - nll["splice"]) / (nll["zero"] - nll["clean"]),
                  "nll_total": {k: v for k, v in nll.items()}}
        print(L, out[L], flush=True)
    (REPO / "reports/g3_smoke.json").write_text(json.dumps(out, indent=2))


# ─── baseline ───────────────────────────────────────────────────────────────
def baseline() -> None:
    model, tok, _ = G.load_multi(MODEL, {})
    done = set()
    if GENS.exists():
        done = {(json.loads(l)["set"], json.loads(l)["id"]) for l in GENS.read_text().splitlines() if l.strip()}
    todo = [p for s in prompt_sets().values() for p in s if (p["set"], p["id"]) not in done]
    t0 = time.time()
    for i, p in enumerate(todo):
        U = np.random.default_rng(SEED + (0 if p["set"] == "D4" else 10_000) + p["id"]).random((N, MAX_NEW)).astype(np.float32)
        outs = MM.sample_rows(model, tok, M.chat_tokens(tok, p["prompt"]), mx.array(U),
                              max_new_tokens=MAX_NEW, temperature=TEMP, top_p=TOP_P)
        with GENS.open("a") as f:
            for k, t in enumerate(outs):
                f.write(json.dumps({**p, "sample": k, "text": t}) + "\n")
        print(f"prompt {i + 1}/{len(todo)} · {(time.time() - t0) / 60:.1f} min", flush=True)
    summarise_baseline()
    densities(model, tok)


def load_gens(which: str | None = None) -> list[dict]:
    rows = [json.loads(l) for l in GENS.read_text().splitlines() if l.strip()]
    return [r for r in rows if which is None or r["set"] == which]


def habit(t: str) -> bool:
    d = detect_v3(t)
    return bool(d["family"] or d["cousin"])


def summarise_baseline() -> None:
    rows = load_gens()
    res = {}
    for key, sub in (("D4 M", [r for r in rows if r["set"] == "D4" and r["type"] == "M"]),
                     ("D4 P", [r for r in rows if r["set"] == "D4" and r["type"] == "P"]),
                     ("D2", [r for r in rows if r["set"] == "D2"])):
        res[key] = {"answers": len(sub), "habit": sum(habit(r["text"]) for r in sub),
                    "family": sum(bool(detect_v3(r["text"])["family"]) for r in sub),
                    "cousin": sum(bool(detect_v3(r["text"])["cousin"]) for r in sub)}
    (REPO / "reports/g3_baseline.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def densities(model, tok) -> None:
    """Per-feature firing rate over D4 answer tokens, at each layer (for the matched control)."""
    rows = load_gens("D4")[::5]  # one sample per prompt
    out = {}
    for L in LAYERS:
        sae = G.load_sae(MODEL, L)
        W_enc, b_enc, thr = (np.array(t.detach().float().cpu().numpy()) for t in (sae.W_enc, sae.b_enc, sae.threshold))
        cap = G.capture(model, L)
        fires = np.zeros(N_FEAT); n_tok = 0
        for r in rows:
            pt = M.chat_tokens(tok, r["prompt"])
            ids = pt + tok.encode(r["text"], add_special_tokens=False)
            model(mx.array([ids]))
            h = np.array(cap.h.astype(mx.float32))[0, len(pt):]
            pre = h @ W_enc + b_enc
            fires += (pre > thr).sum(axis=0); n_tok += h.shape[0]
        G.release_capture(model, L)
        out[L] = (fires / n_tok).tolist()
        print("densities", L, "done", flush=True)
    (REPO / "reports/g3_densities.json").write_text(json.dumps(out))


# ─── attribute ──────────────────────────────────────────────────────────────
def cut_prefixes(rows: list[dict], max_per_kind: int, only_m: bool) -> dict[str, list[dict]]:
    """At most one decision point per prompt and kind (Gemma 3 uses the habit so often
    that taking the first N in file order would draw them from a handful of prompts)."""
    out = {"family": [], "cousin": []}
    seen = set()
    used = {"family": set(), "cousin": set()}
    for r in rows:
        if only_m and r.get("type") != "M":
            continue
        t = normalise(r["text"])
        hits = detect_v3(t)
        for kind in ("family", "cousin"):
            if len(out[kind]) >= max_per_kind or r["prompt"] in used[kind]:
                continue
            for h in hits[kind]:
                m = CUT[kind].search(t, max(0, h.span[0] - 1), h.span[1])
                if not m or m.start() < 5:
                    continue
                cut = t[:m.start()]
                if (r["prompt"], cut) not in seen:
                    seen.add((r["prompt"], cut))
                    used[kind].add(r["prompt"])
                    out[kind].append({"prompt": r["prompt"], "answer_prefix": cut})
                break
    return out


def attribute(layer: int) -> None:
    sae = G.load_sae(MODEL, layer)
    W_enc, b_enc, thr = (np.array(t.detach().float().cpu().numpy()) for t in (sae.W_enc, sae.b_enc, sae.threshold))
    model, tok, wraps = G.load_multi(MODEL, {layer: list(range(N_FEAT))})
    w = wraps[layer]
    tids = target_ids(tok)
    sets = cut_prefixes(load_gens("D4"), 40, only_m=True)
    print({k: len(v) for k, v in sets.items()}, "targets", tids, flush=True)
    captured = {}
    orig = type(w).__call__

    def spy(self, x, mask=None, cache=None):
        h = self.layer(x, mask, cache)
        captured["h"] = np.array(h.astype(mx.float32))
        return h

    results, t0 = {}, time.time()
    for kind, items in sets.items():
        drops, base = {}, []
        for it in items:
            ids = M.chat_tokens(tok, it["prompt"]) + tok.encode(it["answer_prefix"], add_special_tokens=False)
            type(w).__call__ = spy
            w.row_mask = None
            logits = model(mx.array([ids]))[:, -1, :].astype(mx.float32)
            type(w).__call__ = orig
            p0 = float(mx.sum(mx.softmax(logits, axis=-1)[0, tids[kind]]))
            h = captured["h"][0, -WINDOW:].astype(np.float64)
            pre = h @ W_enc.astype(np.float64) + b_enc
            total = np.where(pre > thr, pre, 0.0).sum(axis=0)
            active = np.argsort(-total)[:MAX_CANDIDATES]
            active = active[total[active] > 0]
            for s in range(0, len(active), 31):
                feats = active[s:s + 31]
                m = np.zeros((len(feats) + 1, N_FEAT), dtype=np.float32)  # row 0: nothing switched off
                m[np.arange(1, len(feats) + 1), feats] = 1.0
                w.row_mask = mx.array(m)
                lg = model(mx.array([ids] * (len(feats) + 1)))[:, -1, :].astype(mx.float32)
                p = np.array(mx.sum(mx.softmax(lg, axis=-1)[:, tids[kind]], axis=-1))
                for f, pf in zip(feats, p[1:]):
                    drops.setdefault(int(f), []).append(float(p[0]) - float(pf))
            w.row_mask = None
            base.append(p0)
            print(f"L{layer} {kind} {len(base)}/{len(items)} · P {p0:.2f} · {len(active)} cand · {(time.time() - t0) / 60:.1f} min", flush=True)
        ranked = sorted(((f, float(np.mean(d)), len(d), float(np.mean(d)) * math.sqrt(len(d))) for f, d in drops.items()),
                        key=lambda x: -x[3])
        results[kind] = {"n_prefixes": len(items), "mean_p_commit": float(np.mean(base)) if base else None,
                         "ranked": [{"feature_idx": f, "mean_drop": md, "n_active": n, "score": sc} for f, md, n, sc in ranked[:500]]}
    (REPO / f"reports/g3_attribution_L{layer}.json").write_text(json.dumps(results, indent=2))


# ─── ladder ─────────────────────────────────────────────────────────────────
def ladder(layer: int) -> None:
    attrib = json.loads((REPO / f"reports/g3_attribution_L{layer}.json").read_text())
    chat = [r["feature_idx"] for r in attrib["family"]["ranked"][:25]]
    density = np.array(json.loads((REPO / "reports/g3_densities.json").read_text())[str(layer)])
    rng = random.Random(7)
    pool = [f for f in range(N_FEAT) if f not in set(chat)]
    random_sets = [rng.sample(pool, 25) for _ in range(N_DRAWS)]
    by_density = sorted(pool, key=lambda f: density[f])
    dens_arr = density[by_density]
    matched_sets = []
    for _ in range(N_DRAWS):
        s = set()
        for f in chat:
            j = int(np.searchsorted(dens_arr, density[f]))
            cands = [by_density[k] for k in range(max(0, j - 5), min(len(by_density), j + 5)) if by_density[k] not in s]
            s.add(rng.choice(cands))
        matched_sets.append(sorted(s))
    conds = [("normal", [])] + [(f"top {k}", chat[:k]) for k in (1, 2, 5, 10, 25)]
    conds += [(f"random {i}", s) for i, s in enumerate(random_sets)]
    conds += [(f"matched {i}", s) for i, s in enumerate(matched_sets)]
    conds += [(f"all but {f}", [g for g in chat if g != f]) for f in chat]
    conds += [(f"only {f}", [f]) for f in chat]
    union = sorted(set().union(*[set(s) for _, s in conds]))
    model, tok, wraps = G.load_multi(MODEL, {layer: union})
    MM.set_row_masks(wraps, [{layer: set(s)} for _, s in conds], 1)
    tids = target_ids(tok)
    sets = cut_prefixes(load_gens("D2"), 60, only_m=False)
    print({k: len(v) for k, v in sets.items()}, "held-out prefixes;", len(conds), "conditions", flush=True)
    result = {"layer": layer, "chat_25": chat, "matched_draw0": matched_sets[0],
              "n_prefixes": {k: len(v) for k, v in sets.items()}, "kinds": {}}
    for kind, items in sets.items():
        P = []
        for it in items:
            ids = M.chat_tokens(tok, it["prompt"]) + tok.encode(it["answer_prefix"], add_special_tokens=False)
            P.append(np.array(mx.sum(mx.softmax(model(mx.array([ids] * len(conds)))[:, -1, :].astype(mx.float32), axis=-1)[:, tids[kind]], axis=-1)))
        if not P:
            continue
        mean = np.array(P).mean(axis=0)
        drop = {c: float(1 - mean[j] / mean[0]) for j, (c, _) in enumerate(conds)}
        rand = [drop[f"random {i}"] for i in range(N_DRAWS)]
        matched = [drop[f"matched {i}"] for i in range(N_DRAWS)]
        result["kinds"][kind] = {
            "baseline_p": float(mean[0]),
            "ladder": {k: drop[f"top {k}"] for k in (1, 2, 5, 10, 25)},
            "random_25": {"mean": float(np.mean(rand)), "max": float(np.max(rand))},
            "matched_25": {"mean": float(np.mean(matched)), "max": float(np.max(matched))},
            "leave_one_out_cost": sorted(((f, float(drop["top 25"] - drop[f"all but {f}"])) for f in chat), key=lambda x: -x[1]),
            "single_drop": sorted(((f, drop[f"only {f}"]) for f in chat), key=lambda x: -x[1]),
        }
        print(kind, json.dumps({k: v for k, v in result["kinds"][kind].items() if k in ("baseline_p", "ladder", "random_25", "matched_25")}), flush=True)
    (REPO / f"reports/g3_ladder_L{layer}.json").write_text(json.dumps(result, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["smoke", "baseline", "attribute", "ladder", "summary"])
    ap.add_argument("--layer", type=int, default=22)
    a = ap.parse_args()
    {"smoke": smoke, "baseline": baseline, "summary": summarise_baseline,
     "attribute": lambda: attribute(a.layer), "ladder": lambda: ladder(a.layer)}[a.stage]()


if __name__ == "__main__":
    main()
