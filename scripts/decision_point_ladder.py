"""The chat-chosen features at the decision point: ladder, controls, leave-one-out.

Exploratory. Prefixes are cut from the NORMAL answers of the pre-registered
confirmation v3 (D2 prompts, which played no part in choosing the features),
just before the model commits to the construction:

  "A city's coffee culture"  -> P(" isn" / " aren" / " not" ...)   family
  "...the annual village fete, it's" -> P(" more" / " much" / ...) cousin

For each prefix, one batch holds every condition (same sequence, one row each):

  normal
  top 1 / 2 / 5 / 10 / 25 of the chat-chosen features
  20 draws of 25 random features
  20 draws of 25 features matched to the chat-25's firing rates
  leave-one-out: all 25 except feature i, for each i
  singles: feature i alone, for each i

Switch-off is the error-term-preserving ablation at layer 20, every position
except BOS (v2_mlx_multi). Reports mean P(commit) per condition and relative
drops. Writes reports/decision_point_ladder.{json,md}.
"""

from __future__ import annotations

import json
import random
import sys

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import reroute_attribution as RA  # noqa: E402
import v2_engine as E  # noqa: E402
import v2_mlx as M  # noqa: E402
import v2_mlx_multi as MM  # noqa: E402

sys.path.insert(0, str(E.REPO / "src"))
from classifier.detect_v3 import detect_v3, normalise  # noqa: E402

REPO = E.REPO
N_DRAWS = 20


def heldout_prefixes(max_per_kind: int = 60):
    rows = [json.loads(l) for l in (REPO / "reports/confirmation_v3_generations.jsonl").read_text().splitlines() if l.strip()]
    out = {"family": [], "cousin": []}
    seen = set()
    for r in rows:
        if r["condition"] != "normal":
            continue
        t = normalise(r["text"])
        hits = detect_v3(t)
        for kind in ("family", "cousin"):
            if len(out[kind]) >= max_per_kind:
                continue
            for h in hits[kind]:
                m = RA.CUT[kind].search(t, max(0, h.span[0] - 1), h.span[1])
                if not m or m.start() < 5:
                    continue
                cut = t[:m.start()]
                if (r["prompt"], cut) not in seen:
                    seen.add((r["prompt"], cut))
                    out[kind].append({"prompt": r["prompt"], "answer_prefix": cut})
                break
    return out


def main() -> None:
    prereg = json.loads((REPO / "reports/confirmation_v3_prereg.json").read_text())
    chat = prereg["chat_features"]
    layout = json.loads((REPO / "web/demo/layout.json").read_text())["features"]
    density = {int(f["idx"]): float(f.get("density", 0.0)) for f in layout}
    rng = random.Random(7)
    pool = [f for f in range(16384) if f not in set(chat)]
    random_sets = [rng.sample(pool, 25) for _ in range(N_DRAWS)]
    # matched: for each chat feature, a non-chat feature with the nearest firing rate (sampled among the 10 nearest)
    by_density = sorted(pool, key=lambda f: density.get(f, 0.0))
    dens_arr = np.array([density.get(f, 0.0) for f in by_density])
    matched_sets = []
    for _ in range(N_DRAWS):
        s = set()
        for f in chat:
            j = int(np.searchsorted(dens_arr, density.get(f, 0.0)))
            cands = [by_density[k] for k in range(max(0, j - 5), min(len(by_density), j + 5)) if by_density[k] not in s]
            s.add(rng.choice(cands))
        matched_sets.append(sorted(s))

    conds = [("normal", [])]
    conds += [(f"top {k}", chat[:k]) for k in (1, 2, 5, 10, 25)]
    conds += [(f"random {i}", s) for i, s in enumerate(random_sets)]
    conds += [(f"matched {i}", s) for i, s in enumerate(matched_sets)]
    conds += [(f"all but {f}", [g for g in chat if g != f]) for f in chat]
    conds += [(f"only {f}", [f]) for f in chat]
    union = sorted(set().union(*[set(s) for _, s in conds]))

    model, tok, wraps = MM.load_multi("google/gemma-2-2b-it", {20: union})
    MM.set_row_masks(wraps, [{20: set(s)} for _, s in conds], 1)
    target_ids = {k: sorted({tok.encode(x, add_special_tokens=False)[0] for x in v}) for k, v in RA.TARGETS.items()}
    sets = heldout_prefixes()
    print({k: len(v) for k, v in sets.items()}, "prefixes;", len(conds), "conditions", flush=True)

    probs = {k: [] for k in sets}
    for kind, items in sets.items():
        for i, it in enumerate(items):
            ids = M.chat_tokens(tok, it["prompt"]) + tok.encode(it["answer_prefix"], add_special_tokens=False)
            lg = model(mx.array([ids] * len(conds)))[:, -1, :].astype(mx.float32)
            p = np.array(mx.sum(mx.softmax(lg, axis=-1)[:, target_ids[kind]], axis=-1))
            probs[kind].append(p)
        print(kind, "done", flush=True)

    result = {"n_prefixes": {k: len(v) for k, v in sets.items()}, "conditions": [c for c, _ in conds], "kinds": {}}
    for kind, P in probs.items():
        P = np.array(P)  # (prefixes, conditions)
        mean = P.mean(axis=0)
        base = mean[0]
        drop = {c: float(1 - mean[j] / base) for j, (c, _) in enumerate(conds)}
        rand = [drop[f"random {i}"] for i in range(N_DRAWS)]
        matched = [drop[f"matched {i}"] for i in range(N_DRAWS)]
        loo = sorted(((f, float(drop["top 25"] - drop[f"all but {f}"])) for f in chat), key=lambda x: -x[1])
        result["kinds"][kind] = {
            "baseline_p": float(base),
            "ladder": {k: drop[f"top {k}"] for k in (1, 2, 5, 10, 25)},
            "random_25": {"mean": float(np.mean(rand)), "max": float(np.max(rand)), "draws": rand},
            "matched_25": {"mean": float(np.mean(matched)), "max": float(np.max(matched)), "draws": matched},
            "leave_one_out_cost": loo,
            "single_drop": sorted(((f, drop[f"only {f}"]) for f in chat), key=lambda x: -x[1]),
            "per_prefix_top25_drop_share_positive": float(np.mean(P[:, 0] > P[:, conds.index(("top 25", chat[:25]))])),
        }
    (REPO / "reports/decision_point_ladder.json").write_text(json.dumps(result, indent=2))
    lab = json.loads((REPO / "data/labels_cache.json").read_text())
    md = ["# The chat-chosen 25 at the decision point (exploratory)", "",
          "Held-out prefixes cut from normal answers in confirmation v3 (D2 prompts). P(commit) = probability the",
          "next token starts the negation (family) or the minimiser (cousin). Drops are relative to normal.", ""]
    for kind, d in result["kinds"].items():
        md += [f"## {kind} ({result['n_prefixes'][kind]} prefixes, baseline P(commit) {d['baseline_p']:.2f})", "",
               "| features switched off | drop in P(commit) |", "|---|---:|"]
        for k, v in d["ladder"].items():
            md.append(f"| top {k} | {100 * v:.0f}% |")
        md.append(f"| 25 random (20 draws) | mean {100 * d['random_25']['mean']:.1f}%, max {100 * d['random_25']['max']:.1f}% |")
        md.append(f"| 25 firing-rate-matched (20 draws) | mean {100 * d['matched_25']['mean']:.1f}%, max {100 * d['matched_25']['max']:.1f}% |")
        md += ["", "Leave-one-out (drop lost when feature i is left on):", ""]
        for f, c in d["leave_one_out_cost"][:6]:
            l = lab.get(str(f), {}); l = l.get("text") if isinstance(l, dict) else l
            md.append(f"- {f}: {100 * c:.1f} points · {l}")
        md += ["", "Alone (drop when only feature i is switched off):", ""]
        for f, c in d["single_drop"][:6]:
            l = lab.get(str(f), {}); l = l.get("text") if isinstance(l, dict) else l
            md.append(f"- {f}: {100 * c:.0f}% · {l}")
        md.append("")
    (REPO / "reports/decision_point_ladder.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
