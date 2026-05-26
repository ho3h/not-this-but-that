"""Joint-ablation ladder for the redundancy hypothesis.

PRD-level question: when we ablate not just feat 3223 but progressively larger
sets of co-causal features at the pre-pivot position, does P(pivot) collapse
(localizable to a constellation) or plateau (true Hydra at the SAE level)?

Three sources of joint sets — the "graph as hypothesis generator":
  - **attribution_top{2,5,10,25}** — most causal at the pivot decision, from
    `reports/pivot_attribution.json`.
  - **decoder_neighbors_{10}** — 3223 plus its top-9 decoder-cosine neighbors
    (graph edge DECODER_SIMILAR).
  - **coact_partners_{10}** — 3223 plus its top-9 co-activating partners
    (graph edge CO_ACTIVATES_WITH, ranked by PMI).

Each named condition is paired against `n_random` random-feature draws of the
same size. The plateau-vs-collapse shape of the ladder is the answer.

Method mirrors causal_m2.py — truncate D1 'with'-prompts at the pivot, hook at
the SAE post-encode at the last position, sum softmax over pivot token ids.

Usage:
    uv run python scripts/joint_ablation.py --n-random 3
    uv run python scripts/joint_ablation.py --n-random 5 --max-samples 80
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sae_lens import HookedSAETransformer, SAE

from neograph.config import SAE as GEMMA_SAE
from neograph.cypher import NeographClient
from neograph.util import get_logger

log = get_logger("joint_ablation")

REPO_ROOT = Path(__file__).resolve().parent.parent
D1_PATH = REPO_ROOT / "data" / "D1_contrast_pairs.jsonl"
ATTRIB_PATH = REPO_ROOT / "reports" / "pivot_attribution.json"
LABELS_PATH = REPO_ROOT / "data" / "labels_cache.json"
OUT_JSON = REPO_ROOT / "reports" / "joint_ablation.json"
OUT_MD = REPO_ROOT / "reports" / "joint_ablation.md"


PIVOT_STRINGS = {
    "C1": [", it", ", they", ", he", ", she", ", we", "—", "; it"],
    "C2": [", but", ", yet", ", also", " but ", " also "],
    "C3": ["—", ", it", " but ", ", but"],
}

PIVOT_RE = re.compile(
    r"[,;—–\-]\s*(?:it|that|this|he|she|we|they|these|those|there|but|also|yet)\b",
    re.IGNORECASE,
)

ANCHOR_FEATURE = 3223


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def truncate_to_pivot(text: str) -> str | None:
    m = PIVOT_RE.search(text)
    if not m:
        return None
    return text[: m.start()].rstrip()


def load_d1(variants: set[str]) -> list[dict]:
    rows = []
    for line in D1_PATH.read_text().splitlines():
        if not line.strip() or line.startswith('{"_meta'):
            continue
        d = json.loads(line)
        if d["variant"] in variants:
            rows.append(d)
    return rows


def collect_pivot_token_ids(model, variants: set[str]) -> dict[str, list[int]]:
    out = {}
    for v in variants:
        ids = set()
        for s in PIVOT_STRINGS[v]:
            for t in model.to_tokens(s, prepend_bos=False)[0].tolist():
                ids.add(int(t))
            for t in model.to_tokens(" " + s.strip(), prepend_bos=False)[0].tolist():
                ids.add(int(t))
        out[v] = sorted(ids)
    return out


def load_labels() -> dict[int, str]:
    if not LABELS_PATH.exists():
        return {}
    raw = json.loads(LABELS_PATH.read_text())
    return {int(k): v.get("text", "") for k, v in raw.items()}


def load_attribution_ladder() -> dict[str, list[int]]:
    """Read pivot_attribution.json and assemble the top-N ladders."""
    data = json.loads(ATTRIB_PATH.read_text())
    promotes = [r["feature_idx"] for r in data["top_promotes_pivot"]]
    if promotes[0] != ANCHOR_FEATURE:
        raise RuntimeError(
            f"Expected anchor {ANCHOR_FEATURE} as top promoter, got {promotes[0]}. "
            f"Re-run pivot_attribution.py before this experiment."
        )
    return {
        "single_3223": promotes[:1],
        "attrib_top2": promotes[:2],
        "attrib_top5": promotes[:5],
        "attrib_top10": promotes[:10],
        "attrib_top25": promotes[:25],
    }


def load_suppressor_set(n: int = 10) -> list[int]:
    """Features whose ablation *raises* P(pivot). Used as a directional sanity check."""
    data = json.loads(ATTRIB_PATH.read_text())
    return [r["feature_idx"] for r in data["top_suppresses_pivot"][:n]]


def load_graph_sets(anchor: int = ANCHOR_FEATURE, k: int = 9) -> dict[str, list[int]]:
    """Pull decoder-similar and co-activating partners of `anchor` from Neo4j.

    Returns {anchor + top-k by each measure}. Falls back to empty dict if the
    graph is unreachable or returns nothing, with a warning.
    """
    out: dict[str, list[int]] = {}
    try:
        with NeographClient() as c:
            r = c.run(
                """
                MATCH (a:SAEFeature {index: $idx})-[r:DECODER_SIMILAR]-(b:SAEFeature)
                WHERE a.sae_id CONTAINS 'L20/16k' AND b.sae_id CONTAINS 'L20/16k'
                RETURN DISTINCT b.index AS idx, r.cosine AS cos
                ORDER BY cos DESC LIMIT $k
                """,
                idx=anchor, k=k,
            )
            dec = [int(row["idx"]) for row in r]
            if dec:
                out["decoder_neighbors10"] = [anchor] + dec

            r = c.run(
                """
                MATCH (a:SAEFeature {index: $idx})-[r:CO_ACTIVATES_WITH]-(b:SAEFeature)
                WHERE a.sae_id CONTAINS 'L20/16k' AND b.sae_id CONTAINS 'L20/16k'
                RETURN DISTINCT b.index AS idx, r.pmi AS pmi, r.jaccard AS jac
                ORDER BY jac DESC LIMIT $k
                """,
                idx=anchor, k=k,
            )
            coact = [int(row["idx"]) for row in r]
            if coact:
                out["coact_partners10"] = [anchor] + coact
    except Exception as e:
        log.warning(f"graph fetch failed, skipping graph-driven sets: {e}")
    return out


def measure_pivot_prob(model, sae, prefix: str, pivot_ids: list[int],
                       feat_indices: list[int] | None) -> float:
    """Return P(any pivot token | prefix) at the next position, optionally
    ablating `feat_indices` at the last position of the SAE post-encode."""
    tokens = model.to_tokens(prefix, prepend_bos=True)
    hook_name = f"{sae.cfg.metadata.hook_name}.hook_sae_acts_post"

    if not feat_indices:
        with torch.no_grad():
            logits = model.run_with_hooks_with_saes(tokens, saes=[sae], fwd_hooks=[])
    else:
        idxs = torch.tensor(feat_indices, device=tokens.device, dtype=torch.long)

        def ablate(act, **kwargs):
            act = act.clone()
            act[..., -1, idxs] = 0.0
            return act

        with torch.no_grad():
            logits = model.run_with_hooks_with_saes(
                tokens, saes=[sae], fwd_hooks=[(hook_name, ablate)]
            )

    probs = F.softmax(logits[0, -1, :].float().cpu(), dim=-1)
    return float(probs[pivot_ids].sum().item())


def run(args):
    dev = device()
    log.info(f"loading gemma-2-2b on {dev}…")
    model = HookedSAETransformer.from_pretrained("gemma-2-2b", device=dev)
    model.eval()

    log.info("loading Gemma Scope SAE")
    sae = SAE.from_pretrained(
        release=GEMMA_SAE.release, sae_id=GEMMA_SAE.sae_id, device=dev,
    )
    if isinstance(sae, tuple):
        sae = sae[0]
    sae.eval()
    d_sae = sae.cfg.d_sae

    variants = set(args.variants)
    pairs = load_d1(variants)
    samples = []
    for p in pairs:
        pref = truncate_to_pivot(p["with"])
        if pref is None:
            continue
        samples.append({"variant": p["variant"], "prefix": pref})
    samples = samples[: args.max_samples]
    log.info(f"truncated samples: {len(samples)} across variants {sorted(variants)}")

    pivot_ids = collect_pivot_token_ids(model, variants)
    log.info(f"pivot token counts: {{{', '.join(f'{v}: {len(ids)}' for v, ids in pivot_ids.items())}}}")

    # Assemble conditions
    named_sets = load_attribution_ladder()
    named_sets.update(load_graph_sets())
    named_sets["suppressors_top10"] = load_suppressor_set(10)

    sizes_needing_random = sorted({len(v) for v in named_sets.values()})
    rng = np.random.default_rng(args.seed)
    excluded = set().union(*[set(v) for v in named_sets.values()])
    pool = np.array([i for i in range(d_sae) if i not in excluded], dtype=np.int64)

    random_sets: dict[str, list[int]] = {}
    for k in sizes_needing_random:
        for draw in range(args.n_random):
            name = f"random_k{k}_d{draw}"
            random_sets[name] = sorted(int(x) for x in rng.choice(pool, size=k, replace=False))

    all_conditions = {**named_sets, **random_sets}
    log.info(
        f"conditions: {len(named_sets)} named + {len(random_sets)} random "
        f"(sizes {sizes_needing_random}, {args.n_random} draws each) = "
        f"{len(all_conditions)} total"
    )

    # Per-sample loop
    per_sample: list[dict] = []
    t0 = time.perf_counter()
    for si, s in enumerate(samples):
        pids = pivot_ids[s["variant"]]
        base_p = measure_pivot_prob(model, sae, s["prefix"], pids, None)
        row = {"variant": s["variant"], "prefix": s["prefix"], "baseline_p": base_p}
        cond_p: dict[str, float] = {}
        for cond_name, feats in all_conditions.items():
            cond_p[cond_name] = measure_pivot_prob(model, sae, s["prefix"], pids, feats)
        row["cond_p"] = cond_p
        per_sample.append(row)

        if (si + 1) % 10 == 0 or si == len(samples) - 1:
            elapsed = time.perf_counter() - t0
            rate = (si + 1) / elapsed
            eta = (len(samples) - si - 1) / rate
            log.info(f"  {si + 1}/{len(samples)} samples  ({rate:.2f}/s, ETA {eta:.0f}s)")

    # Aggregate per condition
    baseline_mean = float(np.mean([r["baseline_p"] for r in per_sample]))

    def cond_stats(name: str) -> dict:
        vals = np.array([r["cond_p"][name] for r in per_sample])
        drops = np.array([r["baseline_p"] - r["cond_p"][name] for r in per_sample])
        return {
            "n_features": len(all_conditions[name]),
            "features": all_conditions[name],
            "mean_p_pivot": float(vals.mean()),
            "mean_drop": float(drops.mean()),
            "median_drop": float(np.median(drops)),
            "rel_drop": float(drops.mean() / baseline_mean) if baseline_mean > 0 else 0.0,
            "n": int(vals.shape[0]),
        }

    by_cond = {name: cond_stats(name) for name in all_conditions}

    # Aggregate random nulls by size
    random_by_size: dict[int, dict] = {}
    for k in sizes_needing_random:
        draws = [by_cond[f"random_k{k}_d{d}"]["mean_drop"] for d in range(args.n_random)]
        random_by_size[k] = {
            "n_draws": args.n_random,
            "mean": float(np.mean(draws)),
            "std": float(np.std(draws, ddof=1)) if len(draws) > 1 else 0.0,
            "max": float(np.max(draws)),
            "draws": draws,
        }

    # Named-set vs random null comparison
    named_named = list(named_sets.keys())
    head_to_head = []
    for name in named_named:
        s = by_cond[name]
        null = random_by_size[s["n_features"]]
        head_to_head.append({
            "name": name,
            "n_features": s["n_features"],
            "mean_drop": s["mean_drop"],
            "rel_drop": s["rel_drop"],
            "null_mean": null["mean"],
            "null_max": null["max"],
            "null_std": null["std"],
            "exceeds_null_max": s["mean_drop"] > null["max"],
        })

    labels = load_labels()
    result = {
        "anchor_feature": ANCHOR_FEATURE,
        "n_samples": len(samples),
        "variants": sorted(variants),
        "baseline_mean_p_pivot": baseline_mean,
        "n_random_draws_per_size": args.n_random,
        "by_condition": by_cond,
        "random_by_size": random_by_size,
        "head_to_head": head_to_head,
        "labels": {str(i): labels.get(i, "") for s in named_sets.values() for i in s},
        "per_sample": per_sample,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))

    # Markdown
    md = [
        "# Joint-ablation ladder — the redundancy hypothesis",
        "",
        f"**Anchor feature:** {ANCHOR_FEATURE} (Neuronpedia: *phrases conveying exceptions or negations*)  ",
        f"**N truncated D1 samples:** {len(samples)} ({', '.join(sorted(variants))})  ",
        f"**Baseline mean P(pivot):** {baseline_mean:.4f}  ",
        f"**Random-null draws per size:** {args.n_random}  ",
        "**Intervention:** zero-ablate the named feature set at the SAE post-encode at the pre-pivot last position.",
        "",
        "## The ladder (named sets vs size-matched random null)",
        "",
        "| Condition | N feats | mean P(pivot) | mean drop | rel drop | null mean drop | null max | exceeds null max? |",
        "|---|---:|---:|---:|---:|---:|---:|:--:|",
    ]
    # Order: ladders by size, then graph-driven, then suppressors
    ordered = [
        "single_3223", "attrib_top2", "attrib_top5", "attrib_top10", "attrib_top25",
        "decoder_neighbors10", "coact_partners10", "suppressors_top10",
    ]
    for name in ordered:
        if name not in by_cond:
            continue
        s = by_cond[name]
        null = random_by_size[s["n_features"]]
        flag = "✓" if s["mean_drop"] > null["max"] else "·"
        md.append(
            f"| `{name}` | {s['n_features']} | {s['mean_p_pivot']:.4f} | "
            f"{s['mean_drop']:+.4f} | {s['rel_drop']:+.2%} | "
            f"{null['mean']:+.4f} | {null['max']:+.4f} | {flag} |"
        )
    md.append("")
    md.append("**`single_3223` reproduces Phase 4 necessity.** "
              "The ladder asks whether larger named sets continue to drop P(pivot) "
              "beyond what a size-matched random ablation would do.")
    md.append("")

    md.append("## Random-null distribution by size")
    md.append("")
    md.append("| Size | mean drop | std drop | max drop |")
    md.append("|---:|---:|---:|---:|")
    for k in sorted(random_by_size):
        n = random_by_size[k]
        md.append(f"| {k} | {n['mean']:+.5f} | {n['std']:.5f} | {n['max']:+.5f} |")
    md.append("")

    md.append("## Named-set feature lists")
    md.append("")
    for name in ordered:
        if name not in named_sets:
            continue
        md.append(f"### `{name}` (n={len(named_sets[name])})")
        for f in named_sets[name]:
            lbl = labels.get(int(f), "")
            md.append(f"- **{f}** — {lbl[:90] if lbl else '(no label)'}")
        md.append("")

    # Verdict
    md.append("## Verdict — Hydra vs localized constellation")
    md.append("")
    attrib_drops = [by_cond[k]["mean_drop"] for k in
                    ["single_3223", "attrib_top2", "attrib_top5", "attrib_top10", "attrib_top25"]
                    if k in by_cond]
    rel_drops = [by_cond[k]["rel_drop"] for k in
                 ["single_3223", "attrib_top2", "attrib_top5", "attrib_top10", "attrib_top25"]
                 if k in by_cond]
    if rel_drops:
        md.append(f"- attribution ladder mean drops: {[f'{d:+.4f}' for d in attrib_drops]}")
        md.append(f"- attribution ladder rel drops:  {[f'{d:+.2%}' for d in rel_drops]}")
        if rel_drops[-1] < 0.5:
            md.append("")
            md.append("**Hydra-shaped:** even ablating 25 features fails to halve P(pivot). "
                      "The construction is implemented redundantly across an extended "
                      "constellation that exceeds the size of any single SAE-feature set "
                      "tractable to enumerate. The single-feature finding stands as a "
                      "*necessary path*, not a *sufficient lever*.")
        elif rel_drops[-1] < 0.85:
            md.append("")
            md.append("**Partial collapse:** ablating the top-25 cluster significantly "
                      "reduces P(pivot) but does not zero it. The construction is "
                      "concentrated in a constellation rather than scattered; with the "
                      "joint set identified, a stronger intervention (e.g. larger set, "
                      "directional ablation across the cluster) is the next step.")
        else:
            md.append("")
            md.append("**Localized constellation:** the joint top-25 ablation collapses "
                      "P(pivot) to a small fraction of baseline. The construction is "
                      "localized to this set, which is the answer to 'where does it live?' "
                      "This is the kill story.")

    OUT_MD.write_text("\n".join(md))

    print()
    print(f"baseline mean P(pivot)        = {baseline_mean:.4f}")
    print("Named-set drops:")
    for name in ordered:
        if name not in by_cond:
            continue
        s = by_cond[name]
        null = random_by_size[s["n_features"]]
        flag = "✓ exceeds null max" if s["mean_drop"] > null["max"] else "· within null"
        print(f"  {name:<22}  n={s['n_features']:>3}  "
              f"drop={s['mean_drop']:+.4f} ({s['rel_drop']:+.2%})  "
              f"null_max={null['max']:+.4f}  {flag}")
    print(f"\n→ {OUT_MD}")
    print(f"→ {OUT_JSON}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["C1", "C2", "C3"])
    ap.add_argument("--max-samples", type=int, default=80)
    ap.add_argument("--n-random", type=int, default=3,
                    help="Random size-matched control draws per ladder rung.")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
