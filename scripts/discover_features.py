"""Phase 3 — feature discovery on D1.

For each contrast pair, run Gemma 2 2B + the Gemma Scope L20 width-16k SAE
on both `with` and `without` sentences. Take the SAE activation at the
**last token** of each (a robust, defensible "the model has finished
processing" position — pivot-aligned comparison is asymmetric since the
'without' has no pivot, see report). Aggregate per-feature differential
activation across pairs; rank features by t-statistic; output top-K with
Neuronpedia labels.

PRD §8 Phase 3 kill check: a feature (or small supernode) shows clean,
label-interpretable separation between with/without. If not — pivot to a
2–3-feature supernode per Anthropic Biology, or stop.

Usage:
    uv run python scripts/discover_features.py
    uv run python scripts/discover_features.py --top-k 20 --layer 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sae_lens import HookedSAETransformer, SAE

from neograph.config import SAE as GEMMA_SAE
from neograph.util import get_logger

log = get_logger("phase3.discover")

REPO_ROOT = Path(__file__).resolve().parent.parent
D1_PATH = REPO_ROOT / "data" / "D1_contrast_pairs.jsonl"
LABELS_PATH = REPO_ROOT / "data" / "labels_cache.json"
OUT_JSON = REPO_ROOT / "reports" / "phase3_discovery.json"
OUT_MD = REPO_ROOT / "reports" / "phase3_discovery.md"


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_d1() -> list[dict]:
    return [
        json.loads(line)
        for line in D1_PATH.read_text().splitlines()
        if line.strip() and not line.startswith('{"_meta')
    ]


def load_labels() -> dict[int, str]:
    if not LABELS_PATH.exists():
        return {}
    raw = json.loads(LABELS_PATH.read_text())
    return {int(k): v.get("text", "") for k, v in raw.items()}


def encode_last_token_acts(model, sae, text: str, hook_acts_post: str) -> np.ndarray:
    tokens = model.to_tokens(text, prepend_bos=True)
    with torch.no_grad():
        _, cache = model.run_with_cache_with_saes(tokens, saes=[sae])
    feat_acts = cache[hook_acts_post][0, -1, :].float().cpu().numpy()
    return feat_acts


def run(layer: int, top_k: int) -> None:
    dev = device()
    log.info(f"loading gemma-2-2b on {dev}…")
    model = HookedSAETransformer.from_pretrained("gemma-2-2b", device=dev)
    model.eval()

    log.info(f"loading Gemma Scope SAE release={GEMMA_SAE.release} id={GEMMA_SAE.sae_id}")
    sae = SAE.from_pretrained(
        release=GEMMA_SAE.release, sae_id=GEMMA_SAE.sae_id, device=dev,
    )
    if isinstance(sae, tuple):
        sae = sae[0]
    sae.eval()
    hook_acts_post = f"{sae.cfg.hook_name}.hook_sae_acts_post"
    d_sae = sae.cfg.d_sae
    log.info(f"SAE: hook={sae.cfg.hook_name} d_sae={d_sae}")

    pairs = load_d1()
    log.info(f"D1 loaded: {len(pairs)} pairs")

    acts_with = np.zeros((len(pairs), d_sae), dtype=np.float32)
    acts_without = np.zeros((len(pairs), d_sae), dtype=np.float32)
    variants = [p["variant"] for p in pairs]

    t0 = time.perf_counter()
    for i, p in enumerate(pairs):
        acts_with[i] = encode_last_token_acts(model, sae, p["with"], hook_acts_post)
        acts_without[i] = encode_last_token_acts(model, sae, p["without"], hook_acts_post)
        if (i + 1) % 25 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta = (len(pairs) - i - 1) / rate
            log.info(f"  {i + 1}/{len(pairs)} pairs encoded ({rate:.2f}/s, ETA {eta:.1f}s)")

    log.info(f"encoding done in {time.perf_counter() - t0:.1f}s; ranking features…")

    # Per-feature differential statistics across all pairs.
    diff = acts_with - acts_without              # (n_pairs, d_sae)
    mean_diff = diff.mean(axis=0)                # (d_sae,)
    n = diff.shape[0]
    se = diff.std(axis=0, ddof=1) / np.sqrt(n)
    se = np.maximum(se, 1e-8)
    t_stat = mean_diff / se                      # (d_sae,)

    # Per-variant breakdown
    per_variant_diff = {}
    for v in ("C1", "C2", "C3", "C4"):
        mask = np.array([x == v for x in variants])
        if mask.sum() == 0:
            continue
        d = diff[mask]
        if d.shape[0] < 2:
            continue
        m = d.mean(axis=0)
        s = d.std(axis=0, ddof=1) / np.sqrt(d.shape[0])
        s = np.maximum(s, 1e-8)
        per_variant_diff[v] = {
            "mean_diff": m,
            "t_stat": m / s,
            "n": int(d.shape[0]),
        }

    # Rank features by t-statistic, both directions (over- and under-firing).
    top_pos = np.argsort(-t_stat)[:top_k]  # features that fire MORE on with
    top_neg = np.argsort(t_stat)[:top_k]   # features that fire MORE on without

    labels = load_labels()

    def feature_record(idx: int) -> dict:
        rec = {
            "feature_idx": int(idx),
            "mean_diff": float(mean_diff[idx]),
            "t_stat": float(t_stat[idx]),
            "label": labels.get(int(idx), ""),
        }
        for v, st in per_variant_diff.items():
            rec[f"t_stat_{v}"] = float(st["t_stat"][idx])
            rec[f"mean_diff_{v}"] = float(st["mean_diff"][idx])
        return rec

    top_pos_records = [feature_record(int(i)) for i in top_pos]
    top_neg_records = [feature_record(int(i)) for i in top_neg]

    result = {
        "layer": layer,
        "sae_release": GEMMA_SAE.release,
        "sae_id": GEMMA_SAE.sae_id,
        "d_sae": int(d_sae),
        "n_pairs": int(n),
        "pairs_per_variant": {
            v: int(sum(1 for x in variants if x == v))
            for v in ("C1", "C2", "C3", "C4")
        },
        "position": "last_token",
        "top_features_with_higher": top_pos_records,
        "top_features_without_higher": top_neg_records,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2))

    # Markdown report
    md = [
        "# Phase 3 — Feature discovery on D1",
        "",
        f"**Model:** gemma-2-2b · **SAE:** {GEMMA_SAE.release} / {GEMMA_SAE.sae_id} "
        f"(d_sae = {d_sae})",
        f"**Pairs:** {n} ({', '.join(f'{v}={c}' for v, c in result['pairs_per_variant'].items())})",
        "**Position:** last token of each sentence.  ",
        "**Stat:** per-feature `(mean(act_with − act_without) / SE)` across all pairs.",
        "",
        "## Caveat\n",
        "The pivot-aligned comparison the PRD (§5 M2) calls for is asymmetric "
        "here — the *without* paraphrases have no pivot, by construction. "
        "Phase 3 uses the **last-token** position for both sides, which captures "
        "the cumulative effect of the construction on the residual stream after "
        "the model has finished processing the sentence. This is the same "
        "convention as Marks et al. 'Sparse Feature Circuits'. The pivot-specific "
        "measurement happens in Phase 4 via M2 (P(pivot token) on `with`-style "
        "prompts that have already opened the negation).\n",
        "",
        "## Top features that fire MORE on `with` (construction-recruiting)\n",
        "| Rank | Feature | t-stat | mean Δ | t·C1 | t·C2 | t·C3 | t·C4 | Label |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, r in enumerate(top_pos_records):
        md.append(
            f"| {i+1} | {r['feature_idx']} | {r['t_stat']:+.2f} | {r['mean_diff']:+.3f} | "
            f"{r.get('t_stat_C1', 0):+.2f} | {r.get('t_stat_C2', 0):+.2f} | "
            f"{r.get('t_stat_C3', 0):+.2f} | {r.get('t_stat_C4', 0):+.2f} | "
            f"{(r['label'] or '—')[:70]} |"
        )
    md.append("")
    md.append("## Top features that fire MORE on `without` (paraphrase-recruiting)\n")
    md.append("| Rank | Feature | t-stat | mean Δ | t·C1 | t·C2 | t·C3 | t·C4 | Label |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for i, r in enumerate(top_neg_records):
        md.append(
            f"| {i+1} | {r['feature_idx']} | {r['t_stat']:+.2f} | {r['mean_diff']:+.3f} | "
            f"{r.get('t_stat_C1', 0):+.2f} | {r.get('t_stat_C2', 0):+.2f} | "
            f"{r.get('t_stat_C3', 0):+.2f} | {r.get('t_stat_C4', 0):+.2f} | "
            f"{(r['label'] or '—')[:70]} |"
        )
    md.append("")

    # Kill-check assessment
    md.append("## Phase 3 kill check\n")
    top_t = top_pos_records[0]["t_stat"] if top_pos_records else 0
    cross_variant = (
        top_pos_records[0].get("t_stat_C1", 0) > 0
        and top_pos_records[0].get("t_stat_C2", 0) > 0
        and top_pos_records[0].get("t_stat_C3", 0) > 0
    ) if top_pos_records else False
    if top_t >= 5 and cross_variant:
        md.append(f"**PASS** — top feature t-stat = {top_t:+.2f}, positive across "
                  f"C1/C2/C3. Candidate for Phase 4 causal validation.")
    elif top_t >= 3:
        md.append(f"**MARGINAL** — top feature t-stat = {top_t:+.2f}; check the "
                  f"label and per-variant signs. If the feature isn't C1/C2/C3-positive "
                  f"across the board, pivot to a 2–3-feature supernode per Anthropic Biology.")
    else:
        md.append(f"**FAIL** — top feature t-stat = {top_t:+.2f}. The construction "
                  f"is not localized to a single feature at this layer. Pivot to a "
                  f"supernode or sweep more layers; or stop and report the negative result.")

    OUT_MD.write_text("\n".join(md))

    print(f"\nTop 5 features with-higher (by t-stat):")
    for i, r in enumerate(top_pos_records[:5]):
        print(f"  #{i+1}  feat {r['feature_idx']:5d}  t={r['t_stat']:+6.2f}  "
              f"Δ={r['mean_diff']:+6.3f}  · {(r['label'] or '—')[:60]}")
    print(f"\n→ {OUT_MD}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=GEMMA_SAE.layer)
    ap.add_argument("--top-k", type=int, default=15)
    args = ap.parse_args()
    run(layer=args.layer, top_k=args.top_k)


if __name__ == "__main__":
    main()
