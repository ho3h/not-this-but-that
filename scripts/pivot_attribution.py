"""Per-feature attribution to P(pivot) at the truncated D1 position.

Phase 3 used last-token activations on *completed* sentences and found that
feature 9841 differentially fires when the construction has been used. But
Phase 4 showed that ablating 9841 at the *truncated* position (right before
the pivot) doesn't move P(pivot) — because 9841 isn't active there. It's a
consequence feature, not a cause.

This script does the opposite: it ablates *every* active feature at the
truncated position one at a time and measures ΔP(pivot). The features with
the largest attribution are the *causal* candidates, and may not be the same
features Phase 3 discovered.

Method (mirrors causal_attribution_v2's per-feature zero-ablation, retargeted
at pivot probability instead of factual completion):
  1. For each truncated D1 with-prompt, get the SAE activations at the last
     position; collect the set of features active across all prompts.
  2. For each active feature, ablate at the last position, recompute
     P(pivot), record the drop.
  3. Aggregate per-feature: mean drop, n_prompts_where_active, attribution
     ratio.

Writes:
  reports/pivot_attribution.json
  reports/pivot_attribution.md  (top-K features by mean ablation drop)

Usage:
    uv run python scripts/pivot_attribution.py --max-prompts 60 --top-k 25
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
from neograph.util import get_logger

log = get_logger("phase4.pivot_attrib")

REPO_ROOT = Path(__file__).resolve().parent.parent
D1_PATH = REPO_ROOT / "data" / "D1_contrast_pairs.jsonl"
LABELS_PATH = REPO_ROOT / "data" / "labels_cache.json"
DEFAULT_OUT_JSON = REPO_ROOT / "reports" / "pivot_attribution.json"
DEFAULT_OUT_MD = REPO_ROOT / "reports" / "pivot_attribution.md"

PIVOT_STRINGS = {
    "C1": [", it", ", they", ", he", ", she", ", we", "—", "; it"],
    "C2": [", but", ", yet", ", also", " but ", " also "],
    "C3": ["—", ", it", " but ", ", but"],
}
ACTIVATION_THRESHOLD = 1e-3


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def truncate_to_pivot(text: str) -> str | None:
    m = re.search(
        r"[,;—–\-]\s*(?:it|that|this|he|she|we|they|these|those|there|but|also|yet)\b",
        text, re.IGNORECASE,
    )
    if not m:
        return None
    return text[: m.start()].rstrip()


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-prompts", type=int, default=80,
                    help="Cap on truncated D1 with-prompts to process.")
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--variants", nargs="+", default=["C1", "C2", "C3"])
    ap.add_argument("--checkpoint-every", type=int, default=10,
                    help="Write partial pivot_attribution.json every N samples.")
    ap.add_argument("--sae-layer", type=int, default=20,
                    help="Which Gemma Scope canonical SAE layer to scan. "
                         "Default 20. Outputs go to pivot_attribution_L<N>.json "
                         "(L20 keeps the canonical filename for back-compat).")
    args = ap.parse_args()

    dev = device()
    log.info(f"loading gemma-2-2b on {dev}…")
    model = HookedSAETransformer.from_pretrained("gemma-2-2b", device=dev)
    model.eval()

    layer = int(args.sae_layer)
    sae_id = f"layer_{layer}/width_16k/canonical"
    log.info(f"loading Gemma Scope SAE at layer {layer} ({sae_id})")
    sae = SAE.from_pretrained(
        release=GEMMA_SAE.release, sae_id=sae_id, device=dev,
    )
    if isinstance(sae, tuple):
        sae = sae[0]
    sae.eval()
    hook_acts_post = f"{sae.cfg.metadata.hook_name}.hook_sae_acts_post"
    d_sae = sae.cfg.d_sae

    # Per-layer output paths (L20 keeps canonical name for back-compat)
    if layer == 20:
        OUT_JSON = DEFAULT_OUT_JSON
        OUT_MD = DEFAULT_OUT_MD
    else:
        OUT_JSON = REPO_ROOT / "reports" / f"pivot_attribution_L{layer}.json"
        OUT_MD = REPO_ROOT / "reports" / f"pivot_attribution_L{layer}.md"

    # Truncated samples
    samples = []
    for line in D1_PATH.read_text().splitlines():
        if not line.strip() or line.startswith('{"_meta'):
            continue
        d = json.loads(line)
        if d["variant"] not in set(args.variants):
            continue
        prefix = truncate_to_pivot(d["with"])
        if prefix is None:
            continue
        samples.append({"variant": d["variant"], "prefix": prefix})
    samples = samples[: args.max_prompts]
    log.info(f"processing {len(samples)} truncated prompts")

    pivot_ids = collect_pivot_token_ids(model, set(args.variants))

    # Per-feature attribution accumulator
    attrib_sum = np.zeros(d_sae, dtype=np.float64)
    attrib_count = np.zeros(d_sae, dtype=np.int32)
    baseline_ps = []

    def _write_checkpoint(processed_n: int) -> None:
        """Snapshot current state to disk so killed runs leave usable data."""
        ma = np.zeros(d_sae, dtype=np.float64)
        nz = attrib_count > 0
        ma[nz] = attrib_sum[nz] / attrib_count[nz]
        sc = ma * np.sqrt(attrib_count)
        nonzero_idx = np.where(attrib_count > 0)[0]
        full = sorted(
            [(int(i), float(ma[i]), int(attrib_count[i]), float(sc[i]))
             for i in nonzero_idx], key=lambda t: -t[3],
        )
        labels_local = load_labels()
        top_pos_local = np.argsort(-sc)[: args.top_k]
        top_neg_local = np.argsort(sc)[: args.top_k]

        def _fmt(idx: int) -> dict:
            return {"feature_idx": int(idx), "mean_attribution_drop": float(ma[idx]),
                    "n_prompts_active": int(attrib_count[idx]),
                    "scored": float(sc[idx]), "label": labels_local.get(int(idx), "")}

        partial = {
            "n_samples": processed_n,
            "n_samples_target": len(samples),
            "checkpoint": True,
            "baseline_p_pivot_mean": float(np.mean(baseline_ps)) if baseline_ps else 0.0,
            "top_promotes_pivot": [_fmt(int(i)) for i in top_pos_local],
            "top_suppresses_pivot": [_fmt(int(i)) for i in top_neg_local],
            "n_features_with_signal": int(len(full)),
            "full_ranked_by_score": [
                {"feature_idx": fi, "mean_attribution_drop": mav,
                 "n_prompts_active": ct, "scored": scv}
                for (fi, mav, ct, scv) in full
            ],
        }
        OUT_JSON.write_text(json.dumps(partial, indent=2))

    t0 = time.perf_counter()
    for si, s in enumerate(samples):
        tokens = model.to_tokens(s["prefix"], prepend_bos=True)
        pids = pivot_ids[s["variant"]]

        # baseline + active features
        with torch.no_grad():
            base_logits, cache = model.run_with_cache_with_saes(tokens, saes=[sae])
        base_probs = F.softmax(base_logits[0, -1, :].float().cpu(), dim=-1)
        base_p = float(base_probs[pids].sum())
        baseline_ps.append(base_p)

        feat_acts = cache[hook_acts_post][0, -1, :].float().cpu().numpy()
        active = np.where(feat_acts > ACTIVATION_THRESHOLD)[0]
        if len(active) == 0:
            continue

        # ablate each active feature individually
        for f in active:
            f_int = int(f)
            idx_tensor = torch.tensor([f_int], device=tokens.device, dtype=torch.long)

            def ablate(act, **kwargs):
                act = act.clone()
                act[..., -1, idx_tensor] = 0.0
                return act

            with torch.no_grad():
                abl_logits = model.run_with_hooks_with_saes(
                    tokens, saes=[sae],
                    fwd_hooks=[(hook_acts_post, ablate)],
                )
            abl_probs = F.softmax(abl_logits[0, -1, :].float().cpu(), dim=-1)
            abl_p = float(abl_probs[pids].sum())
            attrib_sum[f_int] += (base_p - abl_p)
            attrib_count[f_int] += 1

        elapsed = time.perf_counter() - t0
        rate = (si + 1) / elapsed
        eta = (len(samples) - si - 1) / rate
        if (si + 1) % 5 == 0 or si == len(samples) - 1:
            log.info(f"  {si + 1}/{len(samples)} ({rate:.2f}/s, ETA {eta:.0f}s)  "
                     f"active feats this prompt: {len(active)}")
        if (si + 1) % args.checkpoint_every == 0 or si == len(samples) - 1:
            _write_checkpoint(si + 1)
            log.info(f"  checkpoint written at {si + 1}/{len(samples)}")

    # Compute per-feature mean attribution where it was active
    mean_attrib = np.zeros(d_sae, dtype=np.float64)
    nonzero = attrib_count > 0
    mean_attrib[nonzero] = attrib_sum[nonzero] / attrib_count[nonzero]

    # Rank by attribution * sqrt(n) (so rarely-active features don't dominate
    # purely by single-sample noise).
    score = mean_attrib * np.sqrt(attrib_count)
    top_pos = np.argsort(-score)[: args.top_k]
    top_neg = np.argsort(score)[: args.top_k]

    labels = load_labels()

    def fmt(idx: int) -> dict:
        return {
            "feature_idx": int(idx),
            "mean_attribution_drop": float(mean_attrib[idx]),
            "n_prompts_active": int(attrib_count[idx]),
            "scored": float(score[idx]),
            "label": labels.get(int(idx), ""),
        }

    # Full ranked list (only features that fired at least once) so downstream
    # consumers can pull any top-N without re-running the scan.
    nonzero_idx = np.where(attrib_count > 0)[0]
    full_ranked = sorted(
        [(int(i), float(mean_attrib[i]), int(attrib_count[i]), float(score[i]))
         for i in nonzero_idx],
        key=lambda t: -t[3],
    )

    result = {
        "n_samples": len(samples),
        "baseline_p_pivot_mean": float(np.mean(baseline_ps)),
        "top_promotes_pivot": [fmt(int(i)) for i in top_pos],
        "top_suppresses_pivot": [fmt(int(i)) for i in top_neg],
        "n_features_with_signal": int(len(full_ranked)),
        "full_ranked_by_score": [
            {"feature_idx": fi, "mean_attribution_drop": ma,
             "n_prompts_active": ct, "scored": sc}
            for (fi, ma, ct, sc) in full_ranked
        ],
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))

    md = [
        "# Per-feature attribution to P(pivot) at the truncated D1 position",
        "",
        f"**N samples:** {len(samples)} truncated 'with'-prompts ({sorted(set(args.variants))})  ",
        f"**Baseline P(pivot) mean:** {result['baseline_p_pivot_mean']:.4f}  ",
        "**Score:** mean attribution drop × √(n prompts where feature was active).  ",
        "Positive score = ablating the feature DROPS P(pivot) → feature *promotes* the construction at the decision point.  ",
        "Negative score = ablating the feature RAISES P(pivot) → feature *suppresses* the construction.",
        "",
        "## Top features that PROMOTE the pivot (ablation drops P(pivot))",
        "",
        "| Rank | Feature | mean Δ | n active | score | Label |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for i, r in enumerate(result["top_promotes_pivot"]):
        md.append(f"| {i+1} | {r['feature_idx']} | {r['mean_attribution_drop']:+.5f} | "
                  f"{r['n_prompts_active']} | {r['scored']:+.4f} | {(r['label'] or '—')[:70]} |")
    md.append("")
    md.append("## Top features that SUPPRESS the pivot (ablation raises P(pivot))")
    md.append("")
    md.append("| Rank | Feature | mean Δ | n active | score | Label |")
    md.append("|---:|---:|---:|---:|---:|---|")
    for i, r in enumerate(result["top_suppresses_pivot"]):
        md.append(f"| {i+1} | {r['feature_idx']} | {r['mean_attribution_drop']:+.5f} | "
                  f"{r['n_prompts_active']} | {r['scored']:+.4f} | {(r['label'] or '—')[:70]} |")

    OUT_MD.write_text("\n".join(md))
    print(f"\n→ {OUT_MD}")


if __name__ == "__main__":
    main()
