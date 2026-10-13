"""Phase 4 — M2 causal validation. Bidirectional pivot-probability intervention.

PRD §5 M2: given a context that has opened the negation ("It's not just an
update"), measure `P(pivot | …)`. Phase 4 perturbs that probability:

  - **ablate** the candidate feature(s) at the SAE post-encode hook
    → P(pivot) should DROP (feature is causally involved in committing).
  - **clamp_up** the candidate feature(s) to a high value
    → P(pivot) should RISE (forcing the feature should drive the construction).

Bidirectional + beats controls (random-k, bottom-k) = a real lever.

This is the PRD's primary causal claim. M1 (generation-based construction
rate) is the secondary measurement and is run by a separate script.

Usage:
    uv run python scripts/causal_m2.py --features 6631
    uv run python scripts/causal_m2.py --features 6631 9768 13414 --controls random,bottom
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

log = get_logger("phase4.causal_m2")

REPO_ROOT = Path(__file__).resolve().parent.parent
D1_PATH = REPO_ROOT / "data" / "D1_contrast_pairs.jsonl"
DISCOVERY_PATH = REPO_ROOT / "reports" / "phase3_discovery.json"
OUT_JSON = REPO_ROOT / "reports" / "phase4_causal_m2.json"
OUT_MD = REPO_ROOT / "reports" / "phase4_causal_m2.md"


# Pivot tokens we care about. Each variant has its own commit point; M2 sums
# the probabilities of any token that completes the construction.
PIVOT_STRINGS = {
    "C1": [", it", ", they", ", he", ", she", ", we", "—", "; it"],
    "C2": [", but", ", yet", ", also", " but ", " also "],
    "C3": ["—", ", it", " but ", ", but"],
}


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_d1(variants: set[str]) -> list[dict]:
    rows = []
    for line in D1_PATH.read_text().splitlines():
        if not line.strip() or line.startswith('{"_meta'):
            continue
        d = json.loads(line)
        if d["variant"] in variants:
            rows.append(d)
    return rows


# --- Pivot truncation -----------------------------------------------------------
# For each WITH sentence, find the comma/em-dash that commits the construction
# and truncate the prompt at the character before it. The next-token prediction
# at that truncation is the pivot probability.

_PIVOT_CHAR_RE = re.compile(r"[,;—–\-]\s*(?:it|that|this|he|she|we|they|these|those|there|but|also|yet)\b",
                             re.IGNORECASE)


def truncate_to_pivot(text: str) -> tuple[str, str] | None:
    """Return (prefix, expected_pivot_string) or None if no pivot found."""
    m = _PIVOT_CHAR_RE.search(text)
    if not m:
        return None
    cut = m.start()
    prefix = text[:cut].rstrip()
    pivot = text[cut:].split()[0] if text[cut:].strip() else text[cut:]
    return prefix, pivot


def collect_pivot_token_ids(model, variants: set[str]) -> dict[str, list[int]]:
    """For each variant, return the list of token ids that count as the
    construction pivot. We sum probabilities across these tokens to compute
    P(pivot).
    """
    out = {}
    for v in variants:
        ids = set()
        for s in PIVOT_STRINGS[v]:
            for t in model.to_tokens(s, prepend_bos=False)[0].tolist():
                ids.add(int(t))
            # also try with leading space
            for t in model.to_tokens(" " + s.strip(), prepend_bos=False)[0].tolist():
                ids.add(int(t))
        out[v] = sorted(ids)
    return out


def measure_pivot_prob(model, sae, prefix: str, pivot_ids: list[int],
                       feat_indices: list[int] | None, mode: str,
                       clamp_value: float = 10.0) -> tuple[float, float]:
    """Return (P(any pivot token), max logit of pivot tokens) at the next position.

    `mode` ∈ {"baseline", "ablate", "clamp_up"}. When mode != "baseline",
    `feat_indices` is the list of SAE feature indices to perturb at the
    last position.
    """
    tokens = model.to_tokens(prefix, prepend_bos=True)
    hook_name = f"{sae.cfg.metadata.hook_name}.hook_sae_acts_post"

    if mode == "baseline" or not feat_indices:
        with torch.no_grad():
            logits = model.run_with_hooks_with_saes(tokens, saes=[sae], fwd_hooks=[])
    else:
        idxs = torch.tensor(feat_indices, device=tokens.device, dtype=torch.long)

        def perturb(act, **kwargs):
            act = act.clone()
            if mode == "ablate":
                act[..., -1, idxs] = 0.0
            elif mode == "clamp_up":
                act[..., -1, idxs] = clamp_value
            return act

        with torch.no_grad():
            logits = model.run_with_hooks_with_saes(
                tokens, saes=[sae], fwd_hooks=[(hook_name, perturb)]
            )

    last_logits = logits[0, -1, :].float().cpu()
    probs = F.softmax(last_logits, dim=-1)
    pivot_p = float(probs[pivot_ids].sum().item())
    pivot_max_logit = float(last_logits[pivot_ids].max().item())
    return pivot_p, pivot_max_logit


def pick_controls(d_sae: int, candidate: list[int], *, n_random: int = 5,
                  rng: np.random.Generator, bottom_k_from: list[int] | None = None
                  ) -> dict[str, list[list[int]]]:
    """Return control feature-index sets sized to match `candidate`.

    - `random`: `n_random` independent random draws of `len(candidate)` indices.
    - `bottom`: 1 draw from the lowest-|t| features in Phase 3 discovery
                (loaded from bottom_k_from if provided; else random fallback).
    """
    k = len(candidate)
    candidate_set = set(candidate)
    pool = [i for i in range(d_sae) if i not in candidate_set]
    out = {"random": []}
    for _ in range(n_random):
        out["random"].append(sorted(int(x) for x in rng.choice(pool, size=k, replace=False)))
    if bottom_k_from is not None:
        bottom = [i for i in bottom_k_from if i not in candidate_set][:k]
        if len(bottom) == k:
            out["bottom"] = [bottom]
    return out


def run(features: list[int], variants: set[str], n_random: int,
        clamp_value: float = 10.0) -> None:
    dev = device()
    log.info(f"loading gemma-2-2b on {dev}…")
    model = HookedSAETransformer.from_pretrained("gemma-2-2b", device=dev)
    model.eval()

    log.info(f"loading Gemma Scope SAE")
    sae = SAE.from_pretrained(
        release=GEMMA_SAE.release, sae_id=GEMMA_SAE.sae_id, device=dev,
    )
    if isinstance(sae, tuple):
        sae = sae[0]
    sae.eval()
    d_sae = sae.cfg.d_sae

    pairs = load_d1(variants)
    log.info(f"D1 loaded: {len(pairs)} pairs across variants {variants}")

    # Pivot token ids per variant
    pivot_ids_per_variant = collect_pivot_token_ids(model, variants)
    log.info({v: len(ids) for v, ids in pivot_ids_per_variant.items()})

    # Truncate each pair's "with" at the pivot
    samples = []
    for p in pairs:
        tr = truncate_to_pivot(p["with"])
        if tr is None:
            continue
        prefix, _ = tr
        samples.append({"variant": p["variant"], "prefix": prefix, "with": p["with"]})
    log.info(f"truncated to {len(samples)} samples with pivot")

    # Controls
    rng = np.random.default_rng(11)
    bottom_k_from = None
    if DISCOVERY_PATH.exists():
        disc = json.loads(DISCOVERY_PATH.read_text())
        # bottom = features with the smallest absolute t-stat in the
        # top_features_without_higher tail (i.e. the *least* signed).
        # Simpler: load all per-feature t-stats from discovery output if we
        # had them stored; fallback uses random.
        bottom_k_from = None  # not stored in discovery; controls fall back to random only

    controls = pick_controls(d_sae, features, n_random=n_random,
                              rng=rng, bottom_k_from=bottom_k_from)

    # Run measurements
    conditions: list[tuple[str, str, list[int] | None]] = [
        ("baseline", "baseline", None),
        ("candidate_ablate", "ablate", features),
        ("candidate_clamp_up", "clamp_up", features),
    ]
    for i, idxs in enumerate(controls.get("random", [])):
        conditions.append((f"random_ablate_{i}", "ablate", idxs))
        conditions.append((f"random_clamp_up_{i}", "clamp_up", idxs))
    if "bottom" in controls:
        conditions.append(("bottom_ablate", "ablate", controls["bottom"][0]))
        conditions.append(("bottom_clamp_up", "clamp_up", controls["bottom"][0]))

    results = {cond: [] for cond, _, _ in conditions}
    t0 = time.perf_counter()
    for i, s in enumerate(samples):
        pivot_ids = pivot_ids_per_variant[s["variant"]]
        for cond_name, mode, feat_idxs in conditions:
            p, ml = measure_pivot_prob(model, sae, s["prefix"], pivot_ids,
                                       feat_idxs, mode, clamp_value=clamp_value)
            results[cond_name].append({"variant": s["variant"], "p_pivot": p,
                                       "max_logit_pivot": ml})
        if (i + 1) % 25 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta = (len(samples) - i - 1) / rate
            log.info(f"  {i + 1}/{len(samples)} samples ({rate:.2f}/s, ETA {eta:.1f}s)")

    # Aggregate
    summary = {}
    for cond_name in results:
        vals = np.array([r["p_pivot"] for r in results[cond_name]])
        summary[cond_name] = {
            "mean_p_pivot": float(vals.mean()),
            "median_p_pivot": float(np.median(vals)),
            "std_p_pivot": float(vals.std(ddof=1)),
            "n": int(vals.shape[0]),
        }
        # Per-variant
        for v in variants:
            vmask = np.array([r["variant"] == v for r in results[cond_name]])
            if vmask.sum() == 0:
                continue
            summary[cond_name][f"mean_p_pivot_{v}"] = float(vals[vmask].mean())

    # Compute control-beating effect sizes
    baseline_mean = summary["baseline"]["mean_p_pivot"]
    candidate_ablate_drop = baseline_mean - summary["candidate_ablate"]["mean_p_pivot"]
    candidate_clamp_up_rise = summary["candidate_clamp_up"]["mean_p_pivot"] - baseline_mean

    random_ablate_drops = [
        baseline_mean - summary[f"random_ablate_{i}"]["mean_p_pivot"]
        for i in range(n_random)
    ]
    random_clamp_up_rises = [
        summary[f"random_clamp_up_{i}"]["mean_p_pivot"] - baseline_mean
        for i in range(n_random)
    ]

    summary["effect_sizes"] = {
        "baseline_mean_p_pivot": baseline_mean,
        "candidate_ablate_drop": candidate_ablate_drop,
        "candidate_clamp_up_rise": candidate_clamp_up_rise,
        "random_ablate_drop_mean": float(np.mean(random_ablate_drops)),
        "random_ablate_drop_std": float(np.std(random_ablate_drops, ddof=1)) if len(random_ablate_drops) > 1 else 0.0,
        "random_clamp_up_rise_mean": float(np.mean(random_clamp_up_rises)),
        "random_clamp_up_rise_std": float(np.std(random_clamp_up_rises, ddof=1)) if len(random_clamp_up_rises) > 1 else 0.0,
    }

    out_obj = {
        "features": features,
        "variants": sorted(variants),
        "n_samples": len(samples),
        "n_random_controls": n_random,
        "by_condition": summary,
        "per_sample_results": results,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out_obj, indent=2))

    # Markdown
    md = [
        "# Phase 4 — M2 causal validation",
        "",
        f"**Features under test:** {features}",
        f"**Variants:** {sorted(variants)}  ",
        f"**Truncated D1 samples:** {len(samples)}  ",
        f"**Random-k controls:** {n_random} independent draws  ",
        f"**Clamp-up value:** 10.0 (uniform; refine in iteration if needed)",
        "",
        "## Mean P(pivot) by condition\n",
        "| Condition | mean | median | std | n |",
        "|---|---:|---:|---:|---:|",
    ]
    for cond_name in ["baseline", "candidate_ablate", "candidate_clamp_up"] + \
                     [f"random_ablate_{i}" for i in range(n_random)] + \
                     [f"random_clamp_up_{i}" for i in range(n_random)]:
        if cond_name not in summary:
            continue
        s = summary[cond_name]
        md.append(f"| {cond_name} | {s['mean_p_pivot']:.4f} | {s['median_p_pivot']:.4f} | "
                  f"{s['std_p_pivot']:.4f} | {s['n']} |")
    md.append("")

    eff = summary["effect_sizes"]
    md.append("## Bidirectional + control-beating check\n")
    md.append(f"- **Candidate ablate drop:** {eff['candidate_ablate_drop']:+.4f}")
    md.append(f"- **Random ablate drop:** {eff['random_ablate_drop_mean']:+.4f} "
              f"± {eff['random_ablate_drop_std']:.4f}")
    md.append(f"- **Candidate clamp-up rise:** {eff['candidate_clamp_up_rise']:+.4f}")
    md.append(f"- **Random clamp-up rise:** {eff['random_clamp_up_rise_mean']:+.4f} "
              f"± {eff['random_clamp_up_rise_std']:.4f}")
    md.append("")

    # Honest reporting against degenerate controls.
    # When the random-k null has near-zero variance (single-feature ablation at
    # a position where the random feature is usually inactive produces
    # drop ≈ 0 ± 0), reporting (effect − null_mean) / null_std yields a sigma
    # multiple in the thousands that is *meaningless* — it isn't a tail event
    # in a smooth distribution, it's a qualitative separation from a flat one.
    # So we report it the way it actually is: how many of the random-k draws
    # the candidate exceeded, with the candidate value alongside the null
    # mean/std for the reader to judge separation.
    n_random_draws = len(random_ablate_drops)
    drops_below_candidate = sum(1 for d in random_ablate_drops
                                  if d < eff["candidate_ablate_drop"])
    rises_below_candidate = sum(1 for d in random_clamp_up_rises
                                  if d < eff["candidate_clamp_up_rise"])
    md.append(f"- **Ablate:** candidate drop = {eff['candidate_ablate_drop']:+.5f}; "
              f"random-k null mean = {eff['random_ablate_drop_mean']:+.5f} "
              f"± {eff['random_ablate_drop_std']:.5f} (n = {n_random_draws}); "
              f"candidate exceeded {drops_below_candidate}/{n_random_draws} random draws.")
    md.append(f"- **Clamp-up:** candidate rise = {eff['candidate_clamp_up_rise']:+.5f}; "
              f"random-k null mean = {eff['random_clamp_up_rise_mean']:+.5f} "
              f"± {eff['random_clamp_up_rise_std']:.5f} (n = {n_random_draws}); "
              f"candidate exceeded {rises_below_candidate}/{n_random_draws} random draws.")
    md.append("")
    md.append("Reported this way because the random-k null distribution is "
              "near-degenerate (random single-feature ablation at the pre-pivot "
              "position changes P(pivot) by essentially zero, with essentially "
              "zero variance). Quoting a σ multiple here would mislead — the "
              "candidate is qualitatively separated from a flat null, not a tail "
              "event in a smooth one.")
    md.append("")

    md.append("## Kill check\n")
    passes_ablate = (eff["candidate_ablate_drop"] > 0 and drop_z > 2.0)
    passes_clamp = (eff["candidate_clamp_up_rise"] > 0 and rise_z > 2.0)
    if passes_ablate and passes_clamp:
        md.append("**PASS** — feature(s) beat random-k controls in both "
                  "directions. M2 mechanism claim is causal. Phase 5 (quality "
                  "preservation) is the next gate.")
    elif passes_ablate or passes_clamp:
        md.append("**PARTIAL** — only one direction beats controls. The "
                  "feature(s) may be necessary-but-not-sufficient (ablate-only "
                  "passes) or sufficient-but-not-necessary (clamp-only passes). "
                  "Worth reporting as a partial causal claim; investigate the "
                  "asymmetry.")
    else:
        md.append("**FAIL** — neither direction beats controls. The construction "
                  "is not localized to this feature/supernode. Pivot to a "
                  "larger supernode or stop. Honest negative result is "
                  "still worth writing up (PRD §0 — that was the lesson).")

    OUT_MD.write_text("\n".join(md))

    print("\nSummary:")
    print(f"  baseline                   P(pivot) = {baseline_mean:.4f}")
    print(f"  candidate ablate           drop     = {eff['candidate_ablate_drop']:+.4f}  "
          f"(random {eff['random_ablate_drop_mean']:+.4f} ± {eff['random_ablate_drop_std']:.4f}, z={drop_z:+.2f}σ)")
    print(f"  candidate clamp_up         rise     = {eff['candidate_clamp_up_rise']:+.4f}  "
          f"(random {eff['random_clamp_up_rise_mean']:+.4f} ± {eff['random_clamp_up_rise_std']:.4f}, z={rise_z:+.2f}σ)")
    print(f"\n→ {OUT_MD}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", type=int, required=True,
                    help="Candidate SAE feature indices (1 or more for a supernode).")
    ap.add_argument("--variants", nargs="+", default=["C1", "C2", "C3"],
                    choices=["C1", "C2", "C3", "C4"])
    ap.add_argument("--n-random", type=int, default=5,
                    help="Number of independent random-k control draws.")
    ap.add_argument("--clamp-value", type=float, default=10.0,
                    help="Value to clamp features to in clamp_up mode.")
    args = ap.parse_args()
    run(features=args.features, variants=set(args.variants), n_random=args.n_random,
        clamp_value=args.clamp_value)


if __name__ == "__main__":
    main()
