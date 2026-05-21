"""Phase 6 — base vs instruct comparison of the validated feature.

PRD §8 P6: "Compare the validated feature between gemma-2-2b and
gemma-2-2b-it: activation levels on identical prompts, and whether
ablation has the same effect in both."

The genealogy hypothesis: the construction feature exists in base Gemma
but lies relatively dormant; instruct-tuning amplifies its recruitment.
A clean positive result is: feature 9841 fires harder in instruct than
in base on the same contexts, AND ablating it drops P(pivot) more in
instruct than in base.

Caveat (PRD §8 P6 explicit): the Gemma Scope SAE is trained on the
*base* model. Applying it to the instruct model is a known transfer
issue. We verify reconstruction quality on the instruct model first;
any number reported below is conditional on that check.

Usage:
    uv run python scripts/genealogy_compare.py --features 9841
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

log = get_logger("phase6.genealogy")

REPO_ROOT = Path(__file__).resolve().parent.parent
D2_PATH = REPO_ROOT / "data" / "D2_neutral_prompts.json"
D1_PATH = REPO_ROOT / "data" / "D1_contrast_pairs.jsonl"
OUT_JSON = REPO_ROOT / "reports" / "phase6_genealogy.json"
OUT_MD = REPO_ROOT / "reports" / "phase6_genealogy.md"

PIVOT_STRINGS = {
    "C1": [", it", ", they", ", he", ", she", ", we", "—", "; it"],
    "C2": [", but", ", yet", ", also", " but ", " also "],
    "C3": ["—", ", it", " but ", ", but"],
}


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _pivot_re():
    return re.compile(
        r"[,;—–\-]\s*(?:it|that|this|he|she|we|they|these|those|there|but|also|yet)\b",
        re.IGNORECASE,
    )


def truncate_to_pivot(text: str) -> str | None:
    m = _pivot_re().search(text)
    if not m:
        return None
    return text[: m.start()].rstrip()


def reconstruction_quality(model, sae, sentences: list[str], hook_acts_post: str,
                            hook_resid: str) -> dict:
    """SAE round-trip on the model: variance explained, mean L2 error.

    For each sentence we run a forward pass and capture both the post-SAE
    activations and the original residual stream pre-SAE. Compare encoder
    → decoder reconstruction to the input.
    """
    var_explained = []
    l2_err = []
    for s in sentences:
        tokens = model.to_tokens(s, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache_with_saes(tokens, saes=[sae])
        orig = cache[hook_resid][0].float()  # (seq, d_model)
        # The SAE input/output convention can vary; recompute the
        # reconstruction directly via the SAE's encode/decode.
        recon = sae(orig).float()
        # variance explained: 1 - var(residual error) / var(orig)
        err = orig - recon
        ve = 1.0 - (err.var().item() / max(orig.var().item(), 1e-8))
        var_explained.append(ve)
        l2_err.append(float(err.norm(dim=-1).mean().item()))
    return {
        "n": len(sentences),
        "var_explained_mean": float(np.mean(var_explained)),
        "var_explained_median": float(np.median(var_explained)),
        "l2_err_mean": float(np.mean(l2_err)),
    }


def feature_activations(model, sae, sentences: list[str], feat_indices: list[int],
                         hook_acts_post: str) -> np.ndarray:
    """Return (n_sentences, n_feats) of last-token activations."""
    out = np.zeros((len(sentences), len(feat_indices)), dtype=np.float32)
    idx = np.array(feat_indices, dtype=np.int64)
    for i, s in enumerate(sentences):
        tokens = model.to_tokens(s, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache_with_saes(tokens, saes=[sae])
        a = cache[hook_acts_post][0, -1, :].float().cpu().numpy()
        out[i] = a[idx]
    return out


def measure_pivot_drop(model, sae, prefixes: list[str], variants: list[str],
                       feat_indices: list[int], pivot_ids: dict[str, list[int]],
                       hook_acts_post: str) -> dict:
    """For each (prefix, variant), measure baseline P(pivot) and ablated P(pivot).
    Returns mean drop."""
    idxs = torch.tensor(feat_indices, device=model.cfg.device, dtype=torch.long)

    def ablate(act, **kwargs):
        act = act.clone()
        act[..., -1, idxs] = 0.0
        return act

    drops = []
    for prefix, v in zip(prefixes, variants):
        tokens = model.to_tokens(prefix, prepend_bos=True)
        with torch.no_grad():
            base_logits = model.run_with_hooks_with_saes(tokens, saes=[sae], fwd_hooks=[])
            abl_logits = model.run_with_hooks_with_saes(
                tokens, saes=[sae],
                fwd_hooks=[(hook_acts_post, ablate)],
            )
        ids = pivot_ids[v]
        base_p = float(F.softmax(base_logits[0, -1, :].float().cpu(), dim=-1)[ids].sum())
        abl_p = float(F.softmax(abl_logits[0, -1, :].float().cpu(), dim=-1)[ids].sum())
        drops.append({"variant": v, "base_p": base_p, "ablate_p": abl_p,
                      "drop": base_p - abl_p})

    arr = np.array([d["drop"] for d in drops])
    return {
        "n": len(drops),
        "mean_drop": float(arr.mean()),
        "median_drop": float(np.median(arr)),
        "std_drop": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "baseline_mean_p": float(np.mean([d["base_p"] for d in drops])),
        "ablate_mean_p": float(np.mean([d["ablate_p"] for d in drops])),
        "per_sample": drops,
    }


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


def run_one_model(hf_name: str, sae, feat_indices: list[int],
                  truncated_with: list[tuple[str, str]], d2_prompts: list[str],
                  d3_chunks: list[str]) -> dict:
    dev = device()
    log.info(f"loading {hf_name}…")
    model = HookedSAETransformer.from_pretrained(hf_name, device=dev)
    model.eval()
    hook_acts_post = f"{sae.cfg.metadata.hook_name}.hook_sae_acts_post"
    hook_resid = sae.cfg.metadata.hook_name

    # Reconstruction quality on D3
    log.info("measuring SAE reconstruction quality on this model…")
    rq = reconstruction_quality(model, sae, d3_chunks[:4], hook_acts_post, hook_resid)
    log.info(f"  var explained mean = {rq['var_explained_mean']:.3f}")

    # Feature activation on D2
    log.info(f"measuring feature {feat_indices} activation on {len(d2_prompts)} D2 prompts…")
    acts = feature_activations(model, sae, d2_prompts[:50], feat_indices, hook_acts_post)
    feat_summary = {
        f"feat_{f}": {
            "mean": float(acts[:, i].mean()),
            "median": float(np.median(acts[:, i])),
            "fraction_active": float((acts[:, i] > 1e-3).mean()),
        }
        for i, f in enumerate(feat_indices)
    }

    # Ablation effect on truncated D1 with-prompts
    pivot_ids = collect_pivot_token_ids(model, {v for _, v in truncated_with})
    prefixes = [p for p, _ in truncated_with]
    variants = [v for _, v in truncated_with]
    log.info(f"measuring ablation effect on {len(prefixes)} truncated D1 prompts…")
    ab = measure_pivot_drop(model, sae, prefixes, variants, feat_indices, pivot_ids,
                             hook_acts_post)

    # Free
    del model
    if dev == "mps":
        torch.mps.empty_cache()

    return {
        "model": hf_name,
        "reconstruction": rq,
        "feature_acts_on_d2": feat_summary,
        "ablation_pivot_drop_on_d1": {
            "mean_drop": ab["mean_drop"],
            "median_drop": ab["median_drop"],
            "std_drop": ab["std_drop"],
            "baseline_p": ab["baseline_mean_p"],
            "ablate_p": ab["ablate_mean_p"],
            "n": ab["n"],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", type=int, required=True)
    args = ap.parse_args()

    dev = device()
    log.info(f"loading Gemma Scope SAE (trained on base; we'll apply to both)…")
    sae = SAE.from_pretrained(
        release=GEMMA_SAE.release, sae_id=GEMMA_SAE.sae_id, device=dev,
    )
    if isinstance(sae, tuple):
        sae = sae[0]
    sae.eval()

    # D2 prompts
    d2 = json.loads(D2_PATH.read_text())["prompts"]

    # Truncated D1 with-prompts for ablation pivot drop
    truncated_with = []
    for line in D1_PATH.read_text().splitlines():
        if not line.strip() or line.startswith('{"_meta'):
            continue
        d = json.loads(line)
        if d["variant"] not in ("C1", "C2", "C3"):
            continue
        prefix = truncate_to_pivot(d["with"])
        if prefix is None:
            continue
        truncated_with.append((prefix, d["variant"]))
    log.info(f"truncated D1 with-prompts: {len(truncated_with)}")

    # D3 chunks for reconstruction
    d3_path = REPO_ROOT / "data" / "D3_fluency.txt"
    d3_text = d3_path.read_text()
    # Coarse paragraph split
    d3_chunks = [c.strip() for c in d3_text.split("\n\n") if len(c.strip()) > 200]

    results = {}
    for hf_name in ["gemma-2-2b", "gemma-2-2b-it"]:
        results[hf_name] = run_one_model(
            hf_name=hf_name, sae=sae, feat_indices=args.features,
            truncated_with=truncated_with, d2_prompts=d2, d3_chunks=d3_chunks,
        )

    # Compare
    base = results["gemma-2-2b"]
    inst = results["gemma-2-2b-it"]
    for f in args.features:
        key = f"feat_{f}"
        b = base["feature_acts_on_d2"][key]["mean"]
        i = inst["feature_acts_on_d2"][key]["mean"]
        log.info(f"feat {f} on D2: base mean={b:.3f}, instruct mean={i:.3f}, "
                 f"ratio={i/max(b,1e-6):.2f}×")

    OUT_JSON.write_text(json.dumps(results, indent=2))

    # Markdown
    md = [
        "# Phase 6 — Genealogy (base vs instruct)",
        "",
        f"**Features under test:** {args.features}",
        "",
        "## SAE reconstruction quality\n",
        "Gemma Scope is trained on the base model. PRD §8 P6 says we must "
        "verify reconstruction on the instruct model before trusting any "
        "instruct-side number.\n",
        "",
        "| Model | var explained (mean) | var explained (median) | L2 err (mean) |",
        "|---|---:|---:|---:|",
        f"| gemma-2-2b (base) | {base['reconstruction']['var_explained_mean']:.3f} | "
        f"{base['reconstruction']['var_explained_median']:.3f} | "
        f"{base['reconstruction']['l2_err_mean']:.3f} |",
        f"| gemma-2-2b-it (instruct) | {inst['reconstruction']['var_explained_mean']:.3f} | "
        f"{inst['reconstruction']['var_explained_median']:.3f} | "
        f"{inst['reconstruction']['l2_err_mean']:.3f} |",
        "",
    ]
    inst_ve = inst["reconstruction"]["var_explained_mean"]
    if inst_ve > 0.6:
        md.append(f"Instruct VE = {inst_ve:.3f} — SAE transfer is acceptable; "
                  f"instruct-side numbers below can be trusted with light hedging.")
    else:
        md.append(f"**Instruct VE = {inst_ve:.3f}** — SAE transfer is poor. Numbers "
                  f"below are reported but should NOT be quoted without this caveat.")
    md.append("")

    md.append("## Feature activation on D2 (mean of last-token activations, n=50)\n")
    md.append("| Feature | base mean | instruct mean | ratio | base %active | instruct %active |")
    md.append("|---:|---:|---:|---:|---:|---:|")
    for f in args.features:
        key = f"feat_{f}"
        b = base["feature_acts_on_d2"][key]
        i = inst["feature_acts_on_d2"][key]
        ratio = i["mean"] / max(b["mean"], 1e-6)
        md.append(f"| {f} | {b['mean']:.3f} | {i['mean']:.3f} | {ratio:.2f}× | "
                  f"{b['fraction_active']:.2%} | {i['fraction_active']:.2%} |")
    md.append("")

    md.append("## Ablation effect on truncated D1 with-prompts\n")
    md.append("| Model | baseline P(pivot) | ablate P(pivot) | drop (mean ± std) | n |")
    md.append("|---|---:|---:|---:|---:|")
    for label, r in [("gemma-2-2b (base)", base), ("gemma-2-2b-it (instruct)", inst)]:
        a = r["ablation_pivot_drop_on_d1"]
        md.append(f"| {label} | {a['baseline_p']:.4f} | {a['ablate_p']:.4f} | "
                  f"{a['mean_drop']:+.4f} ± {a['std_drop']:.4f} | {a['n']} |")
    md.append("")

    base_drop = base["ablation_pivot_drop_on_d1"]["mean_drop"]
    inst_drop = inst["ablation_pivot_drop_on_d1"]["mean_drop"]
    md.append(f"**Drop ratio (instruct / base): {inst_drop / max(base_drop, 1e-6):.2f}×**")
    md.append("")

    md.append("## Genealogy verdict\n")
    feat_ratios = [inst["feature_acts_on_d2"][f"feat_{f}"]["mean"] /
                    max(base["feature_acts_on_d2"][f"feat_{f}"]["mean"], 1e-6)
                    for f in args.features]
    mean_act_ratio = float(np.mean(feat_ratios))
    drop_ratio = inst_drop / max(base_drop, 1e-6)
    if mean_act_ratio > 1.5 and drop_ratio > 1.5:
        md.append(f"**SUPPORTED** — feature fires {mean_act_ratio:.2f}× harder in "
                  f"instruct on identical D2 contexts, and ablation drops P(pivot) "
                  f"{drop_ratio:.2f}× more in instruct. The 'dormant in base, "
                  f"amplified by instruct' framing has both an activation and a "
                  f"causal signature behind it.")
    elif mean_act_ratio > 1.5:
        md.append(f"**PARTIAL** — feature fires {mean_act_ratio:.2f}× harder in "
                  f"instruct, but the ablation gap ({drop_ratio:.2f}×) is smaller. "
                  f"Instruct recruits the feature more, but the causal effect is "
                  f"similar — the construction's commit isn't more feature-dependent "
                  f"in instruct, just more frequent.")
    elif drop_ratio > 1.5:
        md.append(f"**PARTIAL** — feature activation is similar between base "
                  f"({mean_act_ratio:.2f}× ratio) but ablation drops {drop_ratio:.2f}× "
                  f"more in instruct. The feature is more *causally load-bearing* in "
                  f"instruct without firing more on average.")
    else:
        md.append(f"**NOT SUPPORTED** — feature activation ratio "
                  f"{mean_act_ratio:.2f}× and ablation ratio {drop_ratio:.2f}× are "
                  f"both close to 1×. The Phase 2 behavioural gap doesn't come "
                  f"through this feature.")

    OUT_MD.write_text("\n".join(md))
    print(f"\n→ {OUT_MD}")


if __name__ == "__main__":
    main()
