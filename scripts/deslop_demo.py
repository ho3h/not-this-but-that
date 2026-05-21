"""Phase 7 — de-slop demo. Generate from D2 with intervention; before/after.

PRD §8 Phase 7: "Package the validated intervention as an inference-time
steering vector: clamp construction features toward zero, regenerate.
Build a tiny before/after demo. Run a small human (or LLM-judge) eval:
are steered outputs rated 'less AI-sounding' while staying fluent?"

We generate token-by-token via the HookedSAETransformer so we can keep
the clamp hook installed throughout sampling. Two conditions:

  - baseline: no clamp
  - ablate:   feature(s) clamped to 0 at the SAE post-encode hook

For each prompt and seed we run both conditions, then:
  - score both with the construction classifier (M1 drop)
  - embed both with MiniLM (meaning cosine — high = meaning preserved)
  - print a small side-by-side artefact

Usage:
    uv run python scripts/deslop_demo.py --features 9841 --prompts 12 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sae_lens import HookedSAETransformer, SAE

from classifier import detect_construction
from neograph.config import SAE as GEMMA_SAE
from neograph.util import get_logger

log = get_logger("phase7.deslop")

REPO_ROOT = Path(__file__).resolve().parent.parent
D2_PATH = REPO_ROOT / "data" / "D2_neutral_prompts.json"
OUT_JSON = REPO_ROOT / "reports" / "phase7_deslop.json"
OUT_MD = REPO_ROOT / "reports" / "phase7_deslop.md"


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def make_clamp_hook(feat_indices: list[int], value: float):
    idx_tensor = None

    def hook(act, **kwargs):
        nonlocal idx_tensor
        if idx_tensor is None:
            idx_tensor = torch.tensor(feat_indices, device=act.device, dtype=torch.long)
        act = act.clone()
        act[..., idx_tensor] = value
        return act

    return hook


def sample_with_intervention(model, sae, prompt: str, max_new_tokens: int,
                              temperature: float, top_p: float, seed: int,
                              feat_indices: list[int] | None,
                              clamp_value: float = 0.0) -> str:
    torch.manual_seed(seed)
    dev = next(model.parameters()).device
    if dev.type == "mps":
        torch.mps.manual_seed(seed)

    tokens = model.to_tokens(prompt, prepend_bos=True).to(dev)
    hook_name = f"{sae.cfg.metadata.hook_name}.hook_sae_acts_post"
    hooks = []
    if feat_indices:
        hooks.append((hook_name, make_clamp_hook(feat_indices, clamp_value)))

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model.run_with_hooks_with_saes(tokens, saes=[sae], fwd_hooks=hooks)
        last = logits[0, -1, :].float() / max(temperature, 1e-6)
        probs = F.softmax(last, dim=-1)
        # top-p sampling
        sp, si = torch.sort(probs, descending=True)
        csum = torch.cumsum(sp, dim=-1)
        keep = csum < top_p
        keep[..., 0] = True  # always keep top-1
        filt = torch.zeros_like(probs)
        filt[si[keep]] = sp[keep]
        filt = filt / filt.sum()
        next_id = torch.multinomial(filt, num_samples=1)
        tokens = torch.cat([tokens, next_id.unsqueeze(0)], dim=1)
        if next_id.item() == model.tokenizer.eos_token_id:
            break

    text = model.tokenizer.decode(tokens[0, len(model.to_tokens(prompt, prepend_bos=True)[0]):],
                                    skip_special_tokens=True)
    return text


def construction_present(text: str) -> bool:
    hits = detect_construction(text, strict=False)
    return any(h.variant.value in ("C1", "C2", "C3") for h in hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", type=int, required=True)
    ap.add_argument("--prompts", type=int, default=12,
                    help="How many D2 prompts to use.")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    dev = device()
    log.info(f"loading gemma-2-2b-it on {dev} (instruct has higher C3 rate per Phase 2)…")
    model = HookedSAETransformer.from_pretrained("gemma-2-2b-it", device=dev)
    model.eval()

    log.info("loading Gemma Scope SAE")
    sae = SAE.from_pretrained(
        release=GEMMA_SAE.release, sae_id=GEMMA_SAE.sae_id, device=dev,
    )
    if isinstance(sae, tuple):
        sae = sae[0]
    sae.eval()

    d2 = json.loads(D2_PATH.read_text())["prompts"][: args.prompts]
    log.info(f"generating on {len(d2)} prompts × {args.seeds} seeds × 2 conditions…")

    rows = []
    t0 = time.perf_counter()
    for pi, prompt in enumerate(d2):
        for seed in range(args.seeds):
            # baseline
            baseline = sample_with_intervention(
                model, sae, prompt, args.max_new_tokens,
                args.temperature, args.top_p, seed, feat_indices=None,
            )
            # ablate
            ablated = sample_with_intervention(
                model, sae, prompt, args.max_new_tokens,
                args.temperature, args.top_p, seed, feat_indices=args.features,
                clamp_value=0.0,
            )
            rows.append({
                "prompt": prompt, "seed": seed,
                "baseline": baseline, "ablated": ablated,
                "baseline_has_construction": construction_present(baseline),
                "ablated_has_construction": construction_present(ablated),
            })
        log.info(f"  {pi + 1}/{len(d2)} prompts done "
                 f"({(pi + 1) * args.seeds / (time.perf_counter() - t0):.2f}/s)")

    n = len(rows)
    baseline_rate = sum(1 for r in rows if r["baseline_has_construction"]) / n
    ablated_rate = sum(1 for r in rows if r["ablated_has_construction"]) / n

    # Meaning cosine
    log.info("computing embedding cosine…")
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
        be = embedder.encode([r["baseline"] for r in rows], normalize_embeddings=True,
                              show_progress_bar=False)
        ae = embedder.encode([r["ablated"] for r in rows], normalize_embeddings=True,
                              show_progress_bar=False)
        cos = (be * ae).sum(axis=1)
        meaning = {"mean": float(np.mean(cos)), "median": float(np.median(cos)),
                   "p10": float(np.percentile(cos, 10)),
                   "p90": float(np.percentile(cos, 90))}
    except Exception as e:
        log.warning(f"sentence_transformers unavailable: {e}")
        meaning = None

    result = {
        "features": args.features,
        "n_pairs": n,
        "baseline_construction_rate": baseline_rate,
        "ablated_construction_rate": ablated_rate,
        "absolute_drop": baseline_rate - ablated_rate,
        "relative_drop": (baseline_rate - ablated_rate) / max(baseline_rate, 1e-6),
        "meaning_cosine": meaning,
        "sampling": {"temperature": args.temperature, "top_p": args.top_p,
                     "max_new_tokens": args.max_new_tokens},
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))

    md = [
        "# Phase 7 — De-slop demo (Gemma 2 2B-it, feature ablation)",
        "",
        f"**Features ablated:** {args.features}",
        f"**Pairs:** {n} ({len(d2)} D2 prompts × {args.seeds} seeds)",
        f"**Sampling:** temperature={args.temperature}, top_p={args.top_p}, "
        f"max_new_tokens={args.max_new_tokens}",
        "",
        "## Construction rate (any-core C1∪C2∪C3)\n",
        f"- baseline (no intervention): **{baseline_rate:.2%}** ({sum(1 for r in rows if r['baseline_has_construction'])}/{n} generations contain the construction)",
        f"- ablated (feature → 0):     **{ablated_rate:.2%}** ({sum(1 for r in rows if r['ablated_has_construction'])}/{n})",
        f"- absolute drop:             **{result['absolute_drop']:+.2%}**",
        f"- relative drop:             **{result['relative_drop']:+.2%}**",
        "",
    ]
    if meaning is not None:
        md.append("## Meaning preservation (MiniLM cosine of baseline vs ablated)\n")
        md.append(f"- mean cosine = {meaning['mean']:.3f}, median = {meaning['median']:.3f}")
        md.append(f"- 10th pct = {meaning['p10']:.3f}, 90th pct = {meaning['p90']:.3f}")
        md.append("")
        if meaning["mean"] >= 0.75:
            md.append("Meaning is largely preserved (mean cosine ≥ 0.75) — the ablation "
                      "is a scalpel, not a sledgehammer.")
        elif meaning["mean"] >= 0.55:
            md.append("Meaning is partially preserved (mean cosine 0.55–0.75) — the "
                      "ablation pulls the generation somewhere coherent but related.")
        else:
            md.append("Meaning has drifted (mean cosine < 0.55) — the ablation "
                      "doesn't just remove the construction, it changes the topic. "
                      "Phase 5 quality check would call this a failure.")
        md.append("")

    md.append("## Examples (first 6 pairs)\n")
    for r in rows[:6]:
        bch = "✓C" if r["baseline_has_construction"] else "·"
        ach = "✓C" if r["ablated_has_construction"] else "·"
        md.append(f"### `{r['prompt']}` (seed {r['seed']})\n")
        md.append(f"**Baseline** {bch}:  \n> {r['baseline'].strip()[:400]}")
        md.append("")
        md.append(f"**Ablated** {ach}:  \n> {r['ablated'].strip()[:400]}")
        md.append("")
        md.append("---\n")

    OUT_MD.write_text("\n".join(md))
    print(f"\nbaseline construction rate: {baseline_rate:.2%}")
    print(f"ablated  construction rate: {ablated_rate:.2%}")
    print(f"absolute drop:              {result['absolute_drop']:+.2%}")
    if meaning is not None:
        print(f"meaning cosine (mean):      {meaning['mean']:.3f}")
    print(f"\n→ {OUT_MD}")


if __name__ == "__main__":
    main()
