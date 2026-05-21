"""Phase 5 — quality preservation under intervention (M3).

PRD §5 M3: three checks gate the *product* claim.

  1. **Fluency** — perplexity of held-out text (D3) under the intervened
     model. Must not blow up.
  2. **Meaning** — embedding cosine between baseline generation and
     intervened generation on D1 'with'-style prompts. Must stay high.
  3. **Coherence** — LLM-judge 1–5 rating. Stub here; wired in iteration.

The mechanism claim survives an M3 failure; the de-slop *product* claim
does not. Phase 4 says we have a lever; Phase 5 says whether the lever
is a scalpel.

Usage:
    uv run python scripts/quality_preservation.py --features 6631
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

from neograph.config import SAE as GEMMA_SAE
from neograph.util import get_logger

log = get_logger("phase5.quality")

REPO_ROOT = Path(__file__).resolve().parent.parent
D3_PATH = REPO_ROOT / "data" / "D3_fluency.txt"
D1_PATH = REPO_ROOT / "data" / "D1_contrast_pairs.jsonl"
OUT_JSON = REPO_ROOT / "reports" / "phase5_quality.json"
OUT_MD = REPO_ROOT / "reports" / "phase5_quality.md"


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def make_clamp_hook(feat_indices: list[int], value: float):
    idx_tensor = None

    def hook(act, **kwargs):
        nonlocal idx_tensor
        if idx_tensor is None:
            idx_tensor = torch.tensor(feat_indices, device=act.device, dtype=torch.long)
        act = act.clone()
        # Apply at ALL positions in the sequence — the construction can
        # commit anywhere mid-generation. Phase 4 M2 only clamps the last
        # position because that's the next-token decision; Phase 5
        # generation needs the feature suppressed throughout.
        act[..., idx_tensor] = value
        return act

    return hook


def measure_perplexity(model, sae, text: str, *, feat_indices: list[int] | None,
                        value: float = 0.0) -> float:
    """Token-level NLL → perplexity. If feat_indices, install the clamp hook."""
    tokens = model.to_tokens(text, prepend_bos=True)
    if tokens.shape[1] < 2:
        return float("nan")
    hook_name = f"{sae.cfg.hook_name}.hook_sae_acts_post"

    if feat_indices is None:
        with torch.no_grad():
            logits = model.run_with_hooks_with_saes(tokens, saes=[sae], fwd_hooks=[])
    else:
        hook = make_clamp_hook(feat_indices, value)
        with torch.no_grad():
            logits = model.run_with_hooks_with_saes(
                tokens, saes=[sae], fwd_hooks=[(hook_name, hook)]
            )

    # Shift: logits[i] predicts tokens[i+1]
    log_probs = F.log_softmax(logits[0, :-1, :].float(), dim=-1)
    target = tokens[0, 1:]
    token_nll = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    mean_nll = float(token_nll.mean().cpu().item())
    return float(np.exp(mean_nll))


def chunk_text(text: str, max_chars: int = 1200) -> list[str]:
    """Split D3 into ~1k-token chunks for perplexity averaging."""
    text = text.strip()
    chunks, cur = [], []
    cur_len = 0
    for para in text.split("\n\n"):
        if cur_len + len(para) > max_chars and cur:
            chunks.append("\n\n".join(cur))
            cur, cur_len = [], 0
        cur.append(para)
        cur_len += len(para) + 2
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def run(features: list[int]) -> None:
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

    # === FLUENCY (D3 perplexity, baseline vs ablate vs clamp_up) ============
    d3 = D3_PATH.read_text()
    chunks = chunk_text(d3)
    log.info(f"D3: {len(chunks)} chunks, ~{sum(len(c) for c in chunks)} chars")

    perplexity = {"baseline": [], "ablate": [], "clamp_up": []}
    t0 = time.perf_counter()
    for i, chunk in enumerate(chunks):
        perplexity["baseline"].append(measure_perplexity(model, sae, chunk, feat_indices=None))
        perplexity["ablate"].append(measure_perplexity(model, sae, chunk, feat_indices=features, value=0.0))
        perplexity["clamp_up"].append(measure_perplexity(model, sae, chunk, feat_indices=features, value=10.0))
        if (i + 1) % 2 == 0:
            log.info(f"  fluency: {i + 1}/{len(chunks)} chunks "
                     f"({(i + 1) / (time.perf_counter() - t0):.2f}/s)")

    fluency_summary = {
        cond: {
            "geomean_perplexity": float(np.exp(np.mean(np.log(vals)))),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "n_chunks": len(vals),
        }
        for cond, vals in perplexity.items()
    }
    base_ppl = fluency_summary["baseline"]["geomean_perplexity"]
    ablate_ratio = fluency_summary["ablate"]["geomean_perplexity"] / base_ppl
    clamp_ratio = fluency_summary["clamp_up"]["geomean_perplexity"] / base_ppl
    fluency_summary["ablate_ppl_ratio"] = ablate_ratio
    fluency_summary["clamp_up_ppl_ratio"] = clamp_ratio
    log.info(f"D3 perplexity ratios: ablate={ablate_ratio:.3f}× clamp_up={clamp_ratio:.3f}×")

    # === MEANING — embedding cosine on D1 'with' under ablation ============
    # We don't generate here (that would be Phase 4's M1 territory); instead
    # we compare the *original* 'with' sentence to its de-slopped paraphrase
    # 'without' across D1 — that gives a baseline cosine for how much
    # meaning is preserved by the construction's removal. The Phase 7
    # de-slop demo will rerun this comparison on generated paraphrases.
    log.info("loading sentence embedder for meaning check…")
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
    except Exception as e:
        log.warning(f"sentence_transformers unavailable ({e}); skipping meaning check")
        embedder = None

    meaning_summary = None
    if embedder is not None:
        pairs = [json.loads(l) for l in D1_PATH.read_text().splitlines()
                 if l.strip() and not l.startswith('{"_meta')]
        with_emb = embedder.encode([p["with"] for p in pairs], normalize_embeddings=True,
                                    show_progress_bar=False)
        without_emb = embedder.encode([p["without"] for p in pairs], normalize_embeddings=True,
                                       show_progress_bar=False)
        cos = (with_emb * without_emb).sum(axis=1)
        meaning_summary = {
            "mean_cosine_with_vs_without": float(np.mean(cos)),
            "median": float(np.median(cos)),
            "p10": float(np.percentile(cos, 10)),
            "p90": float(np.percentile(cos, 90)),
            "n_pairs": int(cos.size),
            "note": "D1 baseline: meaning preserved by the paraphraser (us). "
                    "Phase 7 will compare baseline gen vs intervened gen.",
        }
        log.info(f"meaning cosine (D1 with vs without): mean={meaning_summary['mean_cosine_with_vs_without']:.3f}")

    # === COHERENCE — stub ====================================================
    coherence_summary = {
        "status": "stub",
        "note": "Phase 5 v2 wires Anthropic Claude as the LLM judge on a sample of "
                "Phase 4 M1 ablate generations (baseline vs intervened, blind to "
                "condition), 1-5 coherence rating. Skipped here.",
    }

    # === Aggregate + report =================================================
    result = {
        "features": features,
        "fluency": fluency_summary,
        "meaning": meaning_summary,
        "coherence": coherence_summary,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2))

    md = [
        "# Phase 5 — Quality preservation (M3)",
        "",
        f"**Features under test:** {features}",
        "",
        "## Fluency (D3 perplexity)\n",
        "| Condition | geom-mean PPL | mean | median | n chunks |",
        "|---|---:|---:|---:|---:|",
    ]
    for cond in ("baseline", "ablate", "clamp_up"):
        s = fluency_summary[cond]
        md.append(f"| {cond} | {s['geomean_perplexity']:.3f} | {s['mean']:.3f} | "
                  f"{s['median']:.3f} | {s['n_chunks']} |")
    md.append("")
    md.append(f"- Ablate / baseline = **{ablate_ratio:.3f}×**")
    md.append(f"- Clamp-up / baseline = **{clamp_ratio:.3f}×**")
    md.append("")
    if max(ablate_ratio, clamp_ratio) < 1.20:
        md.append("**Fluency PASS** — perplexity ratio under 1.2× baseline.")
    elif max(ablate_ratio, clamp_ratio) < 1.50:
        md.append("**Fluency MARGINAL** — perplexity ratio between 1.2 and 1.5×; "
                  "the lever is closer to a sledgehammer than a scalpel.")
    else:
        md.append("**Fluency FAIL** — perplexity > 1.5× baseline. The de-slop "
                  "product claim does not survive; the mechanism finding can.")
    md.append("")
    md.append("## Meaning (D1 with vs without baseline cosine)\n")
    if meaning_summary:
        md.append(f"- mean cosine = {meaning_summary['mean_cosine_with_vs_without']:.3f}, "
                  f"median = {meaning_summary['median']:.3f}, "
                  f"p10 = {meaning_summary['p10']:.3f}, "
                  f"p90 = {meaning_summary['p90']:.3f} (n = {meaning_summary['n_pairs']})")
        md.append("")
        md.append("*Baseline reference only*: this is how much meaning the D1 "
                  "paraphraser preserved across the construction's removal. Phase 7 "
                  "compares baseline generation vs intervened generation under the "
                  "same metric — the actual product claim is the Phase 7 number.")
    else:
        md.append("sentence-transformers unavailable; install to enable.")
    md.append("")
    md.append("## Coherence — LLM-judge\n")
    md.append(coherence_summary["note"])

    OUT_MD.write_text("\n".join(md))
    print(f"\n→ {OUT_MD}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", type=int, required=True)
    args = ap.parse_args()
    run(features=args.features)


if __name__ == "__main__":
    main()
