"""Phase 2 scoring — compute M1 across models with bootstrap CIs.

Reads `reports/phase2_generations_<model>.jsonl` (one row per generation),
runs the construction classifier on each generation, and produces:

  - per-model construction rate per variant (C1, C2, C3, C4, any_core)
  - sentence-level bootstrap 95% CIs (PRD §8 Phase 2 + HANDOVER §5 P2 gotcha:
    the resampling unit is the *sentence*, not the generation, because a
    300-token generation can contain multiple sentences and the per-sentence
    rate is the cleaner statistic)
  - base-vs-instruct gap test for Gemma 2 2B (the Phase 6 genealogy
    motivating expectation)

Writes:
  - reports/phase2_baseline.json
  - reports/phase2_baseline.md

Usage:
    uv run python scripts/score_phase2.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from classifier import Variant, detect_construction
from neograph.util import get_logger

log = get_logger("phase2.score")

REPO_ROOT = Path(__file__).resolve().parent.parent
GEN_DIR = REPO_ROOT / "reports"
N_BOOTSTRAP = 2000
RNG_SEED = 7


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(text: str) -> list[str]:
    """Cheap sentence split — good enough for M1 since the construction's
    hinges are sentence-local. Avoids an extra spaCy parse per generation."""
    text = text.strip()
    if not text:
        return []
    return [s for s in SENT_SPLIT.split(text) if s.strip()]


def load_generations(model: str) -> list[dict]:
    path = GEN_DIR / f"phase2_generations_{model}.jsonl"
    if not path.exists():
        log.warning(f"missing: {path}")
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score_sentences(generations: list[dict], *, strict: bool) -> list[dict]:
    """Yield one row per sentence with its construction labels."""
    rows = []
    for gen in generations:
        sents = split_sentences(gen["generation"])
        for sent_idx, sent in enumerate(sents):
            hits = detect_construction(sent, strict=strict)
            present = {h.variant.value for h in hits}
            rows.append({
                "model": gen["model"],
                "prompt_idx": gen["prompt_idx"],
                "seed": gen["seed"],
                "sent_idx": sent_idx,
                "sentence": sent,
                "C1": "C1" in present,
                "C2": "C2" in present,
                "C3": "C3" in present,
                "C4": "C4" in present,
                "any_core": bool(present & {"C1", "C2", "C3"}),
                "any": bool(present),
            })
    return rows


def bootstrap_ci(values: np.ndarray, *, n_bootstrap: int = N_BOOTSTRAP,
                 rng: np.random.Generator) -> tuple[float, float, float]:
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    idxs = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    boots = values[idxs].mean(axis=1)
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def summarise_model(sent_rows: list[dict], *, model: str,
                    rng: np.random.Generator) -> dict:
    if not sent_rows:
        return {"model": model, "n_sentences": 0}
    arr = {k: np.array([r[k] for r in sent_rows], dtype=bool) for k in
           ("C1", "C2", "C3", "C4", "any_core", "any")}
    summary = {"model": model, "n_sentences": len(sent_rows),
               "n_generations": len({(r["prompt_idx"], r["seed"]) for r in sent_rows})}
    for k, vals in arr.items():
        mean, lo, hi = bootstrap_ci(vals.astype(float), rng=rng)
        summary[k] = {"rate": mean, "ci95_lo": lo, "ci95_hi": hi,
                       "n_positive": int(vals.sum())}
    return summary


def write_markdown(summaries: list[dict], out_path: Path) -> None:
    lines = []
    lines.append("# Phase 2 — Behavioral baseline (M1)\n")
    lines.append(
        "Per-model construction rates on D2 neutral prompts. M1 is computed "
        "**per sentence**, not per generation (see HANDOVER §5 P2 gotcha — a "
        "300-token generation contains multiple sentences and the resampling "
        "unit is the sentence). CIs are 95% bootstrap with 2 000 resamples.\n"
    )
    lines.append("## Setup\n")
    lines.append("- 102 D2 prompts × 5 seeds = 510 generations per model")
    lines.append("- Sampling: temperature=0.8, top_p=0.95, max_new_tokens=150")
    lines.append("- Classifier: strict mode (regex + spaCy dependency filter)")
    lines.append("")

    lines.append("## Construction rates\n")
    lines.append("| Model | n_sent | C1 | C2 | C3 | C4 | any_core (C1∪C2∪C3) |")
    lines.append("|---|---:|---|---|---|---|---|")
    for s in summaries:
        if s.get("n_sentences", 0) == 0:
            lines.append(f"| {s['model']} | 0 | — | — | — | — | — |")
            continue
        def fmt(v):
            return f"{v['rate']:.3f} ({v['ci95_lo']:.3f}–{v['ci95_hi']:.3f})"
        lines.append(
            f"| {s['model']} | {s['n_sentences']} | "
            f"{fmt(s['C1'])} | {fmt(s['C2'])} | {fmt(s['C3'])} | "
            f"{fmt(s['C4'])} | {fmt(s['any_core'])} |"
        )
    lines.append("")

    # Base-vs-instruct: the motivating expectation for Phase 6.
    by_name = {s["model"]: s for s in summaries}
    base, it = by_name.get("gemma_2b"), by_name.get("gemma_2b_it")
    if base and it and base.get("n_sentences") and it.get("n_sentences"):
        lines.append("## Base vs instruct (Gemma 2 2B)\n")
        lines.append("The Phase 6 genealogy hypothesis (PRD §8 P6) predicts the "
                     "construction is *dormant in base, amplified by instruct*. "
                     "Phase 2 surfaces the gap.\n")
        for k in ("C1", "C2", "C3", "any_core"):
            b, i = base[k]["rate"], it[k]["rate"]
            ratio = (i / b) if b > 0 else float("inf") if i > 0 else 1.0
            lines.append(f"- **{k}**: base = {b:.3f} [{base[k]['ci95_lo']:.3f},"
                         f"{base[k]['ci95_hi']:.3f}], "
                         f"instruct = {i:.3f} [{it[k]['ci95_lo']:.3f},"
                         f"{it[k]['ci95_hi']:.3f}], ratio = {ratio:.2f}×")
        lines.append("")
        gap = it["any_core"]["rate"] - base["any_core"]["rate"]
        lines.append(f"**any_core gap (instruct − base): {gap:+.3f}.**")
        if it["any_core"]["ci95_lo"] > base["any_core"]["ci95_hi"]:
            lines.append("CIs do not overlap — the gap is clean. Phase 6 has a "
                         "live story to chase.")
        elif gap > 0:
            lines.append("Gap is positive but CIs overlap — the genealogy "
                         "hypothesis is alive but not yet decisive at this N.")
        else:
            lines.append("Gap is non-positive — Phase 6 genealogy is in trouble. "
                         "Within-model mechanism (Phases 3–5) can still hold.")
    lines.append("")

    out_path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["pythia_70m", "gpt2", "gemma_2b", "gemma_2b_it"])
    ap.add_argument("--out-json", type=Path,
                    default=REPO_ROOT / "reports" / "phase2_baseline.json")
    ap.add_argument("--out-md", type=Path,
                    default=REPO_ROOT / "reports" / "phase2_baseline.md")
    ap.add_argument("--sentences-out", type=Path,
                    default=REPO_ROOT / "reports" / "phase2_scored_sentences.jsonl")
    ap.add_argument("--no-strict", action="store_true",
                    help="Disable the spaCy dep filter (regex-only).")
    args = ap.parse_args()

    rng = np.random.default_rng(RNG_SEED)
    summaries = []
    all_sent_rows: list[dict] = []
    for model in args.models:
        gens = load_generations(model)
        log.info(f"[{model}] {len(gens)} generations")
        sent_rows = score_sentences(gens, strict=not args.no_strict)
        log.info(f"[{model}] {len(sent_rows)} sentences")
        all_sent_rows.extend(sent_rows)
        summaries.append(summarise_model(sent_rows, model=model, rng=rng))

    args.out_json.write_text(json.dumps(summaries, indent=2))
    write_markdown(summaries, args.out_md)
    with args.sentences_out.open("w") as f:
        for r in all_sent_rows:
            f.write(json.dumps(r) + "\n")

    print(f"\n{'model':15s} {'n_sent':>6s} {'C1':>10s} {'C2':>10s} {'C3':>10s} "
          f"{'any_core':>10s}")
    for s in summaries:
        if s.get("n_sentences", 0) == 0:
            print(f"{s['model']:15s} {'0':>6s} {'(no gens)':>10s}")
            continue
        print(f"{s['model']:15s} {s['n_sentences']:6d} "
              f"{s['C1']['rate']:>10.3f} {s['C2']['rate']:>10.3f} "
              f"{s['C3']['rate']:>10.3f} {s['any_core']['rate']:>10.3f}")
    print(f"\n→ {args.out_md}")


if __name__ == "__main__":
    main()
