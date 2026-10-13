"""Which training stage adds "not X but Y"? OLMo 2 1B at four public checkpoints (exploratory).

allenai/OLMo-2-0425-1B (pretrained base) -> -SFT (instruction tuning) -> -DPO (preference
tuning) -> -Instruct (final, adds RL with verifiable rewards). Every stage answers the same
152 held-out turn-up prompts (93 meaning, 59 process), 4 samples each, 250 new tokens,
T=0.8, top-p 0.95, with the same uniforms per prompt in every stage. Every stage gets the
identical input, built with the Instruct tokenizer's chat template; the base model never saw
that template in training, so its row is a reference point, not a like-for-like comparison.
Text after any turn or role tag the model invents ("<|user|>", "<|teacher|>") is cut before scoring.

Scored with detector v3 (family or cousin): share of answers using the construction, and
uses per 1,000 words (answer lengths differ by stage). CIs: 5,000 prompt-clustered bootstrap
resamples. Writes reports/why_stages.{json,md} and reports/why_stages_generations.jsonl.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g3_mlx as G  # noqa: E402
import v2_mlx_multi as MM  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from classifier.detect_v3 import detect_v3  # noqa: E402

STAGES = [("base", "allenai/OLMo-2-0425-1B"), ("SFT", "allenai/OLMo-2-0425-1B-SFT"),
          ("DPO", "allenai/OLMo-2-0425-1B-DPO"), ("Instruct", "allenai/OLMo-2-0425-1B-Instruct")]
N, MAX_NEW, SEED, BOOT = 4, 250, 20261010, 5000
CUT = re.compile(r"<\|[^|<>]{1,30}\|>")  # any turn or role tag the model invents ("<|user|>", "<|teacher|>")


def prompts() -> list[tuple[str, str]]:
    seen = {}
    for line in (REPO / "reports/turn_up_generations.jsonl").read_text().splitlines():
        r = json.loads(line)
        seen.setdefault(r["prompt"], r["type"])
    return list(seen.items())


def score(text: str) -> tuple[int, int]:
    d = detect_v3(text)
    return len(d["family"]) + len(d["cousin"]), len(text.split())


def local_path(repo: str) -> str:
    """MLX loads safetensors only; the SFT and DPO checkpoints ship pytorch_model.bin, so convert once."""
    from huggingface_hub import snapshot_download

    snap = Path(snapshot_download(repo))
    if any(snap.glob("*.safetensors")):
        return str(snap)
    out = Path.home() / ".cache/not-this-but-that" / repo.replace("/", "--")
    if not (out / "model.safetensors").exists():
        import shutil
        import torch
        from safetensors.torch import save_file

        out.mkdir(parents=True, exist_ok=True)
        sd = torch.load(snap / "pytorch_model.bin", map_location="cpu", weights_only=True)
        save_file({k: v.contiguous().clone() for k, v in sd.items()}, out / "model.safetensors", metadata={"format": "pt"})
        for f in snap.iterdir():
            if f.suffix in (".json", ".txt") and f.name != "model.safetensors.index.json":
                shutil.copy(f, out / f.name)
    return str(out)


def generate(gen: Path) -> list[dict]:
    from mlx_lm import load as mlx_load
    from transformers import AutoTokenizer

    chat_tok = AutoTokenizer.from_pretrained(STAGES[-1][1])
    items = prompts()
    rows = [json.loads(l) for l in gen.read_text().splitlines() if l.strip()] if gen.exists() else []
    done = {r["stage"] for r in rows}
    for stage, repo in STAGES:
        if stage in done:
            print(f"{stage}: cached", flush=True)
            continue
        model, tok = mlx_load(local_path(repo))
        G.fold_ropes(model)
        probe = "<|user|>\nWhy do people bake bread?\n<|assistant|>\n"
        assert tok.encode(probe) == chat_tok.encode(probe), f"{stage} tokenizer differs from Instruct's"
        for i, (p, typ) in enumerate(items):
            ids = chat_tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)["input_ids"]
            U = mx.array(np.random.default_rng(SEED + i).random((N, MAX_NEW)).astype(np.float32))
            outs = MM.sample_rows(model, tok, list(ids), U, max_new_tokens=MAX_NEW, temperature=0.8, top_p=0.95)
            for k, text in enumerate(outs):
                text = CUT.split(text)[0].strip()
                h, w = score(text)
                rows.append({"stage": stage, "id": i, "type": typ, "prompt": p, "sample": k, "text": text,
                             "hits": h, "words": w})
            if (i + 1) % 25 == 0:
                print(f"{stage} {i + 1}/{len(items)}", flush=True)
        gen.write_text("\n".join(json.dumps(r) for r in rows) + "\n")  # checkpoint after every stage
        del model
        mx.clear_cache()
    return rows


def summarise(rows: list[dict]) -> dict:
    stages = [s for s, _ in STAGES]
    ids = sorted({r["id"] for r in rows})
    typ = {r["id"]: r["type"] for r in rows}
    cell = {}
    for r in rows:
        c = cell.setdefault((r["stage"], r["id"]), [0, 0, 0, 0])
        c[0] += r["hits"] > 0; c[1] += 1; c[2] += r["hits"]; c[3] += r["words"]
    rng = np.random.default_rng(SEED)
    out = {}
    for t in ("all", "M", "P"):
        sel = np.array([i for i in ids if t == "all" or typ[i] == t])
        A = {s: np.array([cell[(s, i)] for i in sel], dtype=float) for s in stages}
        share = lambda a: a[:, 0].sum() / a[:, 1].sum()  # noqa: E731
        per_k = lambda a: 1000 * a[:, 2].sum() / max(a[:, 3].sum(), 1)  # noqa: E731
        res = {s: {"answers_with_hit": int(A[s][:, 0].sum()), "answers": int(A[s][:, 1].sum()),
                   "share": float(share(A[s])), "per_1000_words": float(per_k(A[s])),
                   "mean_words": float(A[s][:, 3].sum() / A[s][:, 1].sum())} for s in stages}
        for a, b in list(zip(stages[1:], stages[:-1])) + [("Instruct", "SFT")]:
            boots = []
            for _ in range(BOOT):
                ix = rng.integers(0, len(sel), len(sel))
                den = share(A[b][ix])
                boots.append(share(A[a][ix]) / den if den > 0 else np.inf)
            fin = np.array([v for v in boots if np.isfinite(v)])
            res[f"{a}/{b}"] = {"ratio": float(share(A[a]) / share(A[b])) if share(A[b]) > 0 else float("inf"),
                               "ci95": [float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))] if len(fin) else None}
        out[t] = res
    return out


def main() -> None:
    gen = REPO / "reports/why_stages_generations.jsonl"
    if "--rescore" in sys.argv and gen.exists():
        rows = [json.loads(l) for l in gen.read_text().splitlines() if l.strip()]
        for r in rows:
            r["text"] = CUT.split(r["text"])[0].strip()
            r["hits"], r["words"] = score(r["text"])
    else:
        rows = generate(gen)
    res = summarise(rows)
    (REPO / "reports/why_stages.json").write_text(json.dumps(res, indent=2))
    md = ["# Which training stage adds \"not X but Y\"? OLMo 2 1B (exploratory)", "",
          "152 held-out prompts × 4 samples per stage, 250 new tokens. Detector v3, family or cousin.", "",
          "| prompts | stage | answers using it | share | per 1,000 words | mean words |", "|---|---|---:|---:|---:|---:|"]
    for t, label in (("all", "all 152"), ("M", "93 meaning"), ("P", "59 process")):
        for s, _ in STAGES:
            v = res[t][s]
            md.append(f"| {label} | {s} | {v['answers_with_hit']} / {v['answers']} | {100 * v['share']:.1f}% | "
                      f"{v['per_1000_words']:.2f} | {v['mean_words']:.0f} |")
    md += ["", "Stage-to-stage ratios of the share (95% prompt-clustered bootstrap CI):", ""]
    for t, label in (("all", "all 152"), ("M", "93 meaning"), ("P", "59 process")):
        for k in ("SFT/base", "DPO/SFT", "Instruct/DPO", "Instruct/SFT"):
            v = res[t][k]
            ci = f"{v['ci95'][0]:.2f}× to {v['ci95'][1]:.2f}×" if v["ci95"] else "n/a"
            md.append(f"- {label}, {k}: **{v['ratio']:.2f}×** ({ci})")
    md += ["", "Caveat: the base model gets the same chat-formatted input as the tuned stages, which it was not "
           "trained on. Compare SFT, DPO and Instruct with each other; read base as a reference point.", "",
           "## Reading", "",
           "Instruction tuning (SFT) multiplies the share of answers using the construction about fivefold, but "
           "only by making answers longer: per 1,000 words the rate does not move. Preference tuning (DPO) and the "
           "final RL stage then raise the rate per word on meaning questions, at a length that no longer changes "
           "(DPO and Instruct answers all reach the 250-token cap), while process questions stay flat. Each single "
           "step is within noise at this sample size; SFT to Instruct taken together is the comparison to read."]
    (REPO / "reports/why_stages.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
