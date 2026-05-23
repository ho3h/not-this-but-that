"""Common infrastructure for the kill-the-AI-ism gauntlet.

Each attack script loads Gemma 2 2B-it, sets up an intervention, generates
on the gauntlet's fresh prompt set, scores with the referee, computes
fluency cost via held-out perplexity, and saves 5 EYEBALL before/after
sentence pairs for the Medium post. This module gives them the common
plumbing so the attack scripts stay short.

EYEBALL discipline: 3-5 representative sentence pairs per attack, saved
to reports/gauntlet/eyeball_<attack>.md as soon as they're available.
Numbers are the score; sentences are the post.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gauntlet.forms import CORE_FORMS, Form
from gauntlet.referee import detect, score_text

REPO = Path(__file__).resolve().parent.parent.parent

HF_NAME = "google/gemma-2-2b-it"
D3_PATH = REPO / "data" / "D3_fluency.txt"
EYEBALL_DIR = REPO / "reports" / "gauntlet"

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT.split(text.strip()) if len(s.strip()) > 30]


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class GauntletModel:
    """Wraps Gemma 2 2B-it with a tokenizer and generation utilities."""
    model: object
    tokenizer: object
    dev: str

    @classmethod
    def load(cls) -> "GauntletModel":
        dev = device()
        print(f"[runner] loading {HF_NAME} on {dev}…")
        tokenizer = AutoTokenizer.from_pretrained(HF_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(HF_NAME, dtype=torch.float16).to(dev)
        model.eval()
        return cls(model=model, tokenizer=tokenizer, dev=dev)

    def chat_input(self, user_prompt: str, system_prompt: str | None = None) -> dict:
        """Build the chat-template input. Gemma 2 has no separate system role —
        if a system instruction is supplied, prepend it to the user content."""
        if system_prompt:
            user_content = f"{system_prompt}\n\n{user_prompt}"
        else:
            user_content = user_prompt
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            add_generation_prompt=True, tokenize=False,
        )
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return {"input_ids": enc.input_ids.to(self.dev),
                "attention_mask": enc.attention_mask.to(self.dev)}

    def generate(self, user_prompt: str, *, seed: int, max_new_tokens: int = 200,
                 temperature: float = 0.8, top_p: float = 0.95,
                 system_prompt: str | None = None,
                 logits_processor_list=None,
                 logits_processor=None) -> str:
        """Generate from a prompt with deterministic seeding."""
        torch.manual_seed(seed)
        if self.dev == "mps":
            torch.mps.manual_seed(seed)
        enc = self.chat_input(user_prompt, system_prompt=system_prompt)
        input_len = enc["input_ids"].shape[1]
        kwargs = {}
        if logits_processor_list is not None:
            kwargs["logits_processor"] = logits_processor_list
        elif logits_processor is not None:
            from transformers import LogitsProcessorList
            kwargs["logits_processor"] = LogitsProcessorList([logits_processor])
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=temperature, top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )
        return self.tokenizer.decode(out[0, input_len:], skip_special_tokens=True)


# ─── scoring ─────────────────────────────────────────────────────────────


def score_generations(generations: list[dict]) -> dict:
    """Compute construction kill-rate stats on a list of {prompt, seed, generation}."""
    sent_rows = []
    for g in generations:
        for i, s in enumerate(split_sentences(g["generation"])):
            sc = score_text(s, strict=True)
            sent_rows.append({
                "prompt_idx": g.get("prompt_idx", -1),
                "seed": g.get("seed", -1),
                "sent_idx": i,
                "sentence": s,
                **{f.value: sc[f.value] for f in CORE_FORMS},
                "any_core": sc["any_core"],
                "any": sc["any"],
            })
    n = len(sent_rows) or 1
    out = {
        "n_generations": len(generations),
        "n_sentences": len(sent_rows),
        "any_core_rate": sum(r["any_core"] for r in sent_rows) / n,
        "any_rate": sum(r["any"] for r in sent_rows) / n,
    }
    for f in CORE_FORMS:
        out[f"{f.value}_rate"] = sum(r[f.value] for r in sent_rows) / n
        out[f"{f.value}_count"] = sum(1 for r in sent_rows if r[f.value])
    out["sentences"] = sent_rows
    return out


# ─── fluency: perplexity on D3 ────────────────────────────────────────────


@torch.no_grad()
def perplexity_on_d3(gm: GauntletModel, *, hook_factory: Callable | None = None,
                    n_chunks: int = 4, chunk_chars: int = 800) -> float:
    """Held-out perplexity on D3 paragraphs. `hook_factory` (optional) installs
    an intervention hook on the model; the hook is removed after measurement.
    Returns geometric-mean perplexity across chunks.
    """
    text = D3_PATH.read_text()
    chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 200][:n_chunks]
    chunks = [c[:chunk_chars] for c in chunks]

    handles = []
    if hook_factory is not None:
        handles = hook_factory(gm.model)
    try:
        log_ppls = []
        for c in chunks:
            enc = gm.tokenizer(c, return_tensors="pt").to(gm.dev)
            input_ids = enc.input_ids
            labels = input_ids.clone()
            out = gm.model(input_ids=input_ids, labels=labels)
            loss = float(out.loss.item())
            log_ppls.append(loss)
        return float(math.exp(sum(log_ppls) / len(log_ppls)))
    finally:
        for h in handles:
            h.remove()


# ─── EYEBALL — sentence-level before/after for the post ─────────────────


def write_eyeball(attack_id: str, attack_name: str, *,
                  baseline_pairs: list[dict], intervened_pairs: list[dict],
                  notes: str = "") -> None:
    """Write a small markdown file of 5 before/after sentence pairs per attack."""
    EYEBALL_DIR.mkdir(parents=True, exist_ok=True)
    out = EYEBALL_DIR / f"eyeball_{attack_id.lower()}.md"
    lines = [
        f"# EYEBALL — {attack_id} ({attack_name})",
        "",
        "Five sentence pairs from the same (prompt, seed) under baseline vs intervention. "
        "Numbers are in the gauntlet report; this is the prose-level beat for the post.",
        "",
    ]
    if notes:
        lines += [notes, ""]
    # pair them up by (prompt_idx, seed)
    by_key = {(p["prompt_idx"], p["seed"]): p for p in baseline_pairs}
    pairs_shown = 0
    for ip in intervened_pairs:
        key = (ip["prompt_idx"], ip["seed"])
        bp = by_key.get(key)
        if bp is None:
            continue
        lines.append(f"### prompt {ip['prompt_idx']} · seed {ip['seed']}")
        lines.append("")
        lines.append("**Baseline:**")
        lines.append(f"> {bp['generation'][:500]}")
        lines.append("")
        lines.append(f"**{attack_id}:**")
        lines.append(f"> {ip['generation'][:500]}")
        lines.append("")
        lines.append("---")
        lines.append("")
        pairs_shown += 1
        if pairs_shown >= 5:
            break
    out.write_text("\n".join(lines))
    print(f"[runner] wrote {pairs_shown} EYEBALL pairs to {out}")


# ─── reporting ───────────────────────────────────────────────────────────


def attack_report(attack_id: str, attack_name: str, *,
                  baseline_stats: dict, intervened_stats: dict,
                  baseline_ppl: float | None, intervened_ppl: float | None,
                  extra: dict | None = None) -> dict:
    """Compose the per-attack result dict for the gauntlet."""
    rep = {
        "attack_id": attack_id,
        "attack_name": attack_name,
        "baseline": {k: v for k, v in baseline_stats.items() if k != "sentences"},
        "intervened": {k: v for k, v in intervened_stats.items() if k != "sentences"},
        "deltas": {
            "any_core_drop": baseline_stats["any_core_rate"] - intervened_stats["any_core_rate"],
            "any_drop": baseline_stats["any_rate"] - intervened_stats["any_rate"],
            **{
                f"{f.value}_drop": baseline_stats[f"{f.value}_rate"] - intervened_stats[f"{f.value}_rate"]
                for f in CORE_FORMS
            },
        },
    }
    if baseline_ppl is not None and intervened_ppl is not None:
        rep["fluency"] = {
            "baseline_perplexity": baseline_ppl,
            "intervened_perplexity": intervened_ppl,
            "perplexity_ratio": intervened_ppl / baseline_ppl,
        }
    if extra:
        rep["extra"] = extra
    return rep


def save_report(attack_id: str, report: dict, generations_baseline: list[dict],
                 generations_intervened: list[dict]) -> None:
    EYEBALL_DIR.mkdir(parents=True, exist_ok=True)
    p = EYEBALL_DIR / f"{attack_id.lower()}_result.json"
    p.write_text(json.dumps({
        "report": report,
        "baseline_generations": generations_baseline,
        "intervened_generations": generations_intervened,
    }, indent=2))
    print(f"[runner] wrote {p}")


# ─── checkpointing — survive partial-run failure ──────────────────────────
#
# A4-A7 are long jobs (minutes to hours). They used to only persist results
# at the very end of the run. If they died at generation 179/180 we'd lose
# everything. Now each attack writes its current (baseline, intervened)
# generation lists to a checkpoint file after every prompt, and on startup
# loads any prior checkpoint to skip already-completed (prompt_idx, seed)
# pairs. Resume is implicit.


def checkpoint_path(attack_id: str, tag: str = "") -> Path:
    EYEBALL_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    return EYEBALL_DIR / f"{attack_id.lower()}_checkpoint{suffix}.json"


def save_checkpoint(attack_id: str, baseline_gens: list[dict],
                     intervened_gens: list[dict], tag: str = "",
                     extra: dict | None = None) -> None:
    p = checkpoint_path(attack_id, tag)
    payload = {
        "attack_id": attack_id,
        "tag": tag,
        "baseline_generations": baseline_gens,
        "intervened_generations": intervened_gens,
    }
    if extra is not None:
        payload["extra"] = extra
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(p)  # atomic on POSIX


def load_checkpoint(attack_id: str, tag: str = "") -> dict | None:
    p = checkpoint_path(attack_id, tag)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        print(f"[runner] warning: failed to load {p}: {e}")
        return None


def done_keys(gens: list[dict]) -> set[tuple[int, int]]:
    """Return set of (prompt_idx, seed) already produced in a generations list."""
    return {(g["prompt_idx"], g["seed"]) for g in gens}


def print_report(report: dict) -> None:
    print(f"\n=== {report['attack_id']} — {report['attack_name']} ===")
    b = report["baseline"]
    i = report["intervened"]
    print(f"  any_core rate:   baseline {b['any_core_rate']:.3%}  →  intervened {i['any_core_rate']:.3%}  "
          f"(drop {report['deltas']['any_core_drop']:+.3%})")
    for f in CORE_FORMS:
        print(f"  {f.value} rate:        baseline {b[f.value + '_rate']:.3%}  →  intervened {i[f.value + '_rate']:.3%}")
    if "fluency" in report:
        fl = report["fluency"]
        print(f"  fluency: PPL {fl['baseline_perplexity']:.2f} → {fl['intervened_perplexity']:.2f}  "
              f"(ratio {fl['perplexity_ratio']:.2f}×)")
