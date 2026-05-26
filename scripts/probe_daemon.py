"""Probe daemon — model + SAE + graph in one persistent process.

Why this exists: iterative ablation experiments are bottlenecked by the
~10–20 s cost of loading Gemma 2 2B + Gemma Scope SAE. The daemon loads
both once and answers many ablation queries via a JSON HTTP API. Use
either curl or `scripts/probe.py` as a thin client.

Endpoint:  POST http://localhost:8765/probe
Body:      {"cmd": "<name>", ...}
Response:  {"ok": bool, "result": ..., "error": str?}

Commands:
  ping                                              health check
  labels      {"features": [int]}                   Neuronpedia labels
  graph       {"query": "decoder_neighbors"|        resolve a structural prior
              "coact_partners"|"community",          to a list of feature ids
              "anchor": int, "k": int}
  graph_cypher{"query": str, "params": {...}}       raw Cypher passthrough
  attribution {"top_n": int,                        load top-N from
              "kind": "promote"|"suppress"}         pivot_attribution.json
  measure_pivot {"ablate": [int]|null,              M2 measurement: mean
                "variants": [str], "max_samples": int} P(pivot) under joint ablation
  ladder      {"conditions": {name: [int]},         batch many M2 conditions
              "n_random_per_size": int,             with random size-matched controls
              "variants": [str], "max_samples": int,
              "seed": int}
  generate    {"prompt": str, "ablate": [int]|null, generate from prompt with
              "max_new_tokens": int,                sustained ablation
              "temperature": float, "top_p": float,
              "seed": int}
  stop                                              graceful shutdown

Usage:
  uv run python scripts/probe_daemon.py            # foreground
  curl -s -X POST -H 'Content-Type: application/json' \\
       -d '{"cmd": "ping"}' http://localhost:8765/probe

The daemon is a single in-process model; concurrent requests are
serialized via a coarse lock. Don't try to fan it out, just send the
next request when the current one returns.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sae_lens import HookedSAETransformer, SAE

from classifier import detect_construction
from neograph.config import SAE as GEMMA_SAE
from neograph.cypher import NeographClient
from neograph.util import get_logger

log = get_logger("probe.daemon")

REPO_ROOT = Path(__file__).resolve().parent.parent
D1_PATH = REPO_ROOT / "data" / "D1_contrast_pairs.jsonl"
D2_PATH = REPO_ROOT / "data" / "D2_neutral_prompts.json"
ATTRIB_PATH = REPO_ROOT / "reports" / "pivot_attribution.json"
LABELS_PATH = REPO_ROOT / "data" / "labels_cache.json"
LABEL_EMB_PATH = REPO_ROOT / "reports" / "label_embeddings.npy"
LABEL_EMB_IDX_PATH = REPO_ROOT / "reports" / "label_embedding_idx.json"

HOST = "127.0.0.1"
PORT = 8765

PIVOT_STRINGS = {
    "C1": [", it", ", they", ", he", ", she", ", we", "—", "; it"],
    "C2": [", but", ", yet", ", also", " but ", " also "],
    "C3": ["—", ", it", " but ", ", but"],
}
PIVOT_RE = re.compile(
    r"[,;—–\-]\s*(?:it|that|this|he|she|we|they|these|those|there|but|also|yet)\b",
    re.IGNORECASE,
)


def truncate_to_pivot(text: str) -> str | None:
    m = PIVOT_RE.search(text)
    return text[: m.start()].rstrip() if m else None


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


class ProbeEngine:
    """Holds model + SAE + cached helpers; thread-safe via a coarse lock."""

    def __init__(self, load_it: bool = True) -> None:
        self._lock = threading.Lock()
        self._dev = device()
        log.info(f"loading gemma-2-2b on {self._dev}…")
        self.model = HookedSAETransformer.from_pretrained("gemma-2-2b", device=self._dev)
        self.model.eval()
        self.model_it = None
        if load_it:
            log.info(f"loading gemma-2-2b-it on {self._dev}…")
            self.model_it = HookedSAETransformer.from_pretrained(
                "gemma-2-2b-it", device=self._dev,
            )
            self.model_it.eval()
        log.info("loading Gemma Scope SAE (L20)")
        sae = SAE.from_pretrained(
            release=GEMMA_SAE.release, sae_id=GEMMA_SAE.sae_id, device=self._dev,
        )
        if isinstance(sae, tuple):
            sae = sae[0]
        sae.eval()
        self.sae = sae
        self.hook_name = f"{sae.cfg.metadata.hook_name}.hook_sae_acts_post"
        self.d_sae = sae.cfg.d_sae
        # Cache for per-layer SAEs loaded on demand (Q7).
        self._sae_by_layer: dict[int, object] = {20: sae}

        self.labels = self._load_labels()
        self._d1_samples: list[dict] | None = None
        self._d2_prompts: list[str] | None = None
        self._pivot_ids: dict[str, list[int]] | None = None
        log.info("ready.")

    def _select_model(self, which: str):
        if which == "it":
            if self.model_it is None:
                raise RuntimeError("gemma-2-2b-it not loaded (daemon started with --no-it)")
            return self.model_it
        if which == "base":
            return self.model
        raise ValueError(f"unknown model: {which} (use 'base' or 'it')")

    def _get_sae(self, layer: int):
        """Lazy-load a Gemma Scope canonical SAE at the requested layer."""
        if layer in self._sae_by_layer:
            return self._sae_by_layer[layer]
        log.info(f"loading Gemma Scope SAE at layer {layer}")
        sae = SAE.from_pretrained(
            release=GEMMA_SAE.release,
            sae_id=f"layer_{layer}/width_16k/canonical",
            device=self._dev,
        )
        if isinstance(sae, tuple):
            sae = sae[0]
        sae.eval()
        self._sae_by_layer[layer] = sae
        return sae

    def _ensure_d2(self) -> list[str]:
        if self._d2_prompts is not None:
            return self._d2_prompts
        self._d2_prompts = json.loads(D2_PATH.read_text())["prompts"]
        return self._d2_prompts

    @staticmethod
    def _load_labels() -> dict[int, str]:
        if not LABELS_PATH.exists():
            return {}
        raw = json.loads(LABELS_PATH.read_text())
        return {int(k): v.get("text", "") for k, v in raw.items()}

    def _ensure_d1(self) -> list[dict]:
        if self._d1_samples is not None:
            return self._d1_samples
        rows = []
        for line in D1_PATH.read_text().splitlines():
            if not line.strip() or line.startswith('{"_meta'):
                continue
            d = json.loads(line)
            pref = truncate_to_pivot(d["with"])
            if pref is None:
                continue
            rows.append({"variant": d["variant"], "prefix": pref})
        self._d1_samples = rows
        return rows

    def _ensure_pivot_ids(self) -> dict[str, list[int]]:
        if self._pivot_ids is not None:
            return self._pivot_ids
        out = {}
        for v, strs in PIVOT_STRINGS.items():
            ids = set()
            for s in strs:
                for t in self.model.to_tokens(s, prepend_bos=False)[0].tolist():
                    ids.add(int(t))
                for t in self.model.to_tokens(" " + s.strip(), prepend_bos=False)[0].tolist():
                    ids.add(int(t))
            out[v] = sorted(ids)
        self._pivot_ids = out
        return out

    # === Commands ===

    def cmd_ping(self, _: dict) -> dict:
        return {"device": self._dev, "d_sae": self.d_sae, "model": "gemma-2-2b"}

    def cmd_labels(self, args: dict) -> dict:
        feats = [int(f) for f in args.get("features", [])]
        return {str(f): self.labels.get(f, "") for f in feats}

    def cmd_attribution(self, args: dict) -> dict:
        """Return top-N attribution-ranked features.

        Prefers the `full_ranked_by_score` field (all features with signal,
        sorted) when present; falls back to the truncated `top_promotes_pivot`
        / `top_suppresses_pivot` lists for older attribution outputs.
        Supports `slice` (e.g. {"start": 25, "end": 50}) for ranked windows.
        """
        data = json.loads(ATTRIB_PATH.read_text())
        kind = args.get("kind", "promote")
        if "full_ranked_by_score" in data:
            full = list(data["full_ranked_by_score"])
            if kind == "suppress":
                full = sorted(full, key=lambda r: r["scored"])  # most-negative first
            else:
                full = sorted(full, key=lambda r: -r["scored"])  # most-positive first
        else:
            key = "top_promotes_pivot" if kind == "promote" else "top_suppresses_pivot"
            full = data[key]
        if "slice" in args:
            s = args["slice"]
            rows = full[int(s.get("start", 0)) : int(s["end"])]
        else:
            top = int(args.get("top_n", 10))
            rows = full[:top]
        return {
            "features": [int(r["feature_idx"]) for r in rows],
            "details": [
                {"index": int(r["feature_idx"]),
                 "mean_drop": float(r["mean_attribution_drop"]),
                 "n_active": int(r["n_prompts_active"]),
                 "score": float(r["scored"]),
                 "label": self.labels.get(int(r["feature_idx"]), r.get("label", ""))}
                for r in rows
            ],
            "n_with_signal": int(data.get("n_features_with_signal", len(full))),
        }

    def cmd_graph(self, args: dict) -> dict:
        q = args["query"]
        anchor = int(args.get("anchor", 3223))
        k = int(args.get("k", 10))
        with NeographClient() as c:
            if q == "decoder_neighbors":
                r = c.run(
                    """
                    MATCH (a:SAEFeature {index: $idx})-[r:DECODER_SIMILAR]-(b:SAEFeature)
                    WHERE a.sae_id CONTAINS 'L20/16k' AND b.sae_id CONTAINS 'L20/16k'
                    RETURN DISTINCT b.index AS idx, r.cosine AS score
                    ORDER BY score DESC LIMIT $k
                    """, idx=anchor, k=k,
                )
            elif q == "coact_partners":
                rank = args.get("rank_by", "jaccard")  # "jaccard" or "pmi"
                order_by = "r.jaccard" if rank == "jaccard" else "r.pmi"
                r = c.run(
                    f"""
                    MATCH (a:SAEFeature {{index: $idx}})-[r:CO_ACTIVATES_WITH]-(b:SAEFeature)
                    WHERE a.sae_id CONTAINS 'L20/16k' AND b.sae_id CONTAINS 'L20/16k'
                    RETURN DISTINCT b.index AS idx, {order_by} AS score
                    ORDER BY score DESC LIMIT $k
                    """, idx=anchor, k=k,
                )
            elif q == "community":
                cid = int(args.get("cid", 12))
                lim = int(args.get("limit", 50))
                r = c.run(
                    """
                    MATCH (f:SAEFeature) WHERE f.communityId = $cid
                          AND f.sae_id CONTAINS 'L20/16k'
                    RETURN f.index AS idx, f.activation_density AS score
                    ORDER BY score DESC LIMIT $lim
                    """, cid=cid, lim=lim,
                )
            else:
                raise ValueError(f"unknown query: {q}")
        feats = [int(row["idx"]) for row in r]
        scores = [float(row["score"]) for row in r]
        return {
            "features": feats,
            "scores": scores,
            "labels": {str(f): self.labels.get(f, "") for f in feats},
        }

    def cmd_graph_cypher(self, args: dict) -> dict:
        with NeographClient() as c:
            rows = c.run(args["query"], **(args.get("params") or {}))
        return {"rows": rows}

    def _measure_pivot(self, prefix: str, pivot_ids: list[int],
                       feat_indices: list[int] | None,
                       sae=None, use_sae: bool = True) -> float:
        """Return P(any pivot token | prefix) at next position.

        - If `use_sae` is False, runs the model with no SAE inserted (raw
          residual stream). `feat_indices` must be None in that case
          because feature ablation requires the SAE to decompose.
        - If `use_sae` is True (default), runs with `sae` inserted (defaults
          to layer-20 canonical) and optionally zero-ablates `feat_indices`
          at the last position of that SAE's post-encode.
        """
        tokens = self.model.to_tokens(prefix, prepend_bos=True)
        if not use_sae:
            if feat_indices:
                raise ValueError("can't ablate SAE features without inserting the SAE")
            with torch.no_grad():
                logits = self.model(tokens)
            probs = F.softmax(logits[0, -1, :].float().cpu(), dim=-1)
            return float(probs[pivot_ids].sum().item())

        sae = sae or self.sae
        hook_name = f"{sae.cfg.metadata.hook_name}.hook_sae_acts_post"
        if not feat_indices:
            with torch.no_grad():
                logits = self.model.run_with_hooks_with_saes(
                    tokens, saes=[sae], fwd_hooks=[]
                )
        else:
            idxs = torch.tensor(feat_indices, device=tokens.device, dtype=torch.long)

            def ablate(act, **kwargs):
                act = act.clone()
                act[..., -1, idxs] = 0.0
                return act

            with torch.no_grad():
                logits = self.model.run_with_hooks_with_saes(
                    tokens, saes=[sae], fwd_hooks=[(hook_name, ablate)],
                )
        probs = F.softmax(logits[0, -1, :].float().cpu(), dim=-1)
        return float(probs[pivot_ids].sum().item())

    def cmd_measure_pivot(self, args: dict) -> dict:
        """Measure P(pivot) under joint ablation.

        Optional args:
          use_sae: bool = True   when False, ablate must be null; measures
                                  P(pivot) on the raw model (no SAE inserted).
          sae_layer: int = 20    which Gemma Scope canonical SAE to use.
        """
        variants = set(args.get("variants", ["C1", "C2", "C3"]))
        max_samples = int(args.get("max_samples", 80))
        ablate = args.get("ablate")
        ablate = [int(x) for x in ablate] if ablate else None
        use_sae = bool(args.get("use_sae", True))
        layer = int(args.get("sae_layer", 20))
        sae = self._get_sae(layer) if use_sae else None

        samples = [s for s in self._ensure_d1() if s["variant"] in variants][:max_samples]
        pids = self._ensure_pivot_ids()

        per = []
        for s in samples:
            p_base = self._measure_pivot(s["prefix"], pids[s["variant"]], None,
                                          sae=sae, use_sae=use_sae)
            p_abl = self._measure_pivot(s["prefix"], pids[s["variant"]], ablate,
                                         sae=sae, use_sae=use_sae)
            per.append({"variant": s["variant"], "p_baseline": p_base,
                        "p_ablated": p_abl, "drop": p_base - p_abl})
        arr = np.array([r["drop"] for r in per])
        return {
            "n": len(per),
            "ablate": ablate or [],
            "use_sae": use_sae,
            "sae_layer": layer if use_sae else None,
            "baseline_mean": float(np.mean([r["p_baseline"] for r in per])),
            "ablated_mean": float(np.mean([r["p_ablated"] for r in per])),
            "mean_drop": float(arr.mean()),
            "median_drop": float(np.median(arr)),
            "rel_drop": float(arr.mean() / np.mean([r["p_baseline"] for r in per])),
            "per_sample": per,
        }

    def _measure_pivot_multi(self, prefix: str, pivot_ids: list[int],
                              ablate_by_layer: dict[int, list[int]] | None) -> float:
        """Measure P(pivot) ablating features at multiple SAEs simultaneously.

        `ablate_by_layer` maps layer int -> list of feature indices to ablate at
        that layer's SAE post-encode at the last position. Passing an empty
        dict / None gives the multi-SAE baseline (all SAEs inserted, no
        ablation) so we measure the joint reconstruction overhead too.
        """
        tokens = self.model.to_tokens(prefix, prepend_bos=True)
        layers = sorted(ablate_by_layer) if ablate_by_layer else []
        saes = [self._get_sae(L) for L in layers]
        hooks = []
        for L, sae in zip(layers, saes):
            idxs = torch.tensor(ablate_by_layer[L], device=tokens.device, dtype=torch.long)
            hook_name = f"{sae.cfg.metadata.hook_name}.hook_sae_acts_post"

            def make_hook(idxs_local):
                def ablate(act, **kwargs):
                    act = act.clone()
                    act[..., -1, idxs_local] = 0.0
                    return act
                return ablate

            hooks.append((hook_name, make_hook(idxs)))

        with torch.no_grad():
            logits = self.model.run_with_hooks_with_saes(
                tokens, saes=saes, fwd_hooks=hooks,
            )
        probs = F.softmax(logits[0, -1, :].float().cpu(), dim=-1)
        return float(probs[pivot_ids].sum().item())

    def cmd_measure_pivot_multi(self, args: dict) -> dict:
        """Multi-layer joint ablation. Required arg:
          ablate_by_layer: {"12": [...], "20": [...], "25": [...]}
        """
        raw = args["ablate_by_layer"]
        ablate_by_layer = {int(k): [int(x) for x in v] for k, v in raw.items()}
        variants = set(args.get("variants", ["C1", "C2", "C3"]))
        max_samples = int(args.get("max_samples", 80))

        samples = [s for s in self._ensure_d1() if s["variant"] in variants][:max_samples]
        pids = self._ensure_pivot_ids()

        # Multi-SAE baseline (all SAEs inserted, no ablation)
        baseline_layers = {L: [] for L in ablate_by_layer}

        per = []
        for s in samples:
            p_base = self._measure_pivot_multi(s["prefix"], pids[s["variant"]], baseline_layers)
            p_abl = self._measure_pivot_multi(s["prefix"], pids[s["variant"]], ablate_by_layer)
            per.append({"variant": s["variant"], "p_baseline": p_base,
                        "p_ablated": p_abl, "drop": p_base - p_abl})
        arr = np.array([r["drop"] for r in per])
        baseline_arr = np.array([r["p_baseline"] for r in per])
        return {
            "n": len(per),
            "ablate_by_layer": {str(L): v for L, v in ablate_by_layer.items()},
            "n_features_total": sum(len(v) for v in ablate_by_layer.values()),
            "multi_sae_baseline_mean": float(baseline_arr.mean()),
            "ablated_mean": float(np.mean([r["p_ablated"] for r in per])),
            "mean_drop": float(arr.mean()),
            "median_drop": float(np.median(arr)),
            "rel_drop": float(arr.mean() / baseline_arr.mean()) if baseline_arr.mean() > 0 else 0.0,
            "per_sample": per,
        }

    def cmd_ladder(self, args: dict) -> dict:
        """Run many named conditions in one pass, plus random size-matched controls.

        Optional `sae_layer` (default 20) lets you run the ladder against a
        different Gemma Scope canonical SAE, for Q7 layer-transfer probes.
        """
        named: dict[str, list[int]] = {
            k: [int(x) for x in v] for k, v in args["conditions"].items()
        }
        variants = set(args.get("variants", ["C1", "C2", "C3"]))
        max_samples = int(args.get("max_samples", 80))
        n_random = int(args.get("n_random_per_size", 3))
        seed = int(args.get("seed", 11))
        layer = int(args.get("sae_layer", 20))
        sae = self._get_sae(layer)

        sizes = sorted({len(v) for v in named.values()})
        rng = np.random.default_rng(seed)
        excluded = set().union(*[set(v) for v in named.values()])
        pool = np.array([i for i in range(sae.cfg.d_sae) if i not in excluded], dtype=np.int64)
        random_conditions = {}
        for k in sizes:
            for d in range(n_random):
                random_conditions[f"random_k{k}_d{d}"] = sorted(
                    int(x) for x in rng.choice(pool, size=k, replace=False)
                )

        all_conditions = {**named, **random_conditions}
        samples = [s for s in self._ensure_d1() if s["variant"] in variants][:max_samples]
        pids = self._ensure_pivot_ids()

        per_sample = []
        for s in samples:
            base = self._measure_pivot(s["prefix"], pids[s["variant"]], None, sae=sae)
            row = {"variant": s["variant"], "p_baseline": base, "cond_p": {}}
            for name, feats in all_conditions.items():
                row["cond_p"][name] = self._measure_pivot(
                    s["prefix"], pids[s["variant"]], feats, sae=sae,
                )
            per_sample.append(row)

        baseline_mean = float(np.mean([r["p_baseline"] for r in per_sample]))
        by_cond = {}
        for name, feats in all_conditions.items():
            drops = np.array([r["p_baseline"] - r["cond_p"][name] for r in per_sample])
            by_cond[name] = {
                "n_features": len(feats),
                "features": feats,
                "mean_p_pivot": float(np.mean([r["cond_p"][name] for r in per_sample])),
                "mean_drop": float(drops.mean()),
                "rel_drop": float(drops.mean() / baseline_mean) if baseline_mean > 0 else 0.0,
            }
        random_by_size = {}
        for k in sizes:
            draws = [by_cond[f"random_k{k}_d{d}"]["mean_drop"] for d in range(n_random)]
            random_by_size[k] = {
                "mean": float(np.mean(draws)),
                "max": float(np.max(draws)),
                "std": float(np.std(draws, ddof=1)) if len(draws) > 1 else 0.0,
            }
        return {
            "n_samples": len(samples),
            "sae_layer": layer,
            "baseline_mean_p_pivot": baseline_mean,
            "by_condition": by_cond,
            "random_by_size": random_by_size,
            "named_conditions": list(named.keys()),
        }

    def _generate_once(self, model, prompt: str, ablate: list[int] | None,
                        max_new: int, temperature: float, top_p: float, seed: int) -> str:
        torch.manual_seed(seed)
        if self._dev == "mps":
            torch.mps.manual_seed(seed)
        tokens = model.to_tokens(prompt, prepend_bos=True).to(next(model.parameters()).device)
        prompt_len = tokens.shape[1]
        hooks = []
        if ablate:
            idxs = torch.tensor(ablate, device=tokens.device, dtype=torch.long)

            def clamp(act, **kwargs):
                act = act.clone()
                act[..., idxs] = 0.0
                return act

            hooks.append((self.hook_name, clamp))

        for _ in range(max_new):
            with torch.no_grad():
                logits = model.run_with_hooks_with_saes(
                    tokens, saes=[self.sae], fwd_hooks=hooks,
                )
            last = logits[0, -1, :].float() / max(temperature, 1e-6)
            probs = F.softmax(last, dim=-1)
            sp, si = torch.sort(probs, descending=True)
            csum = torch.cumsum(sp, dim=-1)
            keep = csum < top_p
            keep[..., 0] = True
            filt = torch.zeros_like(probs)
            filt[si[keep]] = sp[keep]
            filt = filt / filt.sum()
            next_id = torch.multinomial(filt, num_samples=1)
            tokens = torch.cat([tokens, next_id.unsqueeze(0)], dim=1)
            if next_id.item() == model.tokenizer.eos_token_id:
                break

        return model.tokenizer.decode(tokens[0, prompt_len:], skip_special_tokens=True)

    def cmd_generate(self, args: dict) -> dict:
        which = args.get("model", "base")
        model = self._select_model(which)
        ablate = args.get("ablate")
        ablate = [int(x) for x in ablate] if ablate else None
        completion = self._generate_once(
            model,
            prompt=args["prompt"],
            ablate=ablate,
            max_new=int(args.get("max_new_tokens", 120)),
            temperature=float(args.get("temperature", 0.8)),
            top_p=float(args.get("top_p", 0.95)),
            seed=int(args.get("seed", 0)),
        )
        return {"completion": completion, "model": which,
                "n_ablated": len(ablate) if ablate else 0}

    def cmd_m1_eval(self, args: dict) -> dict:
        """Generate from D2 prompts under sustained joint ablation, score with
        M1 classifier. Returns construction rate baseline-vs-ablated."""
        which = args.get("model", "it")
        model = self._select_model(which)
        ablate = args.get("ablate")
        ablate = [int(x) for x in ablate] if ablate else None
        n_prompts = int(args.get("n_prompts", 12))
        seeds = list(range(int(args.get("seeds", 3))))
        max_new = int(args.get("max_new_tokens", 120))
        temperature = float(args.get("temperature", 0.8))
        top_p = float(args.get("top_p", 0.95))
        strict = bool(args.get("strict", False))

        d2 = self._ensure_d2()[:n_prompts]
        rows = []
        for prompt in d2:
            for seed in seeds:
                base_text = self._generate_once(
                    model, prompt, None, max_new, temperature, top_p, seed,
                )
                abl_text = self._generate_once(
                    model, prompt, ablate, max_new, temperature, top_p, seed,
                )
                base_hits = detect_construction(base_text, strict=strict)
                abl_hits = detect_construction(abl_text, strict=strict)
                rows.append({
                    "prompt": prompt, "seed": seed,
                    "baseline": base_text, "ablated": abl_text,
                    "baseline_variants": sorted({h.variant.value for h in base_hits}),
                    "ablated_variants": sorted({h.variant.value for h in abl_hits}),
                    "baseline_any_core": any(h.variant.value in ("C1", "C2", "C3") for h in base_hits),
                    "ablated_any_core": any(h.variant.value in ("C1", "C2", "C3") for h in abl_hits),
                })
        n = len(rows)
        base_rate = sum(1 for r in rows if r["baseline_any_core"]) / n
        abl_rate = sum(1 for r in rows if r["ablated_any_core"]) / n
        return {
            "n_pairs": n,
            "model": which,
            "ablate": ablate or [],
            "baseline_construction_rate": base_rate,
            "ablated_construction_rate": abl_rate,
            "absolute_drop": base_rate - abl_rate,
            "relative_drop": (base_rate - abl_rate) / max(base_rate, 1e-9),
            "examples": rows[:6],  # first 6 pairs for eyeballing
            "all_rows": rows,
        }

    def cmd_generate_with_activations(self, args: dict) -> dict:
        """Generate a completion AND capture top-K SAE features active at each token.

        Returns a list of per-token records:
          [{token_idx, token_str, top_features: [{idx, activation}, ...]}]

        The frontend uses these records to animate per-token feature firing
        in the UMAP cloud. Capture happens AT each generation step on the
        last-position SAE acts.

        Args:
          prompt: str
          model: "base" | "it" (default "it")
          ablate: optional list[int] sustained ablation
          max_new_tokens: int (default 60)
          temperature, top_p, seed: usual
          top_k_features: how many features per token to record (default 20)
        """
        which = args.get("model", "it")
        model = self._select_model(which)
        ablate = args.get("ablate")
        ablate = [int(x) for x in ablate] if ablate else None
        max_new = int(args.get("max_new_tokens", 60))
        temperature = float(args.get("temperature", 0.8))
        top_p = float(args.get("top_p", 0.95))
        seed = int(args.get("seed", 0))
        top_k = int(args.get("top_k_features", 20))

        torch.manual_seed(seed)
        if self._dev == "mps":
            torch.mps.manual_seed(seed)

        tokens = model.to_tokens(args["prompt"], prepend_bos=True).to(
            next(model.parameters()).device
        )
        prompt_len = tokens.shape[1]
        records: list[dict] = []

        for step in range(max_new):
            hooks = []
            captured: dict = {}
            if ablate:
                idxs = torch.tensor(ablate, device=tokens.device, dtype=torch.long)

                def clamp(act, **kwargs):
                    act = act.clone()
                    act[..., idxs] = 0.0
                    return act

                hooks.append((self.hook_name, clamp))

            # Capture pre-clamp activations at last position. Capture happens
            # on the same hook (acts_post) but as a separate observer hook.
            def capture(act, **kwargs):
                # act shape: (batch, seq, d_sae) at this site
                last = act[0, -1, :].float().cpu()
                vals, idx = torch.topk(last, top_k)
                captured["top_features"] = [
                    {"idx": int(i), "act": float(v)} for v, i in zip(vals.tolist(), idx.tolist())
                ]

            hooks.append((self.hook_name, capture))

            with torch.no_grad():
                logits = model.run_with_hooks_with_saes(
                    tokens, saes=[self.sae], fwd_hooks=hooks,
                )

            last_logits = logits[0, -1, :].float() / max(temperature, 1e-6)
            probs = F.softmax(last_logits, dim=-1)
            sp, si = torch.sort(probs, descending=True)
            csum = torch.cumsum(sp, dim=-1)
            keep = csum < top_p
            keep[..., 0] = True
            filt = torch.zeros_like(probs)
            filt[si[keep]] = sp[keep]
            filt = filt / filt.sum()
            next_id = torch.multinomial(filt, num_samples=1)

            token_str = model.tokenizer.decode([int(next_id.item())], skip_special_tokens=True)
            records.append({
                "step": step,
                "token_id": int(next_id.item()),
                "token_str": token_str,
                "top_features": captured.get("top_features", []),
            })

            tokens = torch.cat([tokens, next_id.unsqueeze(0)], dim=1)
            if next_id.item() == model.tokenizer.eos_token_id:
                break

        completion = model.tokenizer.decode(tokens[0, prompt_len:], skip_special_tokens=True)
        return {
            "completion": completion,
            "model": which,
            "n_ablated": len(ablate) if ablate else 0,
            "prompt": args["prompt"],
            "records": records,
            "n_tokens": len(records),
        }

    def cmd_m1_d1_continuation(self, args: dict) -> dict:
        """Higher-power M1 test: take D1 truncated 'with' prefixes (positioned
        to commit to the construction), let the IT model continue with
        sustained joint ablation, score the continuation with the classifier.

        Higher baseline construction rate than D2 → more statistical power
        per generation.
        """
        which = args.get("model", "it")
        model = self._select_model(which)
        ablate = args.get("ablate")
        ablate = [int(x) for x in ablate] if ablate else None
        n_prefixes = int(args.get("n_prefixes", 40))
        seeds = list(range(int(args.get("seeds", 2))))
        max_new = int(args.get("max_new_tokens", 60))
        temperature = float(args.get("temperature", 0.8))
        top_p = float(args.get("top_p", 0.95))
        variants = set(args.get("variants", ["C1", "C2", "C3"]))

        samples = [s for s in self._ensure_d1() if s["variant"] in variants][:n_prefixes]
        rows = []
        for s in samples:
            for seed in seeds:
                base_text = self._generate_once(
                    model, s["prefix"], None, max_new, temperature, top_p, seed,
                )
                abl_text = self._generate_once(
                    model, s["prefix"], ablate, max_new, temperature, top_p, seed,
                )
                # Score the continuation only (what came after the prefix).
                base_hits = detect_construction(base_text, strict=False)
                abl_hits = detect_construction(abl_text, strict=False)
                rows.append({
                    "prefix": s["prefix"], "variant": s["variant"], "seed": seed,
                    "baseline": base_text, "ablated": abl_text,
                    "baseline_variants": sorted({h.variant.value for h in base_hits}),
                    "ablated_variants": sorted({h.variant.value for h in abl_hits}),
                    "baseline_any_core": any(h.variant.value in ("C1", "C2", "C3") for h in base_hits),
                    "ablated_any_core": any(h.variant.value in ("C1", "C2", "C3") for h in abl_hits),
                })
        n = len(rows)
        base_rate = sum(1 for r in rows if r["baseline_any_core"]) / n
        abl_rate = sum(1 for r in rows if r["ablated_any_core"]) / n
        return {
            "n_pairs": n,
            "model": which,
            "ablate": ablate or [],
            "baseline_continuation_rate": base_rate,
            "ablated_continuation_rate": abl_rate,
            "absolute_drop": base_rate - abl_rate,
            "relative_drop": (base_rate - abl_rate) / max(base_rate, 1e-9),
            "examples": rows[:8],
            "all_rows": rows,
        }

    # ── Concept retrieval (RAG-for-activations) ────────────────────────────
    # Given a prompt, find the SAE features whose auto-interp labels are most
    # semantically similar. Returns a feature set ready to be silenced or
    # promoted. The novel pattern: instead of retrieving documents to put
    # in the context window, retrieve concept-features to clamp on the
    # residual stream directly.
    _label_embs = None        # (16384, 384) float32
    _label_feature_idx = None # list[int] aligned with rows of _label_embs
    _label_text = None        # list[str] aligned
    _embedder = None

    def _ensure_concept_index(self):
        if self._label_embs is not None:
            return
        if not LABEL_EMB_PATH.exists():
            raise RuntimeError(
                f"label embeddings not built yet — run "
                f"`scripts/build_label_embeddings.py` first "
                f"(missing {LABEL_EMB_PATH})"
            )
        import numpy as np
        log.info("loading label embedding index…")
        embs = np.load(LABEL_EMB_PATH)
        idx = json.loads(LABEL_EMB_IDX_PATH.read_text())
        ProbeEngine._label_embs = embs
        ProbeEngine._label_feature_idx = idx["feature_indices"]
        ProbeEngine._label_text = idx["labels"]
        log.info(f"  loaded {embs.shape[0]} label embeddings, dim={embs.shape[1]}")

    def _ensure_embedder(self):
        if self._embedder is not None:
            return self._embedder
        from sentence_transformers import SentenceTransformer
        from neograph.config import EMBEDDING_MODEL
        log.info(f"loading text embedder ({EMBEDDING_MODEL}) on CPU…")
        ProbeEngine._embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        return self._embedder

    def cmd_concept_retrieve(self, args: dict) -> dict:
        """Retrieve SAE features whose labels are semantically similar to the prompt.

        Args:
          prompt: str — the user prompt to retrieve concepts for
          k: int = 25 — how many top-matched features to return
          expand_neighbours: int = 0 — if > 0, for each top match, add its top-N
            decoder-cosine neighbours from Neo4j (graph traversal expansion)
          include_terms: str | null — optional extra string to add to prompt
            for matching (e.g. "negation" to bias toward those concepts)

        Returns:
          features: list[int]   — feature indices to ablate / promote
          matches:  list[dict]  — each with idx, label, score
          expansion_added: int  — how many features came from graph expansion
        """
        import numpy as np
        self._ensure_concept_index()
        embedder = self._ensure_embedder()
        prompt = args.get("prompt", "").strip()
        if not prompt:
            return {"features": [], "matches": [], "expansion_added": 0,
                    "note": "empty prompt"}
        if args.get("include_terms"):
            prompt = f"{prompt}\n{args['include_terms']}"
        k = int(args.get("k", 25))
        expand = int(args.get("expand_neighbours", 0))

        # Embed prompt → cosine similarity to all 16k label embeddings
        q = embedder.encode([prompt], normalize_embeddings=True)[0]
        sims = self._label_embs @ q   # (16384,) — already L2-normalised
        top_rows = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        # Sort the top-K by score
        top_rows = top_rows[np.argsort(-sims[top_rows])]

        matches = []
        for row_idx in top_rows:
            feat_idx = self._label_feature_idx[row_idx]
            matches.append({
                "idx": int(feat_idx),
                "label": self._label_text[row_idx],
                "score": float(sims[row_idx]),
            })
        seen = {m["idx"] for m in matches}
        expansion_added = 0

        # Optional: expand each match via graph traversal (decoder-cosine neighbours)
        if expand > 0 and matches:
            with NeographClient() as c:
                for m in list(matches):  # iterate snapshot
                    r = c.run(
                        """
                        MATCH (a:SAEFeature {index: $idx})-[r:DECODER_SIMILAR]-(b:SAEFeature)
                        WHERE a.sae_id CONTAINS 'L20/16k' AND b.sae_id CONTAINS 'L20/16k'
                        RETURN DISTINCT b.index AS idx, r.cosine AS cos
                        ORDER BY cos DESC LIMIT $n
                        """,
                        idx=m["idx"], n=expand,
                    )
                    for row in r:
                        nidx = int(row["idx"])
                        if nidx not in seen:
                            seen.add(nidx)
                            expansion_added += 1
        return {
            "features": list(seen),
            "matches": matches,
            "expansion_added": expansion_added,
            "n_total": len(seen),
        }

    def cmd_compose_behaviours(self, args: dict) -> dict:
        """Demo 2: combine multiple :Behaviour coalitions with per-behaviour
        intensities. Returns a weighted feature set you can pass to the
        ablate-and-generate flow.

        Args:
          intensities: {behaviour_name: float in [-100, 100]}
            Positive = silence; negative = (would-be) promote (not wired here).
            Magnitude = how aggressively to draw from that behaviour.

        Returns:
          features:        list[int]  — the silenced set (the union for intensity > 0)
          per_behaviour:   list[dict] — what each behaviour contributed
          cypher:          str         — the actual Cypher used
        """
        raw = args.get("intensities", {})
        intensities = {k: float(v) for k, v in raw.items()}
        # Silence (positive) — promote-clamp would need separate clamp_up wiring
        silence = {k: v for k, v in intensities.items() if v > 0}
        if not silence:
            return {"features": [], "per_behaviour": [], "cypher": "// no behaviours selected"}

        # Per behaviour, take top-N features by INCLUDES weight where N scales
        # with intensity (intensity 100 → all features in the coalition,
        # intensity 50 → top half, etc.).
        per_behaviour = []
        union_features = set()
        cypher_parts = []
        with NeographClient() as c:
            for name, intensity in silence.items():
                # Behaviour coalitions are 25 features; pick by intensity %
                frac = max(0.0, min(intensity / 100.0, 1.0))
                rows = c.run(
                    """
                    MATCH (b:Behaviour {name: $name})-[r:INCLUDES]->(f:SAEFeature)
                      WHERE f.sae_id CONTAINS 'L20/16k'
                    RETURN f.index AS idx, r.weight AS weight, r.rank AS rank
                    ORDER BY r.rank
                    """,
                    name=name,
                )
                full = [(int(r["idx"]), float(r["weight"]), int(r["rank"])) for r in rows]
                take = max(1, int(round(frac * len(full))))
                contributed = [f[0] for f in full[:take]]
                union_features.update(contributed)
                per_behaviour.append({
                    "name": name, "intensity": intensity, "took": take,
                    "of": len(full), "features": contributed,
                })
                cypher_parts.append(
                    f"MATCH (b:Behaviour {{name: '{name}'}})-[r:INCLUDES]->(f:SAEFeature)\n"
                    f"  WITH f, r.rank AS rank ORDER BY rank LIMIT {take}\n"
                    f"  RETURN f.index"
                )

        composed_cypher = (
            "// Compose " + ", ".join(silence.keys()) + " at the given intensities:\n"
            + " UNION ".join(cypher_parts)
        )
        return {
            "features": sorted(union_features),
            "per_behaviour": per_behaviour,
            "cypher": composed_cypher,
            "n_total": len(union_features),
        }

    def cmd_write_audit(self, args: dict) -> dict:
        """Demo 3: persist an intervention into Neo4j as a queryable subgraph.

        Body:
          session_id:    str
          prompt:        str
          baseline_text: str
          ablated_text:  str
          n_silenced:    int
          sources: [ {kind: str, label: str, features: [int]} ]

        Writes:
          (:Session {id})-[:RAN]->(:Intervention {id, prompt, ts, ...})
          (:Intervention)-[:USED_SOURCE]->(:Source {kind, label})
          (:Source)-[:SELECTED]->(:SAEFeature)
          (:Intervention)-[:SILENCED]->(:SAEFeature)

        Returns intervention_id for later read-back via the audit endpoint.
        """
        import uuid
        intervention_id = "i_" + uuid.uuid4().hex[:10]
        with NeographClient() as c:
            c.run(
                """
                MERGE (s:Session {id: $sid})
                  ON CREATE SET s.started_at = datetime()
                CREATE (i:Intervention {
                  id: $iid,
                  prompt: $prompt,
                  ts: datetime(),
                  n_silenced: $n_silenced,
                  baseline_text: $baseline_text,
                  ablated_text: $ablated_text
                })
                CREATE (s)-[:RAN]->(i)
                """,
                sid=args["session_id"],
                iid=intervention_id,
                prompt=args.get("prompt", ""),
                n_silenced=int(args.get("n_silenced", 0)),
                baseline_text=args.get("baseline_text", ""),
                ablated_text=args.get("ablated_text", ""),
            )
            all_features = set()
            for src in args.get("sources", []):
                feats = [int(f) for f in src.get("features", [])]
                all_features.update(feats)
                c.run(
                    """
                    MATCH (i:Intervention {id: $iid})
                    CREATE (so:Source {kind: $kind, label: $label, intervention_id: $iid})
                    CREATE (i)-[:USED_SOURCE]->(so)
                    WITH so
                    UNWIND $features AS feat_idx
                      MATCH (f:SAEFeature {index: feat_idx})
                        WHERE f.sae_id CONTAINS 'L20/16k'
                      MERGE (so)-[:SELECTED]->(f)
                    """,
                    iid=intervention_id,
                    kind=src["kind"],
                    label=src["label"],
                    features=feats,
                )
            # SILENCED edges (the union — the actually-applied set)
            c.run(
                """
                MATCH (i:Intervention {id: $iid})
                UNWIND $features AS feat_idx
                  MATCH (f:SAEFeature {index: feat_idx})
                    WHERE f.sae_id CONTAINS 'L20/16k'
                  MERGE (i)-[:SILENCED]->(f)
                """,
                iid=intervention_id, features=sorted(all_features),
            )
        return {
            "intervention_id": intervention_id,
            "session_id": args["session_id"],
            "n_sources": len(args.get("sources", [])),
            "n_silenced": len(all_features),
        }

    def cmd_read_audit(self, args: dict) -> dict:
        """Read back an intervention's lineage by id."""
        iid = args["intervention_id"]
        with NeographClient() as c:
            rows = c.run(
                """
                MATCH (i:Intervention {id: $iid})
                OPTIONAL MATCH (i)-[:USED_SOURCE]->(so:Source)-[:SELECTED]->(f:SAEFeature)
                OPTIONAL MATCH (f)-[:LABELED_AS {primary: true}]->(l:AutoInterpLabel)
                RETURN i.id AS id, i.prompt AS prompt, i.ts AS ts,
                       so.kind AS source_kind, so.label AS source_label,
                       f.index AS feature, l.text AS feature_label
                """,
                iid=iid,
            )
        sources = {}
        prompt = ts = None
        for r in rows:
            prompt = prompt or r["prompt"]
            ts = ts or str(r["ts"])
            if not r["source_kind"]:
                continue
            key = r["source_kind"] + "::" + (r["source_label"] or "")
            if key not in sources:
                sources[key] = {"kind": r["source_kind"], "label": r["source_label"],
                                 "features": []}
            sources[key]["features"].append({
                "idx": r["feature"], "label": r["feature_label"]
            })
        return {"intervention_id": iid, "prompt": prompt, "ts": ts,
                 "sources": list(sources.values())}

    def cmd_behaviours_list(self, args: dict) -> dict:
        """List named :Behaviour coalitions stored in the graph."""
        with NeographClient() as c:
            rows = c.run(
                """
                MATCH (b:Behaviour)
                OPTIONAL MATCH (b)-[r:INCLUDES]->(f:SAEFeature)
                WITH b, count(r) AS n_features, collect(f.index)[..30] AS sample
                RETURN b.name AS name, b.description AS description,
                       b.anchor_feature AS anchor, b.coalition_size AS size,
                       sample
                ORDER BY b.name
                """
            )
        return {"behaviours": [dict(r) for r in rows]}

    def cmd_surgical_deslop(self, args: dict) -> dict:
        """Demo 1 primitive: silence ONLY the features that are BOTH (a)
        semantically retrieved from the prompt AND (b) members of a named
        Behaviour coalition.

        Args:
          prompt: str
          behaviour: str — default "ai-ism"
          k: int — top-K concept matches to consider (default 50)

        Returns:
          intersection_features: list[int]  — the surgical set (what to silence)
          retrieved_features:    list[int]  — what concept-retrieval returned
          behaviour_features:    list[int]  — the full coalition
          matches: list[dict]               — labels + sims of retrieved that intersect
          cypher: str                        — the actual Cypher used (for the UI's "show query")
        """
        import numpy as np
        self._ensure_concept_index()
        embedder = self._ensure_embedder()
        prompt = args.get("prompt", "").strip()
        behaviour = args.get("behaviour", "ai-ism")
        k = int(args.get("k", 50))
        if not prompt:
            return {"error": "empty prompt"}

        # Retrieve concepts (top-K) ---------------------------------------------
        q = embedder.encode([prompt], normalize_embeddings=True)[0]
        sims = self._label_embs @ q
        top_rows = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        top_rows = top_rows[np.argsort(-sims[top_rows])]
        retrieved = [(int(self._label_feature_idx[r]),
                       float(sims[r]),
                       self._label_text[r]) for r in top_rows]
        retrieved_idx = [r[0] for r in retrieved]

        # Intersect with the behaviour coalition via Cypher ---------------------
        cypher = """
            MATCH (b:Behaviour {name: $name})-[r:INCLUDES]->(f:SAEFeature)
              WHERE f.index IN $retrieved AND f.sae_id CONTAINS 'L20/16k'
            RETURN f.index AS idx, r.weight AS weight, r.rank AS rank
            ORDER BY r.rank
        """
        full_cypher = """
            MATCH (b:Behaviour {name: $name})-[:INCLUDES]->(f:SAEFeature)
            RETURN f.index AS idx
        """
        with NeographClient() as c:
            inter_rows = c.run(cypher, name=behaviour, retrieved=retrieved_idx)
            full_rows = c.run(full_cypher, name=behaviour)
        intersection = [int(r["idx"]) for r in inter_rows]
        behaviour_features = [int(r["idx"]) for r in full_rows]

        retrieved_lookup = {idx: (sim, lbl) for idx, sim, lbl in retrieved}
        matches = []
        for r in inter_rows:
            idx = int(r["idx"])
            sim, lbl = retrieved_lookup.get(idx, (0.0, ""))
            matches.append({
                "idx": idx, "score": sim, "label": lbl,
                "behaviour_rank": int(r["rank"]),
                "behaviour_weight": float(r["weight"]),
            })

        # Ranked fallback: for EVERY coalition feature, compute its similarity
        # to the prompt. This lets the UI offer a "silence the top-N most
        # relevant coalition features" affordance when strict intersection
        # returns 0 (which happens when the prompt's concepts don't surface
        # the coalition members in top-K retrieval). Demo-day visceral guarantee.
        feat_to_row = {int(self._label_feature_idx[i]): i
                       for i in range(len(self._label_feature_idx))}
        ranked = []
        for fidx in behaviour_features:
            row = feat_to_row.get(fidx)
            if row is None:
                continue
            ranked.append({
                "idx": fidx,
                "score": float(sims[row]),
                "label": self._label_text[row],
            })
        ranked.sort(key=lambda r: -r["score"])

        return {
            "behaviour": behaviour,
            "k_retrieved": k,
            "retrieved_features": retrieved_idx,
            "behaviour_features": behaviour_features,
            "intersection_features": intersection,
            "matches": matches,
            "ranked_coalition": ranked,  # all coalition feats sorted by sim
            "cypher": cypher.strip(),
            "n_retrieved": len(retrieved_idx),
            "n_behaviour": len(behaviour_features),
            "n_intersection": len(intersection),
        }

    def cmd_minimal_sufficient(self, args: dict) -> dict:
        """Greedy backwards elimination from a seed feature set.

        Starting from `features`, iteratively remove the feature whose
        omission costs the least (smallest reduction in mean drop). Stop
        when removing any feature would lose more than `tolerance` of the
        full-set drop, or when `min_size` is reached.

        Returns the elimination trajectory and the final minimal set.
        """
        feats = sorted(set(int(x) for x in args["features"]))
        tolerance = float(args.get("tolerance", 0.10))  # 10% loss allowed
        min_size = int(args.get("min_size", 1))
        variants = set(args.get("variants", ["C1", "C2", "C3"]))
        max_samples = int(args.get("max_samples", 80))
        layer = int(args.get("sae_layer", 20))
        sae = self._get_sae(layer)

        samples = [s for s in self._ensure_d1() if s["variant"] in variants][:max_samples]
        pids = self._ensure_pivot_ids()

        # Baselines once
        baseline_ps = [
            self._measure_pivot(s["prefix"], pids[s["variant"]], None, sae=sae)
            for s in samples
        ]
        baseline_mean = float(np.mean(baseline_ps))

        def mean_drop(feature_set: list[int]) -> float:
            if not feature_set:
                return 0.0
            ablated = [
                self._measure_pivot(s["prefix"], pids[s["variant"]],
                                     feature_set, sae=sae)
                for s in samples
            ]
            return float(np.mean(baseline_ps) - np.mean(ablated))

        # Full-set drop = anchor we don't want to fall too far below
        full_drop = mean_drop(feats)
        target = full_drop * (1.0 - tolerance)

        trajectory = [{
            "size": len(feats),
            "features": list(feats),
            "mean_drop": full_drop,
            "rel_drop": full_drop / baseline_mean if baseline_mean else 0.0,
        }]
        current = list(feats)
        while len(current) > min_size:
            # Try removing each feature; keep the removal whose drop is largest
            best_remove = None
            best_remaining_drop = -1.0
            for f in current:
                test_set = [x for x in current if x != f]
                d = mean_drop(test_set)
                if d > best_remaining_drop:
                    best_remaining_drop = d
                    best_remove = f
            if best_remaining_drop < target:
                break  # would drop below tolerance threshold
            current = [x for x in current if x != best_remove]
            trajectory.append({
                "size": len(current),
                "removed": best_remove,
                "features": list(current),
                "mean_drop": best_remaining_drop,
                "rel_drop": best_remaining_drop / baseline_mean if baseline_mean else 0.0,
            })

        return {
            "baseline_mean_p_pivot": baseline_mean,
            "full_set_drop": full_drop,
            "tolerance": tolerance,
            "target_floor_drop": target,
            "final_set_size": len(current),
            "final_features": current,
            "trajectory": trajectory,
        }

    # === Dispatch ===

    def dispatch(self, body: dict) -> dict:
        cmd = body.get("cmd")
        handler = getattr(self, f"cmd_{cmd}", None)
        if handler is None:
            return {"ok": False, "error": f"unknown cmd: {cmd}"}
        try:
            with self._lock:
                t0 = time.perf_counter()
                result = handler(body)
                elapsed = time.perf_counter() - t0
            return {"ok": True, "result": result, "elapsed_s": round(elapsed, 3)}
        except Exception as e:
            log.exception(f"cmd {cmd} failed")
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# === HTTP plumbing ===

_engine: ProbeEngine | None = None
_should_stop = threading.Event()


DEMO_DIR = REPO_ROOT / "web" / "demo"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # silence default access log

    def do_OPTIONS(self):
        # CORS preflight
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        # Serve the demo from /demo, /demo/layout.json, etc.
        if self.path == "/" or self.path == "/demo":
            return self._serve_file(DEMO_DIR / "index.html", "text/html")
        if self.path.startswith("/demo/"):
            rel = self.path[len("/demo/"):].split("?")[0]
            path = DEMO_DIR / rel
            # Constrain to demo dir
            try:
                path.resolve().relative_to(DEMO_DIR.resolve())
            except ValueError:
                self.send_error(403); return
            mime = "application/json" if path.suffix == ".json" \
                   else "text/css" if path.suffix == ".css" \
                   else "application/javascript" if path.suffix == ".js" \
                   else "text/html"
            return self._serve_file(path, mime)
        self.send_error(404)

    def _serve_file(self, path, mime: str):
        if not path.exists():
            self.send_error(404); return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != "/probe":
            self.send_error(404); return
        n = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(n).decode())
        except Exception as e:
            self._send_json({"ok": False, "error": f"bad json: {e}"}, 400); return
        if body.get("cmd") == "stop":
            self._send_json({"ok": True, "result": "stopping"}, 200)
            _should_stop.set()
            return
        resp = _engine.dispatch(body)
        self._send_json(resp, 200 if resp.get("ok") else 500)

    def _send_json(self, obj: dict, status: int):
        payload = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)


def main():
    global _engine
    _engine = ProbeEngine()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    log.info(f"probe daemon listening on http://{HOST}:{PORT}/probe")
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        while not _should_stop.wait(timeout=0.5):
            pass
    except KeyboardInterrupt:
        log.info("interrupt — shutting down")
    server.shutdown()
    log.info("bye")


if __name__ == "__main__":
    main()
