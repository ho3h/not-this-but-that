"""SAE fidelity at layer 20, done properly (revisits the Tier 0b kill).

Tier 0b (reports/tier_0b_kill.md) measured variance explained of −1.33 for the
Gemma Scope layer-20 SAE and was never resolved. Gemma Scope was not trained
on the BOS token, whose residual norm dwarfs every other position, so a VE
computed over all positions is dominated by one token the SAE never saw.

This script reports, for the base model on plain text and the -it model on
chat-formatted text, with and without the BOS position:
  - fraction of variance explained (VE), pooled over tokens
  - mean cosine(h, h_hat) and mean L0 (active features per token)
  - next-token cross-entropy: clean vs SAE spliced in (v1 behaviour) vs
    zero-ablating the whole residual, giving the standard "loss recovered"

Writes reports/sae_fidelity.{json,md}.
"""

from __future__ import annotations

import json
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import v2_engine as E  # noqa: E402

REPO = E.REPO


def texts_base() -> list[str]:
    prose = (REPO / "data" / "D3_fluency.txt").read_text()
    paras = [p.strip() for p in prose.split("\n\n") if len(p.strip()) > 200]
    rows = json.loads((REPO / "reports" / "q5c_d2_high_power.json").read_text())["rows"]
    gens = [r["prompt"] + "\n\n" + r["baseline"] for r in rows if r["seed"] == 0][:60]
    return paras + gens


def chats_it(tok) -> list[torch.Tensor]:
    rows = json.loads((REPO / "reports" / "q5c_d2_high_power.json").read_text())["rows"]
    out = []
    for r in [r for r in rows if r["seed"] == 0][:60]:
        msgs = [{"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": r["baseline"].strip()}]
        ids = tok.apply_chat_template(msgs, return_tensors="pt")
        out.append(ids if isinstance(ids, torch.Tensor) else ids["input_ids"])
    return out


class Capture:
    def __init__(self):
        self.h = None

    def __call__(self, module, inputs, output):
        self.h = (output[0] if isinstance(output, tuple) else output).detach()


class Replace:
    """Replace the layer-20 residual (except BOS unless include_bos) with f(h)."""

    def __init__(self, fn, include_bos: bool):
        self.fn, self.include_bos = fn, include_bos

    def __call__(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        start = 0 if self.include_bos else 1
        new = h.clone()
        new[:, start:] = self.fn(h[:, start:].float()).to(h.dtype)
        return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new


def ce(model, ids: torch.Tensor) -> tuple[float, int]:
    with torch.no_grad():
        logits = model(ids).logits[0, :-1].float()
    tgt = ids[0, 1:]
    loss = F.cross_entropy(logits, tgt, reduction="sum")
    return float(loss), int(tgt.numel())


def run(model, sae, seqs: list[torch.Tensor]) -> dict:
    layer = model.model.layers[E.LAYER]
    stats = {}
    for tag, start in (("incl_bos", 0), ("excl_bos", 1)):
        stats[tag] = {"sse": 0.0, "sst_parts": [], "cos": [], "l0": [], "n": 0}
    hs = []
    cap = Capture()
    hnd = layer.register_forward_hook(cap)
    for ids in seqs:
        with torch.no_grad():
            model(ids.to(model.device))
        hs.append(cap.h[0].float())
    hnd.remove()

    result = {}
    for tag, start in (("incl_bos", 0), ("excl_bos", 1)):
        X = torch.cat([h[start:] for h in hs], dim=0)
        with torch.no_grad():
            A = sae.encode(X)
            Xh = sae.decode(A)
        sse = ((X - Xh) ** 2).sum()
        sst = ((X - X.mean(0, keepdim=True)) ** 2).sum()
        result[tag] = {
            "tokens": int(X.shape[0]),
            "variance_explained": float(1 - sse / sst),
            "mean_cosine": float(F.cosine_similarity(X, Xh, dim=-1).mean()),
            "mean_l0": float((A > 0).float().sum(-1).mean()),
            "mean_resid_norm": float(X.norm(dim=-1).mean()),
        }
    bos_norm = float(torch.stack([h[0].norm() for h in hs]).mean())
    result["bos_resid_norm"] = bos_norm

    # Cross-entropy: clean, SAE spliced (excluding / including BOS), residual zeroed (excl BOS)
    conds = {
        "clean": None,
        "splice_excl_bos": Replace(lambda x: sae.decode(sae.encode(x)), include_bos=False),
        "splice_incl_bos": Replace(lambda x: sae.decode(sae.encode(x)), include_bos=True),
        "zero_excl_bos": Replace(lambda x: torch.zeros_like(x), include_bos=False),
    }
    for name, hook in conds.items():
        tot, n = 0.0, 0
        hnd = layer.register_forward_hook(hook) if hook else None
        for ids in seqs:
            l, k = ce(model, ids.to(model.device))
            tot, n = tot + l, n + k
        if hnd:
            hnd.remove()
        result[f"ce_{name}"] = tot / n
    c, s, z = result["ce_clean"], result["ce_splice_excl_bos"], result["ce_zero_excl_bos"]
    result["loss_recovered_excl_bos"] = (z - s) / (z - c) if z != c else None
    result["ce_increase_excl_bos"] = s - c
    return result


def main() -> None:
    sae = E.load_sae()
    report = {"sae": f"{E.SAE_RELEASE}/{E.SAE_ID}", "layer": E.LAYER}

    tok, model = E.load_model("google/gemma-2-2b")
    seqs = [tok(t, return_tensors="pt").input_ids for t in texts_base()]
    report["base_plain_text"] = run(model, sae, seqs)
    del model
    torch.mps.empty_cache() if E.device() == "mps" else None

    tok, model = E.load_model("google/gemma-2-2b-it")
    report["it_chat_text"] = run(model, sae, chats_it(tok))

    (REPO / "reports" / "sae_fidelity.json").write_text(json.dumps(report, indent=2))
    lines = ["# SAE fidelity at layer 20 (revisits Tier 0b)", "",
             f"SAE: `{report['sae']}`. VE = 1 − SSE/SST pooled over tokens.", "",
             "| model / text | positions | tokens | VE | mean cos | mean L0 |",
             "|---|---|---:|---:|---:|---:|"]
    for key, label in (("base_plain_text", "gemma-2-2b, plain text"), ("it_chat_text", "gemma-2-2b-it, chat format")):
        r = report[key]
        for tag in ("incl_bos", "excl_bos"):
            x = r[tag]
            lines.append(f"| {label} | {tag.replace('_', ' ')} | {x['tokens']} | {x['variance_explained']:.3f} | "
                         f"{x['mean_cosine']:.3f} | {x['mean_l0']:.0f} |")
    lines += ["", "| model / text | CE clean | CE spliced (excl BOS) | CE spliced (incl BOS) | CE residual zeroed | loss recovered |",
              "|---|---:|---:|---:|---:|---:|"]
    for key, label in (("base_plain_text", "gemma-2-2b, plain text"), ("it_chat_text", "gemma-2-2b-it, chat format")):
        r = report[key]
        lr = r["loss_recovered_excl_bos"]
        lines.append(f"| {label} | {r['ce_clean']:.3f} | {r['ce_splice_excl_bos']:.3f} | {r['ce_splice_incl_bos']:.3f} | "
                     f"{r['ce_zero_excl_bos']:.3f} | {lr:.3f} |" if lr is not None else "")
    lines += ["", f"Mean residual norm at BOS: base {report['base_plain_text']['bos_resid_norm']:.0f}, "
                  f"it {report['it_chat_text']['bos_resid_norm']:.0f}; other positions: base "
                  f"{report['base_plain_text']['excl_bos']['mean_resid_norm']:.0f}, it "
                  f"{report['it_chat_text']['excl_bos']['mean_resid_norm']:.0f}."]
    (REPO / "reports" / "sae_fidelity.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
