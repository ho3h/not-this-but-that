"""Re-score the A1-A7 gauntlet generations with the final union detector.

The gauntlet's original referee is a separate F1-F7 per-sentence detector
(validated at P=R=0.857 on its own holdout, reports/gauntlet/g5_referee_validation.md).
The Medium post narrates the gauntlet in per-generation terms using the same
union detector as the headline numbers, so this artifact is the receipt for
those sentences. Per-generation: a generation counts once if the union
detector fires anywhere in it.

Writes reports/gauntlet_union_rescore.json.
"""
from __future__ import annotations
import json, pathlib

from classifier import detect_construction
from classifier.detect_v2 import detect_permissive

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "reports" / "gauntlet_union_rescore.json"

NAMES = {
    "a1": "Ask nicely (system prompt)",
    "a2": "Ban the words (logit ban: but/just/only/rather/more/less/em-dash)",
    "a3": "Show the cure (4 de-slopped exemplars in-context)",
    "a4": "Scalpel mid-act (ablate 3223 when it fires)",
    "a5": "Scalpel pre-emptive (ablate 3223 throughout)",
    "a6": "Orthogonalize (project 3223 direction out of all layers, Arditi-style)",
    "a7": "Steering vector (CAA, alpha=8.0)",
}


def union(t: str) -> bool:
    if any(h.variant.value in ("C1", "C2", "C3")
           for h in detect_construction(t, strict=False)):
        return True
    return bool(detect_permissive(t))


def main():
    out = {}
    for key, label in NAMES.items():
        d = json.loads((REPO / f"reports/gauntlet/{key}_result.json").read_text())
        get = lambda r: r if isinstance(r, str) else (r.get("text") or r.get("generation") or "")
        base = [get(r) for r in d["baseline_generations"]]
        intv = [get(r) for r in d["intervened_generations"]]
        b = sum(1 for t in base if union(t))
        i = sum(1 for t in intv if union(t))
        out[key.upper()] = {
            "attack": label, "n": len(base),
            "baseline_hits": b, "intervened_hits": i,
            "rel_drop": (b - i) / b if b else None,
        }
        print(f"{key.upper()} {label[:50]:50s} {b}/{len(base)} -> {i}/{len(intv)}")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
