"""Blind LLM-judge re-score of the Q5b/Q5c/Q5d generations.

The post's headline numbers rest on regex detectors. This harness adds an
independent instrument: a Claude judge, blinded to condition, classifies
every committed generation for presence of the not-X-but-Y construction
(any surface form). If the judge agrees with the union detector, the finding
no longer rests on regex; if it diverges, we want to know where before a
reader does.

Design:
  - Every generation (baseline + ablated, all three evals: 1,452 texts) is
    judged in RANDOMIZED order with an opaque id — the judge never sees
    condition, eval name, or pairing.
  - Uses the Message Batches API (50% price, fine for offline judging) with
    a structured-output JSON schema, so results parse without retries.
  - Verdict asks for the quoted span, which makes spot-auditing the judge
    itself cheap.

Cost estimate (printed before submission; requires ANTHROPIC_API_KEY):
  ~1,452 judgments x (~250 input + ~80 output tokens)
  claude-opus-4-8 batch:  ~$2.40        claude-haiku-4-5 batch: ~$0.33

Usage:
  .venv/bin/python scripts/llm_judge_rescore.py            # submit + wait + score
  .venv/bin/python scripts/llm_judge_rescore.py --model claude-haiku-4-5
  .venv/bin/python scripts/llm_judge_rescore.py --resume msgbatch_xxx   # re-attach

Writes reports/llm_judge_rescore.{md,json}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys
import time

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from m1_stats_reanalysis import mcnemar_midp

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT_MD = REPO / "reports" / "llm_judge_rescore.md"
OUT_JSON = REPO / "reports" / "llm_judge_rescore.json"
MANIFEST = REPO / "reports" / "llm_judge_manifest.json"  # blinding key, local only

EVALS = [
    ("q5b", "reports/q5b_d1_continuation.json", "prefix_idx"),
    ("q5c", "reports/q5c_d2_high_power.json", "prompt_idx"),
    ("q5d", "reports/q5d_minimal_set_d1_n120.json", "prefix_idx"),
]

JUDGE_SYSTEM = (
    "You judge short machine-generated texts for ONE rhetorical construction: "
    "contrastive correction, the 'not X, but Y' / antithesis family. It denies "
    "one framing and substitutes another. Count these surface forms:\n"
    "  - same-sentence: \"It's not a tool, it's a partner\" / \"isn't X — it's Y\"\n"
    "  - cross-sentence: \"It's not a tool. It's a partner.\"\n"
    "  - minimizer: \"not just X, (but) Y\" / \"more than just X; it's Y\"\n"
    "  - reframing: \"it's not about X, it's about Y\" / \"less X, more Y\"\n"
    "Do NOT count: ordinary concessives (\"X is great, but hard\"), plain "
    "negation without a corrective substitute, hedges (\"I'm not sure\"), or "
    "lists. The denied thing and the substituted thing must be parallel "
    "alternatives. Judge only what is present in the text."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "contains_construction": {"type": "boolean"},
        "quoted_span": {
            "type": ["string", "null"],
            "description": "Exact span from the text if present, else null",
        },
        "form": {
            "type": ["string", "null"],
            "enum": ["same-sentence", "cross-sentence", "minimizer", "reframing", None],
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["contains_construction", "quoted_span", "form", "confidence"],
    "additionalProperties": False,
}


def build_items() -> list[dict]:
    items = []
    for ev, path, key in EVALS:
        rows = json.loads((REPO / path).read_text())["rows"]
        for i, r in enumerate(rows):
            for cond in ("baseline", "ablated"):
                text = r[cond]
                items.append({
                    "custom_id": "j" + hashlib.sha1(
                        f"{ev}:{i}:{cond}:{text}".encode()).hexdigest()[:16],
                    "eval": ev, "row": i, "cond": cond,
                    "prompt_id": r[key], "seed": r.get("seed"),
                    "text": text,
                })
    # Blind: judge sees items in randomized order with opaque ids only
    rng = random.Random(20260609)
    rng.shuffle(items)
    return items


def submit(client: anthropic.Anthropic, items: list[dict], model: str) -> str:
    requests = [
        Request(
            custom_id=it["custom_id"],
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=400,
                system=JUDGE_SYSTEM,
                output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{
                    "role": "user",
                    "content": "Text to judge:\n\n" + it["text"],
                }],
            ),
        )
        for it in items
    ]
    batch = client.messages.batches.create(requests=requests)
    print(f"submitted batch {batch.id} ({len(requests)} judgments)")
    return batch.id


def wait(client: anthropic.Anthropic, batch_id: str):
    while True:
        b = client.messages.batches.retrieve(batch_id)
        if b.processing_status == "ended":
            print(f"batch ended: {b.request_counts.succeeded} ok, "
                  f"{b.request_counts.errored} errored")
            return
        print(f"  …{b.processing_status}, {b.request_counts.processing} processing",
              flush=True)
        time.sleep(30)


def collect(client: anthropic.Anthropic, batch_id: str) -> dict[str, dict]:
    verdicts = {}
    for result in client.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            verdicts[result.custom_id] = {"error": result.result.type}
            continue
        msg = result.result.message
        text = next((b.text for b in msg.content if b.type == "text"), "")
        try:
            verdicts[result.custom_id] = json.loads(text)
        except json.JSONDecodeError:
            verdicts[result.custom_id] = {"error": "unparseable", "raw": text[:200]}
    return verdicts


def score(items: list[dict], verdicts: dict[str, dict]) -> dict:
    from classifier import detect_construction
    from classifier.detect_v2 import detect_permissive

    def union(t: str) -> bool:
        if any(h.variant.value in ("C1", "C2", "C3")
               for h in detect_construction(t, strict=False)):
            return True
        return bool(detect_permissive(t))

    out = {}
    for ev, _, _ in EVALS:
        ev_items = [it for it in items if it["eval"] == ev]
        per_cond = {"baseline": [], "ablated": []}
        agree = disagree = judged = 0
        disagreements = []
        pairs: dict = {}
        for it in ev_items:
            v = verdicts.get(it["custom_id"], {})
            if "contains_construction" not in v:
                continue
            judged += 1
            jhit = bool(v["contains_construction"])
            rhit = union(it["text"])
            per_cond[it["cond"]].append(jhit)
            if jhit == rhit:
                agree += 1
            else:
                disagree += 1
                disagreements.append({
                    "cond": it["cond"], "judge": jhit, "regex": rhit,
                    "span": v.get("quoted_span"),
                    "text": it["text"][:160],
                })
            key = (it["prompt_id"], it["seed"])
            pairs.setdefault(key, {})[it["cond"]] = jhit
        b = sum(1 for p in pairs.values()
                if p.get("baseline") and not p.get("ablated"))
        c = sum(1 for p in pairs.values()
                if not p.get("baseline") and p.get("ablated"))
        nb, na = sum(per_cond["baseline"]), sum(per_cond["ablated"])
        n = len(per_cond["baseline"])
        out[ev] = {
            "n_pairs": n, "judged": judged,
            "judge_baseline_hits": nb, "judge_ablated_hits": na,
            "judge_rel_drop": (nb - na) / nb if nb else None,
            "judge_mcnemar_midp": mcnemar_midp(b, c),
            "agreement_with_union": agree / max(1, agree + disagree),
            "disagreements": disagreements,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-4-8",
                    help="claude-opus-4-8 (default) or claude-haiku-4-5 (budget)")
    ap.add_argument("--resume", default=None, help="existing batch id to re-attach")
    ap.add_argument("--yes", action="store_true", help="skip cost confirmation")
    args = ap.parse_args()

    items = build_items()
    MANIFEST.write_text(json.dumps(items, indent=1))
    print(f"{len(items)} texts to judge (blinded order, manifest at {MANIFEST})")

    client = anthropic.Anthropic()

    if args.resume:
        batch_id = args.resume
    else:
        est = "$2.40" if "opus" in args.model else "$0.33"
        print(f"model={args.model}, batch API — estimated cost ≈ {est}")
        if not args.yes and input("submit? [y/N] ").lower() != "y":
            sys.exit(0)
        batch_id = submit(client, items, args.model)

    wait(client, batch_id)
    verdicts = collect(client, batch_id)
    results = score(items, verdicts)

    OUT_JSON.write_text(json.dumps(
        {"batch_id": batch_id, "model": args.model, "results": results}, indent=2))

    md = [
        "# Blind LLM-judge re-score",
        "",
        f"Judge: `{args.model}` via Message Batches, blinded to condition and",
        "pairing, randomized order, structured-output schema. Spans quoted for",
        "spot-audit. Batch id: `" + batch_id + "`.",
        "",
        "| eval | n pairs | judge baseline | judge ablated | rel drop | mid-p | agreement w/ union |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ev, r in results.items():
        rd = f"{r['judge_rel_drop']:.1%}" if r["judge_rel_drop"] is not None else "—"
        md.append(f"| {ev} | {r['n_pairs']} | {r['judge_baseline_hits']} | "
                  f"{r['judge_ablated_hits']} | {rd} | "
                  f"{r['judge_mcnemar_midp']:.4g} | {r['agreement_with_union']:.1%} |")
    md += ["", "Disagreements with the union detector (judge vs regex):", ""]
    for ev, r in results.items():
        for d in r["disagreements"][:15]:
            md.append(f"- **{ev}/{d['cond']}** judge={d['judge']} regex={d['regex']} "
                      f"span={d['span']!r}: {d['text']}…")
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"→ {OUT_MD}")


if __name__ == "__main__":
    main()
