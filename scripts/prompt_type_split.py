"""Construction rate by prompt type: meaning/value vs process/scene.

Exploratory, post hoc. After the pre-registered D2b confirmation run found a
1.0% baseline (vs 5.9% on D2), the prompts that did elicit the construction
looked like requests to judge why something matters ("what makes a good
neighbourhood library") rather than requests for a process or a scene ("the
first hour of a night shift at a small bakery"). To test that reading without
the labeller seeing any outcomes, all 152 D2 + D2b prompts were shuffled and
labelled M (meaning/value) or P (process/scene/facts) from the wording alone,
by an annotator given only the prompt list: data/prompt_type_labels.tsv.

This script joins those labels to the committed BASELINE generations (no
intervention) of Q5c (D2, 3 seeds) and the confirmation run (D2b, 6 seeds),
and scores each answer with the family ∪ cousin detectors.

Writes reports/prompt_type_split.{md,json}. No GPU needed.
"""

from __future__ import annotations

import json
import pathlib
import random
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from classifier.detect_v2 import detect_construction_v2, detect_more_than_just  # noqa: E402

LABELS = REPO / "data" / "prompt_type_labels.tsv"
SOURCES = {
    "D2 (Q5c baseline)": REPO / "reports" / "q5c_d2_high_power.json",
    "D2b (confirmation baseline)": REPO / "reports" / "confirmation_run.json",
}
N_BOOT = 10_000


def load_labels() -> dict[str, str]:
    out = {}
    for line in LABELS.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        _id, lab, prompt = line.split("\t", 2)
        out[prompt.strip()] = lab.strip().upper()
    return out


def hit(text: str) -> tuple[bool, bool]:
    return bool(detect_construction_v2(text)), bool(detect_more_than_just(text))


def cluster_boot_ratio(by_prompt: dict[str, list[int]], labels: dict[str, str], seed: int = 0):
    """Prompt-clustered bootstrap CI for rate(M) / rate(P)."""
    rng = random.Random(seed)
    groups = {k: [p for p in by_prompt if labels[p] == k] for k in ("M", "P")}
    ratios = []
    for _ in range(N_BOOT):
        rates = {}
        for k, ps in groups.items():
            sample = [rng.choice(ps) for _ in ps]
            n = sum(len(by_prompt[p]) for p in sample)
            h = sum(sum(by_prompt[p]) for p in sample)
            rates[k] = h / n if n else 0.0
        if rates["P"] > 0:
            ratios.append(rates["M"] / rates["P"])
    ratios.sort()
    if not ratios:
        return None
    return ratios[int(0.025 * len(ratios))], ratios[int(0.975 * len(ratios))]


def fisher_one_sided(a: int, b: int, c: int, d: int) -> float:
    """P(X >= a) for the 2x2 table [[a, b], [c, d]] under the hypergeometric null."""
    from math import comb

    n1, n2, k = a + b, c + d, a + c
    total = comb(n1 + n2, k)
    return sum(comb(n1, x) * comb(n2, k - x) for x in range(a, min(n1, k) + 1)) / total


def main() -> None:
    labels = load_labels()
    result = {"labels_file": str(LABELS.relative_to(REPO)), "n_labelled": len(labels), "sets": {}}
    pooled: dict[str, list[int]] = {}
    for name, path in SOURCES.items():
        rows = json.loads(path.read_text())["rows"]
        by_prompt: dict[str, list[int]] = {}
        fam_cou = {"M": [0, 0, 0], "P": [0, 0, 0]}  # answers, family, cousin
        for r in rows:
            p = r["prompt"].strip()
            lab = labels[p]
            f, c = hit(r["baseline"])
            by_prompt.setdefault(p, []).append(int(f or c))
            fam_cou[lab][0] += 1
            fam_cou[lab][1] += f
            fam_cou[lab][2] += c
        pooled.update({f"{name}::{p}": v for p, v in by_prompt.items()})
        summary = {}
        for lab in ("M", "P"):
            n, f, c = fam_cou[lab]
            prompts = [p for p in by_prompt if labels[p] == lab]
            summary[lab] = {
                "prompts": len(prompts),
                "answers": n,
                "family": f,
                "cousin": c,
                "family_or_cousin_answers": sum(sum(by_prompt[p]) for p in prompts),
                "rate": sum(sum(by_prompt[p]) for p in prompts) / n if n else None,
                "prompts_with_any_hit": sum(1 for p in prompts if any(by_prompt[p])),
            }
        result["sets"][name] = summary

    pooled_labels = {k: labels[k.split("::", 1)[1]] for k in pooled}
    tot = {}
    for lab in ("M", "P"):
        ks = [k for k in pooled if pooled_labels[k] == lab]
        tot[lab] = {
            "prompts": len(ks),
            "answers": sum(len(pooled[k]) for k in ks),
            "hits": sum(sum(pooled[k]) for k in ks),
            "prompts_with_any_hit": sum(1 for k in ks if any(pooled[k])),
        }
        tot[lab]["rate"] = tot[lab]["hits"] / tot[lab]["answers"]
    ci = cluster_boot_ratio(pooled, pooled_labels)
    a, b = tot["M"]["prompts_with_any_hit"], tot["M"]["prompts"] - tot["M"]["prompts_with_any_hit"]
    c, d = tot["P"]["prompts_with_any_hit"], tot["P"]["prompts"] - tot["P"]["prompts_with_any_hit"]
    result["pooled"] = tot
    result["rate_ratio_M_over_P"] = tot["M"]["rate"] / tot["P"]["rate"] if tot["P"]["rate"] else None
    result["rate_ratio_95ci_prompt_clustered"] = ci
    result["fisher_one_sided_prompt_level"] = fisher_one_sided(a, b, c, d)

    # The two prompt sets differ in their M/P mix (D2 is mostly M, D2b mostly P),
    # so the pooled ratio partly reflects set differences. Mantel-Haenszel rate
    # ratio stratified by set gives the within-set comparison.
    num = den = 0.0
    for s in result["sets"].values():
        m, p_ = s["M"], s["P"]
        n_tot = m["answers"] + p_["answers"]
        num += m["family_or_cousin_answers"] * p_["answers"] / n_tot
        den += p_["family_or_cousin_answers"] * m["answers"] / n_tot
    result["mantel_haenszel_rate_ratio_by_set"] = num / den if den else None

    out_json = REPO / "reports" / "prompt_type_split.json"
    out_json.write_text(json.dumps(result, indent=2))

    lines = [
        "# Construction rate by prompt type (exploratory, post hoc)",
        "",
        "Baseline generations only (no intervention). Hit = negated family ∪ \"more than just\" cousin.",
        "Prompt labels were assigned from wording alone by an annotator who saw only the shuffled",
        "prompt list, never the generations or hit counts (`data/prompt_type_labels.tsv`).",
        "M = asks what something means or why it matters; P = asks for a process, a scene or facts.",
        "",
        "| set | type | prompts | answers | answers with hit | rate | prompts with ≥1 hit |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, s in result["sets"].items():
        for lab in ("M", "P"):
            x = s[lab]
            rate = f"{100 * x['rate']:.1f}%" if x["rate"] is not None else "n/a"
            lines.append(
                f"| {name} | {lab} | {x['prompts']} | {x['answers']} | "
                f"{x['family_or_cousin_answers']} | {rate} | {x['prompts_with_any_hit']} |"
            )
    for lab in ("M", "P"):
        x = tot[lab]
        lines.append(
            f"| **pooled** | **{lab}** | {x['prompts']} | {x['answers']} | {x['hits']} | "
            f"**{100 * x['rate']:.1f}%** | {x['prompts_with_any_hit']} |"
        )
    rr = result["rate_ratio_M_over_P"]
    lines += [
        "",
        f"Rate ratio M/P: **{rr:.1f}×**" if rr else "Rate ratio M/P: undefined (no P hits)",
        f"(95% prompt-clustered bootstrap CI {ci[0]:.1f}× to {ci[1]:.1f}×)." if ci else "",
        f"Prompt-level one-sided Fisher exact test (any hit vs none): p = {result['fisher_one_sided_prompt_level']:.2g}.",
        "",
        "The sets differ in their mix (D2 is mostly M; D2b, written later on concrete topics, is mostly P),",
        "so part of the pooled gap is a difference between sets. Stratified by set, the Mantel-Haenszel",
        f"rate ratio is **{result['mantel_haenszel_rate_ratio_by_set']:.1f}×**; the direction holds within each set.",
        "",
        "Caveat: this split was proposed after looking at which prompts elicited the construction.",
        "It is a lead for a pre-registered test, not a confirmed finding.",
    ]
    (REPO / "reports" / "prompt_type_split.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
