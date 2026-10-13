"""Discovery campaign — behavioural entry-gate.

⚠ DISCOVERY OUTPUT IS CONTAMINATED BY DEFINITION ⚠

The question Phase 7 surfaced: what makes the model decide to *enter* the
"not X, but Y" construction in the first place? Feat 3223 governs the
commit, not the entry. This campaign studies the entry decision at the
behavioural / text level — no SAE involvement, so it can run while Tier 0b
remains open.

We test ~15 distinct text-level hypotheses about what predicts whether a
given sentence in a Gemma-2-2b-it generation contains the construction.
Data: Phase 2 generations, restricted to the D2-Discovery split (76
prompts × 5 seeds = 380 generations) via the firewall module. The
held-out D2-Confirmation split (26 prompts) is NEVER read by this script.

Per the operating protocol §2.3, every output is labelled [CANDIDATE].
None of these are findings. The Confirmation step (a separate script,
running on the Confirmation split, applying BH-FDR at q=0.10 over N=15)
is what decides what's a candidate finding.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import stats

from classifier import detect_construction
from firewall import load_d2

REPO = Path(__file__).resolve().parent.parent
GEN_PATH = REPO / "reports" / "phase2_generations_gemma_2b_it.jsonl"
OUT_DIR = REPO / "reports" / "discovery" / "entry_behavioural"

# Discovery's loose filter — Confirmation will apply the strict BH-FDR.
DISCOVERY_RAW_P = 0.20

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“])")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    return [s.strip() for s in SENT_SPLIT.split(text) if len(s.strip()) > 30]


def contains_construction(sentence: str) -> bool:
    hits = detect_construction(sentence, strict=True)
    return any(h.variant.value in ("C1", "C2", "C3") for h in hits)


@dataclass
class Sentence:
    text: str
    prompt: str
    prompt_idx: int
    seed: int
    sent_idx: int  # 0-based position within its generation
    n_sents_in_gen: int
    has_construction: bool


def collect_sentences() -> list[Sentence]:
    """Phase 2 Gemma-2-2b-it generations, restricted to the Discovery split."""
    discovery_prompts = set(load_d2(phase="discovery"))
    out = []
    for line in GEN_PATH.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["prompt"] not in discovery_prompts:
            continue
        sents = split_sentences(row["generation"])
        for i, s in enumerate(sents):
            out.append(Sentence(
                text=s, prompt=row["prompt"], prompt_idx=row["prompt_idx"],
                seed=row["seed"], sent_idx=i, n_sents_in_gen=len(sents),
                has_construction=contains_construction(s),
            ))
    return out


# --- Hypothesis library ---------------------------------------------------------
# Each hypothesis is a function `Sentence -> bool` (binary predictor) OR
# `Sentence -> float` (continuous predictor). The test statistic depends on
# the type. The hypothesis count is N — Confirmation will use this N in BH-FDR.

@dataclass
class Hypothesis:
    name: str
    description: str
    fn: Callable[[Sentence], object]
    kind: str  # "binary" or "continuous"


def _is_first_sent(s: Sentence) -> bool:
    return s.sent_idx == 0


def _is_last_sent(s: Sentence) -> bool:
    return s.sent_idx == s.n_sents_in_gen - 1


def _is_middle_sent(s: Sentence) -> bool:
    return 0 < s.sent_idx < s.n_sents_in_gen - 1


def _sent_len_tokens(s: Sentence) -> float:
    return float(len(s.text.split()))


def _looks_bulleted(s: Sentence) -> bool:
    return s.text.lstrip().startswith(("*", "-", "•")) or "**" in s.text[:80]


def _has_bold_emphasis(s: Sentence) -> bool:
    return "**" in s.text


def _starts_with_subj_copula(s: Sentence) -> bool:
    return bool(re.match(r"^(it'?s|that'?s|this\s+is|they'?re|we'?re)\b",
                          s.text.strip(), re.IGNORECASE))


def _starts_with_capital_pronoun(s: Sentence) -> bool:
    return bool(re.match(r"^(It|This|That|They|We)\b", s.text.strip()))


def _contains_just_or_merely(s: Sentence) -> bool:
    return bool(re.search(r"\b(?:just|merely|simply|only)\b", s.text, re.IGNORECASE))


def _prompt_starts_with_explain(s: Sentence) -> bool:
    return s.prompt.strip().lower().startswith("explain")


def _prompt_starts_with_describe(s: Sentence) -> bool:
    return s.prompt.strip().lower().startswith("describe")


def _prompt_starts_with_discuss(s: Sentence) -> bool:
    return s.prompt.strip().lower().startswith("discuss")


def _prompt_starts_with_walk(s: Sentence) -> bool:
    return s.prompt.strip().lower().startswith("walk")


def _prompt_starts_with_reflect(s: Sentence) -> bool:
    return s.prompt.strip().lower().startswith("reflect")


def _prompt_length(s: Sentence) -> float:
    return float(len(s.prompt.split()))


def _gen_length(s: Sentence) -> float:
    return float(s.n_sents_in_gen)


def _sent_rel_position(s: Sentence) -> float:
    """0.0 = first sentence, 1.0 = last sentence."""
    return s.sent_idx / max(s.n_sents_in_gen - 1, 1)


def _contains_emdash(s: Sentence) -> bool:
    return "—" in s.text or "–" in s.text or "--" in s.text


def _has_quote(s: Sentence) -> bool:
    return '"' in s.text or "'" in s.text or "“" in s.text


HYPOTHESES = [
    Hypothesis("H01", "Sentence is the FIRST in its generation", _is_first_sent, "binary"),
    Hypothesis("H02", "Sentence is the LAST in its generation", _is_last_sent, "binary"),
    Hypothesis("H03", "Sentence is in the MIDDLE of its generation", _is_middle_sent, "binary"),
    Hypothesis("H04", "Sentence length in tokens (Mann-Whitney U)", _sent_len_tokens, "continuous"),
    Hypothesis("H05", "Sentence appears inside a bulleted/structured list", _looks_bulleted, "binary"),
    Hypothesis("H06", "Sentence contains bold-emphasis markup (**)", _has_bold_emphasis, "binary"),
    Hypothesis("H07", "Sentence starts with a subject-copula contraction (It's / That's / …)", _starts_with_subj_copula, "binary"),
    Hypothesis("H08", "Sentence starts with a capital pronoun (It / This / They / We)", _starts_with_capital_pronoun, "binary"),
    Hypothesis("H09", "Sentence contains 'just' / 'merely' / 'simply' / 'only'", _contains_just_or_merely, "binary"),
    Hypothesis("H10", "Prompt starts with 'Explain'", _prompt_starts_with_explain, "binary"),
    Hypothesis("H11", "Prompt starts with 'Describe'", _prompt_starts_with_describe, "binary"),
    Hypothesis("H12", "Prompt starts with 'Discuss'", _prompt_starts_with_discuss, "binary"),
    Hypothesis("H13", "Prompt starts with 'Walk'", _prompt_starts_with_walk, "binary"),
    Hypothesis("H14", "Prompt starts with 'Reflect'", _prompt_starts_with_reflect, "binary"),
    Hypothesis("H15", "Prompt length in tokens (Mann-Whitney U)", _prompt_length, "continuous"),
    Hypothesis("H16", "Generation length in sentences (Mann-Whitney U)", _gen_length, "continuous"),
    Hypothesis("H17", "Relative sentence position 0-1 (Mann-Whitney U)", _sent_rel_position, "continuous"),
    Hypothesis("H18", "Sentence contains an em-dash / en-dash", _contains_emdash, "binary"),
    Hypothesis("H19", "Sentence contains a quotation mark", _has_quote, "binary"),
]


def test_binary(sents: list[Sentence], pred: Callable[[Sentence], bool]) -> dict:
    """2×2 Fisher exact (one-tailed, alternative='greater'):
       pred=True / has_construction=True is the cell we hypothesise inflated."""
    yes_c = sum(1 for s in sents if pred(s) and s.has_construction)
    yes_n = sum(1 for s in sents if pred(s) and not s.has_construction)
    no_c = sum(1 for s in sents if not pred(s) and s.has_construction)
    no_n = sum(1 for s in sents if not pred(s) and not s.has_construction)
    table = [[yes_c, yes_n], [no_c, no_n]]
    odds, p = stats.fisher_exact(table, alternative="greater")
    rate_yes = yes_c / max(yes_c + yes_n, 1)
    rate_no = no_c / max(no_c + no_n, 1)
    return {
        "test": "Fisher exact (greater)",
        "table": table,
        "rate_when_pred_true": rate_yes,
        "rate_when_pred_false": rate_no,
        "odds_ratio": float(odds),
        "p_raw": float(p),
        "n_pred_true": yes_c + yes_n,
    }


def test_continuous(sents: list[Sentence], fn: Callable[[Sentence], float]) -> dict:
    """Mann-Whitney U comparing the feature value distribution in
       construction vs non-construction sentences."""
    pos = np.array([fn(s) for s in sents if s.has_construction], dtype=float)
    neg = np.array([fn(s) for s in sents if not s.has_construction], dtype=float)
    if len(pos) < 2:
        return {"test": "Mann-Whitney U", "p_raw": 1.0,
                "median_pos": float("nan"), "median_neg": float("nan"),
                "n_pos": len(pos), "n_neg": len(neg)}
    u, p_two = stats.mannwhitneyu(pos, neg, alternative="two-sided")
    return {
        "test": "Mann-Whitney U (two-sided)",
        "U": float(u),
        "median_pos": float(np.median(pos)),
        "median_neg": float(np.median(neg)),
        "n_pos": len(pos),
        "n_neg": len(neg),
        "p_raw": float(p_two),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sents = collect_sentences()
    n_total = len(sents)
    n_pos = sum(1 for s in sents if s.has_construction)
    n_neg = n_total - n_pos
    print(f"Discovery split: {n_total} sentences, {n_pos} positive (construction), {n_neg} negative")

    results = []
    for h in HYPOTHESES:
        if h.kind == "binary":
            r = test_binary(sents, h.fn)
        else:
            r = test_continuous(sents, h.fn)
        r["id"] = h.name
        r["description"] = h.description
        r["kind"] = h.kind
        results.append(r)

    results.sort(key=lambda r: r["p_raw"])

    n_hypotheses = len(HYPOTHESES)
    bh_q = 0.10

    md = [
        "> ⚠ DISCOVERY OUTPUT — CONTAMINATED BY DEFINITION",
        "> These candidates are a ranked list. None of them is a finding.",
        "> Confirmation runs against `pre_registration.yaml` on the held-out",
        "> Confirmation split, with the FDR threshold computed from the",
        "> number of hypotheses below.",
        f"> Number of hypotheses tried in this campaign: **{n_hypotheses}**",
        "",
        "# Behavioural entry-gate Discovery campaign",
        "",
        "**Question:** What text-level features predict construction-entry in "
        "Gemma 2 2B-it generations?",
        "",
        f"**Data:** Phase 2 generations, D2-Discovery split only "
        f"(76 prompts × 5 seeds = 380 generations → {n_total} sentences). "
        f"D2-Confirmation split (26 prompts) **NEVER read by this script** "
        "(see `src/firewall/__init__.py`).",
        "",
        f"**Outcome:** binary `has_construction` per sentence "
        f"({n_pos} positive / {n_neg} negative = "
        f"{n_pos/max(n_total,1):.2%} base rate).",
        "",
        f"**Discovery filter:** raw p ≤ {DISCOVERY_RAW_P:.2f} → marked "
        f"[CANDIDATE]. Confirmation will apply Benjamini-Hochberg FDR at "
        f"q = {bh_q:.2f} across all {n_hypotheses} hypotheses; the per-test "
        f"BH α floor is "
        f"`(i/{n_hypotheses}) · {bh_q:.2f}` for the i-th ranked p-value.",
        "",
        "## Ranked hypotheses\n",
        "| Rank | ID | Hypothesis | Test | Effect | n cases | p (raw) | α_BH (i/N · q) | Discovery filter |",
        "|---:|:---|:---|:---|:---|---:|---:|---:|:---|",
    ]
    for rank, r in enumerate(results, start=1):
        alpha_bh = (rank / n_hypotheses) * bh_q
        passes_filter = r["p_raw"] <= DISCOVERY_RAW_P
        if r["kind"] == "binary":
            effect = (f"rate true: {r['rate_when_pred_true']:.2%}, "
                       f"false: {r['rate_when_pred_false']:.2%}, "
                       f"OR={r['odds_ratio']:.2f}")
            n_cases = r["n_pred_true"]
        else:
            effect = (f"median pos: {r['median_pos']:.2f}, "
                       f"median neg: {r['median_neg']:.2f}")
            n_cases = f"{r['n_pos']}/{r['n_neg']}"
        marker = "**[CANDIDATE]**" if passes_filter else "(below filter)"
        md.append(
            f"| {rank} | {r['id']} | {r['description']} | {r['test']} | "
            f"{effect} | {n_cases} | {r['p_raw']:.4f} | {alpha_bh:.4f} | {marker} |"
        )
    md.append("")

    surviving = [r for r in results if r["p_raw"] <= DISCOVERY_RAW_P]
    md.append(f"## Discovery filter survivors: {len(surviving)} of {n_hypotheses}\n")
    md.append("These are the **[CANDIDATE]** hypotheses Confirmation should test on the "
              "D2-Confirmation split. Confirmation passes a candidate ONLY IF its p-value "
              "on the Confirmation split is below its BH-FDR-corrected α (the column "
              "above), AND the candidate's effect direction matches Discovery's.\n")
    md.append("**The candidates below are not findings. They are leads.** A `[CANDIDATE]` "
              "is what survived the Discovery filter on contaminated data; it's a hypothesis "
              "worth Confirmation-testing, not evidence of anything.\n")
    for r in surviving:
        md.append(f"- **[CANDIDATE] {r['id']}** — {r['description']} (raw p = {r['p_raw']:.4f})")
    md.append("")

    md.append("## Confounds you'd be irresponsible to ignore\n")
    md.append("Two of the surviving candidates are at least partly **circular with the "
              "classifier's detection lexicon**, which means they're correlated with "
              "construction-presence partly *by construction* of the outcome label, "
              "not because they predict the model's decision:")
    md.append("")
    md.append("- **H09** (contains *just / merely / simply / only*) — these are the "
              "exact lexical hinges the classifier looks for in C3 (`_C3 = … (just|"
              "merely|simply) …`). A sentence containing one is *mechanically* more "
              "likely to be flagged as a construction. The p-value is real, but the "
              "candidate is downgraded to *measuring classifier definition, not entry "
              "decision*. Confirmation should still run it, but a Confirmation pass "
              "doesn't mean it predicts entry — it means C3's lexicon shows up in "
              "C3-positive sentences, which is tautological.")
    md.append("")
    md.append("- **H07** (starts with subject-copula contraction *It's / That's /* …) — "
              "C1's regex opener is `(it'?s|that'?s|this is|…)\\s+not`. A sentence "
              "starting with one of these has more *opportunity* to be classified as "
              "C1, though it must still contain the construction's structural pivot "
              "(`, it's` or `, but`) to be flagged. Less circular than H09 but worth "
              "flagging.")
    md.append("")
    md.append("- **H08** (starts with capital pronoun) — superset of H07; same caveat, "
              "broader.")
    md.append("")
    md.append("**The genuinely independent candidates are H03, H16, H17, H19** — sentence "
              "position, generation length, relative position, presence of quotation "
              "marks. These do not overlap with the classifier's lexicon, so a "
              "Confirmation pass on these would mean something real about the entry "
              "decision.")
    md.append("")

    md.append("## What's next (for the agent picking this up)\n")
    md.append("1. **Do not quote any candidate above as a finding.** Per the operating "
              "protocol §6, the closest allowed phrasing is *\"Discovery surfaced X as "
              "a candidate; it has not yet been Confirmation-tested.\"*")
    md.append("2. **Confirmation runs on the D2-Confirmation split** (26 prompts, "
              "loaded via `firewall.load_d2(phase='confirmation')`). The Phase 2 "
              "generations for those 26 prompts already exist in "
              "`reports/phase2_generations_gemma_2b_it.jsonl` — but if you regenerate, "
              "use the same sampling params (temperature=0.8, top_p=0.95, "
              "max_new_tokens=150, seeds 0-4) so the Confirmation distribution "
              "matches Discovery's.")
    md.append("3. **The Confirmation test for each candidate** is the same statistic "
              "Discovery used, run on the Confirmation sentences only, with the "
              "BH α threshold from the column above. A candidate passes Confirmation "
              "only if p_conf ≤ α_BH AND the effect direction matches Discovery's.")
    md.append("4. **Confirmation should explicitly drop or down-weight the confounded "
              "candidates (H07, H08, H09).** A circular candidate passing Confirmation "
              "is not informative; it would just be measuring the classifier on a "
              "new sample.")
    md.append("5. **If no clean candidate survives Confirmation, that is itself "
              "reportable.** It means text-level predictors of construction-entry "
              "don't replicate out-of-sample at the strict FDR level — entry is "
              "harder to predict behaviourally than the variant composition (Phase 2) "
              "was. That's a real result, not a failure.")

    out_md = OUT_DIR / "CANDIDATES.md"
    out_json = OUT_DIR / "CANDIDATES.json"
    out_md.write_text("\n".join(md))
    out_json.write_text(json.dumps({
        "campaign": "entry_behavioural",
        "n_hypotheses": n_hypotheses,
        "n_sentences": n_total,
        "n_positive": n_pos,
        "discovery_raw_p_filter": DISCOVERY_RAW_P,
        "bh_q": bh_q,
        "results": results,
    }, indent=2))
    print(f"\n→ {out_md}")
    print(f"\nDiscovery survivors ({len(surviving)} of {n_hypotheses}):")
    for r in surviving:
        print(f"  [CANDIDATE] {r['id']}  p={r['p_raw']:.4f}  {r['description']}")


if __name__ == "__main__":
    main()
