# Why do chatbots overuse "not X but Y"? Literature notes (2026-10-09)

Compiled by a research pass on 2026-10-09. [V] = checked on the source page; [P] = partly; [U] = not
verified (do not cite until read). No peer-reviewed study isolates the cause of "not X but Y" itself.

## How common it is
- Paech, Roush, Goldfeder, Shwartz-Ziv, "Antislop", ICLR 2026, arXiv 2510.15061. "It's not X, it's Y"
  about 6.3× a human baseline in some (unnamed) models; Appendix L has regexes for the family. [V]
- Washington Post (Merrill, Chen, Kumer), 13 Nov 2025: 328,744 GPT-4o messages; "not just X, but Y" in
  6% of chats by July 2025 while "delve" fell to about 1 in 1,000. [P, via Boston Globe syndication]
- The Atlantic (Oremus), "The Most Famous AI Writing Tic Is Also the Most Mysterious", Jul 2026.
  Reportedly offers two theories: pretraining data, and raters rewarding apparent nuance. [U]
- Wikipedia "Signs of AI writing", "Negative parallelisms" section. [V; advice page, not peer reviewed]

## Where tics come from
- Juzek & Ward, COLING 2025, "Why Does ChatGPT 'Delve' So Much?": no evidence for architecture or data;
  consistent with RLHF. [V]
- Juzek & Ward, BIAS 2025, arXiv 2508.01930: Llama 3.2-3B base vs instruct; of 814 words boosted by
  preference tuning, 813 also exceed human rates; raters preferred the richer variant 52.4% vs 47.6%. [V]
- Zhang et al., "Verbalized Sampling", ICML 2026, arXiv 2510.01171: typicality bias in preference data
  (raters favour what the base model already finds likely), which sharpens modes. [V]
- Shaib et al., EMNLP 2024, arXiv 2407.00211: 76% of templates in model output appear in pretraining
  data (35% for human text); templates persist through RLHF. [V, abstract]
- Reinhart et al., PNAS 2025: instruction-tuned models drift further from human style than base models. [V]
- Singhal et al., COLM 2024 (length bias in reward models); Zhang et al., ACL 2025 (format bias: lists,
  bold, emojis); Kirk et al., ICLR 2024 (RLHF reduces diversity); Sharma et al., ICLR 2024 (sycophancy);
  Liang et al., arXiv 2507.07484 ("Machine Bullshit": RLHF raises empty rhetoric). [V]
- Annotator-dialect hypothesis (Guardian, Hern, 16 Apr 2024, "delve" and Nigerian English): not
  supported by Juzek & Ward's corpus check; mixed natural-experiment evidence (Chooi, LaTeCH-CLfL 2026). [V]

## Feedback loops
- Barron's, "AI Is Rewriting Corporate Lingo. ChatGPT Is Leaving Its Mark.", 14 Apr 2026. AlphaSense
  raw counts of "it's not just a __, it's a __" in releases, filings and transcripts, as reported
  secondhand: ~46 (2022), 49 (2023), 100 (2024), 208 (2025). [P, paywalled; counts unnormalised]
- TechCrunch (Silberling), 20 Apr 2026, covers it; Pangram CEO: "no smoking gun" for AI use. [V]
- Liang et al., Patterns 2025: LLM-assisted text in ~24% of Newswire press releases (Dec 2023). [V]
- Kobak et al., Science Advances 2025 (≥13.5% of 2024 PubMed abstracts); Holzwarth et al., arXiv
  2608.10715 (89% of open-access PMC papers by end of 2025). [V]
- Yakura et al., arXiv 2409.01754: LLM-favoured words rose in unscripted podcast speech after ChatGPT. [V, abstract]
- Shumailov et al., Nature 2024: model collapse on recursively generated data. [V]
