# Why do chatbots keep saying "not X but Y"? (exploratory, 2026-10-09)

None of this was pre-registered. Each line points to its own report; figure: `figures/fig7_why.png`.
Literature notes: `why_literature.md` (no published study isolates the cause of this construction).

## What we found

1. **People were heading this way already.** In printed English, "is not just" grew about 15-fold
   between 1920 and 2019 while the control "not only" stayed flat (`trend.md`). Pretrained models
   learn it from that text: Gemma 2's base model already uses it (`base_vs_chat_v3.md`).
2. **Instruction tuning lengthens answers, and the construction rides along.** OLMo 2 1B: instruction
   tuning multiplies the share of answers using it about fivefold while its rate per 1,000 words stays
   put (0.78 → 0.85 on meaning questions; `why_stages.md`).
3. **Preference training then turns it up per word, where it already lives.** From the instruction-tuned
   to the final OLMo 2 checkpoint (preference tuning plus an RL stage), use on meaning questions rose
   **1.72×** (95% CI 1.09–2.92) at equal answer length. Process questions: 0.91× (0.36–2.14).
4. **The judges disagree with each other.**
   - Human voters (LMArena, 4,917 battles where one answer used it) picked that answer 55.9% of the
     time, but at similar length 51.4%; length-adjusted odds ratio 1.01 (0.95–1.07) (`why_arena.md`).
   - AI judges in two public preference datasets: 57.3% and 55.4% raw, 51.1% (49.7–52.5) and 51.5%
     (50.5–52.4) at equal length (`why_prefdata.md`). The chosen answers still contain it more often
     (11.1% vs 9.6% in OLMo 2's data), which is all preference training needs to push it up.
   - An open reward model (Skywork-Reward-V2, 1.7B) scored the construction above a rewrite with the
     same content and no contrast **77–78%** of the time (600 answers), and about level with a bare
     grand claim, "S is Y" (53%, 49–57) (`why_rewardmodel.md`). It marks down a sentence that plainly
     endorses the ordinary reading.
5. **It fills a slot.** When we wrote Gemma 2's first sentence as a plain grand claim ("X is B"), 38%
   of answers to meaning questions added a foil, almost always in the very next sentence. When the
   first sentence already used the construction, 11% used it again, usually far down the answer.
   My stated prediction (one use invites more) was wrong (`priming.md`).
6. **It answers a particular kind of question.** Questions about meaning or significance draw it about
   7× more often than questions about process (3.0× within prompt sets; `prompt_type_split.md`).

## A working account

Pretraining supplies the habit, which human writers had been using more each decade. Instruction
tuning makes answers longer, which multiplies the chances. Preference training, through judges that
mark down a plain endorsement of the ordinary reading, nudges each sentence about meaning toward the
version that names the ordinary reading and dismisses it. The result is a shape: a grand claim and its
foil, about once per answer about meaning. In English-language news the construction then rose to
3.5× its 2019–22 level at a July 2025 peak and has fallen every month since (`trend.md`).

## Limits

- Gemma's own reward model and preference data are not public; OLMo 2 and Skywork stand in for them.
- OLMo 2's base model sees the chat format it was never trained on; its row is a reference point.
- Every "why" result is from one run at modest sample sizes; each single OLMo stage step is within noise.
- The news trend is correlational: GDELT matches articles, and its mix of outlets changes over time.
