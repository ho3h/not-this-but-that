# Which training stage adds "not X but Y"? OLMo 2 1B (exploratory)

152 held-out prompts × 4 samples per stage, 250 new tokens. Detector v3, family or cousin.

| prompts | stage | answers using it | share | per 1,000 words | mean words |
|---|---|---:|---:|---:|---:|
| all 152 | base | 9 / 608 | 1.5% | 0.62 | 29 |
| all 152 | SFT | 47 / 608 | 7.7% | 0.65 | 179 |
| all 152 | DPO | 60 / 608 | 9.9% | 0.72 | 198 |
| all 152 | Instruct | 72 / 608 | 11.8% | 0.82 | 199 |
| 93 meaning | base | 7 / 372 | 1.9% | 0.78 | 28 |
| 93 meaning | SFT | 36 / 372 | 9.7% | 0.85 | 173 |
| 93 meaning | DPO | 50 / 372 | 13.4% | 0.99 | 200 |
| 93 meaning | Instruct | 62 / 372 | 16.7% | 1.14 | 200 |
| 59 process | base | 2 / 236 | 0.8% | 0.40 | 32 |
| 59 process | SFT | 11 / 236 | 4.7% | 0.36 | 190 |
| 59 process | DPO | 10 / 236 | 4.2% | 0.28 | 196 |
| 59 process | Instruct | 10 / 236 | 4.2% | 0.30 | 197 |

Stage-to-stage ratios of the share (95% prompt-clustered bootstrap CI):

- all 152, SFT/base: **5.22×** (2.92× to 12.80×)
- all 152, DPO/SFT: **1.28×** (0.88× to 1.90×)
- all 152, Instruct/DPO: **1.20×** (0.85× to 1.70×)
- all 152, Instruct/SFT: **1.53×** (1.02× to 2.35×)
- 93 meaning, SFT/base: **5.14×** (2.50× to 15.00×)
- 93 meaning, DPO/SFT: **1.39×** (0.92× to 2.22×)
- 93 meaning, Instruct/DPO: **1.24×** (0.84× to 1.81×)
- 93 meaning, Instruct/SFT: **1.72×** (1.09× to 2.92×)
- 59 process, SFT/base: **5.50×** (2.50× to 14.00×)
- 59 process, DPO/SFT: **0.91×** (0.40× to 2.00×)
- 59 process, Instruct/DPO: **1.00×** (0.33× to 2.67×)
- 59 process, Instruct/SFT: **0.91×** (0.36× to 2.14×)

Caveat: the base model gets the same chat-formatted input as the tuned stages, which it was not trained on. Compare SFT, DPO and Instruct with each other; read base as a reference point.

## Reading

Instruction tuning (SFT) multiplies the share of answers using the construction about fivefold, but only by making answers longer: per 1,000 words the rate does not move. Preference tuning (DPO) and the final RL stage then raise the rate per word on meaning questions, at a length that no longer changes (DPO and Instruct answers all reach the 250-token cap), while process questions stay flat. Each single step is within noise at this sample size; SFT to Instruct taken together is the comparison to read.
