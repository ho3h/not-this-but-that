# Does a reward model prefer "not X but Y"? (exploratory)

600 single-turn answers (LMArena, UltraFeedback, OLMo 2 preference mix) containing one canonical construction sentence, each scored by Skywork/Skywork-Reward-V2-Qwen3-1.7B as written and with that one sentence rewritten.

| comparison | original scores higher | mean reward difference |
|---|---:|---:|
| original vs "S is X, and it's Y" (same content, no contrast) | **78.3%** (75.3–81.5) | +0.187 (+0.160 to +0.212; reward SD 5.06) |
| original vs "S is X. It's also Y." (same content, fluent, no contrast) | **76.7%** (73.3–80.0) | +0.294 (+0.259 to +0.333; reward SD 5.06) |
| original vs "S is Y" (drops X) | **53.2%** (49.2–57.3) | +0.056 (+0.037 to +0.076; reward SD 5.06) |

50% would mean the reward model is indifferent to the construction.

## Reading

Against rewrites that keep both ideas, the reward model prefers the construction about three times in four; against the bare grand claim ("S is Y") it is close to indifferent. What it marks down is a sentence that plainly endorses the ordinary reading ("Coffee culture is about drinks"). The construction lets an answer mention the ordinary reading and dismiss it in the same breath, which is the version this judge scores highest. Each preference is small (a few hundredths of a standard deviation of reward), but it is consistent, and training against such a judge applies it to every sentence of every answer. One open reward model; Gemma's own was not released.

Pilot on 41 LMArena answers only: reports/why_rewardmodel_pilot41.md.
