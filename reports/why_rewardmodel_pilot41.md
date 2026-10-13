# Does a reward model prefer "not X but Y"? (exploratory)

41 single-turn LMArena answers containing one canonical construction sentence, each scored by Skywork/Skywork-Reward-V2-Qwen3-1.7B as written and with that one sentence rewritten.

| comparison | original scores higher | mean reward difference |
|---|---:|---:|
| original vs "S is X, and it's Y" (same content, no contrast) | **73.2%** (58.5–85.4) | +0.264 (+0.142 to +0.403; reward SD 4.44) |
| original vs "S is Y" (drops X) | **61.0%** (46.3–75.6) | +0.129 (+0.035 to +0.274; reward SD 4.44) |

50% would mean the reward model is indifferent to the construction.
