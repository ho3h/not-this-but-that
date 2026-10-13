# Do people vote for "not X but Y"? (exploratory)

LMArena human-preference battles (55k). Battles where exactly one answer uses the construction (detector v3, family or cousin) and the vote is not a tie.

- Battles: 4917
- Construction answer wins: **55.9%** (95% CI 54.4% to 57.3%)
- Length-adjusted odds ratio: **1.01** (95% CI 0.95 to 1.07)

| construction answer's length vs the other (log ratio) | battles | win rate |
|---|---:|---:|
| -inf..-0.25 | 951 | 32.7% |
| -0.25..+0.25 | 1456 | 51.4% |
| +0.25..+inf | 2510 | 67.3% |

Share of each model's answers using the construction (models with ≥300 answers):

| model | answers | rate |
|---|---:|---:|
| gpt-4-0125-preview | 1160 | 20.5% |
| yi-34b-chat | 1447 | 15.1% |
| gpt-4-1106-preview | 7387 | 14.9% |
| qwen1.5-72b-chat | 551 | 12.3% |
| zephyr-7b-beta | 2401 | 12.0% |
| nous-hermes-2-mixtral-8x7b-dpo | 325 | 12.0% |
| gpt-4-0613 | 6165 | 11.5% |
| zephyr-7b-alpha | 412 | 11.4% |
| starling-lm-7b-alpha | 1134 | 10.7% |
| llama-2-70b-chat | 3428 | 10.6% |
| mixtral-8x7b-instruct-v0.1 | 3545 | 10.3% |
| tulu-2-dpo-70b | 1200 | 10.2% |
| llama-2-13b-chat | 2607 | 10.0% |
| codellama-34b-instruct | 1474 | 10.0% |
| solar-10.7b-instruct-v1.0 | 899 | 9.8% |
| mistral-medium | 3315 | 9.5% |
| wizardlm-13b | 1580 | 9.4% |
| wizardlm-70b | 1644 | 9.4% |
| llama-2-7b-chat | 1793 | 9.3% |
| vicuna-33b | 3720 | 8.4% |
| chatglm-6b | 1261 | 8.0% |
| llama2-70b-steerlm-chat | 667 | 7.8% |
| openhermes-2.5-mistral-7b | 952 | 7.4% |
| stripedhyena-nous-7b | 914 | 7.2% |
| deepseek-llm-67b-chat | 795 | 7.2% |
