# Does preference data reward "not X but Y"? (exploratory)

Chosen vs rejected answers in public preference datasets. Detector v3, family or cousin.

| dataset | pairs | chosen using it | rejected using it | one-sided pairs | construction answer chosen | at equal length |
|---|---:|---:|---:|---:|---:|---:|
| UltraFeedback (GPT-4 ratings) | 60,917 | 7.2% | 5.7% | 6,515 | **57.3%** (56.0–58.5) | 51.1% (49.7–52.5) |
| OLMo 2 preference mix | 100,000 | 11.1% | 9.6% | 13,804 | **55.4%** (54.5–56.2) | 51.5% (50.5–52.4) |

50% would mean the judges are indifferent. "At equal length" removes the advantage longer answers get, since answers that use the construction tend to be longer.
