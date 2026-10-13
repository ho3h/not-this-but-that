# The chat-chosen 25 at the decision point (exploratory)

Held-out prefixes cut from normal answers in confirmation v3 (D2 prompts). P(commit) = probability the
next token starts the negation (family) or the minimiser (cousin). Drops are relative to normal.

## family (38 prefixes, baseline P(commit) 0.27)

| features switched off | drop in P(commit) |
|---|---:|
| top 1 | 46% |
| top 2 | 68% |
| top 5 | 85% |
| top 10 | 97% |
| top 25 | 98% |
| 25 random (20 draws) | mean -0.4%, max 0.7% |
| 25 firing-rate-matched (20 draws) | mean -0.1%, max 1.6% |

Leave-one-out (drop lost when feature i is left on):

- 15646: 8.1 points · phrases indicating conditions or exceptions in narratives
- 10655: 4.0 points · terms related to legal processes and court proceedings
- 10252: 2.2 points · present tense singular verbs
- 13565: 2.1 points · expressions indicating comparison or enhancement
- 1548: 1.2 points · expressions of personal giving or philanthropic actions
- 3351: 1.1 points · pronouns and their associated verbs to identify actions and statements about individuals or groups

Alone (drop when only feature i is switched off):

- 15646: 46% · phrases indicating conditions or exceptions in narratives
- 10252: 38% · present tense singular verbs
- 10655: 22% · terms related to legal processes and court proceedings
- 13565: 21% · expressions indicating comparison or enhancement
- 3351: 15% · pronouns and their associated verbs to identify actions and statements about individuals or groups
- 9775: 8% · variations of the verb "to be" in different contexts

## cousin (36 prefixes, baseline P(commit) 0.55)

| features switched off | drop in P(commit) |
|---|---:|
| top 1 | -0% |
| top 2 | -1% |
| top 5 | 91% |
| top 10 | 92% |
| top 25 | 94% |
| 25 random (20 draws) | mean 0.1%, max 0.8% |
| 25 firing-rate-matched (20 draws) | mean 0.8%, max 1.6% |

Leave-one-out (drop lost when feature i is left on):

- 13565: 91.4 points · expressions indicating comparison or enhancement
- 2913: 1.4 points · phrases related to the importance and perception of faith and religion
- 15646: 1.4 points · phrases indicating conditions or exceptions in narratives
- 10655: 0.3 points · terms related to legal processes and court proceedings
- 9775: 0.2 points · variations of the verb "to be" in different contexts
- 15980: 0.2 points · actions and intentions related to assessment and investigation in research contexts

Alone (drop when only feature i is switched off):

- 13565: 91% · expressions indicating comparison or enhancement
- 2913: 2% · phrases related to the importance and perception of faith and religion
- 1692: 2% · legal and technical terminology related to statutes and inventions
- 12767: 1% · references to historical or significant events related to change or progress
- 15980: 0% · actions and intentions related to assessment and investigation in research contexts
- 12898: 0% · references to societal issues, particularly related to laws and marginalized groups

