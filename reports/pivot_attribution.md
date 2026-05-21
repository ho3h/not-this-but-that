# Per-feature attribution to P(pivot) at the truncated D1 position

**N samples:** 60 truncated 'with'-prompts (['C1', 'C2', 'C3'])  
**Baseline P(pivot) mean:** 0.2451  
**Score:** mean attribution drop × √(n prompts where feature was active).  
Positive score = ablating the feature DROPS P(pivot) → feature *promotes* the construction at the decision point.  
Negative score = ablating the feature RAISES P(pivot) → feature *suppresses* the construction.

## Top features that PROMOTE the pivot (ablation drops P(pivot))

| Rank | Feature | mean Δ | n active | score | Label |
|---:|---:|---:|---:|---:|---|
| 1 | 3223 | +0.07616 | 32 | +0.4308 | phrases conveying exceptions or negations |
| 2 | 9909 | +0.04048 | 54 | +0.2975 | references to digital technology and online interactions |
| 3 | 12898 | +0.02522 | 60 | +0.1954 | references to societal issues, particularly related to laws and margin |
| 4 | 4197 | +0.01402 | 25 | +0.0701 | references to opinions, decisions, and activities around relationships |
| 5 | 6759 | +0.00934 | 53 | +0.0680 | terms related to coverage and protection in the context of services or |
| 6 | 2137 | +0.00851 | 53 | +0.0620 | elements of concern or caution in various contexts |
| 7 | 9816 | +0.03496 | 3 | +0.0605 | topics related to legal and political accountability |
| 8 | 4516 | +0.00876 | 45 | +0.0588 | phrases related to scientific measurements and their implications |
| 9 | 1250 | +0.02367 | 5 | +0.0529 | various professional roles and their associated tasks or features |
| 10 | 2282 | +0.01843 | 8 | +0.0521 | specific medical terms and conditions, particularly related to studies |
| 11 | 11864 | +0.00587 | 60 | +0.0455 | technical terms and phrases related to legal and procedural contexts |
| 12 | 7361 | +0.00692 | 26 | +0.0353 | statements about legal analysis and recommendations |
| 13 | 8530 | +0.00465 | 45 | +0.0312 | tokens related to scientific concepts and conditions, particularly foc |
| 14 | 12524 | +0.02888 | 1 | +0.0289 | data related to physical attributes and measurements |
| 15 | 2184 | +0.00716 | 16 | +0.0286 | positive attributes and expressions of appreciation |
| 16 | 8820 | +0.01069 | 6 | +0.0262 | references to ethical concerns and issues related to accountability |
| 17 | 12923 | +0.01273 | 4 | +0.0255 | references to urgency and priority in actions or events |
| 18 | 9606 | +0.00706 | 11 | +0.0234 | elements related to communication and technology usage |
| 19 | 12506 | +0.00420 | 27 | +0.0218 | concepts related to problems, significant findings, and situations of  |
| 20 | 1608 | +0.00766 | 8 | +0.0217 | legal and health-related terms associated with studies and analyses |
| 21 | 14401 | +0.01887 | 1 | +0.0189 | elements related to emotional depth and complexity in narratives |
| 22 | 6336 | +0.00345 | 27 | +0.0179 | terms related to legal and criminal proceedings |
| 23 | 347 | +0.01175 | 2 | +0.0166 | phrases and concepts related to accountability and compliance |
| 24 | 10822 | +0.01166 | 2 | +0.0165 | indicators of significant or important concepts within the text, often |
| 25 | 10701 | +0.00412 | 16 | +0.0165 | the skewed perceptions and realities of a character named Earl |

## Top features that SUPPRESS the pivot (ablation raises P(pivot))

| Rank | Feature | mean Δ | n active | score | Label |
|---:|---:|---:|---:|---:|---|
| 1 | 2235 | -0.11953 | 1 | -0.1195 | comparative adjectives and their contexts related to ease or difficult |
| 2 | 1523 | -0.11174 | 1 | -0.1117 | references to "hill" in various contexts |
| 3 | 7136 | -0.02731 | 16 | -0.1093 | terms related to biological and medical processes |
| 4 | 13853 | -0.03025 | 9 | -0.0908 | discussions around beliefs and decisions |
| 5 | 6143 | -0.01171 | 58 | -0.0892 | phrases related to medical conditions and treatments |
| 6 | 1869 | -0.02858 | 9 | -0.0857 | statements related to evidence and conclusions in analytical texts |
| 7 | 4028 | -0.03389 | 5 | -0.0758 | phrases related to assessment and accountability in various contexts |
| 8 | 3645 | -0.01744 | 18 | -0.0740 | phrases related to measurements or characteristics of products and exp |
| 9 | 139 | -0.02415 | 7 | -0.0639 | references to actions and their consequences, emphasizing occurrences  |
| 10 | 8836 | -0.01851 | 11 | -0.0614 | themes related to opportunities and abilities, particularly in the con |
| 11 | 12790 | -0.01684 | 12 | -0.0583 | contexts involving work, labor, or employment-related discussions |
| 12 | 13584 | -0.02977 | 3 | -0.0516 | words of acceptance and phrases emphasizing unconditional perspectives |
| 13 | 8080 | -0.04962 | 1 | -0.0496 | references to mountains or mountainous terrain |
| 14 | 1813 | -0.04834 | 1 | -0.0483 | occurrences of the word "feature" and related terms in discussions of  |
| 15 | 15509 | -0.00610 | 59 | -0.0469 | words likely to be near the end of sentences |
| 16 | 13130 | -0.02331 | 4 | -0.0466 | terms related to legal proceedings and the concept of prior knowledge  |
| 17 | 13762 | -0.00590 | 60 | -0.0457 | phrases related to legal and ethical accountability in various context |
| 18 | 2718 | -0.04540 | 1 | -0.0454 | positive superlative adjectives indicating quality |
| 19 | 12992 | -0.01814 | 6 | -0.0444 | descriptors of time-related qualities and intensities |
| 20 | 13600 | -0.03040 | 2 | -0.0430 | content related to essays or articles, particularly focusing on their  |
| 21 | 13703 | -0.03996 | 1 | -0.0400 | references to the color pink and its various associations |
| 22 | 16285 | -0.00963 | 16 | -0.0385 | nouns related to documentation and procedures |
| 23 | 16295 | -0.02676 | 2 | -0.0378 | significant terms and phrases related to scientific studies, particula |
| 24 | 11370 | -0.03717 | 1 | -0.0372 | the word "direct" and its variations in various contexts |
| 25 | 7600 | -0.01045 | 12 | -0.0362 | specific named entities and categories |