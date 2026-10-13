# Per-feature attribution to P(pivot) at the truncated D1 position

**N samples:** 100 truncated 'with'-prompts (['C1', 'C2', 'C3'])  
**Baseline P(pivot) mean:** 0.3060  
**Score:** mean attribution drop × √(n prompts where feature was active).  
Positive score = ablating the feature DROPS P(pivot) → feature *promotes* the construction at the decision point.  
Negative score = ablating the feature RAISES P(pivot) → feature *suppresses* the construction.

## Top features that PROMOTE the pivot (ablation drops P(pivot))

| Rank | Feature | mean Δ | n active | score | Label |
|---:|---:|---:|---:|---:|---|
| 1 | 3223 | +0.10324 | 68 | +0.8513 | phrases conveying exceptions or negations |
| 2 | 9909 | +0.05007 | 94 | +0.4855 | references to digital technology and online interactions |
| 3 | 12898 | +0.02452 | 76 | +0.2138 | references to societal issues, particularly related to laws and margin |
| 4 | 2282 | +0.01834 | 32 | +0.1038 | specific medical terms and conditions, particularly related to studies |
| 5 | 1250 | +0.03005 | 9 | +0.0901 | various professional roles and their associated tasks or features |
| 6 | 4516 | +0.00929 | 82 | +0.0842 | phrases related to scientific measurements and their implications |
| 7 | 6759 | +0.00940 | 63 | +0.0746 | terms related to coverage and protection in the context of services or |
| 8 | 4197 | +0.01348 | 30 | +0.0738 | references to opinions, decisions, and activities around relationships |
| 9 | 7100 | +0.01305 | 27 | +0.0678 | elements related to narrative and storytelling |
| 10 | 2137 | +0.00695 | 71 | +0.0585 | elements of concern or caution in various contexts |
| 11 | 11864 | +0.00583 | 98 | +0.0577 | technical terms and phrases related to legal and procedural contexts |
| 12 | 9816 | +0.02179 | 7 | +0.0576 | topics related to legal and political accountability |
| 13 | 2706 | +0.01795 | 6 | +0.0440 | words associated with gain or loss in competitive scenarios |
| 14 | 7361 | +0.00719 | 30 | +0.0394 | statements about legal analysis and recommendations |
| 15 | 12561 | +0.01455 | 5 | +0.0325 | references to significant events or actions related to authority and g |
| 16 | 8530 | +0.00413 | 55 | +0.0307 | tokens related to scientific concepts and conditions, particularly foc |
| 17 | 12524 | +0.02888 | 1 | +0.0289 | data related to physical attributes and measurements |
| 18 | 2184 | +0.00716 | 16 | +0.0286 | positive attributes and expressions of appreciation |
| 19 | 13265 | +0.02702 | 1 | +0.0270 | quantitative data and metrics related to health conditions and their e |
| 20 | 12923 | +0.01273 | 4 | +0.0255 | references to urgency and priority in actions or events |
| 21 | 9863 | +0.01729 | 2 | +0.0244 | technical terms related to experimental processes and methodologies |
| 22 | 9606 | +0.00640 | 14 | +0.0239 | elements related to communication and technology usage |
| 23 | 9173 | +0.01108 | 4 | +0.0222 | keywords related to gaming and family dynamics |
| 24 | 1608 | +0.00667 | 11 | +0.0221 | legal and health-related terms associated with studies and analyses |
| 25 | 12506 | +0.00350 | 36 | +0.0210 | concepts related to problems, significant findings, and situations of  |
| 26 | 2328 | +0.00698 | 9 | +0.0210 | terms indicating limitations or deficiency in knowledge or compliance |
| 27 | 10168 | +0.01433 | 2 | +0.0203 | references to art and artists |
| 28 | 3987 | +0.01167 | 3 | +0.0202 | elements of social justice issues relating to race and accountability |
| 29 | 13321 | +0.01407 | 2 | +0.0199 | instances of the word 'design' in various contexts |
| 30 | 10701 | +0.00437 | 20 | +0.0196 | the skewed perceptions and realities of a character named Earl |
| 31 | 874 | +0.00500 | 15 | +0.0194 | phrases related to personal reflections on experiences and preferences |
| 32 | 15615 | +0.01878 | 1 | +0.0188 | words associated with statistical trends or changes in populations and |
| 33 | 10470 | +0.00838 | 5 | +0.0187 | concepts relating to assessment and evaluation processes |
| 34 | 6336 | +0.00278 | 45 | +0.0187 | terms related to legal and criminal proceedings |
| 35 | 2670 | +0.01315 | 2 | +0.0186 | technical and procedural terms related to processes and systems |
| 36 | 10821 | +0.01707 | 1 | +0.0171 | instances of numerical values and monetary amounts |
| 37 | 3019 | +0.00171 | 97 | +0.0168 | elements related to operational or procedural contexts in a structured |
| 38 | 10822 | +0.01166 | 2 | +0.0165 | indicators of significant or important concepts within the text, often |
| 39 | 12767 | +0.01119 | 2 | +0.0158 | references to historical or significant events related to change or pr |
| 40 | 10111 | +0.00332 | 21 | +0.0152 | terms related to political and social issues involving control and reg |
| 41 | 1692 | +0.00148 | 100 | +0.0148 | legal and technical terminology related to statutes and inventions |
| 42 | 1882 | +0.01435 | 1 | +0.0143 | concepts related to management |
| 43 | 7976 | +0.01408 | 1 | +0.0141 | terms related to subscription services and their usage |
| 44 | 1959 | +0.00477 | 8 | +0.0135 | terms related to emotional states and interpersonal relationships |
| 45 | 347 | +0.00775 | 3 | +0.0134 | phrases and concepts related to accountability and compliance |
| 46 | 14401 | +0.00767 | 3 | +0.0133 | elements related to emotional depth and complexity in narratives |
| 47 | 7180 | +0.00907 | 2 | +0.0128 | references to various forms of art and cultural commentary |
| 48 | 3251 | +0.00896 | 2 | +0.0127 | terms related to coldness |
| 49 | 7655 | +0.01265 | 1 | +0.0126 | elements related to complex themes and relationships in narratives |
| 50 | 8958 | +0.00230 | 30 | +0.0126 | topics related to food, fashion, and travel experiences |
| 51 | 11883 | +0.00875 | 2 | +0.0124 | expressions related to apologies and requests for forgiveness |
| 52 | 7647 | +0.00477 | 6 | +0.0117 | terms related to roles and relationships in partnerships and affiliati |
| 53 | 12555 | +0.00251 | 21 | +0.0115 | phrases related to accountability and social justice issues |
| 54 | 4949 | +0.01126 | 1 | +0.0113 | references to family members |
| 55 | 8978 | +0.00775 | 2 | +0.0110 | statements about medical and scientific findings involving treatments  |
| 56 | 9042 | +0.00540 | 4 | +0.0108 | concepts related to teamwork and collaboration |
| 57 | 8266 | +0.00237 | 20 | +0.0106 | phrases related to emergency response and community safety issues |
| 58 | 2830 | +0.00738 | 2 | +0.0104 | elements and keywords related to health, medical conditions, and socie |
| 59 | 3807 | +0.00733 | 2 | +0.0104 | references to specific settings or situations involving food or social |
| 60 | 8206 | +0.00579 | 3 | +0.0100 | mentions of customers and customer support |
| 61 | 2230 | +0.00198 | 25 | +0.0099 | references to causality and violation of rules or expectations |
| 62 | 14300 | +0.00551 | 3 | +0.0095 | references to specific films, actors, and film-related awards |
| 63 | 8750 | +0.00294 | 10 | +0.0093 | themes related to social awareness and community issues |
| 64 | 15160 | +0.00253 | 13 | +0.0091 | terms related to injury and their consequences |
| 65 | 838 | +0.00525 | 3 | +0.0091 | specific entities and locations related to places and roles |
| 66 | 12035 | +0.00339 | 7 | +0.0090 | statements related to the existence or presence of phenomena or condit |
| 67 | 13191 | +0.00141 | 40 | +0.0089 | concepts related to community engagement and social responsibility |
| 68 | 4238 | +0.00631 | 2 | +0.0089 | references to luck and fortunate circumstances |
| 69 | 13351 | +0.00609 | 2 | +0.0086 | entities related to events and competitions involving teams or clubs |
| 70 | 3940 | +0.00235 | 13 | +0.0085 | references to individuals in various roles or contexts |
| 71 | 6553 | +0.00189 | 19 | +0.0082 | references to decision-making and its consequences |
| 72 | 323 | +0.00572 | 2 | +0.0081 | data related to climate and precipitation patterns |
| 73 | 10640 | +0.00155 | 25 | +0.0077 | information about critiques and evaluations in research studies |
| 74 | 6631 | +0.00077 | 95 | +0.0075 | the beginning of a text or important markers in a document |
| 75 | 8461 | +0.00742 | 1 | +0.0074 | references to social relationships, particularly among groups or indiv |
| 76 | 15234 | +0.00325 | 5 | +0.0073 | terms related to sports competition and performance metrics |
| 77 | 67 | +0.00511 | 2 | +0.0072 | references to levels of difficulty in games, learning environments, or |
| 78 | 13500 | +0.00720 | 1 | +0.0072 | specific height measurements and properties in a document |
| 79 | 6713 | +0.00509 | 2 | +0.0072 | specific technical terms and mechanisms related to communication syste |
| 80 | 15908 | +0.00716 | 1 | +0.0072 | references to significant events or changes impacting individuals or c |
| 81 | 7332 | +0.00504 | 2 | +0.0071 | references to Christmas and Easter celebrations |
| 82 | 5472 | +0.00223 | 10 | +0.0070 | mentions of physical activities, particularly those related to movemen |
| 83 | 2719 | +0.00121 | 32 | +0.0068 | references to arguments, complaints, or actions pertaining to rights a |
| 84 | 14862 | +0.00337 | 4 | +0.0067 | references to cooperation and community involvement |
| 85 | 14404 | +0.00656 | 1 | +0.0066 | phrases or sentences involving expressions of affection or friendship |
| 86 | 892 | +0.00645 | 1 | +0.0065 | explicit references to food |
| 87 | 14088 | +0.00285 | 5 | +0.0064 | phrases indicating legal proceedings and court-related language |
| 88 | 15300 | +0.00276 | 5 | +0.0062 | information related to business practices and customer service |
| 89 | 14491 | +0.00606 | 1 | +0.0061 | entities related to combat and technology |
| 90 | 11839 | +0.00428 | 2 | +0.0061 | references to foundations or organizations and their contributions or  |
| 91 | 15816 | +0.00428 | 2 | +0.0061 | phrases related to lengths, measurements, and optical characteristics  |
| 92 | 635 | +0.00589 | 1 | +0.0059 | terms and phrases related to innovation |
| 93 | 8832 | +0.00585 | 1 | +0.0059 | phrases related to dynamics and processes in biological or environment |
| 94 | 13792 | +0.00332 | 3 | +0.0058 | expressions of joy and affection, particularly smiles and laughter |
| 95 | 952 | +0.00562 | 1 | +0.0056 | temperature ranges and measurements, particularly in Celsius and Fahre |
| 96 | 14253 | +0.00212 | 7 | +0.0056 | references to specific books and their authors |
| 97 | 3498 | +0.00559 | 1 | +0.0056 | references to various forms of assessments or analyses related to film |
| 98 | 13529 | +0.00558 | 1 | +0.0056 | mentions of public figures and their associated actions or statements  |
| 99 | 2918 | +0.00226 | 6 | +0.0055 | punctuation marks and their relationship to conversational tone and se |
| 100 | 5247 | +0.00390 | 2 | +0.0055 | phrases and words related to evaluation and judgment |

## Top features that SUPPRESS the pivot (ablation raises P(pivot))

| Rank | Feature | mean Δ | n active | score | Label |
|---:|---:|---:|---:|---:|---|
| 1 | 8820 | -0.04331 | 30 | -0.2372 | references to ethical concerns and issues related to accountability |
| 2 | 162 | -0.08688 | 3 | -0.1505 | phrases indicating successful performance or effectiveness |
| 3 | 7136 | -0.03046 | 19 | -0.1328 | terms related to biological and medical processes |
| 4 | 10072 | -0.02118 | 37 | -0.1289 | adjectives describing qualities and characteristics |
| 5 | 6143 | -0.01288 | 89 | -0.1215 | phrases related to medical conditions and treatments |
| 6 | 13130 | -0.03550 | 11 | -0.1177 | terms related to legal proceedings and the concept of prior knowledge  |
| 7 | 12257 | -0.04323 | 7 | -0.1144 | terms related to communication and conflict resolution |
| 8 | 1523 | -0.11174 | 1 | -0.1117 | references to "hill" in various contexts |
| 9 | 1948 | -0.09957 | 1 | -0.0996 | references to the action of selecting or retrieving items or informati |
| 10 | 4028 | -0.03711 | 7 | -0.0982 | phrases related to assessment and accountability in various contexts |
| 11 | 1869 | -0.02802 | 12 | -0.0971 | statements related to evidence and conclusions in analytical texts |
| 12 | 9161 | -0.06731 | 2 | -0.0952 | phrases highlighting positive accomplishments and contributions |
| 13 | 13853 | -0.02582 | 13 | -0.0931 | discussions around beliefs and decisions |
| 14 | 3645 | -0.01555 | 29 | -0.0838 | phrases related to measurements or characteristics of products and exp |
| 15 | 2235 | -0.04100 | 4 | -0.0820 | comparative adjectives and their contexts related to ease or difficult |
| 16 | 139 | -0.02711 | 8 | -0.0767 | references to actions and their consequences, emphasizing occurrences  |
| 17 | 7526 | -0.05339 | 2 | -0.0755 | instances of the word "pass" and its variations in various contexts |
| 18 | 13600 | -0.04065 | 3 | -0.0704 | content related to essays or articles, particularly focusing on their  |
| 19 | 12790 | -0.01710 | 13 | -0.0616 | contexts involving work, labor, or employment-related discussions |
| 20 | 9564 | -0.01570 | 15 | -0.0608 | concepts related to legal judgments and court instructions |
| 21 | 8836 | -0.01723 | 12 | -0.0597 | themes related to opportunities and abilities, particularly in the con |
| 22 | 11504 | -0.01839 | 10 | -0.0581 | phrases related to comparisons and evaluations of size and value in co |
| 23 | 6412 | -0.03321 | 3 | -0.0575 | words related to familiarity and interactions with experiences |
| 24 | 9251 | -0.02337 | 6 | -0.0573 | terms related to medical procedures and protocols |
| 25 | 15015 | -0.01975 | 8 | -0.0559 | phrases related to significant actions or impacts regarding government |
| 26 | 13762 | -0.00623 | 70 | -0.0521 | phrases related to legal and ethical accountability in various context |
| 27 | 12992 | -0.01387 | 14 | -0.0519 | descriptors of time-related qualities and intensities |
| 28 | 4672 | -0.02555 | 4 | -0.0511 | phrases relating to performance and evaluation outcomes |
| 29 | 9790 | -0.01540 | 11 | -0.0511 | elements related to progress, improvement, and negotiation processes |
| 30 | 15910 | -0.02922 | 3 | -0.0506 | concepts and ideas related to responsibilities and observations |
| 31 | 8080 | -0.04962 | 1 | -0.0496 | references to mountains or mountainous terrain |
| 32 | 15509 | -0.00584 | 70 | -0.0489 | words likely to be near the end of sentences |
| 33 | 13339 | -0.01115 | 19 | -0.0486 | expressions of comparative status and confidence, particularly focused |
| 34 | 1813 | -0.04834 | 1 | -0.0483 | occurrences of the word "feature" and related terms in discussions of  |
| 35 | 4162 | -0.02121 | 5 | -0.0474 | verbs that convey change or transition |
| 36 | 2718 | -0.04540 | 1 | -0.0454 | positive superlative adjectives indicating quality |
| 37 | 13584 | -0.02252 | 4 | -0.0450 | words of acceptance and phrases emphasizing unconditional perspectives |
| 38 | 6303 | -0.01087 | 17 | -0.0448 | terminology related to comparisons and standards, particularly in eval |
| 39 | 8428 | -0.01826 | 6 | -0.0447 | terms related to popularity and social dynamics |
| 40 | 16285 | -0.00928 | 21 | -0.0425 | nouns related to documentation and procedures |
| 41 | 14113 | -0.02103 | 4 | -0.0421 | references to historical events and related criticisms of power struct |
| 42 | 9404 | -0.04159 | 1 | -0.0416 | terms related to recovery and rehabilitation |
| 43 | 13703 | -0.03996 | 1 | -0.0400 | references to the color pink and its various associations |
| 44 | 6868 | -0.01629 | 6 | -0.0399 | technical details related to ability and functions associated with equ |
| 45 | 14641 | -0.01961 | 4 | -0.0392 | phrases indicative of accountability and honesty in discourse |
| 46 | 7973 | -0.01597 | 6 | -0.0391 | locations, conditions, or features related to specific settings or env |
| 47 | 5055 | -0.02747 | 2 | -0.0389 | the word "short" and related terms or phrases indicating brevity |
| 48 | 2644 | -0.01293 | 9 | -0.0388 | instances of political or social action and accountability |
| 49 | 11370 | -0.03717 | 1 | -0.0372 | the word "direct" and its variations in various contexts |
| 50 | 7600 | -0.00916 | 16 | -0.0366 | specific named entities and categories |
| 51 | 182 | -0.01462 | 6 | -0.0358 | actions related to buying, selling, and changes in ownership or status |
| 52 | 8939 | -0.03579 | 1 | -0.0358 | variations of the verb "run." |
| 53 | 9768 | -0.00337 | 100 | -0.0337 | terms related to control and authority, particularly in political or s |
| 54 | 15144 | -0.02363 | 2 | -0.0334 | words associated with legal proceedings and consequences |
| 55 | 5083 | -0.02352 | 2 | -0.0333 | terms related to winning, victories, and success |
| 56 | 2501 | -0.01092 | 9 | -0.0328 | terms related to reaction and response |
| 57 | 2660 | -0.02275 | 2 | -0.0322 | phrases related to long-term duration or processes |
| 58 | 16295 | -0.01828 | 3 | -0.0317 | significant terms and phrases related to scientific studies, particula |
| 59 | 10730 | -0.01582 | 4 | -0.0316 | references to loss or the act of losing in various contexts |
| 60 | 5958 | -0.03154 | 1 | -0.0315 | words related to working or labor |
| 61 | 3692 | -0.01268 | 6 | -0.0311 | concepts related to quality and accountability |
| 62 | 7687 | -0.01383 | 5 | -0.0309 | words and phrases related to software installation issues and successe |
| 63 | 14980 | -0.01541 | 4 | -0.0308 | phrases related to responsibility and careful use of language |
| 64 | 5505 | -0.02178 | 2 | -0.0308 | connections or transitions related to conditions and their impacts |
| 65 | 12937 | -0.01250 | 6 | -0.0306 | information related to transparency and visibility in communication |
| 66 | 4476 | -0.02166 | 2 | -0.0306 | occurrences of the word "quick" and its variations |
| 67 | 3044 | -0.03023 | 1 | -0.0302 | terms related to heat and heating processes |
| 68 | 4090 | -0.00511 | 33 | -0.0294 | negative sentiments and descriptors related to experiences |
| 69 | 13652 | -0.00437 | 45 | -0.0293 | references to feelings of aspiration or existential thoughts |
| 70 | 2763 | -0.02929 | 1 | -0.0293 | terminology related to scaling in various contexts |
| 71 | 9454 | -0.02026 | 2 | -0.0287 | references to tools and tool-related terms |
| 72 | 12702 | -0.02851 | 1 | -0.0285 | references to theories and theoretical concepts |
| 73 | 4744 | -0.00779 | 13 | -0.0281 | words indicating positive aspirations or difficult experiences |
| 74 | 765 | -0.01589 | 3 | -0.0275 | phrases related to requests for information and availability of docume |
| 75 | 5746 | -0.01336 | 4 | -0.0267 | references to education and professional development activities |
| 76 | 14609 | -0.01059 | 6 | -0.0259 | terms and phrases related to protests and demonstrations |
| 77 | 10897 | -0.00915 | 8 | -0.0259 | themes related to adaptation and personal struggles |
| 78 | 13474 | -0.00562 | 21 | -0.0258 | occupations or professional roles, particularly those requiring specia |
| 79 | 13582 | -0.00469 | 30 | -0.0257 | concepts related to technical implementations and structures in progra |
| 80 | 11383 | -0.01802 | 2 | -0.0255 | verbs indicating caution or reminders |
| 81 | 3433 | -0.01765 | 2 | -0.0250 | terms related to actions and suggestions in a technical or programming |
| 82 | 15361 | -0.00453 | 30 | -0.0248 | elements related to governance and political issues |
| 83 | 15604 | -0.01009 | 6 | -0.0247 | phrases related to algorithms and their performance metrics |
| 84 | 9549 | -0.00331 | 55 | -0.0245 | descriptive terms related to readiness and completion in various conte |
| 85 | 6071 | -0.02448 | 1 | -0.0245 | geographical locations and references to movement or travel |
| 86 | 14966 | -0.01723 | 2 | -0.0244 | phrases related to communication and requests for support |
| 87 | 6833 | -0.00525 | 20 | -0.0235 | expressions of readiness, capability, and emotional states relating to |
| 88 | 8032 | -0.01355 | 3 | -0.0235 | references to "late" or "lateness" |
| 89 | 8651 | -0.02341 | 1 | -0.0234 | terms related to political institutions and membership |
| 90 | 14383 | -0.02273 | 1 | -0.0227 | discussions related to political accountability and accusations of mal |
| 91 | 156 | -0.01575 | 2 | -0.0223 | statistical tests and procedures used in data analysis |
| 92 | 13069 | -0.01234 | 3 | -0.0214 | various methods, techniques, and strategies used in scientific researc |
| 93 | 9049 | -0.00750 | 8 | -0.0212 | action-oriented phrases indicating progress or development |
| 94 | 14750 | -0.01496 | 2 | -0.0212 | topics related to various uses and effects, particularly in the contex |
| 95 | 6802 | -0.01485 | 2 | -0.0210 | phrases related to weather and outdoor conditions |
| 96 | 202 | -0.00658 | 10 | -0.0208 | descriptions of skills and capabilities |
| 97 | 9509 | -0.01187 | 3 | -0.0206 | references to administrative processes and terminology |
| 98 | 4740 | -0.01187 | 3 | -0.0206 | words and phrases associated with winning and competition |
| 99 | 1275 | -0.02049 | 1 | -0.0205 | terms related to size, ranking, and excellence in various contexts |
| 100 | 10697 | -0.01429 | 2 | -0.0202 | references to books and related terms |