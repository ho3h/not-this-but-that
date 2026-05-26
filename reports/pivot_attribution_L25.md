# Per-feature attribution to P(pivot) at the truncated D1 position

**N samples:** 40 truncated 'with'-prompts (['C1', 'C2', 'C3'])  
**Baseline P(pivot) mean:** 0.1957  
**Score:** mean attribution drop × √(n prompts where feature was active).  
Positive score = ablating the feature DROPS P(pivot) → feature *promotes* the construction at the decision point.  
Negative score = ablating the feature RAISES P(pivot) → feature *suppresses* the construction.

## Top features that PROMOTE the pivot (ablation drops P(pivot))

| Rank | Feature | mean Δ | n active | score | Label |
|---:|---:|---:|---:|---:|---|
| 1 | 12264 | +0.09765 | 32 | +0.5524 | elements of Australian culture in various contexts |
| 2 | 15807 | +0.01507 | 31 | +0.0839 | references to lists, collections, and items within a context of organi |
| 3 | 4868 | +0.04578 | 1 | +0.0458 | code related to task management and scheduling |
| 4 | 3017 | +0.00817 | 27 | +0.0425 | phrases related to requirements and necessity in various contexts |
| 5 | 7602 | +0.04017 | 1 | +0.0402 | programming constructs and structures related to data handling and API |
| 6 | 7561 | +0.02224 | 3 | +0.0385 | specific technical terms related to programming and data structures |
| 7 | 10594 | +0.00644 | 34 | +0.0375 | elements related to blogging and online content |
| 8 | 8375 | +0.02393 | 2 | +0.0338 | statistical achievements and performance details for athletes, particu |
| 9 | 13137 | +0.02773 | 1 | +0.0277 | code-related terminology and function definitions |
| 10 | 7425 | +0.02702 | 1 | +0.0270 | terms and phrases related to programming and data transformation using |
| 11 | 9231 | +0.02492 | 1 | +0.0249 | structured information and key components within a document |
| 12 | 1689 | +0.01170 | 4 | +0.0234 | terms related to gardens and gardening |
| 13 | 2385 | +0.01836 | 1 | +0.0184 | verbs related to tendency or habitual actions |
| 14 | 16369 | +0.00301 | 34 | +0.0175 | occurrences of mathematical or technical notations |
| 15 | 5478 | +0.01212 | 2 | +0.0171 | instructions related to food preparation and storage |
| 16 | 5755 | +0.00437 | 15 | +0.0169 | references to themes of memory and personal reflection |
| 17 | 14455 | +0.00958 | 3 | +0.0166 | occurrences of import statements in code snippets |
| 18 | 11277 | +0.01120 | 2 | +0.0158 | references to specific types of sandwiches |
| 19 | 14623 | +0.01096 | 2 | +0.0155 | football-related outcomes and statistics in a game context |
| 20 | 15298 | +0.00497 | 9 | +0.0149 | HTML table structure elements and their closing tags. |
| 21 | 2273 | +0.01467 | 1 | +0.0147 | keywords related to personal struggles and resilience in situations of |
| 22 | 15971 | +0.00422 | 12 | +0.0146 | HTML elements and their attributes |
| 23 | 2203 | +0.00224 | 40 | +0.0142 | content related to testing and debugging processes |
| 24 | 14593 | +0.01396 | 1 | +0.0140 | structure in programming code or documentation |
| 25 | 14925 | +0.01357 | 1 | +0.0136 | grammatical terms and structures related to language processing and sy |
| 26 | 4894 | +0.00352 | 12 | +0.0122 | Perl programming constructs related to hashes and function definitions |
| 27 | 5591 | +0.01177 | 1 | +0.0118 | phrases related to recommendations and suggestions in various contexts |
| 28 | 11525 | +0.00406 | 8 | +0.0115 | sentences that contain statements or facts |
| 29 | 146 | +0.00800 | 2 | +0.0113 | detailed information about financial disclosures and company performan |
| 30 | 16042 | +0.00275 | 15 | +0.0106 | mathematical structures and equations within the text |
| 31 | 3093 | +0.00265 | 16 | +0.0106 | phrases relating to specific legislative decisions and their outcomes |
| 32 | 5149 | +0.01050 | 1 | +0.0105 | expressions of doubt or uncertainty |
| 33 | 15861 | +0.00575 | 3 | +0.0100 | phrases related to status and state changes in processes or entities |
| 34 | 15211 | +0.00438 | 5 | +0.0098 | words ending in "y" or similar phonetic patterns |
| 35 | 16134 | +0.00964 | 1 | +0.0096 | actions in a sporting context, particularly scoring events and player  |
| 36 | 8456 | +0.00156 | 37 | +0.0095 | technical specifications and attributes of products |
| 37 | 15920 | +0.00260 | 13 | +0.0094 | references to specific locations, sites, or buildings |
| 38 | 8510 | +0.00930 | 1 | +0.0093 | the indefinite article "a" in various contexts |
| 39 | 4629 | +0.00166 | 29 | +0.0089 | forms of the verb "make" in various contexts |
| 40 | 16080 | +0.00315 | 8 | +0.0089 | references to geopolitical conflicts and territorial disputes |
| 41 | 4543 | +0.00891 | 1 | +0.0089 | phrases that indicate searching or seeking something |
| 42 | 14790 | +0.00887 | 1 | +0.0089 | occurrences of exception handling structures in code |
| 43 | 1862 | +0.00426 | 4 | +0.0085 | numeric patterns and symbols commonly used in technical or mathematica |
| 44 | 1462 | +0.00789 | 1 | +0.0079 | terms related to therapeutic applications and treatments |
| 45 | 8668 | +0.00785 | 1 | +0.0078 | fields related to software development and plugin documentation |
| 46 | 16099 | +0.00540 | 2 | +0.0076 | quantitative descriptors indicating abundance or quantity |
| 47 | 1734 | +0.00312 | 6 | +0.0076 | technical terms and classes related to programming and software develo |
| 48 | 8221 | +0.00124 | 38 | +0.0076 | terms related to medical studies, approvals, and clinical trial proces |
| 49 | 6468 | +0.00264 | 8 | +0.0075 | words and phrases related to geographic locations and agricultural pra |
| 50 | 5689 | +0.00358 | 4 | +0.0072 | terms that indicate naming or defining something |
| 51 | 14833 | +0.00163 | 18 | +0.0069 | references to "Pu" related terms, likely indicative of a software or p |
| 52 | 16203 | +0.00673 | 1 | +0.0067 | syntactic structures and punctuation, indicating a focus on the format |
| 53 | 3496 | +0.00673 | 1 | +0.0067 | special formatting characters and syntactical structures in code |
| 54 | 1227 | +0.00324 | 4 | +0.0065 | elements related to emotional distress and interpersonal connections |
| 55 | 9763 | +0.00643 | 1 | +0.0064 | key terms related to chemical analysis and environmental toxicity |
| 56 | 582 | +0.00444 | 2 | +0.0063 | quantifiers indicating abundance or a large quantity |
| 57 | 12824 | +0.00622 | 1 | +0.0062 | structured programming elements, such as classes and functions in code |
| 58 | 553 | +0.00612 | 1 | +0.0061 | phrases relating to scientific and technical processes in research and |
| 59 | 14966 | +0.00246 | 6 | +0.0060 | phrases related to communication and requests for support |
| 60 | 14294 | +0.00338 | 3 | +0.0059 | terms and structures related to programming constructs and data manage |
| 61 | 13470 | +0.00577 | 1 | +0.0058 | mentions of social media platforms and their usage |
| 62 | 14792 | +0.00330 | 3 | +0.0057 | phrases related to the need for representation and support in various  |
| 63 | 8858 | +0.00566 | 1 | +0.0057 | instances of the word "everything" and its variations, indicating a fo |
| 64 | 5660 | +0.00114 | 24 | +0.0056 | words related to signals and identifiers in various contexts |
| 65 | 8061 | +0.00555 | 1 | +0.0056 | LaTeX commands and symbols |
| 66 | 16040 | +0.00305 | 3 | +0.0053 | expressions of personal experience and subjective assessments |
| 67 | 11383 | +0.00233 | 5 | +0.0052 | verbs indicating caution or reminders |
| 68 | 14697 | +0.00257 | 4 | +0.0051 | numerical data related to survival rates and statistical results in st |
| 69 | 3868 | +0.00142 | 13 | +0.0051 | references to human creation stories and their interpretations |
| 70 | 8500 | +0.00501 | 1 | +0.0050 | code related to table view operations in iOS development |
| 71 | 12868 | +0.00203 | 6 | +0.0050 | scientific terminology related to algae and their classification |
| 72 | 6965 | +0.00102 | 24 | +0.0050 | terms related to analytical chemistry techniques and measurements |
| 73 | 13478 | +0.00078 | 40 | +0.0050 | phrases that indicate communication or connection between parties |
| 74 | 5164 | +0.00495 | 1 | +0.0049 | complex data structures or formats, particularly in programming or tec |
| 75 | 5264 | +0.00484 | 1 | +0.0048 | terms related to financial agreements and obligations |
| 76 | 5236 | +0.00477 | 1 | +0.0048 | prepositions, particularly "von", "aus", "in", "mit", and "bei" in Ger |
| 77 | 11959 | +0.00273 | 3 | +0.0047 | references to personal relationships and affiliations |
| 78 | 11659 | +0.00230 | 4 | +0.0046 | errors related to system processes and logging activities |
| 79 | 9145 | +0.00265 | 3 | +0.0046 | terms related to gene silencing and tumor suppression in biological re |
| 80 | 9739 | +0.00454 | 1 | +0.0045 | programming-related import statements and class declarations |
| 81 | 8287 | +0.00090 | 25 | +0.0045 | mathematical expressions enclosed in curly braces, particularly those  |
| 82 | 6025 | +0.00300 | 2 | +0.0042 | references to procedures and procedural terminology |
| 83 | 3238 | +0.00424 | 1 | +0.0042 | keywords related to asynchronous programming patterns in JavaScript |
| 84 | 4829 | +0.00185 | 5 | +0.0041 | references to geopolitical events and characters involved in them |
| 85 | 2715 | +0.00143 | 8 | +0.0040 | event details and organizational information related to festivals and  |
| 86 | 7505 | +0.00140 | 8 | +0.0040 | instances of the word "again" and its variations |
| 87 | 646 | +0.00105 | 14 | +0.0039 | references to pregnancy and reproductive choices, particularly concern |
| 88 | 10807 | +0.00222 | 3 | +0.0038 | terms related to icons and symbols in various contexts |
| 89 | 34 | +0.00270 | 2 | +0.0038 | positive evaluative adjectives and praise words used to describe peopl |
| 90 | 5266 | +0.00269 | 2 | +0.0038 | terms related to agriculture and land management practices |
| 91 | 7070 | +0.00268 | 2 | +0.0038 | references to technical terms and data in a scientific or programming  |
| 92 | 1844 | +0.00139 | 7 | +0.0037 | keywords and phrases related to the health impacts of smoking and toba |
| 93 | 9544 | +0.00260 | 2 | +0.0037 | elements and properties related to CSS and styling in code |
| 94 | 13111 | +0.00259 | 2 | +0.0037 | references to locations and publishing entities |
| 95 | 5115 | +0.00360 | 1 | +0.0036 | procedure and instructions in a technical context |
| 96 | 4019 | +0.00359 | 1 | +0.0036 | formal language and attributes typically found in academic publication |
| 97 | 12908 | +0.00177 | 4 | +0.0035 | references to tragic events involving individuals and their emotional  |
| 98 | 13937 | +0.00205 | 3 | +0.0035 | references to the name "Rose" in various contexts |
| 99 | 12653 | +0.00351 | 1 | +0.0035 | legal terminology related to court proceedings and judgments |
| 100 | 9376 | +0.00351 | 1 | +0.0035 | specific institutional and scientific references in academic contexts |

## Top features that SUPPRESS the pivot (ablation raises P(pivot))

| Rank | Feature | mean Δ | n active | score | Label |
|---:|---:|---:|---:|---:|---|
| 1 | 14325 | -0.06106 | 35 | -0.3613 | punctuation and special characters in a programming context |
| 2 | 10150 | -0.03478 | 40 | -0.2200 | specific numerical values or formatting cues in a structured data cont |
| 3 | 13580 | -0.02227 | 40 | -0.1409 | phrases related to tips or advice |
| 4 | 4795 | -0.01893 | 15 | -0.0733 | numeric statistics in a sports context |
| 5 | 6666 | -0.01098 | 38 | -0.0677 | elements related to regulations and guidelines concerning food and din |
| 6 | 949 | -0.01903 | 10 | -0.0602 | the verb "was" and its various forms or uses in the context of history |
| 7 | 3420 | -0.01657 | 13 | -0.0597 | conditional statements and directives in programming or scripting cont |
| 8 | 11666 | -0.01780 | 11 | -0.0590 | instances of the word "but" in various contexts |
| 9 | 4245 | -0.01373 | 13 | -0.0495 | references to the second person pronoun "you" and variations thereof |
| 10 | 8941 | -0.01012 | 18 | -0.0429 | terms related to programming structures and data types |
| 11 | 6420 | -0.03757 | 1 | -0.0376 | references to software updates and notifications |
| 12 | 117 | -0.02644 | 2 | -0.0374 | repeated instances of specific numerical or categorical markers in dat |
| 13 | 10418 | -0.00576 | 40 | -0.0364 | elements related to distribution or arrangement within a defined space |
| 14 | 14245 | -0.03203 | 1 | -0.0320 | conversational phrases expressing queries or uncertainties |
| 15 | 8099 | -0.00746 | 16 | -0.0298 | mentions of statistical data or numerical findings in a scientific con |
| 16 | 12853 | -0.02942 | 1 | -0.0294 | specific identifiers, metrics, or references related to dimensional an |
| 17 | 13432 | -0.00516 | 27 | -0.0268 | expressions of the word "kind" or related discussions about types or c |
| 18 | 4040 | -0.00457 | 34 | -0.0267 | the beginning of document sections or topics |
| 19 | 9905 | -0.00563 | 19 | -0.0246 | references to detailed information or specifics in a document |
| 20 | 12556 | -0.00508 | 21 | -0.0233 | references to quantities and measurements |
| 21 | 3627 | -0.00393 | 34 | -0.0229 | descriptions of the alignment and orientation of physical systems in a |
| 22 | 13446 | -0.00372 | 35 | -0.0220 | terms related to inner ear structure and hearing disorders |
| 23 | 7662 | -0.01548 | 2 | -0.0219 | technical terms related to engineering and project management |
| 24 | 6564 | -0.00700 | 9 | -0.0210 | mathematical symbols and notations |
| 25 | 2271 | -0.00551 | 13 | -0.0199 | references to the state of Iowa and its associated institutions or eve |
| 26 | 16121 | -0.00971 | 4 | -0.0194 | references to mobile devices and related technology |
| 27 | 1591 | -0.00792 | 6 | -0.0194 | URLs and web links |
| 28 | 453 | -0.00426 | 19 | -0.0186 | financial and annual reporting terms |
| 29 | 8002 | -0.00558 | 10 | -0.0177 | references to biological research and experimental methods |
| 30 | 5470 | -0.00978 | 3 | -0.0169 | phrases related to personal identification and circumstances surroundi |
| 31 | 12447 | -0.00713 | 5 | -0.0159 | elements related to data structure definitions and attributes in progr |
| 32 | 4916 | -0.00386 | 17 | -0.0159 | occurrences of phrases and items referenced in a sequential or singula |
| 33 | 2298 | -0.01577 | 1 | -0.0158 | references to specific characters or portions in texts |
| 34 | 13442 | -0.00589 | 6 | -0.0144 | instances of decision-making and actions taken in a context of authori |
| 35 | 4077 | -0.00687 | 4 | -0.0137 | HTML tags and structure in web development content |
| 36 | 16048 | -0.00789 | 3 | -0.0137 | question-and-answer patterns in text, particularly in interview or int |
| 37 | 94 | -0.00952 | 2 | -0.0135 | terms related to collectibles and trade items |
| 38 | 8942 | -0.00489 | 7 | -0.0129 | scores and numeric measurements in competitive sports or scientific co |
| 39 | 12703 | -0.00899 | 2 | -0.0127 | phrases indicating variability and differences in strategies, effects, |
| 40 | 3107 | -0.00615 | 4 | -0.0123 | expressions of help and appreciation |
| 41 | 5359 | -0.00317 | 15 | -0.0123 | words and phrases related to 'affect' and its variations |
| 42 | 10358 | -0.00261 | 19 | -0.0114 | discussion of results or output in a document |
| 43 | 2792 | -0.00275 | 16 | -0.0110 | references to fashion models and their activities |
| 44 | 15053 | -0.00360 | 9 | -0.0108 | terms related to removing or deleting items |
| 45 | 8562 | -0.00601 | 3 | -0.0104 | statements related to investigations and allegations of electoral misc |
| 46 | 786 | -0.00546 | 3 | -0.0095 | instances of government-related discussions or policies |
| 47 | 3936 | -0.00917 | 1 | -0.0092 | requests and appeals for interaction or engagement from the audience |
| 48 | 11793 | -0.00146 | 39 | -0.0091 | instances of document structure and formatting instructions, specifica |
| 49 | 14402 | -0.00370 | 6 | -0.0091 | structures or syntax reflecting code blocks in programming languages |
| 50 | 1426 | -0.00233 | 15 | -0.0090 | text related to numerical data and regex patterns |
| 51 | 5299 | -0.00441 | 4 | -0.0088 | terms and phrases related to fire and firefighting |
| 52 | 7479 | -0.00173 | 26 | -0.0088 | phrases related to personal experiences and choices |
| 53 | 15048 | -0.00278 | 10 | -0.0088 | occurrences of the word "the." |
| 54 | 8043 | -0.00434 | 4 | -0.0087 | spatial descriptions and movements, particularly related to sports or  |
| 55 | 12072 | -0.00260 | 11 | -0.0086 | repeated numerical patterns or specific numeric values within a techni |
| 56 | 4124 | -0.00427 | 4 | -0.0085 | instances of weapons and tools, particularly knives or sharp objects |
| 57 | 5014 | -0.00590 | 2 | -0.0083 | code annotations and auto-generated comments or directives |
| 58 | 11499 | -0.00823 | 1 | -0.0082 | references to structural modeling and its properties |
| 59 | 13139 | -0.00795 | 1 | -0.0080 | references to reasons and explanations |
| 60 | 13773 | -0.00226 | 12 | -0.0078 | mentions of personal names and entities associated with notable indivi |
| 61 | 12619 | -0.00549 | 2 | -0.0078 | contextual references to locations and environmental interactions |
| 62 | 1432 | -0.00536 | 2 | -0.0076 | references to database operations, particularly involving player-relat |
| 63 | 1853 | -0.00755 | 1 | -0.0076 | phrases and concepts related to monitoring and oversight in various co |
| 64 | 16361 | -0.00167 | 20 | -0.0075 | references to the word "we" |
| 65 | 11799 | -0.00146 | 26 | -0.0075 | JavaScript or TypeScript code snippets that involve fetching and manag |
| 66 | 11822 | -0.00332 | 5 | -0.0074 | the beginning of a document |
| 67 | 12967 | -0.00358 | 4 | -0.0072 | references to personal interests and passions |
| 68 | 15677 | -0.00703 | 1 | -0.0070 | verbs and actions related to progress and achievement |
| 69 | 237 | -0.00494 | 2 | -0.0070 | code comments and markup syntax in programming |
| 70 | 13024 | -0.00334 | 4 | -0.0067 | phrases related to evaluation and critique of procedural effectiveness |
| 71 | 8961 | -0.00658 | 1 | -0.0066 | file formats related to data reports |
| 72 | 5382 | -0.00290 | 5 | -0.0065 | terms related to archaeological findings and materials |
| 73 | 864 | -0.00359 | 3 | -0.0062 | HTML tags and associated attributes or values |
| 74 | 15202 | -0.00300 | 4 | -0.0060 | monetary values or financial symbols |
| 75 | 13825 | -0.00590 | 1 | -0.0059 | references to names and relationships in obituaries |
| 76 | 6526 | -0.00144 | 16 | -0.0058 | variances of the word "far" |
| 77 | 6452 | -0.00402 | 2 | -0.0057 | references to criminal activity and associated items related to drug p |
| 78 | 14274 | -0.00393 | 2 | -0.0056 | programming syntax related to function definitions and return statemen |
| 79 | 2263 | -0.00320 | 3 | -0.0055 | phrases related to housing or accommodation situations |
| 80 | 9287 | -0.00391 | 2 | -0.0055 | references to motorcycles and their specifications |
| 81 | 9546 | -0.00534 | 1 | -0.0053 | scientific terminology related to biological measurements and analysis |
| 82 | 775 | -0.00525 | 1 | -0.0052 | references to significant historical events or figures |
| 83 | 5919 | -0.00524 | 1 | -0.0052 | relationships and correlations in data |
| 84 | 10399 | -0.00258 | 4 | -0.0052 | numerical values associated with settings or configurations |
| 85 | 12325 | -0.00121 | 17 | -0.0050 | past tense verbs and phrases indicating claims or results |
| 86 | 13388 | -0.00248 | 4 | -0.0050 | concepts related to health, safety, and welfare implications in variou |
| 87 | 6161 | -0.00285 | 3 | -0.0049 | references to motorcycles, particularly their features and market posi |
| 88 | 2249 | -0.00243 | 4 | -0.0049 | phrases indicative of interpersonal relationships and dialogue |
| 89 | 6778 | -0.00276 | 3 | -0.0048 | numerical values and their relationships in data |
| 90 | 8048 | -0.00448 | 1 | -0.0045 | punctuation and transitional phrases that indicate connections in reas |
| 91 | 9720 | -0.00446 | 1 | -0.0045 | words and phrases addressing or referring to the reader or the audienc |
| 92 | 4244 | -0.00445 | 1 | -0.0045 | technical programming elements and data types |
| 93 | 8072 | -0.00253 | 3 | -0.0044 | terms related to mental health and societal issues |
| 94 | 10159 | -0.00218 | 4 | -0.0044 | references to emotional complexity and depth in literary descriptions |
| 95 | 13350 | -0.00414 | 1 | -0.0041 | terms related to the "Gro" prefix, particularly in environmental or ge |
| 96 | 11900 | -0.00169 | 6 | -0.0041 | various semantic interactions regarding generic programming tasks in c |
| 97 | 12593 | -0.00239 | 3 | -0.0041 | concepts related to personal journeys and individual experiences |
| 98 | 4084 | -0.00408 | 1 | -0.0041 | terms related to the evaluation of health conditions and treatments |
| 99 | 12461 | -0.00286 | 2 | -0.0040 | numerical values and references to time durations |
| 100 | 5869 | -0.00285 | 2 | -0.0040 | contexts involving a systemic or comprehensive approach to treatment a |