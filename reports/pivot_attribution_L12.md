# Per-feature attribution to P(pivot) at the truncated D1 position

**N samples:** 40 truncated 'with'-prompts (['C1', 'C2', 'C3'])  
**Baseline P(pivot) mean:** 0.2621  
**Score:** mean attribution drop × √(n prompts where feature was active).  
Positive score = ablating the feature DROPS P(pivot) → feature *promotes* the construction at the decision point.  
Negative score = ablating the feature RAISES P(pivot) → feature *suppresses* the construction.

## Top features that PROMOTE the pivot (ablation drops P(pivot))

| Rank | Feature | mean Δ | n active | score | Label |
|---:|---:|---:|---:|---:|---|
| 1 | 10678 | +0.12176 | 4 | +0.2435 | phrases related to careful planning or organization in various context |
| 2 | 7507 | +0.02373 | 40 | +0.1501 | references to academic publications or authors |
| 3 | 8728 | +0.01781 | 36 | +0.1069 | phrases related to exchanges or transactions, particularly in the cont |
| 4 | 5127 | +0.01303 | 37 | +0.0793 | phrases related to providing information or guidance about procedures |
| 5 | 2565 | +0.01638 | 10 | +0.0518 | words that begin with the prefix "und" |
| 6 | 9730 | +0.01177 | 18 | +0.0499 | entities related to companies and their historical establishment |
| 7 | 620 | +0.04861 | 1 | +0.0486 | instances of object creation in code |
| 8 | 8070 | +0.04591 | 1 | +0.0459 | references to headers in a programming context |
| 9 | 13478 | +0.00949 | 22 | +0.0445 | phrases that indicate communication or connection between parties |
| 10 | 4685 | +0.01356 | 8 | +0.0383 | terms related to legal claims of discrimination and harassment |
| 11 | 1048 | +0.03805 | 1 | +0.0380 | references to standards and related concepts in various contexts |
| 12 | 5524 | +0.00730 | 26 | +0.0372 | programming-related operations and functions, particularly in a coding |
| 13 | 15766 | +0.03721 | 1 | +0.0372 | proper nouns, particularly names and locations |
| 14 | 485 | +0.01058 | 11 | +0.0351 | instances of the word "beauty" in various contexts |
| 15 | 12586 | +0.03444 | 1 | +0.0344 | mathematical symbols or notation |
| 16 | 3691 | +0.00733 | 20 | +0.0328 | code structure and class definitions in programming |
| 17 | 9340 | +0.01158 | 8 | +0.0327 | URLs and image paths |
| 18 | 4502 | +0.00572 | 29 | +0.0308 | specific vehicles and electronic devices |
| 19 | 5911 | +0.00562 | 28 | +0.0297 | instances of content related to events or announcements, particularly  |
| 20 | 11919 | +0.01191 | 6 | +0.0292 | elements related to narrative structure and continuity in storytelling |
| 21 | 7729 | +0.02857 | 1 | +0.0286 | terms related to ecological interactions and dynamics |
| 22 | 15336 | +0.02807 | 1 | +0.0281 | terms related to thermodynamic laws and processes |
| 23 | 8786 | +0.00602 | 20 | +0.0269 | references to signing or the act of endorsing something |
| 24 | 4073 | +0.01895 | 2 | +0.0268 | terms related to hypotheses and proposals in research |
| 25 | 14591 | +0.00504 | 26 | +0.0257 | corporate relationships and stock ownership among individuals |
| 26 | 5078 | +0.00481 | 27 | +0.0250 | technical terms and functionalities related to software and programmin |
| 27 | 3585 | +0.00688 | 12 | +0.0238 | elements related to programming languages and their implementations |
| 28 | 13574 | +0.01344 | 2 | +0.0190 | mentions of favorites or preferences |
| 29 | 12920 | +0.01796 | 1 | +0.0180 | references to the letter 'J' in various contexts, typically associated |
| 30 | 12332 | +0.00660 | 7 | +0.0175 | programming-related syntax and command structures |
| 31 | 6604 | +0.00506 | 11 | +0.0168 | references to sacrificial practices and their significance in culture |
| 32 | 7215 | +0.01134 | 2 | +0.0160 | mention of treaties or agreements related to peace processes |
| 33 | 14692 | +0.00605 | 7 | +0.0160 | technical or operational terms related to project management and docum |
| 34 | 8404 | +0.01584 | 1 | +0.0158 | specific hardware components and libraries related to microcontrollers |
| 35 | 2940 | +0.00869 | 3 | +0.0151 | demographic statistics related to age and living situations |
| 36 | 6215 | +0.01501 | 1 | +0.0150 | structured data elements or attributes in a markup or programming cont |
| 37 | 15286 | +0.00660 | 5 | +0.0148 | phrases related to capability or limitation |
| 38 | 8825 | +0.01462 | 1 | +0.0146 | layout parameters and specifications in coding, particularly for Andro |
| 39 | 2731 | +0.01441 | 1 | +0.0144 | elements related to programming and mathematical structures involving  |
| 40 | 12322 | +0.00710 | 4 | +0.0142 | instances of the word "introduction" and its variations |
| 41 | 355 | +0.00418 | 11 | +0.0139 | terms related to humanitarian aid and refugee issues |
| 42 | 2009 | +0.00400 | 12 | +0.0138 | phrases related to quality and performance measurements |
| 43 | 6755 | +0.00517 | 7 | +0.0137 | references to the word "Red" and its various contexts |
| 44 | 1304 | +0.00940 | 2 | +0.0133 | references to online threads and discussions |
| 45 | 5686 | +0.00939 | 2 | +0.0133 | technological and scientific concepts related to materials and enginee |
| 46 | 9913 | +0.00590 | 5 | +0.0132 | error-related messages and exceptions in API responses |
| 47 | 4790 | +0.00396 | 10 | +0.0125 | expressions related to chronic pain and its treatment options |
| 48 | 2725 | +0.01251 | 1 | +0.0125 | terms related to waste management and disposal processes |
| 49 | 11618 | +0.01244 | 1 | +0.0124 | phrases related to complex emotional states and their impact on indivi |
| 50 | 1728 | +0.01238 | 1 | +0.0124 | programming syntax elements, specifically focusing on the structure of |
| 51 | 186 | +0.00863 | 2 | +0.0122 | personal pronouns and possessive forms |
| 52 | 11264 | +0.00521 | 5 | +0.0116 | language related to causes and effects |
| 53 | 991 | +0.01131 | 1 | +0.0113 | references to academic or research institutions |
| 54 | 5878 | +0.00782 | 2 | +0.0111 | elements related to game object management and interactions |
| 55 | 15112 | +0.01087 | 1 | +0.0109 | references to specific songs or musical performances |
| 56 | 1431 | +0.01078 | 1 | +0.0108 | phrases related to planning and scheduling activities or tasks |
| 57 | 12910 | +0.00519 | 4 | +0.0104 | medical-related terminology, particularly surrounding diseases and con |
| 58 | 646 | +0.00599 | 3 | +0.0104 | references to pregnancy and reproductive choices, particularly concern |
| 59 | 16071 | +0.00597 | 3 | +0.0103 | phrases related to customer service and support |
| 60 | 1080 | +0.00730 | 2 | +0.0103 | phrases related to innovation and technology |
| 61 | 11396 | +0.00593 | 3 | +0.0103 | information related to legal roles and positions in Victoria, Australi |
| 62 | 816 | +0.01023 | 1 | +0.0102 | terms related to sexual health and modalities |
| 63 | 5898 | +0.01006 | 1 | +0.0101 | components and processes related to food preparation |
| 64 | 8286 | +0.00580 | 3 | +0.0100 | references to academic institutions and experts in research-related co |
| 65 | 14912 | +0.01001 | 1 | +0.0100 | patterns related to function definitions and method calls in programmi |
| 66 | 12564 | +0.00708 | 2 | +0.0100 | repetitive phrases related to existence or state in various contexts |
| 67 | 4861 | +0.00327 | 9 | +0.0098 | LaTeX formatting and mathematical notations |
| 68 | 6355 | +0.00489 | 4 | +0.0098 | terms related to funding and financial support within various contexts |
| 69 | 1992 | +0.00966 | 1 | +0.0097 | references to chemical or biological substances, specifically those be |
| 70 | 8715 | +0.00558 | 3 | +0.0097 | instances of the verb "pick" and its variations |
| 71 | 1547 | +0.00954 | 1 | +0.0095 | references to successful authors and the series they produce |
| 72 | 7698 | +0.00546 | 3 | +0.0094 | numerical data related to research studies |
| 73 | 8734 | +0.00663 | 2 | +0.0094 | keywords related to programming constructs and data types |
| 74 | 10774 | +0.00642 | 2 | +0.0091 | technical terms and references related to programming or coding, parti |
| 75 | 8049 | +0.00639 | 2 | +0.0090 | phrases indicating the necessity or essentiality of concepts or elemen |
| 76 | 613 | +0.00319 | 8 | +0.0090 | conversational dialogue or interactions between individuals |
| 77 | 14841 | +0.00319 | 8 | +0.0090 | words and phrases related to social advocacy and activism |
| 78 | 12846 | +0.00450 | 4 | +0.0090 | instances of the word "follow" in various contexts |
| 79 | 11239 | +0.00366 | 6 | +0.0090 | various forms of the word "den" and its derivatives |
| 80 | 16050 | +0.00400 | 5 | +0.0090 | legal terms and references related to copyright ownership and contact  |
| 81 | 7601 | +0.00514 | 3 | +0.0089 | terms related to sustainability and alternative therapies |
| 82 | 9921 | +0.00625 | 2 | +0.0088 | references to business operations and recommendations for service prov |
| 83 | 9108 | +0.00881 | 1 | +0.0088 | concepts related to salvation and forgiveness |
| 84 | 3582 | +0.00440 | 4 | +0.0088 | references to medical devices and documentation |
| 85 | 3800 | +0.00496 | 3 | +0.0086 | references to languages or terms related to linguistic studies |
| 86 | 15556 | +0.00854 | 1 | +0.0085 | punctuation marks and special characters in the text |
| 87 | 7913 | +0.00483 | 3 | +0.0084 | references to the concept of peace |
| 88 | 15109 | +0.00411 | 4 | +0.0082 | references to family relationships and children |
| 89 | 15852 | +0.00821 | 1 | +0.0082 | descriptions of flower varieties and their characteristics |
| 90 | 16228 | +0.00819 | 1 | +0.0082 | phrases related to division and fragmentation |
| 91 | 11446 | +0.00401 | 4 | +0.0080 | references to availability and access to various products and informat |
| 92 | 15755 | +0.00566 | 2 | +0.0080 | terms related to adaptability and adjusting to changing conditions |
| 93 | 12176 | +0.00757 | 1 | +0.0076 | situations involving feeling stuck or encountering obstacles |
| 94 | 5249 | +0.00755 | 1 | +0.0075 | references to God and His attributes in a theological context |
| 95 | 10679 | +0.00335 | 5 | +0.0075 | references to government, political actions, and accountability relate |
| 96 | 5120 | +0.00527 | 2 | +0.0074 | elements indicating proofs or propositions in mathematical texts |
| 97 | 1407 | +0.00428 | 3 | +0.0074 | technical terms related to audio and music production |
| 98 | 3852 | +0.00328 | 5 | +0.0073 | statements introducing the existence or presence of entities |
| 99 | 3423 | +0.00515 | 2 | +0.0073 | language related to disclaimers, limitations of liability, and non-end |
| 100 | 13353 | +0.00728 | 1 | +0.0073 | references to "bull" or "bull-related" terms |

## Top features that SUPPRESS the pivot (ablation raises P(pivot))

| Rank | Feature | mean Δ | n active | score | Label |
|---:|---:|---:|---:|---:|---|
| 1 | 12291 | -0.10839 | 1 | -0.1084 | scientific terminology and data analysis concepts |
| 2 | 11547 | -0.10386 | 1 | -0.1039 | numerical values and references to figures in scientific contexts |
| 3 | 14888 | -0.08890 | 1 | -0.0889 | economic indicators and trends related to market fluctuations and perf |
| 4 | 3374 | -0.06214 | 2 | -0.0879 | references to significant events or milestones in history |
| 5 | 519 | -0.08197 | 1 | -0.0820 | blocks of code related to programming concepts |
| 6 | 1205 | -0.08135 | 1 | -0.0813 | technical terms and phrases related to programming errors and debuggin |
| 7 | 14172 | -0.07373 | 1 | -0.0737 | elements and syntax related to programming and code structure |
| 8 | 2620 | -0.01072 | 40 | -0.0678 | references to the Middle East and related regions |
| 9 | 4632 | -0.03939 | 2 | -0.0557 | the word "be" in various forms and contexts |
| 10 | 1155 | -0.04869 | 1 | -0.0487 | references to Uganda and its historical context, particularly related  |
| 11 | 16046 | -0.03282 | 2 | -0.0464 | references to dietary intake and food diversity |
| 12 | 16173 | -0.01616 | 8 | -0.0457 | names of political parties and significant political events |
| 13 | 1105 | -0.02441 | 3 | -0.0423 | code-related terms, particularly those associated with function defini |
| 14 | 4835 | -0.03934 | 1 | -0.0393 | elements related to childhood and play experiences |
| 15 | 7102 | -0.02778 | 2 | -0.0393 | keywords and phrases related to function documentation and metadata |
| 16 | 9105 | -0.03842 | 1 | -0.0384 | references to various Christian denominations and their associated ter |
| 17 | 621 | -0.02184 | 3 | -0.0378 | code elements and structures |
| 18 | 14092 | -0.03393 | 1 | -0.0339 | phrases related to providing or presenting information, particularly i |
| 19 | 5040 | -0.01631 | 4 | -0.0326 | file paths and directory structures |
| 20 | 7517 | -0.00616 | 27 | -0.0320 | connections and associations in text regarding media influence or misi |
| 21 | 11306 | -0.00494 | 40 | -0.0313 | programming constructs and control flow statements in code |
| 22 | 7910 | -0.02930 | 1 | -0.0293 | technical terms related to HTTP request and response headers |
| 23 | 4287 | -0.02837 | 1 | -0.0284 | words related to scientific terms and findings |
| 24 | 2135 | -0.01546 | 3 | -0.0268 | technical terms related to programming or software development configu |
| 25 | 13779 | -0.01831 | 2 | -0.0259 | instances of the word "individual" and related terms describing indivi |
| 26 | 3419 | -0.02512 | 1 | -0.0251 | instances of events that imply wrongdoing or violations of justice |
| 27 | 7711 | -0.00487 | 24 | -0.0238 | references to USB connections and functionalities |
| 28 | 7657 | -0.01063 | 5 | -0.0238 | references to specific biochemical markers and methodologies related t |
| 29 | 2291 | -0.00370 | 40 | -0.0234 | Java method calls and variable manipulations in programming code |
| 30 | 8100 | -0.01572 | 2 | -0.0222 | JavaScript function declarations and their properties in code |
| 31 | 3412 | -0.00408 | 28 | -0.0216 | mentions of various types of resources |
| 32 | 1505 | -0.02086 | 1 | -0.0209 | lists and collections of items or elements in code |
| 33 | 8951 | -0.02043 | 1 | -0.0204 | punctuation and sentence structure markers |
| 34 | 4736 | -0.01019 | 4 | -0.0204 | conjunctions and phrases indicating contrast or alternatives |
| 35 | 10355 | -0.00311 | 40 | -0.0197 | terms related to health interventions and support systems |
| 36 | 5428 | -0.00523 | 14 | -0.0196 | references to criminal activities and investigative themes |
| 37 | 6069 | -0.01365 | 2 | -0.0193 | strings and formulas involving mathematical symbols |
| 38 | 6464 | -0.01364 | 2 | -0.0193 | programming constructs related to API requests and responses |
| 39 | 14768 | -0.00572 | 11 | -0.0190 | expressions indicating perception or subjective experience |
| 40 | 11271 | -0.01894 | 1 | -0.0189 | words and phrases related to improving and refining processes or metho |
| 41 | 4659 | -0.00938 | 4 | -0.0188 | words and phrases related to processes of transferring, creating, or h |
| 42 | 8902 | -0.01867 | 1 | -0.0187 | references to data structures, particularly related to pages, states,  |
| 43 | 8337 | -0.01830 | 1 | -0.0183 | references to specific dishes and ingredients in a culinary context |
| 44 | 10571 | -0.01789 | 1 | -0.0179 | references to legal terms and court proceedings |
| 45 | 1886 | -0.00632 | 8 | -0.0179 | references to medical treatment and conditions related to emergencies |
| 46 | 3057 | -0.00990 | 3 | -0.0171 | event handling functions and their parameters in JavaScript code |
| 47 | 10435 | -0.01698 | 1 | -0.0170 | mathematical functions and symbols relevant to statistical models |
| 48 | 6342 | -0.01692 | 1 | -0.0169 | words related to project management and collaboration |
| 49 | 3590 | -0.00743 | 5 | -0.0166 | references to "X" and related terms in a technical context |
| 50 | 1986 | -0.00657 | 6 | -0.0161 | references to specific scientific terminologies and statistical data r |
| 51 | 15739 | -0.01587 | 1 | -0.0159 | references to interactions or relationships between people or entities |
| 52 | 13582 | -0.00892 | 3 | -0.0155 | concepts related to technical implementations and structures in progra |
| 53 | 10345 | -0.01517 | 1 | -0.0152 | symbols and formatting used in programming or mathematical expressions |
| 54 | 13219 | -0.01484 | 1 | -0.0148 | phrases related to collaboration and networking |
| 55 | 13401 | -0.01048 | 2 | -0.0148 | programming-related constructs and error handling in code |
| 56 | 1246 | -0.00338 | 18 | -0.0144 | references to individuals in leadership or professional roles |
| 57 | 10324 | -0.01433 | 1 | -0.0143 | references to stress and its related effects |
| 58 | 12252 | -0.00996 | 2 | -0.0141 | references to academic articles and their authors |
| 59 | 13700 | -0.00253 | 30 | -0.0138 | references to species comparisons and interactions, particularly betwe |
| 60 | 7449 | -0.00962 | 2 | -0.0136 | the word 'to' and its variations in different contexts |
| 61 | 14829 | -0.01355 | 1 | -0.0136 | expressions of agreement or acknowledgment |
| 62 | 7559 | -0.01316 | 1 | -0.0132 | information related to death and familial relationships |
| 63 | 10453 | -0.00586 | 5 | -0.0131 | references to the word "fa," indicating a focus on common or notable u |
| 64 | 2416 | -0.01303 | 1 | -0.0130 | specific elements and structures commonly found in programming languag |
| 65 | 11943 | -0.00388 | 11 | -0.0129 | programming constructs related to control flow and data structure defi |
| 66 | 7159 | -0.01272 | 1 | -0.0127 | references to time frames and chronological events |
| 67 | 14119 | -0.01271 | 1 | -0.0127 | text that cites legal references and sources |
| 68 | 14358 | -0.01262 | 1 | -0.0126 | elements related to data processing and data manipulation commands |
| 69 | 11620 | -0.01230 | 1 | -0.0123 | instances related to vehicles and driving experiences |
| 70 | 9711 | -0.00547 | 5 | -0.0122 | references to specific brands of whiskey |
| 71 | 7803 | -0.01220 | 1 | -0.0122 | punctuation marks and mathematical symbols in a formal text |
| 72 | 3783 | -0.00855 | 2 | -0.0121 | year numbers, particularly those in the 2000s and 2010s. |
| 73 | 6354 | -0.00539 | 5 | -0.0121 | keywords related to business performance and management |
| 74 | 13456 | -0.01203 | 1 | -0.0120 | references to organizations, particularly charities and educational in |
| 75 | 194 | -0.01194 | 1 | -0.0119 | references to team needs and player evaluations in sports contexts |
| 76 | 12001 | -0.01182 | 1 | -0.0118 | items related to the concept of accessibility and availability of reso |
| 77 | 3788 | -0.01128 | 1 | -0.0113 | names of official institutions, organizations, or administrative bodie |
| 78 | 4953 | -0.01105 | 1 | -0.0111 | keywords related to encoding and data processing in programming |
| 79 | 10509 | -0.00293 | 14 | -0.0110 | the presence of 'not equal to' comparisons in code |
| 80 | 15689 | -0.00489 | 5 | -0.0109 | information about uncertainty and confusion regarding identity or even |
| 81 | 418 | -0.01091 | 1 | -0.0109 | expressions of willingness and preparedness |
| 82 | 1416 | -0.00758 | 2 | -0.0107 | references to brands, products, or specific entities in various contex |
| 83 | 14222 | -0.01061 | 1 | -0.0106 | scientific terminology related to procedures and properties of biologi |
| 84 | 4717 | -0.01060 | 1 | -0.0106 | statements related to football players' performance and management dec |
| 85 | 4755 | -0.00527 | 4 | -0.0105 | references to agricultural or botanical terms |
| 86 | 4808 | -0.00738 | 2 | -0.0104 | terms related to geography and regions of the world |
| 87 | 8766 | -0.00708 | 2 | -0.0100 | phrases related to online shopping and deals |
| 88 | 309 | -0.00447 | 5 | -0.0100 | various structural components related to code or programming syntax |
| 89 | 16057 | -0.00996 | 1 | -0.0100 | instances of special characters or formatting symbols |
| 90 | 4040 | -0.00445 | 5 | -0.0100 | the beginning of document sections or topics |
| 91 | 9659 | -0.00298 | 11 | -0.0099 | technical terms related to electrical signals and interrupt handling |
| 92 | 8659 | -0.00699 | 2 | -0.0099 | elements related to technical programming logic and syntax |
| 93 | 5932 | -0.00561 | 3 | -0.0097 | terms related to seeking help or assistance |
| 94 | 2766 | -0.00963 | 1 | -0.0096 | references to the Republican Party and its members |
| 95 | 10948 | -0.00963 | 1 | -0.0096 | instances of the word "as" used in various contexts |
| 96 | 8437 | -0.00962 | 1 | -0.0096 | phrases referring to recent time frames or occurrences |
| 97 | 3113 | -0.00959 | 1 | -0.0096 | clauses or phrases that introduce or describe a key idea or concept, o |
| 98 | 10969 | -0.00677 | 2 | -0.0096 | elements related to time-sensitive events or deadlines |
| 99 | 9233 | -0.00553 | 3 | -0.0096 | instances of function calls and structures related to iterating throug |
| 100 | 2315 | -0.00671 | 2 | -0.0095 | occurrences of the term 'null' in various contexts |