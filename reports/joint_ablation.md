# Joint-ablation ladder — the redundancy hypothesis

**Anchor feature:** 3223 (Neuronpedia: *phrases conveying exceptions or negations*)  
**N truncated D1 samples:** 80 (C1, C2, C3)  
**Baseline mean P(pivot):** 0.2778  
**Random-null draws per size:** 3  
**Intervention:** zero-ablate the named feature set at the SAE post-encode at the pre-pivot last position.

## The ladder (named sets vs size-matched random null)

| Condition | N feats | mean P(pivot) | mean drop | rel drop | null mean drop | null max | exceeds null max? |
|---|---:|---:|---:|---:|---:|---:|:--:|
| `single_3223` | 1 | 0.2283 | +0.0495 | +17.81% | +0.0000 | +0.0000 | ✓ |
| `attrib_top2` | 2 | 0.1738 | +0.1040 | +37.44% | +0.0000 | +0.0001 | ✓ |
| `attrib_top5` | 5 | 0.1349 | +0.1428 | +51.42% | -0.0000 | +0.0000 | ✓ |
| `attrib_top10` | 10 | 0.1073 | +0.1705 | +61.38% | +0.0000 | +0.0000 | ✓ |
| `attrib_top25` | 25 | 0.0769 | +0.2008 | +72.30% | -0.0000 | +0.0000 | ✓ |
| `decoder_neighbors10` | 10 | 0.2227 | +0.0550 | +19.80% | +0.0000 | +0.0000 | ✓ |
| `coact_partners10` | 10 | 0.2288 | +0.0489 | +17.61% | +0.0000 | +0.0000 | ✓ |
| `suppressors_top10` | 10 | 0.3200 | -0.0422 | -15.20% | +0.0000 | +0.0000 | · |

**`single_3223` reproduces Phase 4 necessity.** The ladder asks whether larger named sets continue to drop P(pivot) beyond what a size-matched random ablation would do.

## Random-null distribution by size

| Size | mean drop | std drop | max drop |
|---:|---:|---:|---:|
| 1 | +0.00000 | 0.00000 | +0.00000 |
| 2 | +0.00003 | 0.00005 | +0.00009 |
| 5 | -0.00000 | 0.00001 | +0.00001 |
| 10 | +0.00001 | 0.00001 | +0.00002 |
| 25 | -0.00002 | 0.00003 | +0.00001 |

## Named-set feature lists

### `single_3223` (n=1)
- **3223** — phrases conveying exceptions or negations

### `attrib_top2` (n=2)
- **3223** — phrases conveying exceptions or negations
- **9909** — references to digital technology and online interactions

### `attrib_top5` (n=5)
- **3223** — phrases conveying exceptions or negations
- **9909** — references to digital technology and online interactions
- **12898** — references to societal issues, particularly related to laws and marginalized groups
- **4197** — references to opinions, decisions, and activities around relationships and equality
- **6759** — terms related to coverage and protection in the context of services or products

### `attrib_top10` (n=10)
- **3223** — phrases conveying exceptions or negations
- **9909** — references to digital technology and online interactions
- **12898** — references to societal issues, particularly related to laws and marginalized groups
- **4197** — references to opinions, decisions, and activities around relationships and equality
- **6759** — terms related to coverage and protection in the context of services or products
- **2137** — elements of concern or caution in various contexts
- **9816** — topics related to legal and political accountability
- **4516** — phrases related to scientific measurements and their implications
- **1250** — various professional roles and their associated tasks or features
- **2282** — specific medical terms and conditions, particularly related to studies and outcomes

### `attrib_top25` (n=25)
- **3223** — phrases conveying exceptions or negations
- **9909** — references to digital technology and online interactions
- **12898** — references to societal issues, particularly related to laws and marginalized groups
- **4197** — references to opinions, decisions, and activities around relationships and equality
- **6759** — terms related to coverage and protection in the context of services or products
- **2137** — elements of concern or caution in various contexts
- **9816** — topics related to legal and political accountability
- **4516** — phrases related to scientific measurements and their implications
- **1250** — various professional roles and their associated tasks or features
- **2282** — specific medical terms and conditions, particularly related to studies and outcomes
- **11864** — technical terms and phrases related to legal and procedural contexts
- **7361** — statements about legal analysis and recommendations
- **8530** — tokens related to scientific concepts and conditions, particularly focusing on environment
- **12524** — data related to physical attributes and measurements
- **2184** — positive attributes and expressions of appreciation
- **8820** — references to ethical concerns and issues related to accountability
- **12923** — references to urgency and priority in actions or events
- **9606** — elements related to communication and technology usage
- **12506** — concepts related to problems, significant findings, and situations of concern
- **1608** — legal and health-related terms associated with studies and analyses
- **14401** — elements related to emotional depth and complexity in narratives
- **6336** — terms related to legal and criminal proceedings
- **347** — phrases and concepts related to accountability and compliance
- **10822** — indicators of significant or important concepts within the text, often activating for word
- **10701** — the skewed perceptions and realities of a character named Earl

### `decoder_neighbors10` (n=10)
- **3223** — phrases conveying exceptions or negations
- **11406** — specific punctuation marks and structural elements, indicating focus on formatting or code
- **5759** — references related to significant individuals or entities in various contexts
- **9816** — topics related to legal and political accountability
- **1250** — various professional roles and their associated tasks or features
- **4956** — proper nouns, especially names of people and places
- **5798** — (no label)
- **9870** — instances of reported speech or expressions of opinion
- **2830** — elements and keywords related to health, medical conditions, and societal discussions
- **8266** — phrases related to emergency response and community safety issues

### `coact_partners10` (n=10)
- **3223** — phrases conveying exceptions or negations
- **2008** — dates and related temporal expressions
- **5319** — specific days and dates
- **7128** — references to specific months and dates
- **9936** — dates and numeric sequences
- **16200** — concepts related to celestial bodies and their influence on life experiences
- **3629** — timestamps and references to specific periods or events
- **7956** — date and time references
- **9618** — expressions related to events, seasons, and community engagement
- **199** — phrases related to memorials and historical references

### `suppressors_top10` (n=10)
- **2235** — comparative adjectives and their contexts related to ease or difficulty
- **1523** — references to "hill" in various contexts
- **7136** — terms related to biological and medical processes
- **13853** — discussions around beliefs and decisions
- **6143** — phrases related to medical conditions and treatments
- **1869** — statements related to evidence and conclusions in analytical texts
- **4028** — phrases related to assessment and accountability in various contexts
- **3645** — phrases related to measurements or characteristics of products and experiences
- **139** — references to actions and their consequences, emphasizing occurrences and substances
- **8836** — themes related to opportunities and abilities, particularly in the context of decisions an

## Verdict — Hydra vs localized constellation

- attribution ladder mean drops: ['+0.0495', '+0.1040', '+0.1428', '+0.1705', '+0.2008']
- attribution ladder rel drops:  ['+17.81%', '+37.44%', '+51.42%', '+61.38%', '+72.30%']

**Partial collapse:** ablating the top-25 cluster significantly reduces P(pivot) but does not zero it. The construction is concentrated in a constellation rather than scattered; with the joint set identified, a stronger intervention (e.g. larger set, directional ablation across the cluster) is the next step.