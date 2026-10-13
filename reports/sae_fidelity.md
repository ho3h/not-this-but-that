# SAE fidelity at layer 20 (revisits Tier 0b)

SAE: `gemma-scope-2b-pt-res-canonical/layer_20/width_16k/canonical`. VE = 1 − SSE/SST pooled over tokens.

| model / text | positions | tokens | VE | mean cos | mean L0 |
|---|---|---:|---:|---:|---:|
| gemma-2-2b, plain text | incl bos | 5083 | -9.315 | 0.923 | 179 |
| gemma-2-2b, plain text | excl bos | 5011 | 0.778 | 0.930 | 80 |
| gemma-2-2b-it, chat format | incl bos | 4222 | -10.114 | 0.911 | 203 |
| gemma-2-2b-it, chat format | excl bos | 4162 | 0.731 | 0.917 | 96 |

| model / text | CE clean | CE spliced (excl BOS) | CE spliced (incl BOS) | CE residual zeroed | loss recovered |
|---|---:|---:|---:|---:|---:|
| gemma-2-2b, plain text | 2.629 | 2.720 | 2.862 | 13.630 | 0.992 |
| gemma-2-2b-it, chat format | 4.403 | 4.160 | 3.585 | 21.327 | 1.014 |

Mean residual norm at BOS: base 2893, it 3806; other positions: base 356, it 416.
