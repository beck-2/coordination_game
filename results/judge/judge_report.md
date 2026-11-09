# LLM judge vs regex (reasoning traces)

Judge: glm-5.2, T=0. Sample A: 140 bad-apple traces (overnight1, rounds >= 2, stratified 10 early + 10 late per model). Cost: 0.58 credits.

## Agreement (sample A)

| measure | regex rate | judge rate | agreement | Cohen's kappa |
|---|---|---|---|---|
| mentions holder | 0.99 | 0.99 | 1.00 | 1.00 |
| holder described as fixed/stubborn | 0.74 | 0.76 | 0.78 | 0.42 |
| holder not a real agent | 0.01 | 0.01 | 0.97 | -0.01 |
| considers switching to holder | 0.43 | 0.58 | 0.79 | 0.59 |

## Judge-based rates by model (sample A; n per model ~20)

| model | mentions holder | fixed/stubborn | not a real agent | considers switching | intent follow / hold / other | intent matches actual pick |
|---|---|---|---|---|---|---|
| glm-5.2 | 1.00 | 0.70 | 0.00 | 0.80 | 16/4/0 | 1.00 |
| glm-53-flash | 1.00 | 0.90 | 0.00 | 0.85 | 15/5/0 | 1.00 |
| deepseek-v4-flash | 1.00 | 0.50 | 0.00 | 0.60 | 11/9/0 | 1.00 |
| qwen3.8-27b | 1.00 | 0.70 | 0.10 | 0.90 | 11/9/0 | 1.00 |
| qwen3.5-397b | 1.00 | 1.00 | 0.00 | 0.85 | 1/19/0 | 1.00 |
| gemma-4-31b-it | 0.95 | 0.65 | 0.00 | 0.05 | 0/20/0 | 1.00 |
| gemma-4-26b-a4b-it | 0.95 | 0.85 | 0.00 | 0.00 | 0/20/0 | 1.00 |

## Game-6 anchor after the switch (sample B; glm-5.2)

| condition | n | old label | new label | neither |
|---|---|---|---|---|
| swap | 20 | 20 | 0 | 0 |
| relabel | 20 | 19 | 0 | 1 |

Regex (sample B, same traces): mentions of old vs new label: swap: old 20, new 7, relabel: old 20, new 14
