# Second judge: Claude vs GLM-5.2 vs regex

Claude labels: 180/180 items (sample A 140, sample B 40). Claude annotated blind to the GLM labels, by reading the same prompts.

## Agreement (sample A)

| measure | regex | GLM | Claude | Claude–GLM agree / κ | Claude–regex agree / κ |
|---|---|---|---|---|---|
| mentions holder | 0.99 | 0.99 | 0.99 | 1.00 / 1.00 | 1.00 / 1.00 |
| holder fixed/stubborn | 0.74 | 0.76 | 0.71 | 0.93 / 0.82 | 0.74 / 0.34 |
| holder not a real agent | 0.01 | 0.01 | 0.03 | 0.96 / -0.02 | 0.96 / -0.02 |
| considers switching to holder | 0.43 | 0.58 | 0.61 | 0.96 / 0.91 | 0.79 / 0.60 |
| intent (3-way) | – | – | – | 0.99 / 0.99 | – |

## Claude-based rates by model (sample A; ~20 per model)

| model | mentions holder | fixed/stubborn | not a real agent | considers switching | intent follow / hold / other |
|---|---|---|---|---|---|
| glm-5.2 | 1.00 | 0.70 | 0.00 | 0.85 | 16/4/0 |
| glm-53-flash | 1.00 | 0.90 | 0.05 | 0.90 | 15/5/0 |
| deepseek-v4-flash | 1.00 | 0.45 | 0.00 | 0.65 | 11/9/0 |
| qwen3.8-27b | 1.00 | 0.75 | 0.05 | 0.85 | 11/9/0 |
| qwen3.5-397b | 1.00 | 0.95 | 0.10 | 0.95 | 1/19/0 |
| gemma-4-31b-it | 0.95 | 0.40 | 0.00 | 0.05 | 0/19/1 |
| gemma-4-26b-a4b-it | 0.95 | 0.85 | 0.00 | 0.00 | 0/20/0 |

## Game-6 anchor (sample B)

| condition | n | Claude old / new / neither | GLM old / new / neither | agree |
|---|---|---|---|---|
| swap | 20 | 20/0/0 | 20/0/0 | 20/20 |
| relabel | 20 | 19/0/1 | 19/0/1 | 20/20 |
