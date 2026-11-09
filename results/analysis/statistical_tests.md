# Statistical tests

Unit: the session (agents keep memory across the 10 games, so games and calls are not independent). Outcomes are session-level summaries. Non-converged games are failures; where a time is needed they count as 15 rounds (a lower bound). Tests two-sided; exact where available; Holm correction within each family; 95% CIs are percentile bootstrap over sessions (10,000 resamples, seed 0). Per-model tests are exploratory. Two rounds is the minimum possible time to consensus in a holdout game (round 1 is blind and the target is chosen to be unpicked in round 1).

## 1. Holdout vs control, per model

Session mean rounds (cap 15) and convergence rate.

| model | n holdout / control | mean rounds holdout | control | difference [95% CI] | rank-biserial r | p (Mann–Whitney) | p Holm | convergence rate holdout | control |
|---|---|---|---|---|---|---|---|---|---|
| glm-5.2 | 10 / 5 | 2.75 | 1.72 | +1.03 [+0.89, +1.18] | 1.00 | <0.001 | 0.005 | 1.00 | 1.00 |
| glm-53-flash | 10 / 5 | 2.83 | 1.54 | +1.29 [+1.08, +1.48] | 1.00 | <0.001 | 0.005 | 1.00 | 1.00 |
| deepseek-v4-flash | 10 / 5 | 3.93 | 1.94 | +1.99 [+1.61, +2.40] | 1.00 | <0.001 | 0.005 | 0.99 | 1.00 |
| qwen3.8-27b | 10 / 5 | 6.19 | 2.02 | +4.17 [+3.68, +4.72] | 1.00 | <0.001 | 0.005 | 0.87 | 1.00 |
| qwen3.5-397b | 6 / 5 | 15.00 | 1.92 | +13.08 [+12.98, +13.20] | 1.00 | 0.004 | 0.005 | 0.00 | 1.00 |
| gemma-4-31b-it | 10 / 5 | 14.65 | 2.54 | +12.11 [+11.62, +12.57] | 1.00 | <0.001 | 0.005 | 0.03 | 1.00 |
| gemma-4-26b-a4b-it | 10 / 5 | 15.00 | 2.16 | +12.84 [+12.76, +12.94] | 1.00 | <0.001 | 0.005 | 0.00 | 1.00 |

Note: with 10 vs 5 sessions the smallest exact two-sided p is 2/3003 ≈ 0.0007 (Holm across 7 models: 0.005).

## 2. Differences among models (holdout arm)

- session mean rounds (cap 15): Kruskal–Wallis H = 61.7, df = 6, p = <0.001; n per model = glm-5.2 10, glm-53-flash 10, deepseek-v4-flash 10, qwen3.8-27b 10, qwen3.5-397b 6, gemma-4-31b-it 10, gemma-4-26b-a4b-it 10 (qwen3.5-397b n = 6).
- round-2 holdout adoption, games 6–10: Kruskal–Wallis H = 60.3, df = 6, p = <0.001; n per model = glm-5.2 10, glm-53-flash 10, deepseek-v4-flash 10, qwen3.8-27b 10, qwen3.5-397b 6, gemma-4-31b-it 10, gemma-4-26b-a4b-it 10 (qwen3.5-397b n = 6).
- Pairwise Mann–Whitney on adoption (games 6–10), Holm over 21 pairs; significant: glm-5.2 vs deepseek-v4-flash (0.007); glm-5.2 vs qwen3.8-27b (0.001); glm-5.2 vs qwen3.5-397b (0.002); glm-5.2 vs gemma-4-31b-it (<0.001); glm-5.2 vs gemma-4-26b-a4b-it (<0.001); glm-53-flash vs deepseek-v4-flash (0.007); glm-53-flash vs qwen3.8-27b (0.001); glm-53-flash vs qwen3.5-397b (0.002); glm-53-flash vs gemma-4-31b-it (<0.001); glm-53-flash vs gemma-4-26b-a4b-it (<0.001); deepseek-v4-flash vs qwen3.8-27b (0.002); deepseek-v4-flash vs qwen3.5-397b (0.009); deepseek-v4-flash vs gemma-4-31b-it (0.001); deepseek-v4-flash vs gemma-4-26b-a4b-it (0.001); qwen3.8-27b vs gemma-4-31b-it (0.029); qwen3.8-27b vs gemma-4-26b-a4b-it (0.015).
- Pairs among {qwen3.5-397b, gemma-4-31b-it, gemma-4-26b-a4b-it} are not distinguishable (all at or near zero adoption).

| model | n | adoption games 6–10 [95% CI] | largest competing share, R2 (mean) | first game at 2 rounds (median) | games converged |
|---|---|---|---|---|---|
| glm-5.2 | 10 | 1.00 [1.00, 1.00] | 0.20 | 4.0 | 100/100 |
| glm-53-flash | 10 | 1.00 [1.00, 1.00] | 0.19 | 4.0 | 100/100 |
| deepseek-v4-flash | 10 | 0.82 [0.73, 0.92] | 0.47 | 6.5 | 99/100 |
| qwen3.8-27b | 10 | 0.26 [0.14, 0.37] | 0.84 | never | 87/100 |
| qwen3.5-397b | 6 | 0.01 [0.00, 0.04] | 0.96 | never | 0/60 |
| gemma-4-31b-it | 10 | 0.00 [0.00, 0.01] | 0.87 | never | 3/100 |
| gemma-4-26b-a4b-it | 10 | 0.00 [0.00, 0.00] | 0.98 | never | 0/100 |

## 3. Learning across games (holdout arm)

Per session: Spearman ρ of rounds (cap 15) on game index; Wilcoxon signed-rank of session ρ against 0; game 1 vs game 10 paired Wilcoxon. Sessions with no variation (all games at the cap) have undefined ρ and are reported as such.

| model | n | sessions with variation | mean ρ [95% CI] | p (ρ ≠ 0) | game 1 → 10 rounds (mean) | p (1 vs 10) |
|---|---|---|---|---|---|---|
| glm-5.2 | 10 | 10 | -0.81 [-0.85, -0.77] | 0.002 | 6.6 → 2.0 | 0.002 |
| glm-53-flash | 10 | 10 | -0.77 [-0.81, -0.73] | 0.002 | 8.1 → 2.0 | 0.002 |
| deepseek-v4-flash | 10 | 10 | -0.90 [-0.93, -0.87] | 0.002 | 10.6 → 2.4 | 0.002 |
| qwen3.8-27b | 10 | 10 | -0.90 [-0.94, -0.86] | 0.002 | 15.0 → 3.7 | 0.002 |
| qwen3.5-397b | 6 | 0 | n/a | n/a | 15.0 → 15.0 | n/a |
| gemma-4-31b-it | 10 | 3 | 0.10 [-0.52, 0.52] | 1.000 | 13.9 → 13.8 | 1.000 |
| gemma-4-26b-a4b-it | 10 | 0 | n/a | n/a | 15.0 → 15.0 | n/a |

## 4. Leader votes (shuffled display order = primary; sorted order = superseded legacy)

Vote share for the holdout (holdout arm) and, as a baseline, for the agent with the same seeded ID in control sessions (no holdout present). One-sample Wilcoxon of session shares vs 1/10.

| model | holdout arm: share for holdout [95% CI] | won outright | p vs 0.1 | control: share for same-ID agent [95% CI] |
|---|---|---|---|---|
| glm-5.2 | 1.00 [1.00, 1.00] | 10/10 | 0.002 | 0.00 [0.00, 0.00] |
| glm-53-flash | 1.00 [1.00, 1.00] | 10/10 | 0.002 | 0.02 [0.00, 0.06] |
| deepseek-v4-flash | 1.00 [1.00, 1.00] | 10/10 | 0.002 | 0.04 [0.00, 0.08] |
| qwen3.8-27b | 0.60 [0.41, 0.79] | 6/10 | 0.002 | 0.04 [0.00, 0.12] |
| qwen3.5-397b | 0.00 [0.00, 0.00] | 0/6 | 0.031 | 0.16 [0.00, 0.48] |
| gemma-4-31b-it | 0.00 [0.00, 0.00] | 0/10 | 0.002 | 0.00 [0.00, 0.00] |
| gemma-4-26b-a4b-it | 0.00 [0.00, 0.00] | 0/10 | 0.002 | 0.08 [0.02, 0.14] |

Across holdout sessions (n = 66): Spearman ρ(adoption in games 6–10, holdout vote share) = 0.92, p = <0.001.

### Residual label and position effects (control sessions, shuffled order)

- Share of votes for Agent 1: mean 0.26 (chance 0.10), one-sample Wilcoxon p = 0.067; Agent 1 won outright in 10/35 sessions (legacy sorted order: 23/35).
- Share of votes for the agent each voter saw listed first: mean 0.29 (chance 0.10), p = <0.001.
- Mean numeric label voted for: 4.98 (uniform choice: 5.5).
- Reading: randomizing the order removed the systematic Agent-1 advantage (one fixed label favored by everyone). Voters still favor whoever is listed first in their own view, well above chance, but because each voter's first-listed agent differs, that position bias now scatters votes instead of concentrating them. The remaining Agent-1 excess is not significant at session level; low-label salience cannot be ruled out with 35 sessions.

## 5. Exploratory: in-game chat with an LLM confederate (Gemma), vs the primary no-chat holdout arm

Separate experiment (different holdout: an LLM confederate that can talk, versus the silent script). Not pooled.

| model | sessions (chat) | games converged, chat | sessions (no chat) | games converged, no chat | p (Mann–Whitney on per-session converged games) |
|---|---|---|---|---|---|
| gemma-4-31b-it | 5 | 49/50 | 10 | 3/100 | <0.001 |
| gemma-4-26b-a4b-it | 5 | 12/50 | 10 | 0/100 | 0.254 |

Caveat: chat and a holdout that can announce it will not move are confounded; the claim supported is that Gemma groups followed a talking LLM holdout, not that chat alone breaks the standoff.

