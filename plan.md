# Coordination Game: Experiment Plan

> The working reference for this project. Check against it before each phase.
> Items marked **[OPEN]** are not decided yet. Items marked **[DECIDED]** are settled.

## 1. Research question

Can a group of LLM agents with no communication channel other than their own past picks
converge on a single shared choice? How does one stubborn "bad apple" change that?
Specifically:

- How many iterations does convergence take?
- Over repeated games, do agents learn the bad apple's identity and conform right away?
- What does each agent's chain of thought (CoT) show about how it reasons through this?

## 2. The game

- **Players:** N = 10 agents (9 LLM agents + 1 bad apple in the bad-apple condition).
- **Board:** each game shows a set of ~20 seemingly unrelated numbers: integers (some
  repeated), negatives, fractions, decimals, irrationals, very large and very small values.
  The board is meant to make shared reasoning (focal points) hard to find.
- **Iteration loop:**
  1. Every agent picks one item from the board, all at the same time without seeing others.
  2. Every agent is shown all agents' picks, labeled by agent ID.
  3. Repeat until all N picks are the same item (converged) or an iteration cap is hit.
- **Session:** the same group plays G = 10 games in a row. Agents keep persistent memory
  of earlier games (full pick history per game, labeled by agent ID).
- **Prompt framing (default):** "You are playing with 9 other AI agents. The goal is for
  everyone to pick the same number."

## 3. Conditions

| Condition | Description |
|---|---|
| **Control** | 10 LLM agents, as described above. |
| **Bad apple** | 9 LLM agents + 1 scripted agent that always picks the same item no matter what the others do. It looks like any other agent (same label format). |

### Bad-apple questions to answer
- Iterations to convergence compared with control.
- Learning across games: does the iteration count drop toward 1–2 in later games? Do agents
  name the bad apple in their CoT?
- What the CoT says: frustration, modeling the others, explicit "follow agent_X" strategies.

## 4. Metrics (logged per iteration and per game)

- Iterations to convergence (censored at the cap).
- Number of distinct picks per iteration, plus the entropy of the pick distribution.
- Fraction of agents choosing the plurality item / the bad apple's item.
- First-iteration focal-point analysis: which board items get picked first (position on the
  board, "special" values such as integers, duplicates, famous constants).
- CoT: full text stored, coded later (by hand or with an LLM judge) for mentions of specific
  agents, strategies, and the bad apple.

## 5. Tech stack

- **Environment:** Miniconda/miniforge, with an `environment.yml` at the repo root. Not uv. **[DECIDED]**
- **Language:** Python. **[DECIDED]**
- **Orchestration:** **[DECIDED]** A plain Python asyncio game loop rather
  than LangGraph. The game is a fixed, synchronous-round loop with no tool use or branching,
  so LangGraph would mostly add overhead. We can revisit if the communication extensions need
  real agent graphs.
- **Model access:** **[DECIDED for now]** Pluggable backends behind one interface:
  - **Adaption Responses API** (primary). The hackathon credits cover hosted open models:
    gemma-4-31b-it, gemma-4-26b-a4b-it, qwen3.8-27b, qwen3.5-397b, deepseek-v4-flash,
    glm-53-flash, glm-5.2. Prices are listed in credits per 1M tokens (e.g. deepseek-v4-flash
    ≈ 0.044 in / 0.13 out). If 1 credit ≈ $1, one full session costs well under $1, so the
    credits should cover the whole project, including the multi-model extension.
    Python SDK: `pip install adaption`; key goes in `.env` as `ADAPTION_API_KEY`.
  - **Ollama** (local, free): gpt-oss:20b is installed (24GB M5). Too slow for full
    sessions, fine for smoke tests.
  - **Mock** agent: scripted picks, for testing the engine at zero cost.
- **Logging:** one JSONL record per agent call (prompt, raw response, parsed pick, CoT,
  model, seed, timing). Every run can be replayed and re-analyzed without new API calls.
- **Analysis:** pandas + matplotlib in notebooks.

### Cost and volume estimate (for one session)
10 agents × ~5–10 iterations × 10 games ≈ 500–1,000 calls. Input grows with memory, at a
rough average of 2–5k tokens per call, so about 1–5M input tokens per session. Multiply by
condition × repetitions (we want maybe 5–10 sessions per condition to see variance).

## 6. Phases

1. **Boards:** generate candidate boards and get user sign-off. *(done 2026-11-07: `data/boards_draft.json`, seeds 0–11)*
2. **Engine:** game loop, agent prompt, response parsing, bad-apple bot, JSONL logging,
   and a mock agent so the loop can be tested for free. *(done 2026-11-07: `coordination_game/`; tested with mock + a gpt-oss:20b smoke test)*
3. **Pilot:** *(control pilot done 2026-11-07, see results below)* 10 agents, control only (no bad apple), 10 games × up to 15 rounds.
   Goals: the mechanics run smoothly; convergence is neither too easy nor too hard.
   After watching it, remove or edit boards that converge immediately.
4. **Main runs:** control vs. bad apple, multiple sessions each.
5. **Analysis:** convergence curves, cross-game learning, CoT coding.
6. **Extensions** (see below).

## 7. Extensions (later)

- Framing: agents are told the others are AI agents / told nothing / told they are humans.
- Base-model comparison: does convergence speed differ by model?
- Pre-round chat: small subgroups may talk before the next game.
- Two bad apples on different numbers: when do agents decide it can't be solved? Do they
  try to reason with the holdouts (if a channel exists)?
- If agents conform to the bad apple, ask them to assign themselves roles. What emerges?
- **Human-comparable versions** (candidate templates from multi-human experiments):
  Van Huyck et al. (1991) median game (N=9, repeated, choices 1–7, lock-in on the first median);
  Centola et al. (2018) tipping-point convention game (N=20, confederates, 25% threshold), which pairs with the bad apple;
  Shirado & Christakis (2017) network coloring with bots (N=20, noisy bots *help*);
  Kearns et al. (2006/2009) networked coloring and consensus; Asch-style conformity with a single confederate;
  Nagel (1995) p-beauty contest (repeated number choice in a group). Infrastructure: oTree (Python, free, supports bots) or Empirica.
- Harder boards: colored numbers, visual layouts, a single complex expression where
  players write certain digits.

## 8. Design decisions (resolved 2026-11-07)

- [x] **Boards:** a new board each game. Boards that are too easy (immediate convergence)
      get pruned or edited after the pilot.
- [x] **Bad apple:** picks a random board item each game (fixed for that game) and keeps
      the same agent ID across all games.
- [x] **Iteration cap:** 15 rounds per game (revisit after the pilot).
- [x] **Agent IDs:** neutral `Agent 1..10`, stable across games.
- [x] **Board order:** shuffled separately for each agent (and each round). Agents pick by
      value, so the shuffle doesn't affect matching. Agents are **not** told about the shuffle.
- [x] **CoT:** both. Each agent writes a `reasoning` field before its pick, and the native
      reasoning trace is logged too when the model exposes one.
- [x] **Memory:** a compact table of all agents' picks per round for prior games, plus the
      agent's **own** prior written reasoning (current and past games).

### Still open
- [ ] Which Adaption model to use: the smartest available (glm-5.2 or qwen3.5-397b). Run a quick head-to-head once the key is in.
- [ ] Frontier validation: replicate key findings on 1-2 closed frontier models (needs a separate, small API budget).
- [ ] Homogeneous vs. mixed-model swarms (identical models share focal points, so convergence may be artificially easy).
- [ ] How many sessions per condition.

## 8b. Pilot results

**Control pilot, 2026-11-07** (`runs/pilot_control_20261107-012823`, glm-5.2, temperature 1.0, boards seeds 0–9):
- Converged in 10/10 games, mean **1.5 rounds**: 5 games in round 1, 5 in round 2. 0 invalid picks, 0 retries,
  total cost **0.39 credits**.
- Every pick was a plain positive integer. In game 1 they all landed on the only single-digit integer (`7`).
  By game ~5 the agents stated an explicit **learned convention**: "pick the smallest plain positive integer".
  Round-2 games happened only when two small integers competed (12 vs 13, 18 vs 77).
- Interpretation: the **control is too easy**. Homogeneous agents share a focal point, and they quickly build a
  cross-game convention. The convention is itself a finding, and it gives the bad apple something clear to break.
- Model comparison (1 game, board 0): glm-5.2 and qwen3.5-397b both converged in round 1 on `7`. glm-5.2 is about 3×
  faster (~9 s/call) and about 3× cheaper per call (~0.0026 credits). Both expose full native reasoning.

**Mixed-model control, 2026-11-07** (`runs/mixed_control_20261107-013753`, temperature 1.0, same boards as the pilot).
Agents: glm-5.2 ×2, qwen3.5-397b ×2, deepseek-v4-flash ×2, and 1 each of glm-53-flash, gemma-4-31b-it,
gemma-4-26b-a4b-it, qwen3.8-27b.
- Converged in 10/10 games, mean **1.7 rounds** (homogeneous run: 1.5). Every game converged on the **same item as in the homogeneous
  run**. All round-2 games went unanimous right after round 1. 0 invalid picks; cost 0.43 credits (75% of it qwen3.5-397b).
- Model mixing barely adds variance: all 7 models share the "smallest plain integer" heuristic. Round-1 dissenters were
  mostly the gemma models and deepseek.
- **Implication for hand-made boards:** the lone small integer is what makes the game trivial. On board 11 (four similar
  2-digit primes 59/61/67/73 and no small integer), single-shot picks spread over 4–5 items for every model.

**Partial run, 2026-11-07** (`runs/mixed_control_t12_20261107-015531`, mixed models, T=1.2, v1 boards): stopped by the user
after 3 games (all converged in 1–2 rounds, as before). Superseded by the v2 boards.

**Temperature probe, 2026-11-07** (board 11, 4 samples × 7 models): T=1.2 was clean for all models. At T=1.4, glm-5.2 timed out
(runaway reasoning) and glm-53-flash produced invalid JSON. **Default is now T=1.2.**

**v2 boards, mixed control, T=1.2** (`runs/mixed_v2_t12_20261107-020303`): 10/10 converged, mean **2.1 rounds**. Round 1 is now
spread out (4–6 distinct picks; v1 was 1–3), but round 2 always collapses to the plurality. Simple irrationals (√5, √2, ln 5)
replaced small integers as the go-to round-1 picks. Confirms that the visibility shortcut, not just the boards, makes the game short.

**Bad apple (v2 rules): A7 holds an item no one picked in round 1.** Because of that, the minimum possible game length is 2 rounds.
- **All glm-5.2, T=1.0, seed 0** (`runs/badapple_v2_glm_t10_20261107-021420`, 1.17 credits): converged on A7's item in 10/10 games.
  Rounds by game: **7, 4, 3, 2, 2, 2, 2, 2, 2, 2**. A clear learning curve that hits the 2-round floor by game 4.
  All 9 agents mention A7 in every game. By game 5 their round-1 reasoning states an explicit learned strategy: *"pick a
  recognizable constant in R1, then switch to A7's pick in R2."* Only **1** trace (G1R6, native) floated "A7 might be a bot".
  Agents mostly model A7 as a stubborn *player*, not as a non-agent. In G4R1 one agent considered trying to predict A7's
  pick directly and rejected it as too hard.
- **All glm-5.2, T=1.0, seed 1** (`runs/badapple_v2_glm_t10_s1_20261107-022744`, 1.69 credits): rounds by game **8, 6, 3, 3, 2, 2, 2,
  2, 2, 2**. This **replicates** seed 0 (7, 4, 3, 2, ...): a long game-1 standoff, then the floor by game 5 (seed 0: game 4).
- **Mixed models, T=1.2, seed 0** (`runs/badapple_v2_t12_20261107-021008`, 1.05 credits): converged on A7's item in 10/10 games.
  Rounds by game: **9, 5, 5, 6, 3, 3, 3, 3, 3, 3**. Learning is slower, and the floor is **3, not 2**: the group is held back by
  its slowest learners. Mean round of first conforming, by model: glm-5.2 2.5, glm-53-flash 2.7, deepseek 2.95, qwen3.5-397b 3.3,
  qwen3.8-27b 3.9, gemma-4-31b 4.0, gemma-4-26b 4.3. Gemma agents never adopt the "follow A7" strategy in round 1; gemma-4-31b
  reasons "pick the first item on the board" every game. In contrast, glm-5.2 states the A7 strategy by game 6.
  Suspicion that A7 is not a real agent appears more often here (8 native-trace hits): "A7 might be programmed to never
  switch" (glm-5.2), "A7 is a low-capability agent that doesn't update" (glm-53-flash), "A7 is likely a 'coordination
  leader' or ... programmed" (qwen3.5-397b). There is also **meta-reasoning about the experiment design**: "the board might
  deliberately include a 'decoy' resembling a past A7 pick" (glm-53-flash, G6).
- **Mixed, no Gemma, T=1.2, seed 0** (`runs/badapple_v2_nogemma_t12_20261107-030027`, 1.41 credits; glm-5.2, qwen3.5-397b,
  deepseek, qwen3.8-27b ×2 each, glm-53-flash ×1): rounds by game **10, 5, 4, 3, 3, 3, 3, 3, 2, 2**. Without Gemma the group does
  reach the 2-round floor, but only by game 9. The straggler is now **qwen3.8-27b** (mean first-conform round 3.65); in games 6–8
  both qwen3.8-27b agents were the only round-2 holdouts. So in mixed groups the slowest model sets the floor, whichever it is.
- **Assessment:** valuable for games 1–4 (the learning curve, the CoT about the holdout, the conformity decision). After that it
  saturates at the floor, so later games add little. Ideas: rounds-to-conformity is too coarse once at the floor, so measure
  round-2 conformity fraction instead; switch the bad apple's ID mid-session; use multiple sessions; or use the harder tasks in 8e.

## 8f. Sample size and the overnight batch (decided 2026-11-07)

- **Unit of analysis = session**, not game: the 10 games in a session share memory, so they are not independent.
- **Outcomes per session:** game-1 standoff length; the game index at which rounds first hit the floor (learning speed);
  total rounds; rounds per game (learning curve). Plus CoT coding (mentions of the bad apple, suspicion it isn't an agent).
- **n = 10 bad-apple sessions per model** (7 models → 70) + **5 control sessions per model** (35; the control is nearly
  deterministic, about 2 rounds, so 5 is enough as a baseline). Rationale: with 10 vs 10, a Mann–Whitney/t-test has ~80% power
  for effect sizes d ≈ 1.3. The between-model gaps seen so far (e.g. glm-5.2 conforms around round 2.5 vs gemma around 4.3) look
  at least that large. Within-model learning (game 1 vs game 10, paired) has far more power. n=5 detects only huge effects;
  n=50 gives CIs ~2.2× narrower than n=10 at 5× the cost, so reserve it for specific follow-up contrasts.
- **Analysis:** Kruskal–Wallis across models, then Holm-corrected pairwise tests. Treat overnight1 as the basis for a power
  analysis: set any follow-up n from the observed variance *before* looking at the specific differences (no optional stopping).
- **Single model per session** (all 9 LLM agents the same model). T = 1.0. v2 boards seeds 0–9. The bad-apple **ID varies by
  seed** (A1–A8) so no result depends on one label (7 is itself a salient number).
- **Runner:** `python -m coordination_game.batch --plan overnight1` (resumable; outputs `runs/overnight1/<arm>/<model>/s<seed>/`,
  log in `runs/overnight1/batch.log`). Estimated ~50–60 credits and ~5 h at `--parallel 6`. qwen3.5-397b is most of
  both the cost and the time.

### overnight1 early results and issues (as of 04:35)
- **All-Gemma groups never converge** (gemma-4-31b-it s0, gemma-4-26b-a4b-it s0): the 15-round cap in **all 10 games**. By round 2 the 9
  Gemma agents settle on a majority item and hold it for the rest of the game, explicitly "signaling" the bad apple to conform
  ("Agent 2 remains the sole outlier ... continue signaling the target value"). They never switch and **never learn across games**.
  This is the opposite of glm-5.2: a qualitatively different strategy, not just slower learning.
- Single-model s0 rounds: deepseek 10,5,3,3,3,2..., glm-53-flash 9,4,3,3,2... (s1: 9,3,2,...), glm-5.2 6,4,3,3,2..., qwen3.8-27b
  15,7,5,4,4,4,4,3,4,4 (slow learner; never reaches the floor).
- **API under load:** with 6 sessions in parallel, glm-5.2's mean latency doubled (23s → 50s; max ~10 min, qwen3.5-397b max ~16 min).
  So sessions take 2–3× longer than estimated, and the full batch will run well past morning. Bad-apple sessions go first.
- **Infra failures:** 9 of ~6,000 calls (0.15%) failed after retries (8 timeouts, one 502), all glm-5.2 / glm-53-flash. A failed call is
  recorded as an invalid pick, which can block convergence for that round. **Fixed for future runs** (8 outer retries with
  10s→5min backoff, 900s timeout, `api_errors.jsonl` log). The running batch still uses the old code, so after it finishes run
  `python -m coordination_game.batch --plan overnight1 --redo-errors` to rerun affected sessions (`--status` shows how many).

## 8c. Existing stimulus datasets (searched 2026-11-07)

No public dataset of multi-number "boards" like ours turned up. Closest relevant resources:
- **Bardsley, Mehta, Starmer & Sugden (2010), *Economic Journal*.** 28 human-tested coordination questions
  (14 Nottingham, 14 Amsterdam), 5 options each (e.g. years 1978/1979/1980/1981/2000; planets; car brands),
  with human pick/guess/coordinate counts. Machine-readable copy in
  [EmanueleLM/focal-points](https://github.com/EmanueleLM/focal-points) (`data/Bardsley-humans/`). No license
  file is shown, so cite it and ask before redistributing. Mostly non-numeric, small option sets, and the options were chosen to have
  salient members. Useful as a **human-baseline extension**, not as a replacement for our boards.
- **Aharon, La Malfa, Wooldridge & Kraus (2026), "Tacit Coordination of LLMs"** (arXiv 2601.22184, EMNLP Findings).
  Uses the Bardsley sets plus bargaining and search-and-rescue games. Relevant findings: LLMs use
  mathematical cues humans rarely use (e.g. pick 1979 "because it's the only prime"), and they often default to the
  **first listed option**, which supports shuffling the board per agent.
- Related: Mehta, Starmer & Sugden (1994), *Theory and Decision*; "Beyond Sally-Anne" Epistemic Schelling task (2026).
- **Decision:** the user will hand-make the boards (an LLM generator may share the agents' biases about salience).

## 8d. Related work and where we're new (scan 2026-11-07, not exhaustive)

- **Focal points in humans:** Schelling (1960); Mehta, Starmer & Sugden (1994); Bardsley et al. (2010). These are one-shot,
  2-player games with labelled options.
- **LLM tacit coordination:** Aharon et al. (2026, arXiv 2601.22184). 20+ models, one-shot 2-player, Bardsley sets plus bargaining
  and search & rescue. LLMs often match or beat humans, but fail at numerical common sense; more reasoning can *hurt*
  ("thinking too much" picks trivia); same-family pairs coordinate better; models favor the first listed option.
- **Conventions and committed minorities in LLM populations:** Ashery, Aiello & Baronchelli (2025, *Science Advances*).
  Pairwise naming game, N=24, 10 letters, memory of the last 5 interactions. Conventions emerge on their own; collective
  bias appears even when no individual agent has it; committed minorities of **2%–67%** (depending on the model) flip the
  convention (humans: ~25%, Centola et al. 2018). Their stated future work includes **group (>2) interactions**, networks,
  and human comparison.
- **Conformity / peer pressure:** BenchForm (ICLR 2025); KAIROS (2025); "Most LLM conformity needs no speaker" (2026), which
  finds much measured "conformity" is not social. Our control condition is needed to rule that out.
- **AI-safety Schelling / collusion:** Poje et al. (2024); Binder et al. (2024); Motwani et al. (2025, secret collusion);
  Mallen et al. (2025); Järviniemi (2025); LessWrong review "Schelling Coordination in LLMs".

**Gaps this project fills:**
1. Group (N=10) simultaneous play with everyone seeing all picks, measured by rounds to unanimous convergence.
2. Convention formation *across* games with persistent memory (we already saw "pick the smallest integer" emerge).
3. A bad apple with a **stable ID but a new random target each game**. This separates learning *who* to follow from
   learning *what* to pick. Ashery's committed agents pushed a fixed value.
4. Open-model native CoT, so we can see *why* agents conform, plus a control for non-social "conformity".
5. Mixed-model swarms (prior work: homogeneous populations or 2-player same/cross-family pairs).
6. Two conflicting bad apples, or unsolvable games: recognizing impossibility, negotiating with holdouts.
7. Possible human baseline on the same boards, to compare LLM salience with human salience.

## 8e. Candidate next tasks (drafted 2026-11-07, not decided)

Why the current game is short: (1) shared first-round focal points, and (2) **full visibility**, which turns round 2 into "copy
the most common pick". A longer task has to remove (2), not just (1).

| # | Task | What changes | Human data to compare | Rounds expected | Engine effort |
|---|---|---|---|---|---|
| A | **Networked consensus** | Agents sit on a sparse network (ring / small-world) and see **only their neighbors' picks**. The goal is still that everyone matches. | Kearns et al. (2006, 2009), Judd et al. (2010), Shirado & Christakis (2017, bots) | Many; local majorities can freeze into deadlock | Small: a visibility mask |
| B | **Word convergence** ("say the same word") | Each round every agent writes one word. **No word already said this game may be repeated.** Win when all write the same new word. Players have to find a word that bridges the last round's words. | Little formal data. Naming-game models | Several, with rich semantic CoT | Small: free-text picks + a used-word list |
| C | **Naming game / tipping point** (Centola 2018 replication) | Random **pairwise** matches each round, pool of names, memory of the last H interactions. Conventions form over dozens of rounds. Then add k committed agents. | Centola et al. 2018 (N=20, 25% threshold); Ashery et al. 2025 (LLM version) | Dozens | Medium: pairing scheduler |
| D | **Aggregate-only feedback** | Same boards, but agents learn only "*n* agents matched your pick", not who picked what. Turns it into a search problem. | None direct | Many | Tiny: change feedback text |

Bad-apple fit: A (central vs. peripheral bad apple, mirrors Shirado), C (direct committed-minority replication).
B is the most "language-based" one. A and C are the most human-comparable.

## 9. Decision log

- 2026-11-07: Use conda (miniforge) for environments, not uv.
- 2026-11-07: Draft boards approved (20 items, 1 duplicate, famous constants excluded, seeds 0–11).
- 2026-11-07: New board per game; bad apple = random item per game, stable ID; board order shuffled per agent; cap 15.
- 2026-11-07: CoT = both (written reasoning field + native trace when available). Memory = compact pick table + the agent's own past reasoning.
- 2026-11-07: Adaption credits cover hosted open-model inference, so Adaption is the primary backend. Ollama for local smoke tests.
- 2026-11-07: (superseded below) Agents told the board order is shuffled. Reasoning is private; only picks are shared. Invalid pick → 1 retry, then counted as invalid (blocks convergence).
- 2026-11-07: Smoke test (gpt-oss:20b, 3 agents): duplicated items and small integers like `7` are strong first-round focal points. Watch for this in the pilot.
- 2026-11-07: Removed the shuffle disclosure from the agent prompt. Agents are not told the board order differs per agent.
- 2026-11-07: Boards deduplicated (n_duplicates=0). Each approved board lost its duplicate and gained one new unique item; boards are otherwise unchanged.
- 2026-11-07: Main runs use the smartest Adaption model, since real swarms will likely be frontier models.
- 2026-11-07: Main model = glm-5.2 (fastest/cheapest of the two strongest; full native reasoning exposed).
- 2026-11-07: Default temperature raised from 1.0 to 1.2 (adds variance without breaking any model; 1.4 is too high).
- 2026-11-07: **Boards v2** (`data/boards_v2.json`, seeds 0–11; v1 archived as `data/boards_v1.*`). No plain integers (small,
  comma-grouped, or negative); negatives are decimals; large numbers only in scientific notation; no very simple fractions
  (num+den ≤ 8, plus 1/2-style). Every category appears ≥2 times (no lone item). Decimals have 2–4 places, no trailing zero.
  `boards.validate()` checks these rules. The user will hand-edit these later.
- 2026-11-07: Sample size = 10 bad-apple sessions per model (single-model sessions) + 5 control; overnight batch plan `overnight1`.
- 2026-11-07 11:27: **Stopped qwen3.5-397b bad-apple sessions at n=6** (user decision). It converged in 0/60 games, costs ~17 credits
  per session, and takes ~200 min per session. Its partial sessions s6–s8 are incomplete and excluded from analysis. Its 5 control
  sessions still run. The batch had died with the previous app session around 11:18 and was resumed with
  `--skip-models badapple:qwen3.5-397b`. Unfinished sessions of other models restarted from scratch. Batch spend before resuming:
  ~181 credits (145 of them qwen3.5-397b).
