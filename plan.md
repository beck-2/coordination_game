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
- **Orchestration:** **[OPEN]**. Leaning toward a plain Python asyncio game loop rather
  than LangGraph. The game is a fixed, synchronous-round loop with no tool use or branching,
  so LangGraph would mostly add overhead. We can revisit if the communication extensions need
  real agent graphs.
- **Model access:** **[OPEN]**. Candidates:
  - Ollama (already installed locally): free, good for development and for runs at scale
    on small open models.
  - A single gateway such as OpenRouter or LiteLLM, so that the base-model comparison
    extension only needs a config change.
  - Adaption platform credits ($1,500): find out what models or APIs they actually cover.
- **Logging:** one JSONL record per agent call (prompt, raw response, parsed pick, CoT,
  model, seed, timing). Every run can be replayed and re-analyzed without new API calls.
- **Analysis:** pandas + matplotlib in notebooks.

### Cost and volume estimate (for one session)
10 agents × ~5–10 iterations × 10 games ≈ 500–1,000 calls. Input grows with memory, at a
rough average of 2–5k tokens per call, so about 1–5M input tokens per session. Multiply by
condition × repetitions (we want maybe 5–10 sessions per condition to see variance).

## 6. Phases

1. **Boards:** generate candidate boards and get user sign-off. *(in progress)*
2. **Engine:** game loop, agent prompt, response parsing, bad-apple bot, JSONL logging,
   and a mock agent so the loop can be tested for free.
3. **Pilot:** a small local model through Ollama, 1 session per condition. Check prompts,
   parsing, and that convergence is even possible.
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
- Harder boards: colored numbers, visual layouts, a single complex expression where
  players write certain digits.

## 8. Open design questions

- [ ] Does each game use a new board, or does the board stay the same?
- [ ] Bad apple's pick: fixed board position, fixed value, or random per game? Is the bad
      apple the same agent ID every game?
- [ ] Iteration cap (proposed: 15).
- [ ] Agent IDs: neutral labels (`Agent 1..10`) kept stable across games, so agents can
      track the bad apple?
- [ ] Board order: shuffled per agent or the same for everyone? (Shared order makes "pick
      the first item" an easy focal point.)
- [ ] CoT: native reasoning model traces, a "reasoning" field before the answer, or both?
- [ ] Memory format: full transcript of earlier games vs. a compact pick table per game.
- [ ] What do the Adaption credits cover?

## 9. Decision log

- 2026-11-07: Use conda (miniforge) for environments, not uv.
