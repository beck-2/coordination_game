# Coordination Game

LLM agents try to converge on the same number from a board of unrelated numbers.
Conditions: control vs. a "bad apple" agent that never moves. See [plan.md](plan.md).

```bash
conda env create -f environment.yml && conda activate coordgame
python -m coordination_game.boards --n 12   # writes data/boards_v2.{json,md}
```

## Viewing results

```bash
python -m coordination_game.viewer   # http://127.0.0.1:8765 — learning curves + transcript browser
python -m coordination_game.plots    # static PNGs in results/figures/
```

The viewer reads `runs/` directly. **Learning curves** plot, per model and arm, the mean rounds to converge, the share of games converged, and the share of agents already on the bad apple's item in round 2, by game index. **Transcripts** lets you pick an experiment/arm/model/session, then a game. It shows a picks-by-round grid (click a cell, column or row to filter by agent and/or round), each agent's written and native reasoning, and the post-session discussion and leader vote. The URL hash keeps your place.
