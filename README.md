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

## Game visualization (`game_viz/`)

A click-through story of one real session (GLM-5.2): agents coordinating with no holdout, then a scripted holdout taking
over, then the agents electing it leader, with quotes from the agents' own reasoning.

- Live page: https://claude.ai/artifact/3kE4v3RJzKZu2RHJeformk (shared from the owner's claude.ai account).
- Open locally: open `game_viz/index.html` in a browser (one self-contained file).
- Rebuild after changing the story or the data: `python game_viz/build_data.py` (needs `runs/`). It reads the logs, writes
  `game_viz/data.json`, and inlines it into `game_viz/index.html` (full page) and `game_viz/artifact.html` (artifact fragment).
  Edit the page itself in `game_viz/template.html`; which moments are quoted is chosen in `build_data.py`.
- Keys: ← / → step through scenes, space plays and pauses. Each scene has its own link (`#bad-r4`, `#vote`, …).
