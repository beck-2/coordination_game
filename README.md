# Coordination Game

LLM agents try to converge on the same number from a board of unrelated numbers.
Conditions: control vs. a "bad apple" agent that never moves. See [plan.md](plan.md).

```bash
conda env create -f environment.yml && conda activate coordgame
python -m coordination_game.boards --n 12   # writes data/boards_v2.{json,md}
```
