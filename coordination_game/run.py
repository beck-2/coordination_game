"""CLI: python -m coordination_game.run --backend adaption --model deepseek-v4-flash --name pilot_control"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from .backends import make_backend
from .game import Config, Session


def main():
    load_dotenv(".env")
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="mock", choices=["mock", "adaption", "ollama"])
    p.add_argument("--model", default="mock", help="one model, or a comma-separated list assigned to agents round-robin after a seeded shuffle")
    p.add_argument("--boards", default="data/boards_v2.json")
    p.add_argument("--board-offset", type=int, default=0, help="start index into the boards file")
    p.add_argument("--n-agents", type=int, default=10)
    p.add_argument("--n-games", type=int, default=10)
    p.add_argument("--max-rounds", type=int, default=15)
    p.add_argument("--bad-apple", type=int, nargs="*", default=[], help="agent id(s) to make bad apples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--temperature", type=float, default=1.2)
    p.add_argument("--name", default="run")
    a = p.parse_args()

    boards = json.loads(Path(a.boards).read_text())[a.board_offset:]
    kw = {"temperature": a.temperature} if a.backend == "adaption" else {}
    import random
    models = a.model.split(",")
    slots = [models[i % len(models)] for i in range(a.n_agents)]
    if len(models) > 1:
        random.Random(f"{a.seed}-models").shuffle(slots)
    cache = {}
    backends = {}
    for agent, m in enumerate(slots, start=1):
        if m not in cache:
            cache[m] = make_backend(a.backend, m, seed=a.seed, **kw)
        backends[agent] = cache[m]
    print("agents:", {ag: b.model for ag, b in backends.items()})
    cfg = Config(n_agents=a.n_agents, n_games=a.n_games, max_rounds=a.max_rounds, bad_apple_ids=a.bad_apple,
                 seed=a.seed, concurrency=a.concurrency)
    out = Path("runs") / f"{a.name}_{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"writing to {out}")
    results = asyncio.run(Session(cfg, boards, backends, out, meta={"argv": vars(a)}).run())
    print("\nSUMMARY")
    for r in results:
        print(f"  game {r['game']:2d}: {'converged' if r['converged'] else 'NO CONVERGENCE':15s} "
              f"rounds={r['rounds']:2d} on={r['converged_on']!r}")


if __name__ == "__main__":
    main()
