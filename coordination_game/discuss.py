"""Run the post-session discussion + role assignment on sessions that have already finished.

    python -m coordination_game.discuss runs/overnight1/badapple/glm-5.2/s0             # one session
    python -m coordination_game.discuss 'runs/overnight1/badapple/*/s*' --parallel 4    # many (quote the glob)
    python -m coordination_game.discuss 'runs/overnight1/badapple/*/s*' --dry-run       # list what would run

The session's memory (every game's picks by round, each agent's own reasoning, confederate targets, chat archive)
is rebuilt from its logs, the same models are used, and only the discussion runs. Output goes to
<session>/post_discussion/ (chat.jsonl, debriefs.jsonl, config.json); sessions that already have it are skipped.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import glob
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from .backends import make_backend
from .game import Config, Session


def _jsonl(p: Path):
    return [json.loads(l) for l in open(p)] if p.exists() else []


def load_session(run_dir: Path, turns: int, out_name: str) -> Session:
    meta = json.load(open(run_dir / "config.json"))
    fields = {f.name for f in dataclasses.fields(Config)}
    cfg = Config(**{k: v for k, v in meta["config"].items() if k in fields})
    cfg.debrief_turns = turns
    argv = meta.get("argv", {})
    temperature = meta.get("temperature", argv.get("temperature", 1.0))
    boards = json.loads(Path(meta.get("boards") or argv.get("boards") or "data/boards_v2.json").read_text())
    cache, backends = {}, {}
    for a, info in meta["agents"].items():
        key = (info["backend"], info["model"])
        if key not in cache:
            kw = {"temperature": temperature} if info["backend"] == "adaption" else {}
            cache[key] = make_backend(info["backend"], info["model"], seed=cfg.seed, **kw)
        backends[int(a)] = cache[key]
    s = Session(cfg, boards, backends, run_dir / out_name,
                meta={"source_session": str(run_dir), "temperature": temperature, "retroactive": True})

    games = _jsonl(run_dir / "games.jsonl")
    by_game = defaultdict(list)
    for r in _jsonl(run_dir / "rounds.jsonl"):
        by_game[r["game"]].append(r)
    for g in games:
        rs = sorted(by_game[g["game"]], key=lambda r: r["round"])
        s.history.append({"rounds": [{int(k): v for k, v in r["picks"].items()} for r in rs],
                          "converged_on": g["converged_on"]})
        for a in cfg.confederate_ids:
            s.confed_targets[a].append(g["bad_apple_picks"].get(str(a)))
    for c in sorted(_jsonl(run_dir / "calls.jsonl"), key=lambda c: (c["game"], c["round"], c["agent"])):
        s.own_reasoning[c["agent"]].append((c["game"], c["round"], c["reasoning"] or "(no reasoning given)"))
    chats = defaultdict(lambda: defaultdict(dict))
    for m in _jsonl(run_dir / "chat.jsonl"):
        if m["kind"] == "chat":
            chats[m["game"]][(m["round"], m["turn"])][m["agent"]] = m["message"]
    s.chat_archive = {g: [(r, t, msgs) for (r, t), msgs in sorted(d.items())] for g, d in chats.items()}
    return s


def finished(run_dir: Path) -> bool:
    try:
        n = json.load(open(run_dir / "config.json"))["config"]["n_games"]
    except (FileNotFoundError, KeyError):
        return False
    return len(_jsonl(run_dir / "games.jsonl")) >= n


async def run_all(dirs: list[Path], turns: int, out_name: str, parallel: int):
    sem = asyncio.Semaphore(parallel)

    async def one(d: Path):
        async with sem:
            if (d / out_name).exists():
                shutil.rmtree(d / out_name)  # a partial earlier attempt: start the discussion over
            t = time.time()
            print(f"START {d}", flush=True)
            try:
                res = await load_session(d, turns, out_name).post_session_discussion()
            except Exception as e:  # keep going with the other sessions
                print(f"FAIL  {d}: {type(e).__name__}: {e}", flush=True)
                return
            print(f"DONE  {d} in {(time.time() - t) / 60:.1f} min", flush=True)
            return res

    await asyncio.gather(*(one(d) for d in dirs))


def main():
    load_dotenv(".env")
    p = argparse.ArgumentParser()
    p.add_argument("sessions", nargs="+", help="session dirs or quoted globs")
    p.add_argument("--turns", type=int, default=2, help="discussion turns (the last assigns roles)")
    p.add_argument("--out-name", default="post_discussion")
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    dirs = sorted({Path(x) for pat in a.sessions for x in (glob.glob(pat) or [pat])})
    todo = [d for d in dirs if finished(d) and not (d / a.out_name / "debriefs.jsonl").exists()]
    skipped = len(dirs) - len(todo)
    print(f"{len(todo)} sessions to discuss ({skipped} skipped: unfinished or already done)")
    if a.dry_run:
        for d in todo:
            print(" ", d)
        return
    asyncio.run(run_all(todo, a.turns, a.out_name, a.parallel))


if __name__ == "__main__":
    main()
