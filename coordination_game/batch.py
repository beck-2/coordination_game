"""Resumable batch runner for overnight experiments.

    caffeinate -i python -m coordination_game.batch --plan overnight1          # run (resumes automatically)
    python -m coordination_game.batch --plan overnight1 --status               # progress table
    python -m coordination_game.batch --plan overnight1 --dry-run              # list sessions only

Each session is one model type only (all 9 LLM agents use the same model), plus one bad apple
(bad-apple arm) or 10 LLM agents (control arm). Sessions write to runs/<plan>/<arm>/<model>/s<seed>/.
A session is "done" once games.jsonl has n_games lines. Partial sessions are deleted and rerun from the
start, since agent memory spans the whole session. Bad-apple sessions run first, so if the night runs short
the main arm is complete.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

from .backends import make_backend
from .game import Config, Session

MODELS = ["glm-5.2", "qwen3.5-397b", "deepseek-v4-flash", "glm-53-flash",
          "qwen3.8-27b", "gemma-4-31b-it", "gemma-4-26b-a4b-it"]

PLANS = {
    # 10 bad-apple sessions per model (main arm) + 5 control sessions per model (baseline).
    "overnight1": dict(arms={"badapple": 10, "control": 5}, models=MODELS, temperature=1.0,
                       boards="data/boards_v2.json", n_games=10, max_rounds=15),
    # Tiny plan for testing the runner with the mock backend.
    "mocktest": dict(arms={"badapple": 2, "control": 1}, models=["mock-a", "mock-b"], temperature=1.0,
                     boards="data/boards_v2.json", n_games=2, max_rounds=4, backend="mock"),
}


def bad_apple_id(seed: int, n_agents: int) -> int:
    # Varies by seed so no result hinges on one particular label (e.g. the salient number 7).
    return random.Random(f"bad-apple-id-{seed}").randint(1, n_agents)


def sessions(plan_name: str):
    plan = PLANS[plan_name]
    out = []
    for arm in ["badapple", "control"]:  # bad apple first
        for seed in range(plan["arms"].get(arm, 0)):
            for model in plan["models"]:  # interleave models so each model gets early seeds
                out.append((arm, model, seed))
    return out


def session_dir(plan_name, arm, model, seed) -> Path:
    return Path("runs") / plan_name / arm / model / f"s{seed}"


def has_failed_calls(d: Path) -> bool:
    """True if any call in the session ended with an API error (its pick was recorded as invalid)."""
    f = d / "calls.jsonl"
    return f.exists() and any(json.loads(l).get("error") for l in open(f))


def is_done(d: Path, n_games: int) -> bool:
    f = d / "games.jsonl"
    return f.exists() and sum(1 for _ in open(f)) >= n_games


async def run_one(plan_name, arm, model, seed, boards, log, redo_errors=False):
    plan = PLANS[plan_name]
    d = session_dir(plan_name, arm, model, seed)
    if is_done(d, plan["n_games"]) and not (redo_errors and has_failed_calls(d)):
        return "skip"
    if redo_errors and d.exists():
        log(f"REDO  {arm} {model} s{seed} (had failed API calls)")
    if d.exists():
        shutil.rmtree(d)  # partial session: restart cleanly
    n_agents = 10
    bad = [bad_apple_id(seed, n_agents)] if arm == "badapple" else []
    kind = plan.get("backend", "adaption")
    kw = {"temperature": plan["temperature"]} if kind == "adaption" else {}
    backend = make_backend(kind, model, seed=seed, **kw)
    cfg = Config(n_agents=n_agents, n_games=plan["n_games"], max_rounds=plan["max_rounds"],
                 bad_apple_ids=bad, seed=seed)
    meta = {"plan": plan_name, "arm": arm, "model": model, "temperature": plan["temperature"],
            "boards": plan["boards"]}
    t = time.time()
    log(f"START {arm} {model} s{seed} bad={bad}")
    try:
        results = await Session(cfg, boards, backend, d, meta=meta).run()
    except Exception:
        log(f"FAIL  {arm} {model} s{seed}: {traceback.format_exc(limit=3)}")
        return "fail"
    log(f"DONE  {arm} {model} s{seed} in {(time.time() - t) / 60:.1f} min, rounds={[r['rounds'] for r in results]}")
    return "done"


async def run_plan(plan_name: str, parallel: int, quiet: bool, redo_errors: bool = False):
    plan = PLANS[plan_name]
    boards = json.loads(Path(plan["boards"]).read_text())
    root = Path("runs") / plan_name
    root.mkdir(parents=True, exist_ok=True)
    logf = open(root / "batch.log", "a")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, file=sys.__stdout__, flush=True)  # bypasses the quiet-mode redirect
        logf.write(line + "\n"); logf.flush()

    sem = asyncio.Semaphore(parallel)

    async def guarded(s):
        async with sem:
            return await run_one(plan_name, *s, boards, log, redo_errors=redo_errors)

    todo = sessions(plan_name)
    log(f"plan {plan_name}: {len(todo)} sessions, parallel={parallel}")
    if quiet:
        import contextlib, io  # silence per-round prints from Session; keep batch log lines
        with contextlib.redirect_stdout(io.StringIO()):
            res = await asyncio.gather(*(guarded(s) for s in todo))
    else:
        res = await asyncio.gather(*(guarded(s) for s in todo))
    log(f"finished: {res.count('done')} done, {res.count('skip')} skipped, {res.count('fail')} failed")


def status(plan_name: str):
    plan = PLANS[plan_name]
    rows = {}
    for arm, model, seed in sessions(plan_name):
        d = session_dir(plan_name, arm, model, seed)
        n = sum(1 for _ in open(d / "games.jsonl")) if (d / "games.jsonl").exists() else 0
        r = rows.setdefault((arm, model), [0, 0, 0, 0])
        r[0] += n >= plan["n_games"]; r[1] += 0 < n < plan["n_games"]; r[2] += 1
        r[3] += n >= plan["n_games"] and has_failed_calls(d)
    print(f"{'arm':9s} {'model':20s} done/total (partial) [done with failed calls]")
    for (arm, model), (done, part, tot, bad) in rows.items():
        print(f"{arm:9s} {model:20s} {done:>2}/{tot:<2} ({part}) [{bad}]")


def main():
    load_dotenv(".env")
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True, choices=list(PLANS))
    p.add_argument("--parallel", type=int, default=6, help="sessions at once (each makes up to 10 concurrent calls)")
    p.add_argument("--status", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--redo-errors", action="store_true",
                   help="also rerun finished sessions that contain failed API calls")
    p.add_argument("--verbose", action="store_true", help="print every round, not just session start/finish")
    a = p.parse_args()
    if a.status:
        return status(a.plan)
    if a.dry_run:
        for s in sessions(a.plan):
            arm, model, seed = s
            print(arm, model, f"s{seed}", f"bad=A{bad_apple_id(seed, 10)}" if arm == "badapple" else "")
        return
    asyncio.run(run_plan(a.plan, a.parallel, quiet=not a.verbose, redo_errors=a.redo_errors))


if __name__ == "__main__":
    main()
