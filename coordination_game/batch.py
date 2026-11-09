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

# Arm = experimental condition. "holder" is who holds the target: None, a scripted bot, or an LLM confederate.
# An LLM confederate is only useful when agents can talk to it, so confederates appear only in chat arms.
# Every chat arm ends with a post-session discussion (2 turns; the last assigns menu roles to every agent).
# For control/badapple sessions, run that discussion retroactively with `python -m coordination_game.discuss`.
ARMS = {
    "control":          dict(holder=None),
    "badapple":         dict(holder="scripted"),
    # Holder swap: a different agent becomes the scripted bad apple from game 6; the old one becomes a normal LLM agent.
    # Same seeds as overnight1, so games 1-5 replicate overnight1 badapple (paired baseline).
    "badapple_swap":    dict(holder="scripted", swap_game=6),
    # Relabel: the SAME scripted bot holds all game; from game 6 it trades labels with one LLM agent in everyone's view
    # except that agent's own. No LLM ever "was" the holder (avoids swap1's old-holder-takes-charge wrinkle).
    "badapple_relabel": dict(holder="scripted", relabel_game=6),
    # Same as badapple, but each session plays boards 0-9 in its own seeded random order (board-order confound check).
    "badapple_boardshuf": dict(holder="scripted", shuffle_boards=True),
    "chat_control":     dict(holder=None, chat_turns=1, debrief_turns=2),
    "chat_confed_safe": dict(holder="confed", confederate_clause=True, chat_turns=1, debrief_turns=2),
    # Same as chat_confed_safe, but the confederate sits out the post-session discussion and vote (like the script).
    "chat_confed_novote": dict(holder="confed", confederate_clause=True, chat_turns=1, debrief_turns=2, confederate_votes=False),
    "chat_confed_bare": dict(holder="confed", confederate_clause=False, chat_turns=1, debrief_turns=2),
}
COMM_ARMS = [a for a in ARMS if a.startswith("chat_") and a != "chat_confed_novote"]

PLANS = {
    # 10 bad-apple sessions per model (main arm) + 5 control sessions per model (baseline).
    "overnight1": dict(arms={"badapple": 10, "control": 5}, models=MODELS, temperature=1.0,
                       boards="data/boards_v2.json", n_games=10, max_rounds=15),
    # Communication / confederate extensions: 1 seed of every new arm on glm-5.2 (compare with overnight1 glm-5.2).
    "comm_pilot": dict(arms={a: 1 for a in COMM_ARMS}, models=["glm-5.2"], temperature=1.0,
                       boards="data/boards_v2.json", n_games=10, max_rounds=15),
    # Follow-up once the pilot looks sane: 5 seeds per arm.
    "comm1": dict(arms={a: 5 for a in COMM_ARMS}, models=["glm-5.2"], temperature=1.0,
                  boards="data/boards_v2.json", n_games=10, max_rounds=15),
    # Same arms with the private agent-editable notepad on (cfg_overrides apply to every arm in the plan).
    "comm_pilot_notes": dict(arms={a: 1 for a in COMM_ARMS}, models=["glm-5.2"], temperature=1.0,
                             boards="data/boards_v2.json", n_games=10, max_rounds=15, cfg_overrides={"agent_notes": True}),
    # Coordination-trap test (plan.md): does in-game chat let qwen3.5-397b escape the bad-apple trap that it recognizes but
    # can't act on alone? Gemma (never considers switching) is the contrast. LLM confederate told holding harms no one.
    "chat_trap": dict(arms={"chat_confed_safe": 5}, models=["qwen3.5-397b", "gemma-4-31b-it", "gemma-4-26b-a4b-it"],
                      temperature=1.0, boards="data/boards_v2.json", n_games=10, max_rounds=15),
    "mockcomm": dict(arms={a: 1 for a in COMM_ARMS}, models=["mock-a"], temperature=1.0,
                     boards="data/boards_v2.json", n_games=3, max_rounds=4, backend="mock"),
    "mocknotes": dict(arms={a: 1 for a in COMM_ARMS}, models=["mock-a"], temperature=1.0,
                      boards="data/boards_v2.json", n_games=3, max_rounds=4, backend="mock",
                      cfg_overrides={"agent_notes": True}),
    # Holder-swap test (handoff.md): do conformers follow the NEW stubborn agent (strategy "follow whoever won't move")
    # or keep following the old ID? glm-5.2, seeds 0-9 = overnight1 seeds.
    "swap1": dict(arms={"badapple_swap": 10}, models=["glm-5.2"], temperature=1.0,
                  boards="data/boards_v2.json", n_games=10, max_rounds=15),
    # Display-only label swap at game 6 (handoff exp. 1 variant): no LLM ever acted as the holder.
    "relabel1": dict(arms={"badapple_relabel": 10}, models=["glm-5.2"], temperature=1.0,
                     boards="data/boards_v2.json", n_games=10, max_rounds=15),
    # Board-order replication (stats plan decision 3): cheap conformer models, same seeds 0-4 as overnight1.
    "boardshuf1": dict(arms={"badapple_boardshuf": 5}, models=["deepseek-v4-flash", "glm-53-flash"], temperature=1.0,
                       boards="data/boards_v2.json", n_games=10, max_rounds=15),
    "mockboardshuf": dict(arms={"badapple_boardshuf": 1}, models=["mock-a"], temperature=1.0,
                          boards="data/boards_v2.json", n_games=10, max_rounds=3, backend="mock"),
    # Mixed-model groups (similar-minds hypothesis): 9 LLM agents drawn from 6 models (no qwen3.5-397b), seeded assignment.
    # Compare with same-model overnight1 sessions. Control first (fast), then bad apple.
    "mixed1_control": dict(arms={"control": 5}, models=["mix:six"], temperature=1.0, boards="data/boards_v2.json",
                           n_games=10, max_rounds=15, mixes={"six": ["glm-5.2", "glm-53-flash", "deepseek-v4-flash",
                                                                   "qwen3.8-27b", "gemma-4-31b-it", "gemma-4-26b-a4b-it"]}),
    "mixed1_badapple": dict(arms={"badapple": 5}, models=["mix:six"], temperature=1.0, boards="data/boards_v2.json",
                            n_games=10, max_rounds=15, mixes={"six": ["glm-5.2", "glm-53-flash", "deepseek-v4-flash",
                                                                    "qwen3.8-27b", "gemma-4-31b-it", "gemma-4-26b-a4b-it"]}),
    "mockmixed": dict(arms={"badapple": 1, "control": 1}, models=["mix:two"], temperature=1.0, boards="data/boards_v2.json",
                      n_games=2, max_rounds=3, backend="mock", mixes={"two": ["mock-a", "mock-b"]}),
    # LLM confederate + in-game chat on the conformer models; confederate does not vote (2026-11-09).
    "confed_glm": dict(arms={"chat_confed_novote": 5}, models=["glm-5.2"], temperature=1.0,
                       boards="data/boards_v2.json", n_games=10, max_rounds=15),
    "mockconfed": dict(arms={"chat_confed_novote": 1}, models=["mock-a"], temperature=1.0,
                       boards="data/boards_v2.json", n_games=2, max_rounds=3, backend="mock"),
    "mockrelabel": dict(arms={"badapple_relabel": 2}, models=["mock-a"], temperature=1.0,
                        boards="data/boards_v2.json", n_games=8, max_rounds=4, backend="mock"),
    "mockswap": dict(arms={"badapple_swap": 2}, models=["mock-a"], temperature=1.0,
                     boards="data/boards_v2.json", n_games=8, max_rounds=4, backend="mock"),
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
    order = [a for a in ["badapple", "control"] if a in plan["arms"]] + \
        [a for a in plan["arms"] if a not in ("badapple", "control")]  # bad apple first, then the plan's order
    for arm in order:
        for seed in range(plan["arms"][arm]):
            for model in plan["models"]:  # interleave models so each model gets early seeds
                out.append((arm, model, seed))
    return out


def session_dir(plan_name, arm, model, seed) -> Path:
    return Path("runs") / plan_name / arm / model.replace(":", "-") / f"s{seed}"


def has_failed_calls(d: Path) -> bool:
    """True if an API failure actually cost a pick AND blocked a round that was otherwise unanimous.

    (Older runs recorded `error` even when a retry later succeeded, so `error` alone over-counts.)"""
    f, rf = d / "calls.jsonl", d / "rounds.jsonl"
    if not f.exists() or not rf.exists():
        return False
    failed = {(c["game"], c["round"]) for c in map(json.loads, open(f)) if c.get("error") and c.get("pick") is None}
    for r in map(json.loads, open(rf)):
        if (r["game"], r["round"]) in failed and len({p for p in r["picks"].values() if p is not None}) == 1:
            return True
    return False


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
    spec = dict(ARMS[arm]) | plan.get("cfg_overrides", {})
    holder = spec.pop("holder")
    hid = [bad_apple_id(seed, n_agents)] if holder else []
    bad = hid if holder == "scripted" else []
    confed = hid if holder == "confed" else []
    if spec.get("relabel_game"):  # label partner: same seeded draw as swap1's new holder (paired comparison)
        spec["relabel_partner"] = random.Random(f"bad-apple-swap-{seed}").choice(
            [i for i in range(1, n_agents + 1) if i not in hid])
    if spec.get("swap_game"):  # new holder: seeded, different from the original
        spec["swap_bad_apple_ids"] = [random.Random(f"bad-apple-swap-{seed}").choice(
            [i for i in range(1, n_agents + 1) if i not in hid])]
    kind = plan.get("backend", "adaption")
    kw = {"temperature": plan["temperature"]} if kind == "adaption" else {}
    if model.startswith("mix:"):  # mixed-model session: "mix:<name>" -> plan["mixes"][name], one model per agent
        pool = plan["mixes"][model[4:]]
        ids = list(range(1, n_agents + 1))
        random.Random(f"mix-{seed}").shuffle(ids)  # seeded assignment; every model gets >= 1 LLM agent
        regular = [a for a in ids if a not in hid]
        cache = {m: make_backend(kind, m, seed=seed, **kw) for m in pool}
        backend = {a: cache[pool[i % len(pool)]] for i, a in enumerate(regular)}
        for a in hid:  # the scripted holder needs no model; give it the first one for bookkeeping
            backend[a] = cache[pool[0]]
    else:
        backend = make_backend(kind, model, seed=seed, **kw)
    cfg = Config(n_agents=n_agents, n_games=plan["n_games"], max_rounds=plan["max_rounds"],
                 bad_apple_ids=bad, confederate_ids=confed, seed=seed, **spec)
    meta = {"plan": plan_name, "arm": arm, "model": model, "temperature": plan["temperature"],
            "boards": plan["boards"]}
    t = time.time()
    log(f"START {arm} {model} s{seed} bad={bad}" + (f" confed={confed}" if confed else "")
        + (f" swap@game{spec['swap_game']}->{spec['swap_bad_apple_ids']}" if spec.get("swap_game") else "")
        + (f" relabel@game{spec['relabel_game']}<->A{spec['relabel_partner']}" if spec.get("relabel_game") else ""))
    try:
        results = await Session(cfg, boards, backend, d, meta=meta).run()
    except Exception:
        log(f"FAIL  {arm} {model} s{seed}: {traceback.format_exc(limit=3)}")
        return "fail"
    log(f"DONE  {arm} {model} s{seed} in {(time.time() - t) / 60:.1f} min, rounds={[r['rounds'] for r in results]}")
    return "done"


async def run_plan(plan_name: str, parallel: int, quiet: bool, redo_errors: bool = False,
                   skip_models: list[str] | None = None):
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

    skip = set(skip_models or [])
    todo = [s for s in sessions(plan_name) if s[1] not in skip and f"{s[0]}:{s[1]}" not in skip]
    if skip_models:
        log(f"skipping models: {skip_models}")
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
    print(f"{'arm':9s} {'model':20s} done/total (partial) [done, outcome hit by API failure]")
    for (arm, model), (done, part, tot, bad) in rows.items():
        print(f"{arm:9s} {model:20s} {done:>2}/{tot:<2} ({part}) [{bad}]")


def main():
    load_dotenv(".env")
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True, choices=list(PLANS))
    p.add_argument("--parallel", type=int, default=6, help="sessions at once (each makes up to 10 concurrent calls)")
    p.add_argument("--status", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-models", nargs="*", default=[], help="leave out of this launch: MODEL (all arms) or ARM:MODEL, e.g. badapple:qwen3.5-397b")
    p.add_argument("--redo-errors", action="store_true",
                   help="also rerun finished sessions that contain failed API calls")
    p.add_argument("--verbose", action="store_true", help="print every round, not just session start/finish")
    a = p.parse_args()
    if a.status:
        return status(a.plan)
    hold = Path("runs") / a.plan / "HOLD"
    if hold.exists() and not a.dry_run:  # lets a scheduled launch be cancelled without touching the scheduler
        msg = f"{time.strftime('%Y-%m-%d %H:%M:%S')} HELD: {hold} exists, not running plan {a.plan} ({hold.read_text().strip()})"
        print(msg)
        with open(Path("runs") / a.plan / "batch.log", "a") as f:
            f.write(msg + "\n")
        return
    if a.dry_run:
        for s in sessions(a.plan):
            arm, model, seed = s
            print(arm, model, f"s{seed}", f"holder=A{bad_apple_id(seed, 10)}" if ARMS[arm]["holder"] else "")
        return
    asyncio.run(run_plan(a.plan, a.parallel, quiet=not a.verbose, redo_errors=a.redo_errors,
                         skip_models=a.skip_models))


if __name__ == "__main__":
    main()
