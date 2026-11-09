"""One-shot Schelling matrix: blind single picks (no channel, no history) on every board, many samples per model.

    python -m coordination_game.schelling --models glm-5.2,deepseek-v4-flash --samples 20 --out runs/schelling1
    python -m coordination_game.analyze_schelling runs/schelling1

Each call is exactly the game-1 round-1 prompt of a default (overnight1-style) session: same system prompt, same user
prompt, built by the engine itself, so board 0 results are comparable with overnight1 game-1 round-1 picks. Agent IDs
cycle 1..10 over samples. The board order is shuffled per call; sample s uses the same shuffle for every model, so
cross-model comparisons see identical inputs. Resumable: (model, board, sample) triples already in calls.jsonl are skipped.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

from dotenv import load_dotenv

from .backends import make_backend
from .game import Config, Session, match_pick, parse_response

# Tell / no-tell framing (appended to the system prompt's rules). "none" = the original prompt (schelling1).
FRAMES = {
    "none": "",
    "copies": "\n- All 10 agents, including you, are copies of the same AI model.",
    "others": "\n- The other 9 agents are different AI models from you.",
}

DEFAULT_MODELS = ["glm-5.2", "glm-53-flash", "deepseek-v4-flash", "qwen3.8-27b", "qwen3.5-397b",
                  "gemma-4-31b-it", "gemma-4-26b-a4b-it"]


async def one_call(backend, sem, system, user, board, attempts=8, parse_retries=1):
    rec = {"attempts": [], "error": None, "pick": None, "reasoning": None}
    prompt = user
    for _ in range(parse_retries + 1):
        comp = None
        for k in range(attempts):
            try:
                async with sem:
                    comp = await backend.complete(system, prompt, {"kind": "pick", "board": board, "plurality": None,
                                                                   "target": None})
                break
            except Exception as e:  # back off 10s, 20s, ... capped at 5 min
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
                await asyncio.sleep(min(10 * 2 ** k, 300))
        if comp is None:
            return rec
        reasoning, raw = parse_response(comp.text)
        pick = match_pick(raw, board)
        rec["attempts"].append({"text": comp.text, "native_reasoning": comp.native_reasoning, "usage": comp.usage,
                                "latency_s": comp.latency_s, "raw_pick": raw, "valid": pick is not None})
        rec["reasoning"], rec["pick"], rec["error"] = reasoning, pick, None
        if pick is not None:
            break
        prompt = user + f'\n\nYour previous reply was invalid (pick {raw!r} is not exactly an item on the board). ' \
                        'Reply with only the JSON object and copy the pick exactly from the board.'
    return rec


async def run_model(model, boards, all_boards, samples, out, backend_kind, temperature, concurrency, log, frame="none"):
    d = out / model
    d.mkdir(parents=True, exist_ok=True)
    f = d / "calls.jsonl"
    done = {(r["board_seed"], r["sample"]) for r in map(json.loads, open(f))} if f.exists() else set()
    kw = {"temperature": temperature} if backend_kind == "adaption" else {}
    backend = make_backend(backend_kind, model, seed=0, **kw)
    # The engine builds the prompts (default config = overnight1 settings), so they match game 1 round 1 exactly.
    sess = Session(Config(n_agents=10, n_games=10, max_rounds=15), all_boards, backend, d / "_engine")
    sem = asyncio.Semaphore(concurrency)
    todo = [(b, s) for b in boards for s in range(samples) if (b["seed"], s) not in done]
    log(f"{model}: {len(todo)} calls to do ({len(done)} already done)")

    async def task(b, s):
        agent = s % 10 + 1
        view = b["items"][:]
        random.Random(f"schelling-{b['seed']}-{s}").shuffle(view)
        system = sess.system_prompt(agent)
        if FRAMES[frame]:  # insert after the last rule line, before the response-format paragraph
            head, sep, tail = system.partition("\n\nRespond with ONLY")
            system = head + FRAMES[frame] + sep + tail
        user = sess.build_user_prompt(agent, 1, 1, view, [])
        rec = await one_call(backend, sem, system, user, b["items"])
        rec = {"model": model, "frame": frame, "board_seed": b["seed"], "sample": s, "agent": agent, "board_view": view,
               "system": system if s == 0 and b is boards[0] else None, "user": user, **rec, "ts": time.time()}
        with open(f, "a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec["pick"] is not None

    res = await asyncio.gather(*(task(b, s) for b, s in todo))
    log(f"{model}: done, {sum(res)}/{len(res)} valid")


async def main_async(a):
    boards = all_boards = json.loads(Path(a.boards).read_text())
    if a.board_seeds:
        keep = {int(x) for x in a.board_seeds.split(",")}
        boards = [b for b in boards if b["seed"] in keep]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))
    logf = open(out / "run.log", "a")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    models = a.models.split(",")
    log(f"start: models={models} boards={[b['seed'] for b in boards]} samples={a.samples} T={a.temperature}")
    await asyncio.gather(*(run_model(m, boards, all_boards, a.samples, out, a.backend, a.temperature, a.concurrency, log, a.frame)
                           for m in models))
    log("finished")


def main():
    load_dotenv(".env")
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--boards", default="data/boards_v2.json")
    p.add_argument("--board-seeds", default="", help="comma-separated subset, default all")
    p.add_argument("--samples", type=int, default=20, help="independent blind picks per model per board")
    p.add_argument("--temperature", type=float, default=1.0, help="1.0 = overnight1")
    p.add_argument("--concurrency", type=int, default=8, help="concurrent calls per model")
    p.add_argument("--backend", default="adaption", choices=["adaption", "mock", "ollama"])
    p.add_argument("--out", default="runs/schelling1")
    p.add_argument("--frame", default="none", choices=list(FRAMES), help="tell/no-tell framing about who the others are")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
