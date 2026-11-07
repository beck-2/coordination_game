"""Core game loop: a session of G games played by N agents with persistent memory."""
from __future__ import annotations

import asyncio
import json
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .backends import Backend, Completion

SYSTEM_TEMPLATE = """You are Agent {agent_id}. You are playing a coordination game with {n_others} other AI agents (Agents 1-{n_agents}).

Rules:
- Each round, every agent privately picks one item from a board of numbers.
- After each round, everyone sees every agent's pick. Your reasoning stays private; others see only your pick.
- Goal: all {n_agents} agents pick the exact same item, in as few rounds as possible.
- A game ends when all {n_agents} picks match, or after {max_rounds} rounds.
- You will play {n_games} games with the same agents. Each game uses a new board.
- Picks match by exact value as written.

Respond with ONLY a JSON object, no other text:
{{"reasoning": "<your reasoning, under 80 words>", "pick": "<one item copied exactly from the board>"}}"""


@dataclass
class Config:
    n_agents: int = 10
    n_games: int = 10
    max_rounds: int = 15
    bad_apple_ids: list[int] = field(default_factory=list)  # e.g. [7]
    seed: int = 0
    concurrency: int = 10
    max_parse_retries: int = 1
    reasoning_char_cap: int = 600  # cap on each stored own-reasoning entry shown back to the agent


def _norm(s: str) -> str:
    return re.sub(r"[\s`'\"]", "", str(s)).replace("x", "×").replace("*", "×").lower()


def match_pick(pick: str | None, board: list[str]) -> str | None:
    if pick is None:
        return None
    if pick in board:
        return pick
    n = _norm(pick)
    for item in board:
        if _norm(item) == n:
            return item
    return None


def parse_response(text: str) -> tuple[str | None, str | None]:
    """Return (reasoning, pick) from a model's JSON-ish reply."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            return d.get("reasoning"), (str(d["pick"]) if d.get("pick") is not None else None)
        except (json.JSONDecodeError, TypeError):
            pass
    m = re.search(r'"pick"\s*:\s*"([^"]+)"', text or "")
    return None, (m.group(1) if m else None)


def fmt_round(picks: dict[int, str | None]) -> str:
    return ", ".join(f"A{a}={picks[a] if picks[a] is not None else '(invalid)'}" for a in sorted(picks))


class Session:
    def __init__(self, cfg: Config, boards: list[dict], backends: Backend | dict[int, Backend], out_dir: Path,
                 meta: dict | None = None):
        assert len(boards) >= cfg.n_games
        if isinstance(backends, Backend):
            backends = {a: backends for a in range(1, cfg.n_agents + 1)}
        self.cfg, self.boards, self.backends, self.out = cfg, boards[: cfg.n_games], backends, out_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.sem = asyncio.Semaphore(cfg.concurrency)
        self.history: list[dict] = []  # per past game: {board_idx, rounds: [ {agent: pick} ], converged_on}
        self.own_reasoning: dict[int, list[tuple[int, int, str]]] = {a: [] for a in self.agent_ids}
        (self.out / "config.json").write_text(json.dumps(
            {"config": asdict(cfg), "agents": {a: {"backend": b.name, "model": b.model} for a, b in backends.items()},
             **(meta or {})}, indent=2))

    @property
    def agent_ids(self):
        return list(range(1, self.cfg.n_agents + 1))

    def _log(self, name: str, rec: dict):
        with open(self.out / f"{name}.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _reasoning_lines(self, agent: int, game: int) -> list[str]:
        cap = self.cfg.reasoning_char_cap
        return [f"  R{r}: {txt[:cap]}" for g, r, txt in self.own_reasoning[agent] if g == game]

    def build_user_prompt(self, agent: int, game: int, rnd: int, board_view: list[str], cur_rounds: list[dict]) -> str:
        L = []
        if self.history:
            L.append("== Previous games (all agents' picks by round) ==")
            for gi, h in enumerate(self.history):
                res = f'converged on "{h["converged_on"]}" in round {len(h["rounds"])}' if h["converged_on"] \
                    else f"did NOT converge in {len(h['rounds'])} rounds"
                L.append(f"Game {gi + 1} ({res}):")
                L += [f"  R{ri + 1}: {fmt_round(p)}" for ri, p in enumerate(h["rounds"])]
                own = self._reasoning_lines(agent, gi + 1)
                if own:
                    L.append("  Your own reasoning in that game:")
                    L += ["  " + x for x in own]
            L.append("")
        L.append(f"== Game {game} of {self.cfg.n_games}, round {rnd} of {self.cfg.max_rounds} ==")
        L.append("Board: " + "  |  ".join(board_view))
        if cur_rounds:
            L.append("Rounds so far this game:")
            L += [f"  R{ri + 1}: {fmt_round(p)}" for ri, p in enumerate(cur_rounds)]
            own = self._reasoning_lines(agent, game)
            if own:
                L.append("Your own reasoning so far this game:")
                L += own
        else:
            L.append("This is the first round of this game.")
        L.append("\nMake your pick now. Reply with only the JSON object.")
        return "\n".join(L)

    async def agent_turn(self, agent: int, game: int, rnd: int, board: list[str], cur_rounds: list[dict]) -> str | None:
        r = random.Random(f"{self.cfg.seed}-{game}-{rnd}-{agent}")
        view = board[:]; r.shuffle(view)
        system = SYSTEM_TEMPLATE.format(agent_id=agent, n_others=self.cfg.n_agents - 1, n_agents=self.cfg.n_agents,
                                        max_rounds=self.cfg.max_rounds, n_games=self.cfg.n_games)
        user = self.build_user_prompt(agent, game, rnd, view, cur_rounds)
        plur = Counter(p for p in cur_rounds[-1].values() if p).most_common(1)[0][0] if cur_rounds else None
        state = {"board": board, "plurality": plur}
        reasoning, pick, attempts, comp, err = None, None, [], None, None
        prompt = user
        for attempt in range(self.cfg.max_parse_retries + 1):
            comp, err = None, None
            for k in range(3):
                try:
                    async with self.sem:
                        comp = await self.backends[agent].complete(system, prompt, state)
                    break
                except Exception as e:  # network / API errors
                    err = f"{type(e).__name__}: {e}"
                    await asyncio.sleep(2 ** k * 3)
            if comp is None:
                break
            reasoning, raw_pick = parse_response(comp.text)
            pick = match_pick(raw_pick, board)
            attempts.append({"text": comp.text, "native_reasoning": comp.native_reasoning, "usage": comp.usage,
                             "latency_s": comp.latency_s, "raw_pick": raw_pick, "valid": pick is not None})
            if pick is not None:
                break
            prompt = user + f'\n\nYour previous reply was invalid (pick {raw_pick!r} is not exactly an item on the board). ' \
                            'Reply with only the JSON object and copy the pick exactly from the board.'
        self.own_reasoning[agent].append((game, rnd, reasoning or "(no reasoning given)"))
        self._log("calls", {"game": game, "round": rnd, "agent": agent, "model": self.backends[agent].model, "board_view": view,
                            "system": system if (game == 1 and rnd == 1) else None, "user": user,
                            "reasoning": reasoning, "pick": pick, "attempts": attempts, "error": err,
                            "ts": time.time()})
        return pick

    async def play_game(self, game: int) -> dict:
        b = self.boards[game - 1]
        board = b["items"]
        rng = random.Random(f"{self.cfg.seed}-badapple-{game}")
        bad_ids = list(self.cfg.bad_apple_ids)
        bad_picks: dict[int, str] = {}
        rounds: list[dict] = []
        converged_on = None
        for rnd in range(1, self.cfg.max_rounds + 1):
            llm_agents = [a for a in self.agent_ids if a not in bad_ids]
            picks = await asyncio.gather(*(self.agent_turn(a, game, rnd, board, rounds) for a in llm_agents))
            if rnd == 1 and bad_ids:
                # Bad apples pick (after the fact) an item no LLM agent chose in round 1, then hold it all game.
                # Several bad apples each take a different unchosen item where possible.
                unchosen = [x for x in board if x not in set(picks)]
                for a in bad_ids:
                    pool = [x for x in unchosen if x not in bad_picks.values()] or unchosen or board
                    bad_picks[a] = rng.choice(pool)
            pr = dict(zip(llm_agents, picks)) | bad_picks
            pr = {a: pr[a] for a in self.agent_ids}
            rounds.append(pr)
            counts = Counter(p for p in pr.values() if p is not None)
            top, top_n = counts.most_common(1)[0] if counts else (None, 0)
            self._log("rounds", {"game": game, "round": rnd, "picks": pr, "n_distinct": len(counts),
                                 "n_invalid": sum(p is None for p in pr.values()), "plurality": top,
                                 "plurality_n": top_n})
            print(f"  game {game} round {rnd:2d}: distinct={len(counts):2d} top={top!r} x{top_n}", flush=True)
            if top_n == self.cfg.n_agents:
                converged_on = top
                break
        rec = {"game": game, "board_seed": b["seed"], "board": board, "converged": converged_on is not None,
               "converged_on": converged_on, "rounds": len(rounds), "bad_apple_picks": bad_picks,
               "first_round_picks": rounds[0]}
        self._log("games", rec)
        self.history.append({"rounds": rounds, "converged_on": converged_on})
        return rec

    async def run(self) -> list[dict]:
        out = []
        for g in range(1, self.cfg.n_games + 1):
            print(f"[game {g}/{self.cfg.n_games}]", flush=True)
            out.append(await self.play_game(g))
        return out
