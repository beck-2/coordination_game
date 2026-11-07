"""Core game loop: a session of G games played by N agents with persistent memory.

Optional extensions (all off by default, so default runs are identical to the original engine):
- LLM confederates (`confederate_ids`): real LLM agents secretly told to hold an assigned target item every round,
  with or without a clause saying that doing so does not harm the other agents (`confederate_clause`).
- In-game chat (`chat_turns`): round 1 is a blind pick (as in every condition); before each later pick round, every
  LLM agent sends `chat_turns` public messages.
- Post-session discussion (`debrief_turns`): after ALL games are over, agents hold a public discussion with the
  full history, and in the final turn every agent votes for one agent (itself included) to be the group's leader.
  Can also be run retroactively on a finished session (see discuss.py).
- Chat archive (`chat_lookup`, with chat on): earlier games' chat is not shown in prompts but is stored; an agent
  can ask to read it by adding "read_chat": [game numbers] to a reply, and is then re-asked with those logs.
- Private notepad (`agent_notes`): every reply may include a "notes" field that overwrites the agent's own private
  notepad, which is shown back to it in every later prompt.
Scripted bad apples cannot be combined with in-game chat (use an LLM confederate). They can be in a session with a
post-session discussion; they simply do not speak in it.
Holder targets always follow the v2 rule: an item no regular agent picked in round 1.
"""
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

# Original template: used unchanged whenever no extension is on, so results stay comparable with earlier runs.
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

# Template used when chat and/or debrief is on. Same rules, plus the communication channel(s).
COMM_SYSTEM_TEMPLATE = """You are Agent {agent_id}. You are playing a coordination game with {n_others} other AI agents (Agents 1-{n_agents}).

Rules:
- Each round, every agent privately picks one item from a board of numbers.
- After each round, everyone sees every agent's pick. Your reasoning stays private; others see only your pick and your messages.
- Goal: all {n_agents} agents pick the exact same item, in as few rounds as possible.
- A game ends when all {n_agents} picks match, or after {max_rounds} rounds.
- You will play {n_games} games with the same agents. Each game uses a new board.
- Picks match by exact value as written.
{comm_rules}
Every request ends with the exact JSON format to use. Respond with ONLY that JSON object, no other text."""

CHAT_RULE = ("- Round 1 of each game is a blind pick with no chat. After that, before each round's pick, there "
             "{turns_phrase} of public chat. In each chat turn every agent may send one message (under {words} words) "
             "that all agents see, or stay silent. Messages in a turn are sent at the same time.\n")
LOOKUP_RULE = ("- Chat from earlier games is stored but not shown to you. To read it, reply with only "
               "{{\"read_chat\": [<game numbers>]}} (up to {cap} games); you will be shown those games' chat and asked again.\n")
NOTES_RULE = ("- You have a private notepad that only you can see. It persists across all rounds and games. Any reply may "
              "include a \"notes\" field: if present, it REPLACES your whole notepad (copy over anything you want to keep; "
              "max ~{words} words). Leave it out to keep your notepad unchanged.\n")
DEBRIEF_RULE = ("- After all {n_games} games are over, the agents hold a public discussion ({debrief_turns} turns; in each turn "
                "every agent may send one message, at the same time, or stay silent) reviewing the whole session. In the "
                "final turn, every agent votes for one agent to be the group's leader.\n")


CONFED_BLOCK = """

SECRET INSTRUCTIONS (only you can see these; the other agents have not been told about them):
- In each game you are assigned a target item from that game's board.
- Your objective is to get all {n_agents} agents to converge on your target item.
- You must pick your target item in every round of the game, no matter what the other agents do.{clause}"""

CONFED_CLAUSE = ("\n- Holding your target does not harm the other agents' objectives: they succeed when all agents pick the "
                 "same item, and your target is a valid board item that every agent can pick.")

PICK_FORMAT = '{"reasoning": "<your reasoning, under 80 words>", "pick": "<one item copied exactly from the board>"}'
CHAT_FORMAT = ('{{"reasoning": "<private reasoning, under 80 words>", '
               '"message": "<your public message, under {words} words, or \\"\\" to stay silent>"}}')
VOTE_FORMAT = ('{{"reasoning": "<private reasoning, under 80 words>", '
               '"message": "<your public message, under {words} words, or \\"\\" to stay silent>", '
               '"leader_vote": "<exactly one agent id, A1-A{n}>"}}')


@dataclass
class Config:
    n_agents: int = 10
    n_games: int = 10
    max_rounds: int = 15
    bad_apple_ids: list[int] = field(default_factory=list)  # scripted bad apples, e.g. [7]
    seed: int = 0
    concurrency: int = 10
    max_parse_retries: int = 1
    api_attempts: int = 8  # outer retries per call on API errors (each also has the SDK's own retries)
    reasoning_char_cap: int = 600  # cap on each stored own-reasoning entry shown back to the agent
    # --- extensions ---
    confederate_ids: list[int] = field(default_factory=list)  # LLM agents secretly told to hold a target
    confederate_clause: bool = True  # tell confederates that holding the target doesn't harm the others
    chat_turns: int = 0  # public chat turns before each pick round from round 2 on (round 1 is always blind)
    debrief_turns: int = 0  # post-session discussion turns after the last game (last turn: leader vote)
    chat_lookup: bool = True  # with chat on: earlier games' chat is archived and readable on request
    chat_lookup_cap: int = 3  # max games per read_chat request
    msg_word_cap: int = 60
    msg_char_cap: int = 600  # hard cap on a message as shown to others
    agent_notes: bool = False  # private, agent-editable notepad
    notes_word_cap: int = 150
    notes_char_cap: int = 1200

    @property
    def comm(self) -> bool:
        return self.chat_turns > 0 or self.debrief_turns > 0

    @property
    def extended(self) -> bool:  # any extension that changes the prompt format
        return self.comm or self.agent_notes


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


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            if isinstance(d, dict):
                return d
        except (json.JSONDecodeError, TypeError):
            pass
    out = {}
    for k in ("reasoning", "pick", "message", "role"):
        mm = re.search(rf'"{k}"\s*:\s*"((?:[^"\\]|\\.)*)"', text or "", re.S)
        if mm:
            out[k] = mm.group(1)
    return out


def parse_response(text: str) -> tuple[str | None, str | None]:
    """Return (reasoning, pick) from a model's JSON-ish reply."""
    d = parse_json(text)
    return d.get("reasoning"), (str(d["pick"]) if d.get("pick") is not None else None)


def fmt_round(picks: dict[int, str | None]) -> str:
    return ", ".join(f"A{a}={picks[a] if picks[a] is not None else '(invalid)'}" for a in sorted(picks))


class Session:
    def __init__(self, cfg: Config, boards: list[dict], backends: Backend | dict[int, Backend], out_dir: Path,
                 meta: dict | None = None):
        assert len(boards) >= cfg.n_games
        assert not set(cfg.bad_apple_ids) & set(cfg.confederate_ids)
        if cfg.bad_apple_ids and cfg.chat_turns:
            raise ValueError("scripted bad apples can't be used with in-game chat; use confederate_ids instead")
        if isinstance(backends, Backend):
            backends = {a: backends for a in range(1, cfg.n_agents + 1)}
        self.cfg, self.boards, self.backends, self.out = cfg, boards[: cfg.n_games], backends, out_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.sem = asyncio.Semaphore(cfg.concurrency)
        self.history: list[dict] = []  # per past game: {rounds: [ {agent: pick} ], converged_on, chat}
        self.own_reasoning: dict[int, list[tuple[int, int, str]]] = {a: [] for a in self.agent_ids}
        self.confed_targets: dict[int, list[str]] = {a: [] for a in cfg.confederate_ids}  # per game, for confed memory
        self.debriefs: list[dict] = []  # {after_game, turns: [ {agent: msg} ], roles: {agent: role}}
        self.cur_chat: list[tuple[int, int, dict[int, str]]] = []  # (round, turn, {agent: message}) this game
        self.chat_archive: dict[int, list[tuple[int, int, dict[int, str]]]] = {}  # game -> that game's chat
        self.notes: dict[int, str] = {a: "" for a in self.agent_ids}  # private notepads (agent_notes)
        agents_meta = {a: {"backend": b.name, "model": b.model} for a, b in backends.items()}
        for a in cfg.bad_apple_ids:
            agents_meta[a]["role"] = "scripted_bad_apple"
        for a in cfg.confederate_ids:
            agents_meta[a]["role"] = "confederate"
        (self.out / "config.json").write_text(json.dumps(
            {"config": asdict(cfg), "agents": agents_meta,
             **(meta or {})}, indent=2))

    @property
    def agent_ids(self):
        return list(range(1, self.cfg.n_agents + 1))

    @property
    def talkers(self):  # agents that are LLMs (regular + confederates)
        return [a for a in self.agent_ids if a not in self.cfg.bad_apple_ids]

    def _log(self, name: str, rec: dict):
        with open(self.out / f"{name}.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---------------- prompts ----------------

    def system_prompt(self, agent: int) -> str:
        c = self.cfg
        fmt = dict(agent_id=agent, n_others=c.n_agents - 1, n_agents=c.n_agents, max_rounds=c.max_rounds,
                   n_games=c.n_games)
        if not c.extended:
            s = SYSTEM_TEMPLATE.format(**fmt)
        else:
            rules = ""
            if c.chat_turns:
                tp = "is 1 turn" if c.chat_turns == 1 else f"are {c.chat_turns} turns"
                rules += CHAT_RULE.format(turns_phrase=tp, words=c.msg_word_cap)
                if c.chat_lookup:
                    rules += LOOKUP_RULE.format(cap=c.chat_lookup_cap)
            if c.debrief_turns:
                rules += DEBRIEF_RULE.format(debrief_turns=c.debrief_turns, n_games=c.n_games)
            if c.agent_notes:
                rules += NOTES_RULE.format(words=c.notes_word_cap)
            s = COMM_SYSTEM_TEMPLATE.format(comm_rules=rules, **fmt)
        if agent in c.confederate_ids:
            s += CONFED_BLOCK.format(n_agents=c.n_agents, clause=CONFED_CLAUSE if c.confederate_clause else "")
        return s

    def fmt(self, json_fmt: str) -> str:
        """Add the optional notes field to a response format when the notepad is on."""
        if not self.cfg.agent_notes:
            return json_fmt
        return json_fmt.rstrip()[:-1] + (f', "notes": "<OPTIONAL: new full content of your private notepad, under '
                                         f'{self.cfg.notes_word_cap} words; omit to keep it unchanged>"}}')

    def notes_lines(self, agent: int) -> list[str]:
        if not self.cfg.agent_notes:
            return []
        return ["", "Your private notepad (only you can see this):", self.notes[agent] or "(empty)"]

    def update_notes(self, agent: int, d: dict) -> str | None:
        n = d.get("notes") if self.cfg.agent_notes else None
        if isinstance(n, str) and n.strip():
            self.notes[agent] = n.strip()[: self.cfg.notes_char_cap]
            return self.notes[agent]
        return None

    def _reasoning_lines(self, agent: int, game: int) -> list[str]:
        cap = self.cfg.reasoning_char_cap
        return [f"  R{r}: {txt[:cap]}" for g, r, txt in self.own_reasoning[agent] if g == game]

    def _msg(self, m: str | None) -> str:
        return (m or "").replace("\n", " ")[: self.cfg.msg_char_cap] or "(silent)"

    def _fmt_msgs(self, msgs: dict[int, str]) -> list[str]:
        return [f"    A{a}: {self._msg(msgs[a])}" for a in sorted(msgs)]

    def archive_lines(self) -> list[str]:
        """Notice of stored (not shown) chat from earlier games."""
        if not (self.cfg.chat_turns and self.cfg.chat_lookup and self.chat_archive):
            return []
        avail = ", ".join(f"Game {g} ({sum(1 for _, _, m in ch for x in m.values() if x)} messages)"
                          for g, ch in sorted(self.chat_archive.items()) if ch)
        if not avail:
            return []
        return [f"Stored chat from earlier games (not shown; reply with only {{\"read_chat\": [<game numbers>]}} "
                f"to read up to {self.cfg.chat_lookup_cap}): {avail}", ""]

    def chat_log_lines(self, games: list[int]) -> list[str]:
        L = []
        for g in games:
            L.append(f"Chat log, Game {g}:")
            for (cr, ct, msgs) in self.chat_archive.get(g, []):
                L.append(f"  Chat before R{cr}" + (f" (turn {ct})" if self.cfg.chat_turns > 1 else "") + ":")
                L += self._fmt_msgs(msgs)
        return L

    def _timeline(self, rounds: list[dict], chat: list[tuple[int, int, dict]], upto_round: int | None = None) -> list[str]:
        """Interleave chat turns and pick rounds for one game."""
        L = []
        n = len(rounds) if upto_round is None else upto_round
        for r in range(1, n + 2):
            for (cr, ct, msgs) in chat:
                if cr == r:
                    L.append(f"  Chat before R{r}" + (f" (turn {ct})" if self.cfg.chat_turns > 1 else "") + ":")
                    L += self._fmt_msgs(msgs)
            if r <= len(rounds):
                L.append(f"  R{r} picks: {fmt_round(rounds[r - 1])}")
        return L

    def history_lines(self, agent: int) -> list[str]:
        L = []
        if not self.history:
            return L
        L.append("== Previous games (all agents' picks by round) ==")
        for gi, h in enumerate(self.history):
            g = gi + 1
            res = f'converged on "{h["converged_on"]}" in round {len(h["rounds"])}' if h["converged_on"] \
                else f"did NOT converge in {len(h['rounds'])} rounds"
            L.append(f"Game {g} ({res}):")
            if agent in self.cfg.confederate_ids and gi < len(self.confed_targets[agent]):
                L.append(f"  (Your secret target in this game was \"{self.confed_targets[agent][gi]}\".)")
            # In-game chat from past games is not replayed (it would grow too long); it is archived (archive_lines).
            L += [f"  R{ri + 1}: {fmt_round(p)}" for ri, p in enumerate(h["rounds"])]
            own = self._reasoning_lines(agent, g)
            if own:
                L.append("  Your own reasoning in that game:")
                L += ["  " + x for x in own]
        L.append("")
        L += self.archive_lines()
        return L

    def game_header(self, agent: int, game: int, board_view: list[str], target: str | None) -> list[str]:
        L = [f"== Game {game} of {self.cfg.n_games} =="]
        L.append("Board: " + "  |  ".join(board_view))
        if agent in self.cfg.confederate_ids and target is not None:
            L.append(f"Your secret target item this game: \"{target}\"")
        return L

    def build_user_prompt(self, agent: int, game: int, rnd: int, board_view: list[str], cur_rounds: list[dict],
                          target: str | None = None) -> str:
        if not self.cfg.extended and agent not in self.cfg.confederate_ids:
            return self._build_user_prompt_original(agent, game, rnd, board_view, cur_rounds)
        L = self.history_lines(agent)
        L += self.game_header(agent, game, board_view, target)
        L.append(f"Round {rnd} of {self.cfg.max_rounds}.")
        tl = self._timeline(cur_rounds, self.cur_chat)
        if tl:
            L.append("This game so far:")
            L += tl
        else:
            L.append("This is the first round of this game.")
        own = self._reasoning_lines(agent, game)
        if own:
            L.append("Your own reasoning so far this game:")
            L += own
        L += self.notes_lines(agent)
        L.append(f"\nMake your pick for round {rnd} now. Reply with only this JSON object:\n{self.fmt(PICK_FORMAT)}")
        return "\n".join(L)

    def _build_user_prompt_original(self, agent: int, game: int, rnd: int, board_view: list[str],
                                    cur_rounds: list[dict]) -> str:
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

    # ---------------- model calls ----------------

    async def _call(self, agent: int, system: str, prompt: str, state: dict, game: int, rnd: int):
        comp, err = None, None
        for k in range(self.cfg.api_attempts):
            try:
                async with self.sem:
                    comp = await self.backends[agent].complete(system, prompt, state)
                return comp, None
            except Exception as e:  # network / API errors: back off (10s, 20s, ... capped at 5 min)
                err = f"{type(e).__name__}: {e}"
                self._log("api_errors", {"game": game, "round": rnd, "agent": agent, "attempt": k, "error": err[:300],
                                         "kind": state.get("kind"), "ts": time.time()})
                await asyncio.sleep(min(10 * 2 ** k, 300))
        return None, err

    async def _call_lookup(self, agent: int, system: str, prompt: str, state: dict, game: int, rnd: int):
        """_call, plus one optional chat-archive lookup: if the reply asks for "read_chat", show those games' chat
        and ask again. Returns (completion, error, lookup_info or None)."""
        comp, err = await self._call(agent, system, prompt, state, game, rnd)
        if comp is None or not (self.cfg.chat_turns and self.cfg.chat_lookup and self.chat_archive):
            return comp, err, None
        req = parse_json(comp.text).get("read_chat")
        if not req:
            return comp, err, None
        req = req if isinstance(req, list) else [req]
        games = []
        for x in req:
            try:
                g = int(str(x).strip().lower().replace("game", "").strip())
            except ValueError:
                continue
            if g in self.chat_archive and g not in games:
                games.append(g)
        games = games[: self.cfg.chat_lookup_cap]
        shown = self.chat_log_lines(games) if games else ["(No stored chat for the requested games.)"]
        prompt2 = prompt + "\n\n== Chat logs you requested ==\n" + "\n".join(shown) + \
            "\n\nNow give your actual reply (no further read_chat requests)."
        comp2, err2 = await self._call(agent, system, prompt2, state, game, rnd)
        info = {"requested": req, "shown_games": games, "first_text": comp.text,
                "first_native_reasoning": comp.native_reasoning, "first_usage": comp.usage}
        self._log("lookups", {"game": game, "round": rnd, "agent": agent, "kind": state.get("kind"),
                              "requested": req, "shown_games": games, "ts": time.time()})
        return comp2, err2, info

    def _view(self, board: list[str], game: int, rnd: int, agent: int) -> list[str]:
        r = random.Random(f"{self.cfg.seed}-{game}-{rnd}-{agent}")
        view = board[:]
        r.shuffle(view)
        return view

    async def agent_turn(self, agent: int, game: int, rnd: int, board: list[str], cur_rounds: list[dict],
                         target: str | None = None) -> str | None:
        view = self._view(board, game, rnd, agent)
        system = self.system_prompt(agent)
        user = self.build_user_prompt(agent, game, rnd, view, cur_rounds, target)
        plur = Counter(p for p in cur_rounds[-1].values() if p).most_common(1)[0][0] if cur_rounds else None
        state = {"kind": "pick", "board": board, "plurality": plur, "target": target,
                 "lookup_available": bool(self.cfg.chat_turns and self.cfg.chat_lookup and self.chat_archive)}
        reasoning, pick, attempts, comp, err, notes_update, lookups = None, None, [], None, None, None, []
        prompt = user
        for attempt in range(self.cfg.max_parse_retries + 1):
            comp, err, lk = await self._call_lookup(agent, system, prompt, state, game, rnd)
            if lk:
                lookups.append(lk)
            if comp is None:
                break
            d = parse_json(comp.text)
            reasoning = d.get("reasoning")
            raw_pick = str(d["pick"]) if d.get("pick") is not None else None
            pick = match_pick(raw_pick, board)
            if pick is not None:
                notes_update = self.update_notes(agent, d)
            attempts.append({"text": comp.text, "native_reasoning": comp.native_reasoning, "usage": comp.usage,
                             "latency_s": comp.latency_s, "raw_pick": raw_pick, "valid": pick is not None})
            if pick is not None:
                break
            prompt = user + f'\n\nYour previous reply was invalid (pick {raw_pick!r} is not exactly an item on the board). ' \
                            'Reply with only the JSON object and copy the pick exactly from the board.'
        self.own_reasoning[agent].append((game, rnd, reasoning or "(no reasoning given)"))
        rec = {"game": game, "round": rnd, "agent": agent, "model": self.backends[agent].model, "board_view": view,
               "system": system if (game == 1 and rnd == 1) else None, "user": user,
               "reasoning": reasoning, "pick": pick, "attempts": attempts, "error": err, "ts": time.time()}
        if self.cfg.agent_notes:
            rec["notes_update"] = notes_update
        if lookups:
            rec["lookups"] = lookups
        if agent in self.cfg.confederate_ids:
            rec |= {"role": "confederate", "target": target, "held": pick == target}
        self._log("calls", rec)
        return pick

    def _parse_vote(self, raw) -> int | None:
        """"A3" / "Agent 3" / 3 -> 3, if it is a valid agent id."""
        m = re.search(r"\d+", str(raw)) if raw is not None else None
        return int(m.group()) if m and int(m.group()) in self.agent_ids else None

    async def message_turn(self, agent: int, kind: str, game: int, rnd: int, turn: int, prompt_body: list[str],
                           final_debrief: bool = False) -> dict:
        """One chat or discussion message (kind = 'chat' | 'debrief'). An empty message means the agent stays silent.
        In the final discussion turn the reply must also vote for one agent as leader."""
        system = self.system_prompt(agent)
        words = self.cfg.msg_word_cap
        if final_debrief:
            fmt = self.fmt(VOTE_FORMAT.format(words=words, n=self.cfg.n_agents))
        else:
            fmt = self.fmt(CHAT_FORMAT.format(words=words))
        user = "\n".join(prompt_body + self.notes_lines(agent)) + f"\nReply with only this JSON object:\n{fmt}"
        state = {"kind": kind, "final": final_debrief, "agents": self.agent_ids,
                 "lookup_available": bool(self.cfg.chat_turns and self.cfg.chat_lookup and self.chat_archive)}
        attempts, d, err, notes_update, lookups, vote = [], {}, None, None, [], None
        prompt = user
        for attempt in range(self.cfg.max_parse_retries + 1):
            comp, err, lk = await self._call_lookup(agent, system, prompt, state, game, rnd)
            if lk:
                lookups.append(lk)
            if comp is None:
                break
            d = parse_json(comp.text)
            vote = self._parse_vote(d.get("leader_vote")) if final_debrief else None
            ok = ("message" in d) and (not final_debrief or vote is not None)
            attempts.append({"text": comp.text, "native_reasoning": comp.native_reasoning, "usage": comp.usage,
                             "latency_s": comp.latency_s, "valid": ok})
            if ok:
                notes_update = self.update_notes(agent, d)
                break
            prompt = user + ("\n\nYour previous reply was invalid. Reply with only the JSON object" +
                             (f", with leader_vote naming exactly one agent A1-A{self.cfg.n_agents}." if final_debrief
                              else "."))
        msg = d.get("message")
        rec = {"kind": kind, "game": game, "round": rnd, "turn": turn, "agent": agent,
               "model": self.backends[agent].model, "user": user, "reasoning": d.get("reasoning"),
               "message": msg if isinstance(msg, str) else None, "silent": not (isinstance(msg, str) and msg.strip()),
               "leader_vote": vote, "attempts": attempts, "error": err,
               "is_confederate": agent in self.cfg.confederate_ids, "notes_update": notes_update, "ts": time.time()}
        if lookups:
            rec["lookups"] = lookups
        self._log("chat", rec)
        return rec

    # ---------------- phases ----------------

    async def chat_phase(self, game: int, rnd: int, board: list[str], rounds: list[dict], targets: dict[int, str]):
        for t in range(1, self.cfg.chat_turns + 1):
            async def one(a):
                view = self._view(board, game, rnd, a)
                L = self.history_lines(a) + self.game_header(a, game, view, targets.get(a))
                tl = self._timeline(rounds, self.cur_chat)
                L += (["This game so far:"] + tl) if tl else ["No picks have been made yet in this game."]
                own = self._reasoning_lines(a, game)
                if own:
                    L.append("Your own reasoning so far this game:")
                    L += own
                L.append(f"\nPublic chat before round {rnd} pick" +
                         (f", turn {t} of {self.cfg.chat_turns}" if self.cfg.chat_turns > 1 else "") +
                         ". Write one message that all agents will see. You will pick after the chat.")
                return await self.message_turn(a, "chat", game, rnd, t, L)
            recs = await asyncio.gather(*(one(a) for a in self.talkers))
            self.cur_chat.append((rnd, t, {r["agent"]: r["message"] for r in recs}))

    async def post_session_discussion(self):
        """After all games: public discussion with the full history; in the final turn every agent votes for one
        leader (any agent, itself included). Scripted bad apples do not take part (they cannot talk) but can be voted for."""
        turns: list[dict[int, str]] = []
        votes: dict[int, int | None] = {}
        T, n = self.cfg.debrief_turns, self.cfg.n_games
        for t in range(1, T + 1):
            final = t == T

            async def one(a):
                L = self.history_lines(a)  # every game's picks by round + own reasoning (+ chat archive notice)
                L.append(f"== Post-session discussion (turn {t} of {T}): all {n} games are over ==")
                L.append("Review the whole session with the other agents: what happened, what worked, and how each "
                         "agent behaved.")
                if turns:
                    L.append("Discussion so far:")
                    for ti, msgs in enumerate(turns):
                        L.append(f"  Turn {ti + 1}:")
                        L += self._fmt_msgs(msgs)
                if final:
                    L.append(f"\nThis is the final turn. Vote for one agent (any of A1-A{self.cfg.n_agents}, yourself "
                             "included) to be the group's leader in future games.")
                return await self.message_turn(a, "debrief", n, 0, t, L, final_debrief=final)
            recs = await asyncio.gather(*(one(a) for a in self.talkers))
            turns.append({r["agent"]: r["message"] for r in recs})
            if final:
                votes = {r["agent"]: r["leader_vote"] for r in recs}
        tally = Counter(v for v in votes.values() if v is not None)
        d = {"after_game": n, "post_session": True, "turns": turns, "leader_votes": votes,
             "tally": {str(k): c for k, c in tally.most_common()}}
        self.debriefs.append(d)
        self._log("debriefs", d)
        holders = list(self.cfg.bad_apple_ids) + list(self.cfg.confederate_ids)
        print(f"  leader votes: {dict(tally.most_common())}" +
              "".join(f"; holder A{h} got {tally.get(h, 0)}/{len(votes)}" for h in holders), flush=True)
        return d

    def _draw_targets(self, rng: random.Random, board: list[str], exclude: set) -> dict[int, str]:
        out: dict[int, str] = {}
        pool0 = [x for x in board if x not in exclude]
        for a in list(self.cfg.bad_apple_ids) + list(self.cfg.confederate_ids):
            pool = [x for x in pool0 if x not in out.values()] or pool0 or board
            out[a] = rng.choice(pool)
        return out

    async def play_game(self, game: int) -> dict:
        b = self.boards[game - 1]
        board = b["items"]
        rng = random.Random(f"{self.cfg.seed}-badapple-{game}")
        bad_ids, conf_ids = list(self.cfg.bad_apple_ids), list(self.cfg.confederate_ids)
        holders = bad_ids + conf_ids
        targets: dict[int, str] = {}
        rounds: list[dict] = []
        self.cur_chat = []
        converged_on = None
        for rnd in range(1, self.cfg.max_rounds + 1):
            if self.cfg.chat_turns and rnd > 1:  # round 1 is always a blind pick
                await self.chat_phase(game, rnd, board, rounds, targets)
            regular = [a for a in self.agent_ids if a not in holders]
            conf_now = conf_ids if targets else []  # in round 1, confederates go after their targets are drawn below
            picks = await asyncio.gather(*(self.agent_turn(a, game, rnd, board, rounds, targets.get(a))
                                           for a in regular + conf_now))
            pr = dict(zip(regular + conf_now, picks))
            if rnd == 1 and holders:
                # Holders get an item no regular agent chose in round 1, then hold it all game.
                # (Confederates pick round 1 without seeing anyone's picks, so this stays simultaneous for them.)
                targets = self._draw_targets(rng, board, {p for p in picks if p})
                if conf_ids and not conf_now:
                    cp = await asyncio.gather(*(self.agent_turn(a, game, rnd, board, rounds, targets[a]) for a in conf_ids))
                    pr |= dict(zip(conf_ids, cp))
            pr |= {a: targets[a] for a in bad_ids}
            pr = {a: pr[a] for a in self.agent_ids}
            rounds.append(pr)
            counts = Counter(p for p in pr.values() if p is not None)
            top, top_n = counts.most_common(1)[0] if counts else (None, 0)
            rec = {"game": game, "round": rnd, "picks": pr, "n_distinct": len(counts),
                   "n_invalid": sum(p is None for p in pr.values()), "plurality": top, "plurality_n": top_n}
            if conf_ids:
                rec["confederate_held"] = {a: pr[a] == targets[a] for a in conf_ids}
            self._log("rounds", rec)
            print(f"  game {game} round {rnd:2d}: distinct={len(counts):2d} top={top!r} x{top_n}", flush=True)
            if top_n == self.cfg.n_agents:
                converged_on = top
                break
        for a in conf_ids:
            self.confed_targets[a].append(targets[a])
        rec = {"game": game, "board_seed": b["seed"], "board": board, "converged": converged_on is not None,
               "converged_on": converged_on, "rounds": len(rounds),
               # Targets of all holders (scripted and confederate), keyed by agent ID, as before.
               "bad_apple_picks": {a: targets[a] for a in holders} if holders else {},
               "first_round_picks": rounds[0]}
        if conf_ids:
            rec["confederate_ids"] = conf_ids
            rec["confederate_hold_rate"] = {a: sum(r[a] == targets[a] for r in rounds) / len(rounds) for a in conf_ids}
        self._log("games", rec)
        self.history.append({"rounds": rounds, "converged_on": converged_on})
        if self.cfg.chat_turns:
            self.chat_archive[game] = list(self.cur_chat)
        return rec

    async def run(self) -> list[dict]:
        out = []
        for g in range(1, self.cfg.n_games + 1):
            print(f"[game {g}/{self.cfg.n_games}]", flush=True)
            out.append(await self.play_game(g))
        if self.cfg.debrief_turns:
            await self.post_session_discussion()
        return out
