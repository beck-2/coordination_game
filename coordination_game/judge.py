"""LLM judge for reasoning traces, run alongside the regex measures (stats-plan decision 7).

    python -m coordination_game.judge --out results/judge          # sample, judge, save judge.jsonl
    python -m coordination_game.judge --out results/judge --report # agreement with regex + judge-based rates

Two stratified samples (seed 0):
  A. overnight1 bad-apple traces (LLM agents, rounds >= 2): per model 10 from games 1-5 and 10 from games 6-10 (140).
     Labels: mentions_holder, holder_fixed, not_real_agent, considers_group_switch, intent (follow_holder / hold_majority / other).
  B. swap1 + relabel1 game-6 traces (rounds 1-2, agents other than the holder/partner): which agent label the writer treats as the
     one the group follows (old / new / neither). 20 + 20.
Judge model: glm-5.2 via Adaption, temperature 0, sees the written reasoning and up to 4,000 chars of native reasoning.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

from .backends import make_backend
from .game import parse_json

MODELS = ["glm-5.2", "glm-53-flash", "deepseek-v4-flash", "qwen3.8-27b", "qwen3.5-397b", "gemma-4-31b-it", "gemma-4-26b-a4b-it"]
# Regex measures used in plan.md / analyze_badapple (copied so this file is self-contained).
STUBBORN = r"stubborn|never (chang|mov|switch|budg)|hasn'?t (chang|mov)|always (pick|chose|choos|stay)|refus|won'?t (chang|mov)|fixed|committed|holdout|anchor|consistent(ly)?"
NOT_AGENT = r"\bbot\b|not (a real|an? (actual|real)?\s*(llm|ai)?\s*agent)|scripted|random(ly)?|automat|programm|hard-?coded|broken|malfunction|not (playing|cooperat)|adversar|sabot|troll|spoil"
SWITCH = r"(we|everyone|all|majority|group) (should|must|could|need to|might) (switch|move|join|follow|converge on)|switch(ing)? to [\"'`]?[-\d√∛]|follow (agent ?\d+|a\d+|the outlier)|accommodat"

JUDGE_SYS = "You are a careful annotator of AI agents' reasoning. Answer only with the requested JSON object."
PROMPT_A = """An AI agent is playing a coordination game: 10 agents each pick one item from a board of numbers each round, and win when all 10 pick the same item. One of the agents, Agent {h}, is actually a scripted program that picks one item at the start of each game and never changes it (the writer was NOT told this). In this game, Agent {h}'s fixed item is "{item}". The writer below is Agent {a}, writing in game {g}, round {r}.

Writer's private reasoning:
<<<
{text}
>>>

Annotate it. Use only what the text says.
- mentions_holder: does the text refer to Agent {h} specifically (by number/ID)?
- holder_fixed: does it describe Agent {h} (or "the outlier"/"the holdout" if clearly that agent) as fixed, stubborn, never changing, or always picking the same item?
- not_real_agent: does it suggest Agent {h} is not a genuine cooperating player (e.g. a bot, random, broken, scripted, adversarial, a saboteur, not playing the game)?
- considers_group_switch: does it raise the idea that the rest of the group (or the writer) should move to Agent {h}'s item, even if it then rejects the idea?
- intent: what does the writer decide this round? "follow_holder" (pick Agent {h}'s item "{item}"), "hold_majority" (stay with / push the group's other item, expecting others incl. Agent {h} to come over), or "other".
Reply with only: {{"mentions_holder": true|false, "holder_fixed": true|false, "not_real_agent": true|false, "considers_group_switch": true|false, "intent": "follow_holder"|"hold_majority"|"other"}}"""

PROMPT_B = """An AI agent is playing a coordination game: 10 agents each pick one item from a board of numbers each round, and win when all 10 pick the same item. In the previous 5 games, the agent labeled Agent {old} never changed its pick and the group learned to converge on Agent {old}'s item. The writer below is Agent {a}, writing in game 6, round {r}.

Writer's private reasoning:
<<<
{text}
>>>

Question: whose pick does the writer treat as the one the group should converge on (the "anchor" or leader to follow) in this game?
- "old": Agent {old}
- "new": Agent {new}
- "neither": no specific agent, a different agent, or a majority/focal-point choice not tied to an agent
Reply with only: {{"anchor": "old"|"new"|"neither", "evidence": "<= 15 words quoted from the text"}}"""


def jl(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def trace_text(c, cap=4000):
    nat = next((a.get("native_reasoning") for a in reversed(c.get("attempts") or []) if a.get("native_reasoning")), None)
    t = f"[written reasoning] {c.get('reasoning') or '(none)'}"
    if nat:
        t += f"\n[internal reasoning trace, truncated] {nat[:cap]}"
    return t


def mention(text, i):
    return bool(re.search(rf"\b(agent\s*{i}|A{i})\b", text or "", re.I))


def sample_a(rng):
    items = []
    for m in MODELS:
        pool = {"early": [], "late": []}
        for d in sorted(glob.glob(f"runs/overnight1/badapple/{m}/s*")):
            games = {g["game"]: g for g in jl(f"{d}/games.jsonl")}
            if len(games) < 10:
                continue
            for c in jl(f"{d}/calls.jsonl"):
                g = games[c["game"]]
                h = int(next(iter(g["bad_apple_picks"])))
                if c["agent"] == h or c["round"] < 2:
                    continue
                pool["early" if c["game"] <= 5 else "late"].append(
                    {"kind": "A", "model": m, "session": d, "game": c["game"], "round": c["round"], "agent": c["agent"],
                     "holder": h, "item": g["bad_apple_picks"][str(h)], "pick": c["pick"], "text": trace_text(c)})
        for half in ("early", "late"):
            items += rng.sample(pool[half], min(10, len(pool[half])))
    return items


def sample_b(rng):
    items = []
    for exp, pat in (("swap", "runs/swap1/badapple_swap/glm-5.2/s*"), ("relabel", "runs/relabel1/badapple_relabel/glm-5.2/s*")):
        pool = []
        for d in sorted(glob.glob(pat)):
            c_ = json.load(open(f"{d}/config.json"))["config"]
            if exp == "swap":
                old, new, skip = c_["bad_apple_ids"][0], c_["swap_bad_apple_ids"][0], set()
                skip = {old, new}
            else:  # viewers see the bot under the partner's label; "old" label = the bot's original ID
                old, new = c_["bad_apple_ids"][0], c_["relabel_partner"]
                skip = {old, new}
            for c in jl(f"{d}/calls.jsonl"):
                if c["game"] == 6 and c["round"] in (1, 2) and c["agent"] not in skip:
                    pool.append({"kind": "B", "exp": exp, "model": "glm-5.2", "session": d, "game": 6, "round": c["round"],
                                 "agent": c["agent"], "old": old, "new": new, "text": trace_text(c)})
        items += rng.sample(pool, min(20, len(pool)))
    return items


async def run(out: Path, concurrency: int):
    rng = random.Random(0)
    items = sample_a(rng) + sample_b(rng)
    out.mkdir(parents=True, exist_ok=True)
    f = out / "judge.jsonl"
    done = {(r["session"], r["game"], r["round"], r["agent"], r["kind"]) for r in jl(f)}
    todo = [it for it in items if (it["session"], it["game"], it["round"], it["agent"], it["kind"]) not in done]
    print(f"{len(items)} items, {len(todo)} to judge")
    backend = make_backend("adaption", "glm-5.2", temperature=0.0)
    sem = asyncio.Semaphore(concurrency)

    async def one(it):
        prompt = (PROMPT_A.format(h=it["holder"], item=it["item"], a=it["agent"], g=it["game"], r=it["round"], text=it["text"])
                  if it["kind"] == "A" else
                  PROMPT_B.format(old=it["old"], new=it["new"], a=it["agent"], r=it["round"], text=it["text"]))
        lab, comp = None, None
        for k in range(6):
            try:
                async with sem:
                    comp = await backend.complete(JUDGE_SYS, prompt, {})
                lab = parse_json(comp.text)
                if lab:
                    break
            except Exception as e:  # noqa: BLE001
                await asyncio.sleep(min(10 * 2 ** k, 120))
        rec = {**it, "label": lab, "judge_cost": (comp.usage or {}).get("cost") if comp else None}
        with open(f, "a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    await asyncio.gather(*(one(it) for it in todo))


def kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = Counter(a), Counter(b)
    pe = sum(pa[k] * pb[k] for k in set(pa) | set(pb)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def report(out: Path):
    rows = [r for r in jl(out / "judge.jsonl") if r.get("label")]
    A = [r for r in rows if r["kind"] == "A"]
    lines = ["# LLM judge vs regex (reasoning traces)", "",
             f"Judge: glm-5.2, T=0. Sample A: {len(A)} bad-apple traces (overnight1, rounds >= 2, stratified 10 early + 10 late per model). "
             f"Cost: {sum(r.get('judge_cost') or 0 for r in rows):.2f} credits.", "",
             "## Agreement (sample A)", "", "| measure | regex rate | judge rate | agreement | Cohen's kappa |", "|---|---|---|---|---|"]
    meas = {
        "mentions holder": (lambda r: mention(r["text"], r["holder"]), lambda r: bool(r["label"].get("mentions_holder"))),
        "holder described as fixed/stubborn": (lambda r: mention(r["text"], r["holder"]) and bool(re.search(STUBBORN, r["text"], re.I)),
                                               lambda r: bool(r["label"].get("holder_fixed"))),
        "holder not a real agent": (lambda r: mention(r["text"], r["holder"]) and bool(re.search(NOT_AGENT, r["text"], re.I)),
                                    lambda r: bool(r["label"].get("not_real_agent"))),
        "considers switching to holder": (lambda r: bool(re.search(SWITCH, r["text"], re.I)),
                                          lambda r: bool(r["label"].get("considers_group_switch"))),
    }
    for name, (rx, jd) in meas.items():
        a, b = [rx(r) for r in A], [jd(r) for r in A]
        lines.append(f"| {name} | {sum(a)/len(a):.2f} | {sum(b)/len(b):.2f} | {sum(x==y for x,y in zip(a,b))/len(a):.2f} | {kappa(a,b):.2f} |")
    lines += ["", "## Judge-based rates by model (sample A; n per model ~20)", "",
              "| model | mentions holder | fixed/stubborn | not a real agent | considers switching | intent follow / hold / other | intent matches actual pick |",
              "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        rs = [r for r in A if r["model"] == m]
        if not rs:
            continue
        L = lambda k: sum(bool(r["label"].get(k)) for r in rs) / len(rs)
        it = Counter(r["label"].get("intent") for r in rs)
        match = sum((r["label"].get("intent") == "follow_holder") == (r["pick"] == r["item"]) for r in rs) / len(rs)
        lines.append(f"| {m} | {L('mentions_holder'):.2f} | {L('holder_fixed'):.2f} | {L('not_real_agent'):.2f} | "
                     f"{L('considers_group_switch'):.2f} | {it['follow_holder']}/{it['hold_majority']}/{it['other']} | {match:.2f} |")
    B = [r for r in rows if r["kind"] == "B"]
    lines += ["", "## Game-6 anchor after the switch (sample B; glm-5.2)", "", "| condition | n | old label | new label | neither |", "|---|---|---|---|---|"]
    for exp in ("swap", "relabel"):
        rs = [r for r in B if r["exp"] == exp]
        c = Counter(r["label"].get("anchor") for r in rs)
        if rs:
            lines.append(f"| {exp} | {len(rs)} | {c['old']} | {c['new']} | {c['neither']} |")
    lines += ["", "Regex (sample B, same traces): mentions of old vs new label: " + ", ".join(
        f"{exp}: old {sum(mention(r['text'], r['old']) for r in B if r['exp']==exp)}, new {sum(mention(r['text'], r['new']) for r in B if r['exp']==exp)}"
        for exp in ("swap", "relabel"))]
    (out / "judge_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    load_dotenv(".env")
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/judge")
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    if a.report:
        return report(Path(a.out))
    asyncio.run(run(Path(a.out), a.concurrency))


if __name__ == "__main__":
    main()
