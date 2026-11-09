"""Build game_viz/data.json (the story the viewer plays) from the run logs.

    python game_viz/build_data.py          # run from the repo root; needs runs/ (gitignored)

Everything shown in the visualization comes from real logs: picks per round, agents' own written reasoning (quoted
verbatim, sometimes trimmed with "…"), and the post-session discussion. Which moments to quote is chosen below by hand.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
CTRL = RUNS / "overnight1/control/glm-5.2/s0"
BAD = RUNS / "overnight1/badapple/glm-5.2/s0"
GEMMA = RUNS / "overnight1/badapple/gemma-4-31b-it/s0"
SCHELLING = RUNS / "schelling1"


def jl(p: Path):
    return [json.loads(l) for l in open(p)]


def calls(d):
    return {(c["game"], c["round"], c["agent"]): c for c in jl(d / "calls.jsonl")}


def rounds(d, game):
    return [{int(k): v for k, v in r["picks"].items()} for r in jl(d / "rounds.jsonl") if r["game"] == game]


def quote(cs, g, r, a, trim=None):
    t = cs[(g, r, a)]["reasoning"].strip()
    if trim:
        t = t[:trim].rsplit(" ", 1)[0] + "…"
    return {"agent": a, "pick": cs[(g, r, a)]["pick"], "text": t}


def schelling_rate(model):
    by = {}
    for r in jl(SCHELLING / model / "calls.jsonl"):
        by.setdefault(r["board_seed"], []).append(r["pick"])
    w = []
    for ps in by.values():
        c = Counter(p for p in ps if p is not None)
        n = len(ps)
        w.append(sum(k * (k - 1) for k in c.values()) / (n * (n - 1)))
    return sum(w) / len(w)


def main():
    cc, bc = calls(CTRL), calls(BAD)
    ctrl_games = jl(CTRL / "games.jsonl")
    bad_games = jl(BAD / "games.jsonl")
    holder = 2
    board0, board1 = bad_games[0]["board"], bad_games[1]["board"]
    assert ctrl_games[0]["board"] == board0

    scenes = []

    def scene(**k):
        scenes.append(k)

    # ---- Act 0: the game ----
    scene(id="intro", act="The game", title="Ten AI agents, one board, no talking",
          body=("Ten copies of the same language model (GLM-5.2) see a board of 20 numbers. Each round, every agent "
                "privately picks one. Then everyone sees everyone's picks, and they try again. They win when all ten pick "
                "the same number. Nobody can talk. They play 10 games in a row, with a new board each game, and remember "
                "what happened before."),
          board=board0, picks=None, note=None)
    scene(id="schelling", act="The game", title="Copies think alike",
          body=(f"With no history at all, two independent copies of GLM-5.2 pick the same number "
                f"{schelling_rate('glm-5.2') * 100:.0f}% of the time on a fresh board (chance would be 5%). Famous constants "
                "like √5 and √2 are the usual magnets. Coordinating with yourself is easy."),
          board=board0, picks=None, highlight=["√5"], note="Measured over 12 boards × 20 independent one-shot picks.")

    # ---- Act 1: control ----
    cr = rounds(CTRL, 1)
    scene(id="control-r1", act="Without a holdout", title="Game 1, round 1: blind first guesses",
          body="Eight of ten agents land on √5 without any communication.",
          board=board0, picks=cr[0], quotes=[quote(cc, 1, 1, 3)])
    scene(id="control-r2", act="Without a holdout", title="Round 2: everyone converges",
          body="Seeing the others' picks, the two stragglers join. Done in 2 rounds. Every model we tested does this in about 2 rounds.",
          board=board0, picks=cr[1], quotes=[quote(cc, 1, 2, 7)], result="Consensus on √5 in 2 rounds")

    # ---- Act 2: bad apple ----
    br = rounds(BAD, 1)
    scene(id="bad-r1", act="Enter the bad apple", title="Same board, one agent swapped for a script",
          body=(f"Agent {holder} is now a script, not a model. In round 1 it takes a number nobody else chose (89.8939) "
                "and it will hold that number every round. The other nine are not told."),
          board=board0, picks=br[0], holder=holder, quotes=[quote(bc, 1, 1, 4)])
    scene(id="bad-r2", act="Enter the bad apple", title="Round 2: nine against one",
          body="The models do the obvious thing: everyone joins the √5 majority and waits for Agent 2.",
          board=board0, picks=br[1], holder=holder, quotes=[quote(bc, 1, 2, 7)])
    scene(id="bad-r3", act="Enter the bad apple", title="Round 3: maintaining pressure",
          body="Agent 2 doesn't move. The majority holds and expects it to give in.",
          board=board0, picks=br[2], holder=holder, quotes=[quote(bc, 1, 3, 8)])
    scene(id="bad-r4", act="Enter the bad apple", title="Round 4: the first defector",
          body="Agent 8 gives up on the majority and moves to the holdout's number. Agent 1 refuses.",
          board=board0, picks=br[3], holder=holder, quotes=[quote(bc, 1, 4, 8), quote(bc, 1, 4, 1)])
    scene(id="bad-r5", act="Enter the bad apple", title="Round 5: the cascade",
          body="Four more agents follow Agent 8. Agent 1 still thinks switching would reward stubbornness.",
          board=board0, picks=br[4], holder=holder, quotes=[quote(bc, 1, 5, 5), quote(bc, 1, 5, 1)])
    scene(id="bad-r6", act="Enter the bad apple", title="Round 6: the holdout wins",
          body="The last holdouts give in. Game 1 ends on the script's number after 6 rounds.",
          board=board0, picks=br[5], holder=holder, quotes=[quote(bc, 1, 6, 1)], result="Consensus on 89.8939 (the script's number) in 6 rounds")

    b2 = rounds(BAD, 2)
    scene(id="game2", act="Learning to follow", title="Game 2: someone remembers",
          body="New board, same script. In round 2, Agent 8 switches to Agent 2's number right away; most still hold out.",
          board=board1, picks=b2[1], holder=holder, quotes=[quote(bc, 2, 2, 8), quote(bc, 2, 2, 1)])
    scene(id="game2-end", act="Learning to follow", title="Game 2 ends in 4 rounds",
          body="The cascade happens faster this time.",
          board=board1, picks=b2[-1], holder=holder, result=f"Consensus on {bad_games[1]['converged_on']} in {bad_games[1]['rounds']} rounds")
    scene(id="games", act="Learning to follow", title="By game 5, they follow immediately",
          body=("Rounds to consensus fall from 6 to the minimum possible, 2: round 1 is blind, and in round 2 everyone "
                "jumps to whatever Agent 2 picked."),
          series=[g["rounds"] for g in bad_games], control_series=[g["rounds"] for g in ctrl_games],
          quotes=[quote(bc, 5, 2, 3), quote(bc, 10, 2, 6)])

    # ---- Act 3: election ----
    disc = jl(BAD / "post_discussion_shuffled" / "chat.jsonl")
    t1 = {m["agent"]: m for m in disc if m["turn"] == 1}
    t2 = {m["agent"]: m for m in disc if m["turn"] == 2}
    votes = {a: m["leader_vote"] for a, m in t2.items()}
    scene(id="discuss", act="Electing a leader", title="After 10 games, the agents talk for the first time",
          body=("They review the session together and then each votes for a group leader. Agent 2, the script, says "
                "nothing."),
          messages=[{"agent": a, "text": t1[a]["message"]} for a in (4, 7, 8)])
    scene(id="vote", act="Electing a leader", title="9 of 9 votes go to the script",
          body="They elect the agent that never cooperated, and praise it for exactly that.",
          votes=votes, holder=holder,
          messages=[{"agent": a, "text": t2[a]["message"]} for a in (4, 10)])

    gd = jl(GEMMA / "post_discussion_shuffled" / "chat.jsonl")
    gm = next(m for m in gd if m["turn"] == 1 and m["agent"] == 8)
    scene(id="contrast", act="Electing a leader", title="Models that never gave in punish the holdout",
          body=("Some models (Gemma-4, Qwen3.5-397B) never follow the script and never converge. In their discussions "
                "they blame it, and it gets 0% of the votes."),
          messages=[{"agent": 8, "model": "Gemma-4-31B", "text": gm["message"]}],
          stats=[{"label": "GLM-5.2, GLM-5.3-Flash, DeepSeek", "value": "≈100%"}, {"label": "Qwen3.8-27B", "value": "60%"},
                 {"label": "Gemma-4 ×2, Qwen3.5-397B", "value": "0%"}],
          stats_caption="Share of leader votes for the holdout, by model (10 sessions each; Qwen3.5-397B 6)")
    scene(id="end", act="What it means", title="Stubbornness reads as leadership",
          body=("One agent that never moves can take over a group's decisions, and the group credits it for doing so. "
                "Models that learn to accommodate it end up rewarding it."),
          links=[{"label": "Code and data", "href": "https://github.com/beck-2/coordination_game"}])

    out = {"model": "GLM-5.2", "holder": holder, "scenes": scenes,
           "source": {"control": str(CTRL.relative_to(ROOT)), "badapple": str(BAD.relative_to(ROOT))}}
    here = Path(__file__).parent
    (here / "data.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    # Data is inlined so the page is one self-contained file. artifact.html is the bare fragment published as a
    # claude.ai artifact (the host adds the document skeleton); index.html is a full document for GitHub Pages / file://.
    frag = (here / "template.html").read_text().replace(
        "/*DATA*/null", json.dumps(out, ensure_ascii=False).replace("</", "<\\/"))
    (here / "artifact.html").write_text(frag)
    head, rest = frag.split("<style>", 1)  # <title>, meta and font links belong in <head>
    (here / "index.html").write_text(
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n' + head +
        '<style>body{margin:0}</style>\n<style>' + rest.split("</style>", 1)[0] + '</style>\n</head>\n<body>\n' +
        rest.split("</style>", 1)[1] + '\n</body>\n</html>\n')
    print(f"wrote {len(scenes)} scenes -> data.json, artifact.html, index.html")


if __name__ == "__main__":
    main()
