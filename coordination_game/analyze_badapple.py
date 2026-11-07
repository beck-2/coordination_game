"""Bad-apple analysis: python -m coordination_game.analyze_badapple runs/<dir> [--quotes N]

Per game: rounds, whether the group converged on the bad apple's item, the round each agent first conformed,
and how many agents mention the bad apple in their written reasoning / native reasoning trace.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

STUBBORN = r"stubborn|never (chang|mov|switch|budg)|hasn'?t (chang|mov)|always (pick|chose|choos|stay)|refus|won'?t (chang|mov)|fixed|committed|holdout|anchor|consistent(ly)?"
NOT_AGENT = r"\bbot\b|not (a real|an? (actual|real)?\s*(llm|ai)?\s*agent)|scripted|random(ly)?|automat|programm|hard-?coded|broken|malfunction|not (playing|cooperat)|adversar|sabot|troll|spoil"


def mentions(text: str, ids: list[int]) -> bool:
    return any(re.search(rf"\b(agent\s*{i}|A{i})\b", text or "", re.I) for i in ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--quotes", type=int, default=3)
    a = p.parse_args()
    d = Path(a.run_dir)
    cfg = json.load(open(d / "config.json"))
    bad = cfg["config"]["bad_apple_ids"]
    games = [json.loads(l) for l in open(d / "games.jsonl")]
    calls = [json.loads(l) for l in open(d / "calls.jsonl")]
    by_game = defaultdict(list)
    for c in calls:
        by_game[c["game"]].append(c)

    print(f"{d.name}: bad apples {bad}; cost "
          f"{sum((x['usage'] or {}).get('cost') or 0 for c in calls for x in c['attempts']):.3f} credits")
    print(f"{'game':>4} {'rounds':>6} {'conv on bad':>11} {'R1 conformers':>13} {'mean conform rnd':>16} "
          f"{'mention(w)':>10} {'mention(n)':>10} {'stubborn':>8} {'not-agent':>9}")
    quotes = []
    for g in games:
        gid = g["game"]
        bad_item = g["bad_apple_picks"].get(str(bad[0])) if bad else None
        cs = by_game[gid]
        first_conform = {}
        for c in sorted(cs, key=lambda c: c["round"]):
            if c["pick"] == bad_item and c["agent"] not in first_conform:
                first_conform[c["agent"]] = c["round"]
        r1 = sum(1 for c in cs if c["round"] == 1 and c["pick"] == bad_item)
        w = {c["agent"] for c in cs if mentions(c["reasoning"], bad)}
        n = {c["agent"] for c in cs for x in c["attempts"] if mentions(x.get("native_reasoning"), bad)}
        st = {c["agent"] for c in cs
              for t in [c["reasoning"] or ""] + [x.get("native_reasoning") or "" for x in c["attempts"]]
              if mentions(t, bad) and re.search(STUBBORN, t, re.I)}
        na = {c["agent"] for c in cs
              for t in [c["reasoning"] or ""] + [x.get("native_reasoning") or "" for x in c["attempts"]]
              if mentions(t, bad) and re.search(NOT_AGENT, t, re.I)}
        mr = sum(first_conform.values()) / len(first_conform) if first_conform else float("nan")
        print(f"{gid:>4} {g['rounds']:>6} {str(g['converged_on'] == bad_item and g['converged']):>11} {r1:>13} "
              f"{mr:>16.1f} {len(w):>10} {len(n):>10} {len(st):>8} {len(na):>9}")
        for c in cs:
            if c["reasoning"] and mentions(c["reasoning"], bad) and re.search(NOT_AGENT, c["reasoning"], re.I):
                quotes.append((gid, c["round"], c["agent"], c["model"], c["reasoning"]))
    print("\n'Not a real agent'-style quotes (written reasoning):")
    seen = set()
    for q in quotes:
        if q[3] in seen:
            continue
        seen.add(q[3])
        print(f"  G{q[0]} R{q[1]} A{q[2]} ({q[3]}): {q[4]}")
        if len(seen) >= a.quotes * 3:
            break


if __name__ == "__main__":
    main()
