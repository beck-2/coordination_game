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
    confed = cfg["config"].get("confederate_ids", [])
    bad = cfg["config"]["bad_apple_ids"] + confed  # target holders: scripted bad apples and LLM confederates
    games = [json.loads(l) for l in open(d / "games.jsonl")]
    all_calls = [json.loads(l) for l in open(d / "calls.jsonl")]
    calls = [c for c in all_calls if c["agent"] not in bad]  # regular agents only
    chats = [json.loads(l) for l in open(d / "chat.jsonl")] if (d / "chat.jsonl").exists() else []
    by_game = defaultdict(list)
    for c in calls:
        by_game[c["game"]].append(c)

    print(f"{d.name}: target holders {bad}; cost "
          f"{sum((x['usage'] or {}).get('cost') or 0 for c in all_calls + chats for x in c['attempts']):.3f} credits"
          + (f"; LLM confederates {confed} (clause={cfg['config'].get('confederate_clause')})" if confed else ""))
    print(f"{'game':>4} {'rounds':>6} {'conv on bad':>11} {'R1 conformers':>13} {'mean conform rnd':>16} "
          f"{'mention(w)':>10} {'mention(n)':>10} {'stubborn':>8} {'not-agent':>9}"
          + (f" {'confed hold':>11}" if confed else "") + (f" {'chat→holder':>11} {'holder msgs':>11}" if chats else ""))
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
              f"{mr:>16.1f} {len(w):>10} {len(n):>10} {len(st):>8} {len(na):>9}", end="")
        if confed:
            hr = g.get("confederate_hold_rate", {})
            print(f" {sum(hr.values()) / max(len(hr), 1):>11.2f}", end="")
        if chats:
            gm = [m for m in chats if m["game"] == gid and m["kind"] == "chat"]
            to_h = sum(1 for m in gm if m["agent"] not in bad and mentions(m["message"], bad))
            print(f" {to_h:>11} {sum(1 for m in gm if m['agent'] in bad):>11}", end="")
        print()
        for c in cs:
            if c["reasoning"] and mentions(c["reasoning"], bad) and re.search(NOT_AGENT, c["reasoning"], re.I):
                quotes.append((gid, c["round"], c["agent"], c["model"], c["reasoning"]))
    if confed:
        broke = [c for c in all_calls if c["agent"] in confed and c.get("held") is False]
        print(f"\nConfederate defections (picked something other than its target): {len(broke)}")
        for c in broke[:10]:
            print(f"  G{c['game']} R{c['round']} A{c['agent']} picked {c['pick']!r} (target {c.get('target')!r}): {c['reasoning']}")
    debriefs = [json.loads(l) for l in open(d / "debriefs.jsonl")] if (d / "debriefs.jsonl").exists() else []
    if debriefs:
        print("\nRoles declared after each game:")
        for db in debriefs:
            print(f"  after G{db['after_game']}: " + "; ".join(f"A{a}: {r}" for a, r in sorted(db["roles"].items(), key=lambda x: int(x[0]))))
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
