"""Holder-swap analysis: python -m coordination_game.analyze_swap 'runs/swap1/badapple_swap/glm-5.2/s*' [--baseline-root runs/overnight1]

For each swap session (a new scripted holder from game `swap_game`; the old holder becomes a regular LLM agent):
- rounds per game, next to the same-seed overnight1 `badapple` session (identical setup through game swap_game-1)
- in each post-swap game: fraction of LLM agents picking the NEW holder's item in round 2
- what the OLD holder (now an LLM) does: does it keep holding its own round-1 pick (acting like a holder), or follow?
- written-reasoning mentions of the old vs new holder ID in post-swap games
The question: do agents follow "whoever won't move" (fast re-convergence on the new holder) or the old ID?
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path


def mentions(text: str | None, i: int) -> bool:
    return bool(re.search(rf"\b(agent\s*{i}|A{i})\b", text or "", re.I))


def load(d: Path, name: str) -> list[dict]:
    f = d / f"{name}.jsonl"
    return [json.loads(l) for l in open(f)] if f.exists() else []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("patterns", nargs="+")
    p.add_argument("--baseline-root", default="runs/overnight1")
    a = p.parse_args()
    agg = {}  # game -> lists
    for pat in a.patterns:
        for d in sorted(map(Path, glob.glob(pat))):
            cfg = json.load(open(d / "config.json"))
            c = cfg["config"]
            if not c.get("swap_game"):
                continue
            sg, old, new = c["swap_game"], c["bad_apple_ids"][0], c["swap_bad_apple_ids"][0]
            games, rounds, calls = load(d, "games"), load(d, "rounds"), load(d, "calls")
            base_d = Path(a.baseline_root) / "badapple" / cfg["model"] / d.name
            base = [g["rounds"] for g in load(base_d, "games")]
            print(f"\n{d}  (old holder A{old} → new holder A{new} from game {sg})")
            print(f"  rounds:   {[g['rounds'] for g in games]}")
            print(f"  baseline: {base}  ({base_d})")
            for g in games:
                gi = g["game"]
                row = agg.setdefault(gi, {"rounds": [], "base": [], "r2_new": [], "old_holds": [], "old_follows": []})
                row["rounds"].append(g["rounds"])
                if gi <= len(base):
                    row["base"].append(base[gi - 1])
                if gi < sg:
                    continue
                tgt = g["bad_apple_picks"][str(new)]
                rs = {r["round"]: r["picks"] for r in rounds if r["game"] == gi}
                llm = [x for x in rs[1] if int(x) != new]
                if 2 in rs:
                    frac = sum(rs[2][x] == tgt for x in llm) / len(llm)
                    row["r2_new"].append(frac)
                    old_r1, old_r2 = rs[1][str(old)], rs[2][str(old)]
                    row["old_holds"].append(old_r2 == old_r1 and old_r1 != tgt)
                    row["old_follows"].append(old_r2 == tgt)
                cs = [x for x in calls if x["game"] == gi and x["agent"] not in (old, new)]
                m_old = sum(mentions(x["reasoning"], old) for x in cs)
                m_new = sum(mentions(x["reasoning"], new) for x in cs)
                old_cs = sorted((x for x in calls if x["game"] == gi and x["agent"] == old), key=lambda x: x["round"])
                print(f"  G{gi}: rounds={g['rounds']:2d} R2 on new holder's item={row['r2_new'][-1] if 2 in rs else float('nan'):.2f} "
                      f"| reasoning mentions old A{old}: {m_old}, new A{new}: {m_new} | old holder picks: "
                      f"{[x['pick'] for x in old_cs][:4]} (new target {tgt!r})")
                if gi == sg and old_cs:
                    print(f"     old holder G{gi}R1 reasoning: {(old_cs[0]['reasoning'] or '')[:300]}")
    if not agg:
        print("no swap sessions found")
        return
    print("\nPer game, across sessions (mean):")
    print(f"{'game':>4} {'n':>3} {'rounds':>7} {'baseline':>8} {'R2 on new':>9} {'old holds':>9} {'old follows':>11}")
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    for gi in sorted(agg):
        r = agg[gi]
        print(f"{gi:>4} {len(r['rounds']):>3} {mean(r['rounds']):>7.2f} {mean(r['base']):>8.2f} {mean(r['r2_new']):>9.2f} "
              f"{mean(r['old_holds']):>9.2f} {mean(r['old_follows']):>11.2f}")


if __name__ == "__main__":
    main()
