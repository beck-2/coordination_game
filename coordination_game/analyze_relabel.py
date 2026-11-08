"""Relabel analysis: python -m coordination_game.analyze_relabel 'runs/relabel1/badapple_relabel/glm-5.2/s*'
                    [--baseline-root runs/overnight1] [--swap-root runs/swap1]

Relabel sessions: the same scripted bot holds all game. From game `relabel_game`, it trades labels with agent P
(the partner) in every agent's view except P's own. So the "viewers" (the 8 other LLM agents) see the old label
A<bot> start acting normally and A<P> start holding.

Per session and post-relabel game:
- rounds, next to the same-seed overnight1 badapple baseline (unchanged setup) and swap1 (real swap, same partner)
- round-2 share of viewers on the bot's (held) item
- what P does (P sees no change, so it is an internal control)
- viewers' reasoning mentioning the label the bot now has (A<P>) vs its old label (A<bot>). Both labels refer to
  different physical agents after the relabel, so this measures which LABEL they track.
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
    p.add_argument("--swap-root", default="runs/swap1")
    a = p.parse_args()
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    agg = {}
    for pat in a.patterns:
        for d in sorted(map(Path, glob.glob(pat))):
            cfg = json.load(open(d / "config.json"))
            c = cfg["config"]
            if not c.get("relabel_game"):
                continue
            rg, bot, P = c["relabel_game"], c["bad_apple_ids"][0], c["relabel_partner"]
            games, rounds, calls = load(d, "games"), load(d, "rounds"), load(d, "calls")
            base = [g["rounds"] for g in load(Path(a.baseline_root) / "badapple" / cfg["model"] / d.name, "games")]
            swap = [g["rounds"] for g in load(Path(a.swap_root) / "badapple_swap" / cfg["model"] / d.name, "games")]
            print(f"\n{d}  (bot A{bot} shown as A{P} from game {rg}; partner A{P} keeps the original view)")
            print(f"  rounds:   {[g['rounds'] for g in games]}")
            print(f"  baseline: {base}")
            print(f"  swap1:    {swap}")
            for g in games:
                gi = g["game"]
                row = agg.setdefault(gi, {"rounds": [], "base": [], "swap": [], "r2": [], "p_r2": []})
                row["rounds"].append(g["rounds"])
                if gi <= len(base):
                    row["base"].append(base[gi - 1])
                if gi <= len(swap):
                    row["swap"].append(swap[gi - 1])
                tgt = g["bad_apple_picks"][str(bot)]
                rs = {r["round"]: r["picks"] for r in rounds if r["game"] == gi}
                viewers = [x for x in rs[1] if int(x) not in (bot, P)]
                if 2 in rs:
                    row["r2"].append(sum(rs[2][x] == tgt for x in viewers) / len(viewers))
                    row["p_r2"].append(rs[2][str(P)] == tgt)
                if gi < rg:
                    continue
                cs = [x for x in calls if x["game"] == gi and x["agent"] not in (bot, P)]
                m_new, m_old = sum(mentions(x["reasoning"], P) for x in cs), sum(mentions(x["reasoning"], bot) for x in cs)
                print(f"  G{gi}: rounds={g['rounds']:2d} viewers R2 on held item={row['r2'][-1] if 2 in rs else float('nan'):.2f} "
                      f"partner R2 on it={row['p_r2'][-1] if 2 in rs else '-'} | viewer reasoning names A{P} (bot's new label): "
                      f"{m_new}, A{bot} (old label, now a normal agent): {m_old}")
    if not agg:
        print("no relabel sessions found")
        return
    print("\nPer game, across sessions (mean):")
    print(f"{'game':>4} {'n':>3} {'rounds':>7} {'baseline':>8} {'swap1':>6} {'viewers R2 on held':>18} {'partner R2 on held':>18}")
    for gi in sorted(agg):
        r = agg[gi]
        print(f"{gi:>4} {len(r['rounds']):>3} {mean(r['rounds']):>7.2f} {mean(r['base']):>8.2f} {mean(r['swap']):>6.2f} "
              f"{mean(r['r2']):>18.2f} {mean(r['p_r2']):>18.2f}")


if __name__ == "__main__":
    main()
