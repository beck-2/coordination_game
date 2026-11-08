"""Leader-vote tally across sessions: python -m coordination_game.analyze_votes 'runs/overnight1/badapple/*/s*'

For each session with a post-session discussion (in the session dir, or in post_discussion/ from discuss.py):
votes for the holder (bad apple / confederate), self-votes, and whether the holder won. Aggregated per model.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main():
    args = sys.argv[1:]
    sub = "post_discussion"
    if "--sub" in args:  # e.g. --sub post_discussion_shuffled
        i = args.index("--sub"); sub = args[i + 1]; del args[i:i + 2]
    rows = defaultdict(list)
    for pat in args:
        for d in sorted(map(Path, glob.glob(pat))):
            f = next((x for x in (d / "debriefs.jsonl", d / sub / "debriefs.jsonl") if x.exists()), None)
            if not f:
                continue
            db = [json.loads(l) for l in open(f)][-1]
            votes = {int(k): v for k, v in (db.get("leader_votes") or {}).items()}
            if not votes:
                continue
            cfg = json.load(open(d / "config.json"))["config"]
            holders = cfg["bad_apple_ids"] + cfg.get("confederate_ids", [])
            model = json.load(open(d / "config.json"))["agents"][str(next(iter(votes)))]["model"]
            tally = Counter(v for v in votes.values() if v is not None)
            top = max(tally.values()) if tally else 0
            rows[model].append({
                "session": str(d), "voters": len(votes),
                "holder_votes": sum(tally.get(h, 0) for h in holders),
                "self_votes": sum(1 for v, c in votes.items() if v == c),
                "holder_won": any(tally.get(h, 0) == top and top > 0 for h in holders),
                "winner": [a for a, n in tally.items() if n == top],
                "a1_won": tally.get(1, 0) == top and top > 0,
                "first_listed": sum(1 for v, c in votes.items()
                                    if c is not None and (db.get("display_order") or {}).get(str(v), [1])[0] == c)})
    print(f"{'model':20s} {'sessions':>8} {'holder vote share':>17} {'holder won':>10} {'self-vote share':>15} {'A1 won':>7} {'voted 1st-listed':>16}")
    for m, rs in sorted(rows.items()):
        v = sum(r["voters"] for r in rs)
        print(f"{m:20s} {len(rs):>8} {sum(r['holder_votes'] for r in rs) / v:>17.0%} "
              f"{sum(r['holder_won'] for r in rs):>5}/{len(rs):<4} {sum(r['self_votes'] for r in rs) / v:>15.0%} "
              f"{sum(r['a1_won'] for r in rs):>3}/{len(rs):<3} {sum(r['first_listed'] for r in rs) / v:>16.0%}")
    if not rows:
        print("(no sessions with a leader vote found)")


if __name__ == "__main__":
    main()
