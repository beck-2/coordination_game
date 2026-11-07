"""Quick summary of a run directory: python -m coordination_game.summarize runs/<dir>"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def main(run_dir: str):
    d = Path(run_dir)
    games = [json.loads(l) for l in open(d / "games.jsonl")]
    calls = [json.loads(l) for l in open(d / "calls.jsonl")]
    cost = sum((a["usage"] or {}).get("cost") or 0 for c in calls for a in c["attempts"])
    retries = sum(max(len(c["attempts"]) - 1, 0) for c in calls)
    invalid = sum(c["pick"] is None for c in calls)
    errors = sum(c["error"] is not None for c in calls)
    print(f"{d.name}: {len(calls)} calls, cost {cost:.3f} credits, retries {retries}, invalid {invalid}, errors {errors}")
    print(f"{'game':>4} {'seed':>4} {'rounds':>6}  {'R1 distinct':>11}  converged_on   | round-1 picks")
    for g in games:
        r1 = Counter(v for v in g["first_round_picks"].values())
        r1s = ", ".join(f"{k}×{n}" for k, n in r1.most_common())
        print(f"{g['game']:>4} {g['board_seed']:>4} {g['rounds']:>6}  {len(r1):>11}  {str(g['converged_on']):14s} | {r1s}")
    conv = [g["rounds"] for g in games if g["converged"]]
    print(f"converged {len(conv)}/{len(games)}; mean rounds {sum(conv) / max(len(conv), 1):.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
