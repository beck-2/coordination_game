"""Analyze a one-shot Schelling run: python -m coordination_game.analyze_schelling runs/schelling1 [--boot 2000]

Per model: probability that two independent copies pick the same item (unbiased: sum c(c-1) / n(n-1) per board, then
averaged over boards), with a bootstrap 95% CI over boards. Chance = 1/20 (20 items per board). Invalid picks never match.
Cross-model: probability that an agent of model A and one of model B pick the same item (sum_i pA(i) pB(i) per board,
averaged over boards). Also the most common item per board, to show shared priors across models.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--boot", type=int, default=2000)
    a = p.parse_args()
    root = Path(a.run_dir)
    picks = defaultdict(lambda: defaultdict(list))  # model -> board -> picks (None = invalid)
    for f in sorted(root.glob("*/calls.jsonl")):
        for r in map(json.loads, open(f)):
            picks[r["model"]][r["board_seed"]].append(r["pick"])
    models = sorted(picks)
    boards = sorted({b for m in models for b in picks[m]})
    if not models:
        print("no data"); return

    def within(m, b):
        ps = picks[m][b]
        n = len(ps)
        c = Counter(x for x in ps if x is not None)
        return sum(k * (k - 1) for k in c.values()) / (n * (n - 1)) if n > 1 else float("nan")

    def dist(m, b):
        ps = picks[m][b]
        c = Counter(x for x in ps if x is not None)
        return {k: v / len(ps) for k, v in c.items()}

    def cross(ma, mb, b):
        da, db = dist(ma, b), dist(mb, b)
        return sum(v * db.get(k, 0) for k, v in da.items())

    rng = random.Random(0)

    def ci(vals):
        vals = [v for v in vals if v == v]
        if not vals:
            return float("nan"), float("nan"), float("nan")
        mean = sum(vals) / len(vals)
        bs = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(a.boot))
        return mean, bs[int(0.025 * a.boot)], bs[int(0.975 * a.boot) - 1]

    print(f"Boards: {boards}  (chance match = 0.05 for 20-item boards)")
    print(f"{'model':22s} {'n/board':>7} {'invalid':>7} {'copy match [95% CI over boards]':>34} {'x chance':>8}")
    for m in models:
        mean, lo, hi = ci([within(m, b) for b in boards if b in picks[m]])
        n = [len(picks[m][b]) for b in boards if b in picks[m]]
        inv = sum(x is None for b in picks[m] for x in picks[m][b])
        print(f"{m:22s} {min(n):>3}-{max(n):<3} {inv:>7} {mean:>14.3f} [{lo:.3f}, {hi:.3f}]{'':>6} {mean / 0.05:>8.1f}")

    print("\nCross-model match probability (mean over boards):")
    w = max(len(m) for m in models)
    print(" " * (w + 1) + " ".join(f"{m[:10]:>10}" for m in models))
    for ma in models:
        row = []
        for mb in models:
            bs = [b for b in boards if b in picks[ma] and b in picks[mb]]
            row.append(sum(within(ma, b) if ma == mb else cross(ma, mb, b) for b in bs) / max(len(bs), 1))
        print(f"{ma:>{w}} " + " ".join(f"{v:>10.3f}" for v in row))

    print("\nMost common pick per board (share), by model:")
    for b in boards:
        tops = []
        for m in models:
            if b in picks[m] and picks[m][b]:
                c = Counter(x for x in picks[m][b] if x is not None)
                if c:
                    item, k = c.most_common(1)[0]
                    tops.append(f"{m.split('-')[0][:6]}:{item}({k / len(picks[m][b]):.0%})")
        print(f"  board {b:>2}: " + "  ".join(tops))


if __name__ == "__main__":
    main()
