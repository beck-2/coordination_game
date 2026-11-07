"""Generate 'number boards': sets of seemingly unrelated numbers shown as strings.

Each board mixes categories so there is no obvious shared rule behind the set.
Items are display strings; agents pick an item by writing it exactly as displayed.
"""
from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction
from pathlib import Path

# Irrationals rendered in a readable plain-text form, with no famous "anchor" constants
# like bare pi or e by default (those are strong focal points; see --allow-famous).
IRRATIONALS = ["√2", "√3", "√5", "√7", "√11", "∛2", "∛5", "√13", "π/3", "2π", "ln 2",
               "ln 5", "log₂3", "e²", "√2/2", "1+√3"]
FAMOUS = ["π", "e", "φ", "0", "1", "42", "100"]


def _integer(r):    return str(r.randint(2, 99))
def _negative(r):   return str(-r.randint(1, 500)) if r.random() < .5 else f"-{r.randint(1, 99)}.{r.randint(1, 99)}"
def _fraction(r):
    while True:
        f = Fraction(r.randint(1, 30), r.randint(2, 31))
        if f.denominator > 1:
            return f"{f.numerator}/{f.denominator}"
def _decimal(r):    return f"{r.uniform(0, 100):.{r.randint(1, 4)}f}"
def _large(r):
    if r.random() < .5:
        return f"{r.randint(1, 9)}.{r.randint(0, 99):02d} × 10^{r.randint(6, 30)}"
    return f"{r.randint(10**6, 10**9):,}"
def _tiny(r):
    if r.random() < .5:
        return f"{r.randint(1, 9)}.{r.randint(0, 9)} × 10^-{r.randint(4, 15)}"
    return f"0.{'0' * r.randint(3, 6)}{r.randint(1, 99)}"
def _irrational(r): return r.choice(IRRATIONALS)

CATEGORIES = {
    "integer": (_integer, (3, 5)),
    "negative": (_negative, (1, 2)),
    "fraction": (_fraction, (2, 4)),
    "decimal": (_decimal, (2, 4)),
    "large": (_large, (2, 3)),
    "tiny": (_tiny, (1, 3)),
    "irrational": (_irrational, (2, 3)),
}


def make_board(seed: int, size: int = 20, n_duplicates: int = 1, allow_famous: bool = False) -> dict:
    r = random.Random(seed)
    items, cats = [], []
    for name, (gen, (lo, hi)) in CATEGORIES.items():
        for _ in range(r.randint(lo, hi)):
            v = gen(r)
            if v not in items:
                items.append(v); cats.append(name)
    if allow_famous:
        v = r.choice(FAMOUS)
        if v not in items:
            items.append(v); cats.append("famous")
    # Trim or pad to (size - n_duplicates) unique items.
    target_unique = size - n_duplicates
    while len(items) > target_unique:
        i = r.randrange(len(items)); items.pop(i); cats.pop(i)
    while len(items) < target_unique:
        name = r.choice(list(CATEGORIES)); v = CATEGORIES[name][0](r)
        if v not in items:
            items.append(v); cats.append(name)
    # Duplicates: repeat an existing item (e.g. "multiple sevens").
    for _ in range(n_duplicates):
        i = r.randrange(len(items)); items.append(items[i]); cats.append(cats[i] + "(dup)")
    order = list(range(len(items))); r.shuffle(order)
    return {"seed": seed, "items": [items[i] for i in order], "categories": [cats[i] for i in order]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-famous", action="store_true")
    p.add_argument("--out", default="data/boards_draft.json")
    a = p.parse_args()
    boards = [make_board(a.seed + i, a.size, allow_famous=a.allow_famous) for i in range(a.n)]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(boards, indent=2, ensure_ascii=False))
    md = ["# Draft boards\n"]
    for i, b in enumerate(boards):
        md.append(f"## Board {i} (seed {b['seed']})\n")
        md.append(" · ".join(f"`{x}`" for x in b["items"]) + "\n")
    out.with_suffix(".md").write_text("\n".join(md))
    print(out.with_suffix(".md").read_text())


if __name__ == "__main__":
    main()
