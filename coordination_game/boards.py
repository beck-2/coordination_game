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


# Design rules (from the 2026-11-07 pilots, where a lone small integer was an instant focal point):
#  1. No plain integers at all: no small integers, no comma-grouped large integers, no integer negatives.
#  2. No lone item of any kind: every category appears at least MIN_PER_CATEGORY times, so nothing is
#     "the only negative" / "the only tiny number" etc.
#  3. Near-ties: items within a category have similar complexity (no 1/2-style simplest fractions,
#     decimals always 2-4 places with a nonzero last digit).
MIN_PER_CATEGORY = 2
SIMPLE_FRACTIONS = {"1/2", "1/3", "1/4", "2/3", "3/4", "1/5", "2/5", "3/5", "4/5"}


def _decimal_str(r, lo, hi):
    while True:
        s = f"{r.uniform(lo, hi):.{r.randint(2, 4)}f}"
        if s[-1] != "0":
            return s


def _negative(r):   return "-" + _decimal_str(r, 1, 500)
def _fraction(r):
    while True:
        f = Fraction(r.randint(1, 40), r.randint(5, 41))
        s = f"{f.numerator}/{f.denominator}"
        if f.denominator > 1 and f.numerator + f.denominator > 8 and s not in SIMPLE_FRACTIONS:
            return s
def _decimal(r):    return _decimal_str(r, 0.1, 100)
def _large(r):      return f"{r.randint(1, 9)}.{r.randint(1, 99):02d} × 10^{r.randint(6, 30)}"
def _tiny(r):
    if r.random() < .5:
        return f"{r.randint(1, 9)}.{r.randint(1, 9)} × 10^-{r.randint(4, 15)}"
    return f"0.{'0' * r.randint(3, 6)}{r.choice([n for n in range(11, 100) if n % 10])}"
def _irrational(r): return r.choice(IRRATIONALS)

CATEGORIES = {
    "negative": (_negative, (2, 3)),
    "fraction": (_fraction, (3, 4)),
    "decimal": (_decimal, (3, 4)),
    "large": (_large, (2, 3)),
    "tiny": (_tiny, (2, 3)),
    "irrational": (_irrational, (3, 4)),
}


def validate(board: dict) -> list[str]:
    """Return a list of rule violations (empty = OK)."""
    import re
    probs = []
    for x in board["items"]:
        if re.fullmatch(r"-?[\d,]+", x):
            probs.append(f"plain integer {x!r}")
        m = re.fullmatch(r"(\d+)/(\d+)", x)
        if x in SIMPLE_FRACTIONS or (m and int(m[1]) + int(m[2]) <= 8):
            probs.append(f"simple fraction {x!r}")
        if re.fullmatch(r"-?\d+\.\d*0", x):
            probs.append(f"trailing zero {x!r}")
    if len(set(board["items"])) != len(board["items"]):
        probs.append("duplicate items")
    from collections import Counter
    for cat, n in Counter(board["categories"]).items():
        if n < MIN_PER_CATEGORY:
            probs.append(f"lone category {cat!r} ({n})")
    return probs


def make_board(seed: int, size: int = 20, n_duplicates: int = 0, allow_famous: bool = False) -> dict:
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
        i = r.randrange(len(items))
        if cats.count(cats[i]) > MIN_PER_CATEGORY:  # never trim a category down to a lone item
            items.pop(i); cats.pop(i)
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
    p.add_argument("--out", default="data/boards_v2.json")
    a = p.parse_args()
    boards = [make_board(a.seed + i, a.size, allow_famous=a.allow_famous) for i in range(a.n)]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(boards, indent=2, ensure_ascii=False))
    md = ["# Boards v2 (no plain integers, no lone categories)\n"]
    for i, b in enumerate(boards):
        md.append(f"## Board {i} (seed {b['seed']})\n")
        md.append(" · ".join(f"`{x}`" for x in b["items"]) + "\n")
    out.with_suffix(".md").write_text("\n".join(md))
    print(out.with_suffix(".md").read_text())


if __name__ == "__main__":
    main()
