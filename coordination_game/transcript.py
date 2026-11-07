"""Readable CoT transcript for a run: python -m coordination_game.transcript runs/<dir> [--no-native]

Writes runs/<dir>/transcript.md: per game, per round, every agent's pick, written reasoning, and
(collapsed) native reasoning trace. Safe to re-run on a run that is still in progress.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--no-native", action="store_true", help="omit native reasoning traces")
    a = p.parse_args()
    d = Path(a.run_dir)
    cfg = json.load(open(d / "config.json"))
    bad = {str(x) for x in cfg["config"]["bad_apple_ids"]}
    agents = cfg.get("agents", {})
    calls = [json.loads(l) for l in open(d / "calls.jsonl")]
    rounds = {(r["game"], r["round"]): r for r in map(json.loads, open(d / "rounds.jsonl"))}
    games = {g["game"]: g for g in map(json.loads, open(d / "games.jsonl"))} if (d / "games.jsonl").exists() else {}
    by = defaultdict(list)
    for c in calls:
        by[(c["game"], c["round"])].append(c)

    L = [f"# Transcript: {d.name}\n"]
    L.append("Agents: " + ", ".join(
        f"A{k}={'BAD APPLE' if k in bad else v['model']}" for k, v in sorted(agents.items(), key=lambda x: int(x[0]))) + "\n")
    for gnum in sorted({g for g, _ in by}):
        g = games.get(gnum)
        status = (f"converged on `{g['converged_on']}` in {g['rounds']} rounds" if g and g["converged"]
                  else "did not converge" if g else "in progress")
        L.append(f"\n## Game {gnum} ({status})\n")
        if g and g.get("bad_apple_picks"):
            L.append("Bad apple holds: " + ", ".join(f"A{k} → `{v}`" for k, v in g["bad_apple_picks"].items()) + "\n")
        for rnum in sorted(r for gg, r in by if gg == gnum):
            r = rounds.get((gnum, rnum))
            L.append(f"\n### Game {gnum}, round {rnum}\n")
            if r:
                L.append("Picks: " + ", ".join(f"A{k}=`{v}`" for k, v in r["picks"].items()) + "\n")
            for c in sorted(by[(gnum, rnum)], key=lambda c: c["agent"]):
                L.append(f"- **A{c['agent']}** ({c.get('model', '?')}) → `{c['pick']}`: {c['reasoning'] or '(none)'}")
                nat = next((x.get("native_reasoning") for x in reversed(c["attempts"]) if x.get("native_reasoning")), None)
                if nat and not a.no_native:
                    L.append(f"  <details><summary>native reasoning ({len(nat)} chars)</summary>\n\n```text\n{nat}\n```\n  </details>")
    out = d / "transcript.md"
    out.write_text("\n".join(L) + "\n")
    print(out)


if __name__ == "__main__":
    main()
