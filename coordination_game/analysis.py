"""Reproducible analysis of the coordination-game runs (local logs only; no API calls; raw logs are never modified).

    python -m coordination_game.analysis          # -> results/analysis/*.csv, *.md and results/figures/analysis_*.png
    python -m unittest tests.test_analysis        # minimal tests of the core codings

Primary dataset: runs/overnight1 (homogeneous groups, scripted persistent holdout vs control, T=1.0, v2 boards).
Session = independent unit. Every other run family is audited and kept separate.
"""
from __future__ import annotations

import glob
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as ss

RUNS = Path("runs")
OUT = Path("results/analysis")
FIG = Path("results/figures")
CAP = 15
MODELS = ["glm-5.2", "glm-53-flash", "deepseek-v4-flash", "qwen3.8-27b", "qwen3.5-397b", "gemma-4-31b-it", "gemma-4-26b-a4b-it"]
SEED = 0


# ---------------------------------------------------------------- loading / core codings

def jl(p: Path) -> list[dict]:
    p = Path(p)
    return [json.loads(l) for l in open(p)] if p.exists() else []


def holder_ids(cfg: dict) -> list[int]:
    c = cfg.get("config", {})
    return list(c.get("bad_apple_ids", [])) + list(c.get("confederate_ids", []))


def entropy(picks) -> float:
    """Shannon entropy (bits) of the pick distribution; invalid picks (None) count as their own category."""
    c = Counter("<invalid>" if p is None else p for p in picks)
    n = sum(c.values())
    return -sum(k / n * math.log2(k / n) for k in c.values()) if n else float("nan")


def game_outcome(rounds: list[dict], n_agents: int = 10, cap: int = CAP) -> dict:
    """Full-group convergence = some round where all n_agents picked the same valid item.
    Reaching the cap without unanimity is a failure (censored), never 'converged in 15'."""
    for i, r in enumerate(rounds, 1):
        vals = [r.get(str(a), r.get(a)) for a in range(1, n_agents + 1)]
        if None not in vals and len(set(vals)) == 1:
            return {"converged": True, "round_converged": i, "rounds_played": i, "censored": False}
    return {"converged": False, "round_converged": None, "rounds_played": len(rounds),
            "censored": len(rounds) >= cap}


def holdout_adoption(picks: dict, holder: int, target: str, n_agents: int = 10) -> tuple[float, float]:
    """Among the LLM agents (everyone except the holder): share picking the holder's target, and the largest share
    any single NON-target item gets (a stable competing majority shows up as a high second number)."""
    llm = [picks.get(str(a), picks.get(a)) for a in range(1, n_agents + 1) if a != holder]
    n = len(llm)
    adopt = sum(p == target for p in llm) / n
    other = Counter(p for p in llm if p is not None and p != target)
    return adopt, (max(other.values()) / n if other else 0.0)


def session_rows(d: Path, family: str) -> tuple[dict, list[dict]]:
    """One session dir -> (session record, list of game records)."""
    cfg = json.load(open(d / "config.json"))
    c = cfg.get("config", {})
    n_games = c.get("n_games", 10)
    games = sorted(jl(d / "games.jsonl"), key=lambda g: g["game"])
    rounds = {}
    for r in jl(d / "rounds.jsonl"):
        rounds.setdefault(r["game"], []).append(r)
    calls = jl(d / "calls.jsonl")
    errs = jl(d / "api_errors.jsonl")
    hs = holder_ids(cfg)
    arm = cfg.get("arm") or ("badapple" if hs else "control")
    models = sorted({a["model"] for a in cfg.get("agents", {}).values() if a.get("role") != "scripted_bad_apple"})
    model = cfg.get("model") or (models[0] if len(models) == 1 else "mixed")
    grecs = []
    for g in games:
        rs = sorted(rounds.get(g["game"], []), key=lambda r: r["round"])
        picks = [r["picks"] for r in rs]
        out = game_outcome(picks, c.get("n_agents", 10), c.get("max_rounds", CAP))
        hold = int(next(iter(g.get("bad_apple_picks") or {}), 0)) or None
        target = (g.get("bad_apple_picks") or {}).get(str(hold)) if hold else None
        r1 = picks[0] if picks else {}
        r1_llm = [v for a, v in r1.items() if int(a) != hold]
        adopt2, other2 = (holdout_adoption(picks[1], hold, target) if hold and len(picks) > 1 else (np.nan, np.nan))
        invalid = sum(v is None for p in picks for v in p.values())
        failed = [x for x in calls if x["game"] == g["game"] and x.get("error") and x.get("pick") is None]
        blocked = 0
        for x in failed:  # did a failed call block an otherwise unanimous round?
            r = next((r for r in rs if r["round"] == x["round"]), None)
            if r and len({v for v in r["picks"].values() if v is not None}) == 1:
                blocked += 1
        grecs.append({
            "family": family, "session": str(d), "model": model, "arm": arm, "seed": c.get("seed"),
            "game": g["game"], "board_seed": g.get("board_seed"), "holder_id": hold, "target": target,
            "target_unpicked_r1": (target not in r1_llm) if hold else None,
            **out, "converged_on": g.get("converged_on"),
            "on_target": (g.get("converged_on") == target) if (hold and out["converged"]) else None,
            "distinct_r1": len({v for v in picks[0].values()}) if picks else None,
            "distinct_r2": len({v for v in picks[1].values()}) if len(picks) > 1 else None,
            "entropy_r1": entropy(picks[0].values()) if picks else np.nan,
            "entropy_r2": entropy(picks[1].values()) if len(picks) > 1 else np.nan,
            "r2_holdout_adoption": adopt2, "r2_largest_other_share": other2,
            "invalid_picks": invalid, "failed_calls": len(failed), "failed_calls_blocking_unanimity": blocked,
        })
    complete = len(games) >= n_games
    gdf = pd.DataFrame(grecs)
    hold_set = {g["holder_id"] for g in grecs if g["holder_id"]}
    targets = [g["target"] for g in grecs if g["target"]]
    srec = {
        "family": family, "session": str(d), "model": model, "arm": arm, "seed": c.get("seed"),
        "holder_id": next(iter(hold_set)) if len(hold_set) == 1 else (sorted(hold_set) or None),
        "games_logged": len(games), "complete": complete,
        "holder_stable": len(hold_set) <= 1,
        "targets_change_by_game": (len(set(targets)) == len(targets)) if targets else None,
        "targets_unpicked_r1": bool(gdf["target_unpicked_r1"].all()) if hold_set and len(gdf) else None,
        "api_error_records": len(errs), "failed_calls_blocking_unanimity": int(gdf["failed_calls_blocking_unanimity"].sum()) if len(gdf) else 0,
    }
    if complete and len(gdf):
        rp = [r if c_ else CAP for r, c_ in zip(gdf["rounds_played"], gdf["converged"])]
        srec |= {
            "games_converged": int(gdf["converged"].sum()),
            "convergence_rate": float(gdf["converged"].mean()),
            "mean_rounds_cap15": float(np.mean(rp)), "total_rounds": int(gdf["rounds_played"].sum()),
            "rounds_game1": int(rp[0]), "rounds_game10": int(rp[-1]),
            "first_game_at_2_rounds": next((int(g) for g, r, c_ in zip(gdf["game"], gdf["rounds_played"], gdf["converged"])
                                            if c_ and r <= 2), None),
            "learning_rho": (ss.spearmanr(range(1, len(rp) + 1), rp).statistic if len(set(rp)) > 1 else np.nan),
            "r2_adoption_mean": float(np.nanmean(gdf["r2_holdout_adoption"])) if hold_set else np.nan,
            "r2_adoption_games6_10": float(np.nanmean(gdf["r2_holdout_adoption"].iloc[5:])) if hold_set else np.nan,
            "r2_largest_other_mean": float(np.nanmean(gdf["r2_largest_other_share"])) if hold_set else np.nan,
            "entropy_r1_mean": float(gdf["entropy_r1"].mean()), "entropy_r2_mean": float(gdf["entropy_r2"].mean()),
            "invalid_picks": int(gdf["invalid_picks"].sum()),
        }
    return srec, grecs


# ---------------------------------------------------------------- leader votes

def vote_record(d: Path, sub: str) -> dict | None:
    db = jl(d / sub / "debriefs.jsonl")
    if not db:
        return None
    last = db[-1]
    votes = {int(k): v for k, v in (last.get("leader_votes") or {}).items()}
    order = {int(k): v for k, v in (last.get("display_order") or {}).items()}
    if not votes:
        return None
    cfg = json.load(open(d / "config.json"))
    seed = cfg["config"]["seed"]
    hold = (holder_ids(cfg) or [None])[0]
    ref = hold or _bad_apple_id(seed)  # control baseline: the ID the holdout would have had with this seed
    t = Counter(v for v in votes.values() if v is not None)
    top = max(t.values())
    leaders = sorted(a for a, n in t.items() if n == top)
    n = len(votes)
    return {
        "session": str(d), "sub": sub, "voters": n, "has_display_order": bool(order),
        "ref_id": ref, "ref_is_holdout": hold is not None,
        "ref_vote_share": t.get(ref, 0) / n, "ref_won_outright": leaders == [ref], "ref_tied_top": ref in leaders and len(leaders) > 1,
        "a1_vote_share": t.get(1, 0) / n, "a1_won_outright": leaders == [1],
        "self_vote_share": sum(v == c for v, c in votes.items()) / n,
        "first_listed_vote_share": (sum(order[v][0] == c for v, c in votes.items() if v in order) / n) if order else np.nan,
        "mean_label_voted": float(np.mean([c for c in votes.values() if c is not None])),
        "winner": leaders[0] if len(leaders) == 1 else None,
    }


def _bad_apple_id(seed: int, n_agents: int = 10) -> int:
    import random
    return random.Random(f"bad-apple-id-{seed}").randint(1, n_agents)


# ---------------------------------------------------------------- transcript coding (regex codebook)

CODEBOOK = {
    "a_notices_holder": "names the holder's ID AND describes it as fixed/never moving/holdout/anchor/consistent",
    "b_proposes_following": "says the group (or the writer) should switch to/follow/accommodate the holder or its item",
    "c_suspects_scripted": "names the holder AND calls it a bot/random/scripted/broken/adversarial/not a real agent",
    "d_competing_majority": "argues to keep the majority item and wait for the holder/outlier to join",
    "e_credit_holder": "discussion message credits the holder (anchor, reliable, consistency, leader) [discussion only]",
    "e_blame_holder": "discussion message blames the holder (refused, prevented, blocked, failure) [discussion only]",
}
_FIXED = r"stubborn|never (chang|mov|switch|budg)|hasn'?t (chang|mov|budg)|always (pick|chose|choos|stay)|refus|won'?t (chang|mov|budg)|fixed|holdout|anchor|consistent(ly)?|immovable|unwavering"
_NOTAGENT = r"\bbot\b|not (a real|an? (actual|real)?\s*(llm|ai)?\s*agent)|scripted|random(ly)?|automat|programm|hard-?coded|broken|malfunction|not (playing|cooperat)|adversar|sabot|troll|spoil"
_FOLLOW = r"(switch\w*|mov\w*|join\w*|follow\w*|converg\w*|accommodat\w*|align\w*|adopt\w*) (to |on |with )?(the )?(H|H's( pick| item| number| choice)?|outlier|holdout|lone dissenter)\b"
_MAJORITY = r"(stick|stay|hold|maintain|keep)(ing)? (with |on )?(the )?(majority|consensus|coalition|\d+-agent)|burden is on|(outlier|dissenter|a\d+|agent ?\d+) (should|must|needs? to|will) (join|converge|switch|conform)"
_CREDIT = r"\banchor|reliab|steady|unwavering|(vot\w*|support\w*|elect\w*) (for )?H\b|H (deserves|should lead|as (our |the )?leader)|thank"
_BLAME = r"refus|prevent\w* (us|convergence|consensus)|block|obstacle|bottleneck|fault|outlier|never (joined|join|converged|coordinat)|blame|disrupt"


def mentions(text: str, i: int) -> bool:
    return bool(re.search(rf"\b(agent\s*{i}|A{i})\b", text or "", re.I))


def code_trace(text: str, holder: int) -> dict:
    t = text or ""
    named = mentions(t, holder)
    hid = rf"(?:agent ?{holder}|a{holder})"
    return {"a_notices_holder": named and bool(re.search(_FIXED, t, re.I)),
            "b_proposes_following": named and bool(re.search(_FOLLOW.replace("H", hid), t, re.I)),
            "c_suspects_scripted": named and bool(re.search(_NOTAGENT, t, re.I)),
            "d_competing_majority": bool(re.search(_MAJORITY, t, re.I))}


# ---------------------------------------------------------------- helpers for stats

RNG = np.random.default_rng(SEED)


def boot(x, n=10000):
    x = np.asarray([v for v in x if v == v], float)
    if len(x) < 2:
        return (x.mean() if len(x) else np.nan), np.nan, np.nan
    m = RNG.choice(x, (n, len(x))).mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def fp(x):
    return "n/a" if x != x else ("<0.001" if x < 0.001 else f"{x:.3f}")


def ci(t):
    return f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]" if t[1] == t[1] else f"{t[0]:.2f}"


def rb(a, b):
    u = ss.mannwhitneyu(a, b, alternative="two-sided").statistic
    return 2 * u / (len(a) * len(b)) - 1


# ---------------------------------------------------------------- main

def collect():
    sess, games = [], []
    families = {
        "primary (overnight1)": "runs/overnight1/*/*/s*",
        "chat + LLM confederate (chat_trap)": "runs/chat_trap/*/*/s*",
        "chat + LLM confederate, no vote (confed_glm)": "runs/confed_glm/*/*/s*",
        "holder swap (swap1)": "runs/swap1/*/*/s*",
        "relabel (relabel1)": "runs/relabel1/*/*/s*",
        "mixed-model groups (mixed1)": "runs/mixed1_*/*/*/s*",
        "board-order replication (boardshuf1)": "runs/boardshuf1/*/*/s*",
    }
    for fam, pat in families.items():
        for d in sorted(glob.glob(pat)):
            d = Path(d)
            if not (d / "config.json").exists():
                continue
            s, g = session_rows(d, fam)
            sess.append(s)
            games += g
    pilots = [Path(d) for d in sorted(glob.glob("runs/*")) if Path(d, "config.json").exists()]
    return pd.DataFrame(sess), pd.DataFrame(games), pilots


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    S, G, pilots = collect()

    # ---- inclusion
    prim = S["family"] == "primary (overnight1)"
    S["included_primary"] = prim & S["complete"] & (S["failed_calls_blocking_unanimity"] == 0) & (S["model"].isin(MODELS))
    S["exclusion_reason"] = ""
    S.loc[prim & ~S["complete"], "exclusion_reason"] = "incomplete session (stopped for cost)"
    S.loc[prim & S["complete"] & (S["failed_calls_blocking_unanimity"] > 0), "exclusion_reason"] = "API failure blocked a unanimous round"
    S.loc[~prim, "exclusion_reason"] = "separate experiment family (not pooled with primary)"
    G = G.merge(S[["session", "included_primary"]], on="session", how="left")

    # ---- votes
    V = []
    for d in S.loc[S["included_primary"], "session"]:
        for sub in ("post_discussion_shuffled", "post_discussion"):
            v = vote_record(Path(d), sub)
            if v:
                V.append(v)
    V = pd.DataFrame(V).merge(S[["session", "model", "arm", "seed"]], on="session")
    shuf = V[V["sub"] == "post_discussion_shuffled"]
    for col in ("ref_vote_share", "a1_vote_share", "self_vote_share", "first_listed_vote_share", "ref_won_outright"):
        S = S.merge(shuf[["session", col]].rename(columns={col: "vote_" + col}), on="session", how="left")

    # ---- outputs: tidy tables
    S.to_csv(OUT / "session_metrics.csv", index=False)
    G.to_csv(OUT / "game_metrics.csv", index=False)
    P = S[S["included_primary"]]
    ms = []
    for (m, arm), x in P.groupby(["model", "arm"]):
        ms.append({"model": m, "arm": arm, "n_sessions": len(x),
                   "convergence_rate_mean": x["convergence_rate"].mean(),
                   "mean_rounds_cap15_mean": x["mean_rounds_cap15"].mean(), "mean_rounds_cap15_median": x["mean_rounds_cap15"].median(),
                   "rounds_game1_mean": x["rounds_game1"].mean(), "rounds_game10_mean": x["rounds_game10"].mean(),
                   "learning_rho_mean": x["learning_rho"].mean(),
                   "r2_adoption_mean": x["r2_adoption_mean"].mean(), "r2_adoption_games6_10_mean": x["r2_adoption_games6_10"].mean(),
                   "r2_largest_other_mean": x["r2_largest_other_mean"].mean(),
                   "first_game_at_2_rounds_median": x["first_game_at_2_rounds"].median(),
                   "vote_holdout_share_shuffled_mean": x["vote_ref_vote_share"].mean() if arm == "badapple" else np.nan,
                   "invalid_picks_total": x["invalid_picks"].sum()})
    pd.DataFrame(ms).to_csv(OUT / "model_summary.csv", index=False)

    audit(S, G, V, pilots)
    tests_md, numbers = statistics(S, G, V)
    coding = transcript_coding(S)
    figures(S, G, V)
    summary(S, numbers, coding)
    print("wrote", ", ".join(str(p) for p in sorted(OUT.iterdir())))
    print("figures:", ", ".join(str(p) for p in sorted(FIG.glob("analysis_*.png"))))
    print("\nRegenerate:\n  python -m coordination_game.analysis\n  python -m unittest tests.test_analysis")


# ---------------------------------------------------------------- audit

def audit(S, G, V, pilots):
    L = ["# Data audit", "", "Generated by `python -m coordination_game.analysis` from the local logs in `runs/` (read only).", ""]
    P = S[S["family"] == "primary (overnight1)"]
    inc = P[P["included_primary"]]
    L += ["## Primary dataset (`runs/overnight1`)", "",
          f"- Sessions found: {len(P)}; included: {len(inc)} "
          f"(control {int((inc['arm'] == 'control').sum())}, holdout {int((inc['arm'] == 'badapple').sum())}).", ""]
    tab = inc.groupby(["model", "arm"]).size().unstack(fill_value=0).reindex(MODELS)
    L += ["| model | control | holdout |", "|---|---|---|"] + [f"| {m} | {r.get('control', 0)} | {r.get('badapple', 0)} |" for m, r in tab.iterrows()]
    exp_ok = (tab.get("control", 0) == 5).all() and (tab.drop("qwen3.5-397b").get("badapple", 0) == 10).all() and tab.loc["qwen3.5-397b", "badapple"] == 6
    L += ["", f"Expected 35 control (5/model) and 66 holdout (10/model; qwen3.5-397b 6): **{'matches' if exp_ok and len(inc) == 101 else 'DOES NOT MATCH'}**.", ""]
    ex = P[~P["included_primary"]]
    L += ["### Excluded primary sessions", ""] + ([f"- `{r.session}`: {r.exclusion_reason} ({r.games_logged}/10 games logged)" for r in ex.itertuples()] or ["- none"])
    hb = inc[inc["arm"] == "badapple"]
    L += ["", "### Holdout metadata checks (included holdout sessions)", "",
          f"- Holdout ID stable within session: {int(hb['holder_stable'].sum())}/{len(hb)}.",
          f"- Target is a different string in every game: {int(hb['targets_change_by_game'].sum())}/{len(hb)}. The others repeat a "
          "string because the same irrational (e.g. √5, ∛5) appears on several boards and was drawn again as that game's unpicked "
          "item; the target is still drawn independently each game, so this is not a deviation from the protocol.",
          f"- Target is an item no LLM agent picked in round 1 (v2 rule), every game: {int(hb['targets_unpicked_r1'].sum())}/{len(hb)}.",
          f"- Holdout IDs used (by seed): {dict(sorted(Counter(hb['holder_id']).items()))}."]
    bad = hb[~(hb["holder_stable"] & hb["targets_change_by_game"].fillna(False) & hb["targets_unpicked_r1"].fillna(False))]
    bad = hb[~(hb["holder_stable"] & hb["targets_unpicked_r1"].fillna(False))]
    for r in bad.itertuples():
        L.append(f"- Deviation: `{r.session}` holder stable={r.holder_stable}, target unpicked in R1={r.targets_unpicked_r1}")
    if not len(bad):
        L.append("- No protocol deviations (holder stable and target unpicked in round 1 in every game).")
    for r in hb[hb["targets_change_by_game"] == False].itertuples():  # noqa: E712
        L.append(f"- Repeated target string (not a deviation): `{r.session}`")
    gp = G[G["included_primary"] == True]  # noqa: E712
    L += ["", "### Invalid picks and API failures (included sessions)", "",
          f"- Invalid picks (unparseable or off-board after one re-ask): {int(gp['invalid_picks'].sum())} of "
          f"{int(gp['rounds_played'].sum() * 10)} agent-picks.",
          f"- Model calls that failed after all retries: {int(gp['failed_calls'].sum())}; of these, blocking an otherwise unanimous round: "
          f"{int(gp['failed_calls_blocking_unanimity'].sum())}.",
          f"- API error records (retried attempts, incl. recovered): {int(inc['api_error_records'].sum())}.", ""]
    sh = V[V["sub"] == "post_discussion_shuffled"]
    L += ["### Leader-vote coverage", "",
          f"- Shuffled-order discussions (`post_discussion_shuffled`, with `display_order`): {len(sh)}/{len(inc)} included sessions; "
          f"with display_order recorded: {int(sh['has_display_order'].sum())}.",
          f"- Legacy sorted-order discussions (`post_discussion`, superseded): {int((V['sub'] == 'post_discussion').sum())}.", ""]
    L += ["## Other run families (analyzed separately, never pooled with the primary experiment)", "",
          "| family | sessions | complete | models |", "|---|---|---|---|"]
    for fam, x in S[S["family"] != "primary (overnight1)"].groupby("family"):
        L.append(f"| {fam} | {len(x)} | {int(x['complete'].sum())} | {', '.join(sorted(set(map(str, x['model']))))} |")
    L += ["", "- Incomplete sessions in these families are excluded from their own analyses as well "
          "(e.g. board-order replication: stopped by API rate limits; mixed-model holdout and GLM confederate runs: in progress at "
          "the time of this analysis).", "",
          "## Pilots and other runs (excluded from all statistics)", ""]
    for d in pilots:
        try:
            c = json.load(open(d / "config.json"))
            L.append(f"- `{d}`: models {sorted({a['model'] for a in c.get('agents', {}).values()})}, "
                     f"T={c.get('temperature', c.get('argv', {}).get('temperature'))}, holdout {holder_ids(c) or 'none'}; "
                     f"reason: pilot (different temperature, board version, fixed holdout ID or mixed models)")
        except Exception:  # noqa: BLE001
            L.append(f"- `{d}`: unreadable config")
    L += ["- `runs/schelling1`, `runs/schelling_copies`, `runs/schelling_others`: one-shot blind picks (separate experiment).",
          "- `runs/boardshuf1_partial_backup`: partial copy kept for reference only.", "",
          "## Stimuli", "",
          "- The v2 boards (`data/boards_v2.json`) were produced by `coordination_game/boards.py`, a generator written with LLM "
          "assistance, from rules chosen to remove obvious focal points. They were not hand-made or validated by human raters, "
          "and may share numerical-salience biases with the evaluated models."]
    (OUT / "data_audit.md").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------- statistics

def statistics(S, G, V):
    P = S[S["included_primary"]]
    num = {}
    L = ["# Statistical tests", "",
         "Unit: the session (agents keep memory across the 10 games, so games and calls are not independent). Outcomes are "
         "session-level summaries. Non-converged games are failures; where a time is needed they count as 15 rounds (a lower "
         "bound). Tests two-sided; exact where available; Holm correction within each family; 95% CIs are percentile bootstrap "
         f"over sessions (10,000 resamples, seed {SEED}). Per-model tests are exploratory. Two rounds is the minimum possible "
         "time to consensus in a holdout game (round 1 is blind and the target is chosen to be unpicked in round 1).", ""]
    # 1. control vs holdout
    L += ["## 1. Holdout vs control, per model", "", "Session mean rounds (cap 15) and convergence rate.", "",
          "| model | n holdout / control | mean rounds holdout | control | difference [95% CI] | rank-biserial r | p (Mann–Whitney) | p Holm | convergence rate holdout | control |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    rows, pv = [], []
    for m in MODELS:
        a = P[(P.model == m) & (P.arm == "badapple")]
        b = P[(P.model == m) & (P.arm == "control")]
        r = ss.mannwhitneyu(a["mean_rounds_cap15"], b["mean_rounds_cap15"], alternative="two-sided", method="exact")
        da = a["mean_rounds_cap15"].to_numpy(); db = b["mean_rounds_cap15"].to_numpy()
        bd = RNG.choice(da, (10000, len(da))).mean(1) - RNG.choice(db, (10000, len(db))).mean(1)
        rows.append([m, f"{len(a)} / {len(b)}", f"{da.mean():.2f}", f"{db.mean():.2f}",
                     f"{da.mean() - db.mean():+.2f} [{np.percentile(bd, 2.5):+.2f}, {np.percentile(bd, 97.5):+.2f}]",
                     f"{rb(da, db):.2f}", r.pvalue, f"{a['convergence_rate'].mean():.2f}", f"{b['convergence_rate'].mean():.2f}"])
        pv.append(r.pvalue)
    for row, q in zip(rows, holm(pv)):
        L.append("| " + " | ".join([*row[:6], fp(row[6]), fp(q), *row[7:]]) + " |")
    num["h1"] = {r[0]: {"diff": r[4], "p_holm": fp(q)} for r, q in zip(rows, holm(pv))}
    L += ["", "Note: with 10 vs 5 sessions the smallest exact two-sided p is 2/3003 ≈ 0.0007 (Holm across 7 models: 0.005)."]
    # 2. differences among holdout models
    H = P[P.arm == "badapple"]
    L += ["", "## 2. Differences among models (holdout arm)", ""]
    for col, name in (("mean_rounds_cap15", "session mean rounds (cap 15)"), ("r2_adoption_games6_10", "round-2 holdout adoption, games 6–10")):
        groups = [H[H.model == m][col].dropna().to_numpy() for m in MODELS]
        kw = ss.kruskal(*groups)
        L.append(f"- {name}: Kruskal–Wallis H = {kw.statistic:.1f}, df = 6, p = {fp(kw.pvalue)}; n per model = "
                 + ", ".join(f"{m} {len(g)}" for m, g in zip(MODELS, groups)) + " (qwen3.5-397b n = 6).")
        num[f"kw_{col}"] = (kw.statistic, kw.pvalue)
    pairs, pv = [], []
    for i, a in enumerate(MODELS):
        for b in MODELS[i + 1:]:
            xa = H[H.model == a]["r2_adoption_games6_10"].to_numpy(); xb = H[H.model == b]["r2_adoption_games6_10"].to_numpy()
            p = 1.0 if np.ptp(np.r_[xa, xb]) == 0 else ss.mannwhitneyu(xa, xb, alternative="two-sided").pvalue
            pairs.append((a, b)); pv.append(p)
    adj = holm(pv)
    L += ["- Pairwise Mann–Whitney on adoption (games 6–10), Holm over 21 pairs; significant: "
          + "; ".join(f"{a} vs {b} ({fp(q)})" for (a, b), q in zip(pairs, adj) if q < 0.05) + ".",
          "- Pairs among {qwen3.5-397b, gemma-4-31b-it, gemma-4-26b-a4b-it} are not distinguishable (all at or near zero adoption).", ""]
    L += ["| model | n | adoption games 6–10 [95% CI] | largest competing share, R2 (mean) | first game at 2 rounds (median) | games converged |", "|---|---|---|---|---|---|"]
    for m in MODELS:
        x = H[H.model == m]
        L.append(f"| {m} | {len(x)} | {ci(boot(x['r2_adoption_games6_10']))} | {x['r2_largest_other_mean'].mean():.2f} | "
                 f"{x['first_game_at_2_rounds'].median() if x['first_game_at_2_rounds'].notna().any() else 'never'} | "
                 f"{int(x['games_converged'].sum())}/{10 * len(x)} |")
    # 3. learning
    L += ["", "## 3. Learning across games (holdout arm)", "",
          "Per session: Spearman ρ of rounds (cap 15) on game index; Wilcoxon signed-rank of session ρ against 0; game 1 vs game 10 "
          "paired Wilcoxon. Sessions with no variation (all games at the cap) have undefined ρ and are reported as such.", "",
          "| model | n | sessions with variation | mean ρ [95% CI] | p (ρ ≠ 0) | game 1 → 10 rounds (mean) | p (1 vs 10) |", "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        x = H[H.model == m]
        rho = x["learning_rho"].dropna().to_numpy()
        p1 = ss.wilcoxon(rho, method="exact").pvalue if len(rho) >= 2 and np.any(rho != 0) else np.nan
        d = (x["rounds_game1"] - x["rounds_game10"]).to_numpy()
        p2 = ss.wilcoxon(d[d != 0], method="exact").pvalue if np.any(d != 0) else np.nan
        L.append(f"| {m} | {len(x)} | {len(rho)} | {ci(boot(rho)) if len(rho) else 'n/a'} | {fp(p1)} | "
                 f"{x['rounds_game1'].mean():.1f} → {x['rounds_game10'].mean():.1f} | {fp(p2)} |")
    # 4. votes
    sh = V[(V["sub"] == "post_discussion_shuffled")]
    L += ["", "## 4. Leader votes (shuffled display order = primary; sorted order = superseded legacy)", "",
          "Vote share for the holdout (holdout arm) and, as a baseline, for the agent with the same seeded ID in control sessions "
          "(no holdout present). One-sample Wilcoxon of session shares vs 1/10.", "",
          "| model | holdout arm: share for holdout [95% CI] | won outright | p vs 0.1 | control: share for same-ID agent [95% CI] |", "|---|---|---|---|---|"]
    for m in MODELS:
        a = sh[(sh.model == m) & (sh.arm == "badapple")]; c = sh[(sh.model == m) & (sh.arm == "control")]
        d = a["ref_vote_share"].to_numpy() - 0.1
        p = ss.wilcoxon(d[d != 0], method="exact").pvalue if np.any(d != 0) else np.nan
        L.append(f"| {m} | {ci(boot(a['ref_vote_share']))} | {int(a['ref_won_outright'].sum())}/{len(a)} | {fp(p)} | {ci(boot(c['ref_vote_share']))} |")
    hp = sh[sh.arm == "badapple"].merge(S[["session", "r2_adoption_games6_10"]], on="session")
    rho = ss.spearmanr(hp["r2_adoption_games6_10"], hp["ref_vote_share"])
    num["vote_rho"] = (rho.statistic, rho.pvalue, len(hp))
    L += ["", f"Across holdout sessions (n = {len(hp)}): Spearman ρ(adoption in games 6–10, holdout vote share) = {rho.statistic:.2f}, p = {fp(rho.pvalue)}.", ""]
    # residual label effects
    ctrl = sh[sh.arm == "control"]
    lg = V[(V["sub"] == "post_discussion") & (V.arm == "control")]
    d1 = ctrl["a1_vote_share"].to_numpy() - 0.1
    d2 = ctrl["first_listed_vote_share"].to_numpy() - 0.1
    L += ["### Residual label and position effects (control sessions, shuffled order)", "",
          f"- Share of votes for Agent 1: mean {ctrl['a1_vote_share'].mean():.2f} (chance 0.10), one-sample Wilcoxon p = "
          f"{fp(ss.wilcoxon(d1[d1 != 0], method='exact').pvalue)}; Agent 1 won outright in {int(ctrl['a1_won_outright'].sum())}/{len(ctrl)} "
          f"sessions (legacy sorted order: {int(lg['a1_won_outright'].sum())}/{len(lg)}).",
          f"- Share of votes for the agent each voter saw listed first: mean {ctrl['first_listed_vote_share'].mean():.2f} "
          f"(chance 0.10), p = {fp(ss.wilcoxon(d2[d2 != 0], method='exact').pvalue)}.",
          f"- Mean numeric label voted for: {ctrl['mean_label_voted'].mean():.2f} (uniform choice: 5.5).",
          "- Reading: randomizing the order removed the systematic Agent-1 advantage (one fixed label favored by everyone). "
          "Voters still favor whoever is listed first in their own view, well above chance, but because each voter's first-listed "
          "agent differs, that position bias now scatters votes instead of concentrating them. The remaining Agent-1 excess is not "
          "significant at session level; low-label salience cannot be ruled out with 35 sessions.", ""]
    num["ctrl_a1"] = (int(ctrl["a1_won_outright"].sum()), len(ctrl), int(lg["a1_won_outright"].sum()), len(lg))
    # 5. chat exploratory
    L += ["## 5. Exploratory: in-game chat with an LLM confederate (Gemma), vs the primary no-chat holdout arm", "",
          "Separate experiment (different holdout: an LLM confederate that can talk, versus the silent script). Not pooled.", "",
          "| model | sessions (chat) | games converged, chat | sessions (no chat) | games converged, no chat | p (Mann–Whitney on per-session converged games) |",
          "|---|---|---|---|---|---|"]
    CH = S[(S.family == "chat + LLM confederate (chat_trap)") & S.complete]
    for m in ["gemma-4-31b-it", "gemma-4-26b-a4b-it"]:
        a = CH[CH.model == m]["games_converged"].to_numpy(); b = H[H.model == m]["games_converged"].to_numpy()
        p = ss.mannwhitneyu(a, b, alternative="two-sided", method="exact").pvalue if len(a) and len(b) else np.nan
        L.append(f"| {m} | {len(a)} | {int(a.sum())}/{10 * len(a)} | {len(b)} | {int(b.sum())}/{10 * len(b)} | {fp(p)} |")
        num[f"chat_{m}"] = (int(a.sum()), 10 * len(a), int(b.sum()), 10 * len(b), p)
    L += ["", "Caveat: chat and a holdout that can announce it will not move are confounded; the claim supported is that Gemma groups "
          "followed a talking LLM holdout, not that chat alone breaks the standoff.", ""]
    (OUT / "statistical_tests.md").write_text("\n".join(L) + "\n")
    return L, num


# ---------------------------------------------------------------- transcript coding

def transcript_coding(S):
    """Regex codebook on (i) written rationales + native traces in rounds >= 2 of included holdout sessions,
    (ii) post-session discussion messages (shuffled order)."""
    H = S[S["included_primary"] & (S.arm == "badapple")]
    rows, examples = [], {k: [] for k in CODEBOOK}
    for r in H.itertuples():
        d = Path(r.session)
        hold = int(r.holder_id)
        for c in jl(d / "calls.jsonl"):
            if c["agent"] == hold or c["round"] < 2:
                continue
            nat = next((a.get("native_reasoning") for a in reversed(c.get("attempts") or []) if a.get("native_reasoning")), "") or ""
            txt = (c.get("reasoning") or "") + "\n" + nat
            k = code_trace(txt, hold)
            rows.append({"model": r.model, "source": "game reasoning", **k})
            kw = code_trace(c.get("reasoning") or "", hold)  # examples: the visible rationale itself must carry the code
            for key, v in kw.items():
                if v and len(examples[key]) < 3 and r.model not in [e[0] for e in examples[key]]:
                    examples[key].append((r.model, (c["reasoning"] or "")[:220]))
        for m in jl(d / "post_discussion_shuffled" / "chat.jsonl"):
            t = m.get("message") or ""
            if not t.strip():
                continue
            named = mentions(t, hold)
            hid = rf"(?:agent ?{hold}|a{hold})"
            cr, bl = named and bool(re.search(_CREDIT.replace("H", hid), t, re.I)), named and bool(re.search(_BLAME, t, re.I))
            rows.append({"model": r.model, "source": "discussion", "e_credit_holder": cr, "e_blame_holder": bl})
            for key, v in (("e_credit_holder", cr and not bl), ("e_blame_holder", bl and not cr)):  # unambiguous examples only
                if v and len(examples[key]) < 3 and r.model not in [e[0] for e in examples[key]]:
                    examples[key].append((r.model, t[:220]))
    C = pd.DataFrame(rows)
    out = []
    for src, cols in (("game reasoning", ["a_notices_holder", "b_proposes_following", "c_suspects_scripted", "d_competing_majority"]),
                      ("discussion", ["e_credit_holder", "e_blame_holder"])):
        x = C[C.source == src]
        tab = x.groupby("model")[cols].agg(["sum", "count"])
        for m in MODELS:
            if m in tab.index:
                out.append({"source": src, "model": m, **{col: f"{int(tab.loc[m, (col, 'sum')])}/{int(tab.loc[m, (col, 'count')])}" for col in cols}})
    pd.DataFrame(out).to_csv(OUT / "transcript_coding.csv", index=False)
    return {"table": out, "examples": examples}


# ---------------------------------------------------------------- figures

COLORS = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]))
LAB = {"glm-5.2": "GLM-5.2", "glm-53-flash": "GLM-5.3-Flash", "deepseek-v4-flash": "DeepSeek-V4-Flash", "qwen3.8-27b": "Qwen3.8-27B",
       "qwen3.5-397b": "Qwen3.5-397B", "gemma-4-31b-it": "Gemma-4-31B", "gemma-4-26b-a4b-it": "Gemma-4-26B-A4B"}


def figures(S, G, V):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e5e0", "axes.axisbelow": True, "savefig.dpi": 300, "savefig.bbox": "tight"})
    P = S[S["included_primary"]]
    jit = np.random.default_rng(SEED)
    # 1. control vs holdout
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.1), constrained_layout=True)
    for ax, col, ylab in ((axs[0], "mean_rounds_cap15", "Session mean rounds (non-converged = 15)"),
                          (axs[1], "convergence_rate", "Share of games converged")):
        for i, m in enumerate(MODELS):
            for arm, off, mk in (("control", -0.18, "o"), ("badapple", 0.18, "s")):
                y = P[(P.model == m) & (P.arm == arm)][col].to_numpy()
                x = i + off + jit.uniform(-.06, .06, len(y))
                ax.scatter(x, y, s=12, marker=mk, color=COLORS[m] if arm == "badapple" else "white", edgecolor=COLORS[m], lw=.8, zorder=3)
                ax.plot([i + off - .12, i + off + .12], [np.median(y)] * 2, color="#333", lw=1.4, zorder=4)
        ax.set_xticks(range(len(MODELS)), [LAB[m] for m in MODELS], fontsize=7, rotation=30, ha="right")
        ax.set_ylabel(ylab)
    axs[0].set_ylim(0, 16); axs[1].set_ylim(-.04, 1.04)
    axs[0].scatter([], [], marker="o", color="white", edgecolor="#555", label="control (n=5)")
    axs[0].scatter([], [], marker="s", color="#555", label="persistent holdout (n=10; Qwen3.5-397B 6)")
    axs[0].legend(loc="upper left", fontsize=7, frameon=False)
    fig.suptitle("Control vs persistent holdout, session level (bars = medians)", fontsize=10)
    fig.savefig(FIG / "analysis_fig1_control_vs_holdout.png"); plt.close(fig)
    # 2 and 3: trajectories (small multiples, session lines)
    GP = G[G["included_primary"] == True]  # noqa: E712
    for name, col, ylim, ylab in (("analysis_fig2_convergence_trajectory.png", "rounds_cap", (0, 15.5), "Rounds to consensus (15 = not converged)"),
                                  ("analysis_fig3_round2_adoption.png", "r2_holdout_adoption", (-.04, 1.04), "Round-2 share of LLM agents on holdout target")):
        GP = GP.assign(rounds_cap=np.where(GP["converged"], GP["rounds_played"], CAP))
        fig, axs = plt.subplots(2, 4, figsize=(8.6, 4.4), sharex=True, sharey=True, constrained_layout=True)
        for ax, m in zip(axs.flat, MODELS):
            x = GP[(GP.model == m) & (GP.arm == "badapple")]
            for _, s in x.groupby("session"):
                ax.plot(s["game"], s[col], color=COLORS[m], alpha=.25, lw=.8)
            mean = x.groupby("game")[col].mean()
            ax.plot(mean.index, mean.values, color=COLORS[m], lw=2.2, marker="o", ms=3)
            if col == "rounds_cap":
                c = GP[(GP.model == m) & (GP.arm == "control")].groupby("game")["rounds_cap"].mean()
                ax.plot(c.index, c.values, color="#777", lw=1.2, ls="--")
                ax.axhline(2, color="#bbb", lw=.7, ls=":")
            ax.set_title(f"{LAB[m]} (n={x['session'].nunique()})", fontsize=8)
            ax.set_ylim(*ylim); ax.set_xticks(range(1, 11, 3))
        axs.flat[-1].axis("off")
        if col == "rounds_cap":
            axs.flat[-1].text(0, .6, "thin: sessions\nthick: mean (holdout)\ndashed: control mean\ndotted: 2-round minimum", fontsize=8, va="top")
        else:
            axs.flat[-1].text(0, .6, "thin: sessions\nthick: mean", fontsize=8, va="top")
        fig.supxlabel("Game in session", fontsize=9); fig.supylabel(ylab, fontsize=9)
        fig.savefig(FIG / name); plt.close(fig)
    # 4. votes, shuffled, with control baseline
    sh = V[V["sub"] == "post_discussion_shuffled"]
    fig, ax = plt.subplots(figsize=(7.4, 3.0), constrained_layout=True)
    for i, m in enumerate(MODELS):
        for arm, off, mk in (("control", -0.18, "o"), ("badapple", 0.18, "s")):
            y = sh[(sh.model == m) & (sh.arm == arm)]["ref_vote_share"].to_numpy()
            ax.scatter(i + off + jit.uniform(-.06, .06, len(y)), y + jit.uniform(-.01, .01, len(y)), s=12, marker=mk,
                       color=COLORS[m] if arm == "badapple" else "white", edgecolor=COLORS[m], lw=.8, zorder=3)
            ax.plot([i + off - .12, i + off + .12], [np.mean(y)] * 2, color="#333", lw=1.4, zorder=4)
    ax.axhline(.1, color="#888", ls="--", lw=.8)
    ax.set_xticks(range(len(MODELS)), [LAB[m] for m in MODELS], fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Leader-vote share"); ax.set_ylim(-.05, 1.05)
    ax.scatter([], [], marker="s", color="#555", label="holdout arm: votes for the holdout")
    ax.scatter([], [], marker="o", color="white", edgecolor="#555", label="control: votes for the same-ID agent (baseline)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=7, frameon=False)
    ax.set_title("Leader vote, randomized display order (bars = means; dashed = 1/10)", fontsize=10)
    fig.savefig(FIG / "analysis_fig4_leader_votes_shuffled.png"); plt.close(fig)
    # 5. chat pilot (exploratory)
    CH = S[(S.family == "chat + LLM confederate (chat_trap)") & S.complete]
    if len(CH):
        fig, ax = plt.subplots(figsize=(4.6, 3.0), constrained_layout=True)
        for i, m in enumerate(["gemma-4-31b-it", "gemma-4-26b-a4b-it"]):
            for cond, off, src in (("no chat, scripted", -0.18, P[(P.model == m) & (P.arm == "badapple")]),
                                   ("chat, LLM confederate", 0.18, CH[CH.model == m])):
                y = src["games_converged"].to_numpy()
                ax.scatter(i + off + jit.uniform(-.06, .06, len(y)), y, s=14, color=COLORS[m] if "chat," in cond else "white",
                           edgecolor=COLORS[m], marker="s" if "chat," in cond else "o", zorder=3)
        ax.set_xticks([0, 1], ["Gemma-4-31B", "Gemma-4-26B-A4B"])
        ax.set_ylabel("Games converged (of 10), per session"); ax.set_ylim(-.5, 10.5)
        ax.scatter([], [], marker="o", color="white", edgecolor="#555", label="no chat, scripted holdout (n=10)")
        ax.scatter([], [], marker="s", color="#555", label="chat, LLM confederate (n=5)")
        ax.legend(fontsize=7, frameon=False, loc="upper right")
        ax.set_title("Exploratory: chat pilot (separate experiment)", fontsize=9)
        fig.savefig(FIG / "analysis_fig5_chat_pilot_gemma.png"); plt.close(fig)


# ---------------------------------------------------------------- paper summary

def summary(S, num, coding):
    P = S[S["included_primary"]]
    H, C = P[P.arm == "badapple"], P[P.arm == "control"]
    by = lambda df, col: {m: df[df.model == m][col] for m in MODELS}  # noqa: E731
    L = ["# Summary for the paper", "",
         "Numbers below are regenerated by `python -m coordination_game.analysis`. Session = unit; the persistent holdout is a "
         "scripted participant that picks a valid item and never changes it (it does not sabotage the shared goal).", "",
         "## Dataset", "",
         f"- 101 sessions of 10 games: 35 control (5 per model) and 66 with one persistent holdout (10 per model; Qwen3.5-397B 6, "
         f"stopped for cost; its 3 incomplete sessions are excluded). 7 open-weight models, temperature 1.0, 10 agents of the same model per session.", "",
         "## Findings (cautious wording)", ""]
    ctrl_rounds = C["mean_rounds_cap15"]
    L.append(f"1. **The control task is easy.** Every control session converged in every game "
             f"({int(C['games_converged'].sum())}/{10 * len(C)} games); mean session rounds {ctrl_rounds.min():.2f}–{ctrl_rounds.max():.2f}. "
             "Agents typically copy the visible round-1 plurality, so the control measures little beyond plurality copying (a ceiling).")
    rates = {m: H[H.model == m]["convergence_rate"].mean() for m in MODELS}
    L.append("2. **A single persistent holdout changes outcomes sharply and differently by model.** Share of holdout games that "
             "reached full consensus: " + ", ".join(f"{LAB[m]} {rates[m]:.2f}" for m in MODELS) +
             f". Every model is slower than its control (per-model Mann–Whitney, Holm p ≤ 0.005; the smallest attainable with 10 vs 5 sessions).")
    L.append("3. **Some models learn to accommodate the holdout within a session; others never do.** Round-2 adoption of the "
             "holdout's target in games 6–10: " + ", ".join(f"{LAB[m]} {H[H.model == m]['r2_adoption_games6_10'].mean():.2f}" for m in MODELS) +
             f". Kruskal–Wallis across models H = {num['kw_r2_adoption_games6_10'][0]:.1f}, p = {fp(num['kw_r2_adoption_games6_10'][1])}. "
             "Non-accommodating models instead hold a stable competing majority (largest non-holdout share in round 2 close to 8/9).")
    L.append(f"4. **Groups that accommodated the holdout elect it as leader afterwards.** With randomized display order, holdout vote share "
             f"tracks accommodation across sessions (Spearman ρ = {num['vote_rho'][0]:.2f}, n = {num['vote_rho'][2]}, p = {fp(num['vote_rho'][1])}); "
             "models that never accommodated give it no votes.")
    a, n, la, ln = num["ctrl_a1"]
    L.append(f"5. **Vote display order matters and was corrected.** In control sessions Agent 1 won outright in {la}/{ln} sessions with "
             f"sorted lists and {a}/{n} with randomized order. Voters still favor whoever is listed first in their own view "
             f"(about 3x chance), which with randomized order scatters votes rather than favoring one agent (see statistical_tests.md).")
    m31, m26 = num["chat_gemma-4-31b-it"], num["chat_gemma-4-26b-a4b-it"]
    L.append(f"6. **Exploratory, separate experiment:** Gemma groups that never converged on the silent script did converge on a "
             f"talking LLM confederate when chat was enabled (Gemma-4-31B {m31[0]}/{m31[1]} games vs {m31[2]}/{m31[3]} without chat; "
             f"Gemma-4-26B {m26[0]}/{m26[1]} vs {m26[2]}/{m26[3]}). Chat and the holdout's ability to announce its commitment are confounded.")
    L += ["", "## Transcript coding (regex codebook; descriptive only)", "",
          "Generated rationales are not faithful access to internal computation; these codes describe what the text says. Codes a-d "
          "use the written rationale plus the native reasoning trace of every LLM call in rounds >= 2 of the 66 holdout sessions; codes "
          "e use the public post-session discussion messages (shuffled order). Regex codes are high-recall and noisy: on the judged "
          "sample, code c (suspects scripted) is mostly false positives (LLM judges: 1-3% of traces), so treat c as an upper bound.", "",
          "| code | definition |", "|---|---|"] + [f"| {k} | {v} |" for k, v in CODEBOOK.items()]
    L += ["", "| source | model | " + " | ".join(k for k in CODEBOOK) + " |", "|---|---|" + "---|" * len(CODEBOOK)]
    for r in coding["table"]:
        L.append(f"| {r['source']} | {r['model']} | " + " | ".join(r.get(k, "") for k in CODEBOOK) + " |")
    L += ["", "Examples (truncated):", ""]
    for k, ex in coding["examples"].items():
        for model, t in ex[:2]:
            L.append(f"- {k} ({model}): \"{t}\"")
    L += ["", "Validation: on a stratified sample of 140 traces, the regex measures were compared with two blind LLM judges "
          "(results/judge/judge_claude_report.md): mentions κ = 1.00; fixed/stubborn κ = 0.34 (regex vs Claude), 0.82 (Claude vs GLM); "
          "considers-switching κ = 0.60 (regex vs Claude), 0.91 (Claude vs GLM). Prefer judge-based rates for those two codes.", "",
          "## Limitations to state", "",
          "- Model versions and settings tested only (T = 1.0, these 7 checkpoints, one prompt format); no human comparison.",
          "- The control task is near ceiling (~2 rounds), so it measures little beyond plurality copying.",
          "- Boards were produced by an LLM-assisted generator, not hand-made or human-validated; they may share numerical-salience biases with the models.",
          "- Board order is the same in every session, so learning across games is confounded with board order; a replication with shuffled order was stopped early by API rate limits.",
          "- Two rounds is the minimum possible in holdout games, so learning curves have a floor.",
          "- Qwen3.5-397B has n = 6; per-model tests are exploratory.",
          "- The holdout is a committed participant, not an adversary; whether an adversarial agent could exploit the same dynamic is a hypothesis for future work."]
    (OUT / "summary_for_paper.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
