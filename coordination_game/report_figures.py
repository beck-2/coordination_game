"""Report figures (one per experiment) from the raw run logs.

    python -m coordination_game.report_figures            # -> results/figures/report/fig*.{pdf,png}

Conventions: session = unit of analysis; error bars are 95% percentile-bootstrap CIs over sessions (2,000 resamples,
seed 0) unless a caption says otherwise; non-converged games count as the 15-round cap. One fixed color + marker per
model across every figure (categorical palette in fixed order; markers are the secondary, non-color encoding).
"""
from __future__ import annotations

import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = Path("runs")
OUT = Path("results/figures/report")
CAP = 15
RNG = np.random.default_rng(0)

MODELS = ["glm-5.2", "glm-53-flash", "deepseek-v4-flash", "qwen3.8-27b", "qwen3.5-397b",
          "gemma-4-31b-it", "gemma-4-26b-a4b-it"]
LABEL = {"glm-5.2": "GLM-5.2", "glm-53-flash": "GLM-5.3-Flash", "deepseek-v4-flash": "DeepSeek-V4-Flash",
         "qwen3.8-27b": "Qwen3.8-27B", "qwen3.5-397b": "Qwen3.5-397B", "gemma-4-31b-it": "Gemma-4-31B",
         "gemma-4-26b-a4b-it": "Gemma-4-26B-A4B"}
# Categorical palette slots 1-7 in fixed order (dataviz reference palette, light mode); markers as secondary encoding.
COLOR = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]))
MARK = dict(zip(MODELS, ["o", "s", "D", "^", "v", "P", "X"]))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
COND = {"baseline": ("#52514e", "o"), "swap": ("#eb6834", "s"), "relabel": ("#2a78d6", "D"),
        "nochat": ("#52514e", "o"), "chat": ("#2a78d6", "s")}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "lines.linewidth": 1.6, "lines.markersize": 4.5, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ---------------- helpers ----------------

def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)] if p.exists() else []


def boot_ci(x, n=2000):
    x = np.asarray([v for v in x if v == v], float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    if len(x) == 1:
        return x[0], x[0], x[0]
    m = RNG.choice(x, (n, len(x))).mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def sessions(pattern: str, n_games: int = 10) -> list[Path]:
    return [Path(d) for d in sorted(glob.glob(pattern)) if len(jl(Path(d) / "games.jsonl")) >= n_games]


def rounds_by_game(d: Path) -> list[int]:
    return [g["rounds"] for g in sorted(jl(d / "games.jsonl"), key=lambda g: g["game"])]


def picks(d: Path) -> dict[tuple[int, int], dict[str, str]]:
    return {(r["game"], r["round"]): r["picks"] for r in jl(d / "rounds.jsonl")}


def r2_share(d: Path, item_of, exclude_of) -> list[float]:
    """Per game: share of agents (minus excluded IDs) whose round-2 pick is the given item."""
    P, out = picks(d), []
    for g in sorted(jl(d / "games.jsonl"), key=lambda g: g["game"]):
        gi, item, ex = g["game"], item_of(g), {str(x) for x in exclude_of(g)}
        r2 = P.get((gi, 2))
        if r2 is None:
            out.append(np.nan)
            continue
        ag = [a for a in r2 if a not in ex]
        out.append(sum(r2[a] == item for a in ag) / len(ag))
    return out


def holder_item(g):
    return next(iter(g["bad_apple_picks"].values()))


def plot_ci_line(ax, x, rows, color, marker, label, offset=0.0, lw=1.6, ms=4.5, ls="-", z=3):
    rows = np.asarray(rows, float)
    stats = [boot_ci(rows[:, j]) for j in range(rows.shape[1])]
    m = np.array([s[0] for s in stats]); lo = np.array([s[1] for s in stats]); hi = np.array([s[2] for s in stats])
    xx = np.asarray(x, float) + offset
    ax.fill_between(xx, lo, hi, color=color, alpha=0.12, lw=0, zorder=z - 1)
    ax.plot(xx, m, color=color, marker=marker, label=label, lw=lw, ms=ms, ls=ls, zorder=z,
            markeredgecolor="white", markeredgewidth=0.6)
    return m


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(OUT / f"{name}.pdf")


def panel(ax, letter):
    ax.text(-0.13, 1.04, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", color=INK)


# ---------------- Figure 1: control vs bad apple learning ----------------

def fig1():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), constrained_layout=True)
    G = np.arange(1, 11)
    for arm, ax, letter, title in (("control", axs[0], "a", "Control (no bad apple)"),
                                   ("badapple", axs[1], "b", "Bad apple present")):
        for m in MODELS:
            ss = sessions(f"runs/overnight1/{arm}/{m}/s*")
            if ss:
                plot_ci_line(ax, G, [rounds_by_game(d) for d in ss], COLOR[m], MARK[m], LABEL[m])
        ax.axhline(CAP, color=INK2, lw=0.8, ls=(0, (4, 3)))
        ax.text(10.4, CAP, "cap", color=INK2, fontsize=7, va="center", ha="left")
        ax.set(title=title, xlabel="Game in session", ylabel="Rounds to consensus", ylim=(0, 16), xticks=G)
        panel(ax, letter)
    ax = axs[2]
    for m in MODELS:
        ss = sessions(f"runs/overnight1/badapple/{m}/s*")
        rows = [r2_share(d, holder_item, lambda g: g["bad_apple_picks"].keys()) for d in ss]
        plot_ci_line(ax, G, rows, COLOR[m], MARK[m], LABEL[m])
    ax.set(title="Yielding to the bad apple", xlabel="Game in session",
           ylabel="Share on bad apple's item, round 2", ylim=(-0.03, 1.03), xticks=G)
    panel(ax, "c")
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.1), handlelength=1.6, columnspacing=1.0)
    save(fig, "fig1_learning")


# ---------------- Figure 2: leader election ----------------

def votes(d: Path, sub: str):
    db = jl(d / sub / "debriefs.jsonl")
    if not db:
        return None
    return {int(k): v for k, v in (db[-1].get("leader_votes") or {}).items()}


def fig2():
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), constrained_layout=True,
                            gridspec_kw={"width_ratios": [1.0, 0.75, 1.0]})
    y = np.arange(len(MODELS))[::-1]

    def dots(ax, arm, voted):  # voted(d) -> agent ID whose vote share is plotted
        for i, m in enumerate(MODELS):
            for sub, dy, filled in (("post_discussion", 0.16, False), ("post_discussion_shuffled", -0.16, True)):
                shares = []
                for d in sessions(f"runs/overnight1/{arm}/{m}/s*"):
                    v = votes(d, sub)
                    if v:
                        t = voted(d)
                        shares.append(sum(x == t for x in v.values()) / len(v))
                mu, lo, hi = boot_ci(shares)
                ax.errorbar(mu, y[i] + dy, xerr=[[mu - lo], [hi - mu]], fmt=MARK[m], color=COLOR[m], ms=5, capsize=0,
                            lw=1.2, mfc=COLOR[m] if filled else "white", mec=COLOR[m], mew=1.2)
        ax.axvline(1 / 10, color=INK2, lw=0.8, ls=(0, (4, 3)))
        ax.set(xlim=(-0.04, 1.04), ylim=(-0.6, len(MODELS) - 0.4))
        ax.grid(axis="y", visible=False)

    ax = axs[0]
    dots(ax, "badapple", lambda d: json.load(open(d / "config.json"))["config"]["bad_apple_ids"][0])
    ax.set_yticks(y, [LABEL[m] for m in MODELS])
    ax.set(xlabel="Votes for the bad apple (share)", title="Bad-apple sessions")
    ax.text(0.115, -0.5, "chance", color=INK2, fontsize=7, va="bottom")
    fig.legend([Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, mew=1.2),
                Line2D([], [], marker="o", ls="", color=INK2)],
               ["discussion lists agents A1–A10 (sorted)", "each agent sees its own shuffled order"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.36, -0.08), handletextpad=0.2)
    panel(ax, "a")
    ax = axs[1]
    dots(ax, "control", lambda d: 1)
    ax.set_yticks(y, [""] * len(MODELS))
    ax.set(xlabel="Votes for Agent 1 (share)", title="Control sessions")
    panel(ax, "b")
    ax = axs[2]
    for m in MODELS:
        xs, ys = [], []
        for d in sessions(f"runs/overnight1/badapple/{m}/s*"):
            v = votes(d, "post_discussion_shuffled")
            if not v:
                continue
            h = json.load(open(d / "config.json"))["config"]["bad_apple_ids"][0]
            r2 = r2_share(d, holder_item, lambda g: g["bad_apple_picks"].keys())
            xs.append(np.nanmean(r2[5:])); ys.append(sum(x == h for x in v.values()) / len(v))
        jit = RNG.uniform(-0.015, 0.015, len(ys))
        ax.scatter(np.array(xs) + jit[::-1], np.array(ys) + jit, s=24, marker=MARK[m], color=COLOR[m],
                   edgecolor="white", lw=0.5, label=LABEL[m], zorder=3, alpha=0.85)
    ax.set(xlim=(-0.05, 1.05), ylim=(-0.06, 1.06), xlabel="Yielding in games 6–10\n(share on bad apple's item, round 2)",
           ylabel="Votes for the bad apple (share)", title="Per session (shuffled list)")
    panel(ax, "c")
    save(fig, "fig2_leader_vote")


# ---------------- Figure 3: Qwen3.5-397B coordination trap ----------------

def fig3():
    m = "qwen3.5-397b"
    solo, paired, total_reg = Counter(), Counter(), Counter()
    switch_rounds = defaultdict(list)
    for d in sessions(f"runs/overnight1/badapple/{m}/s*"):
        P = picks(d)
        for g in jl(d / "games.jsonl"):
            gi, item = g["game"], holder_item(g)
            h = set(g["bad_apple_picks"])
            for (gg, rr), pr in P.items():
                if gg != gi or rr < 2:
                    continue
                k = [a for a, v in pr.items() if a not in h and v == item]
                if len(k) == 1:
                    solo[gi] += 1
                elif len(k) >= 2:
                    paired[gi] += len(k)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True, gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axs[0]
    G = np.arange(1, 11)
    a = np.array([solo[g] for g in G]); b = np.array([paired[g] for g in G])
    ax.bar(G, a, 0.7, color="#e87ba4", label="alone (1 agent)", zorder=2)
    ax.bar(G, b, 0.7, bottom=a, color="#9c3d67", label="with one other agent", zorder=2,
           edgecolor="white", linewidth=1)
    ax.set(xticks=G, xlabel="Game in session", ylabel="Agent-rounds on bad apple's item",
           title="Qwen3.5-397B: lone attempts to yield fade out")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    panel(ax, "a")
    # (b) comparison: share of agent-rounds (rounds >= 2) on the holder item, by model, first vs last 5 games
    ax = axs[1]
    for i, mm in enumerate(MODELS):
        vals = {"early": [], "late": []}
        for d in sessions(f"runs/overnight1/badapple/{mm}/s*"):
            P = picks(d)
            for half, gs in (("early", range(1, 6)), ("late", range(6, 11))):
                n = k = 0
                for g in jl(d / "games.jsonl"):
                    if g["game"] not in gs:
                        continue
                    item, h = holder_item(g), set(g["bad_apple_picks"])
                    for (gg, rr), pr in P.items():
                        if gg == g["game"] and rr >= 2:
                            for a_, v in pr.items():
                                if a_ not in h:
                                    n += 1; k += v == item
                vals[half].append(k / n if n else np.nan)
        e, l = boot_ci(vals["early"]), boot_ci(vals["late"])
        ax.plot([0, 1], [e[0], l[0]], color=COLOR[mm], marker=MARK[mm], lw=1.4, ms=5, label=LABEL[mm],
                markeredgecolor="white", markeredgewidth=0.6)
    ax.set(xticks=[0, 1], xlim=(-0.25, 1.25), ylim=(-0.03, 1.03), ylabel="Share of agent-picks on\nbad apple's item (rounds ≥ 2)",
           title="All models, games 1–5 vs 6–10")
    ax.set_xticklabels(["games 1–5", "games 6–10"])
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7)
    panel(ax, "b")
    save(fig, "fig3_qwen_trap")


# ---------------- Figure 4: in-game chat with a talking confederate (Gemma) ----------------

def fig4():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True, sharey=True)
    G = np.arange(1, 11)
    for ax, m, letter in ((axs[0], "gemma-4-31b-it", "a"), (axs[1], "gemma-4-26b-a4b-it", "b")):
        for key, pat, lab in (("nochat", f"runs/overnight1/badapple/{m}/s*", "no chat, scripted bad apple (n=10)"),
                              ("chat", f"runs/chat_trap/chat_confed_safe/{m}/s*", "chat, LLM confederate (n=5)")):
            ss = sessions(pat)
            rows = [rounds_by_game(d) for d in ss]
            c, mk = COND[key]
            for r in rows:
                ax.plot(G, r, color=c, lw=0.6, alpha=0.35, zorder=1)
            plot_ci_line(ax, G, rows, c, mk, lab, lw=2)
        ax.axhline(CAP, color=INK2, lw=0.8, ls=(0, (4, 3)))
        ax.set(title=LABEL[m], xlabel="Game in session", xticks=G, ylim=(0, 16))
        panel(ax, letter)
    axs[0].set_ylabel("Rounds to consensus")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1))
    save(fig, "fig4_chat_confederate")


# ---------------- Figure 5: holder swap vs relabel ----------------

def fig5():
    G = np.arange(1, 11)
    base = [Path(f"runs/overnight1/badapple/glm-5.2/s{s}") for s in range(10)]
    swap = sessions("runs/swap1/badapple_swap/glm-5.2/s*")
    rel = sessions("runs/relabel1/badapple_relabel/glm-5.2/s*")

    def r2_baseline(d):
        return r2_share(d, holder_item, lambda g: g["bad_apple_picks"].keys())

    def r2_swap(d):  # current scripted holder's item; exclude only the current holder
        return r2_share(d, lambda g: g["bad_apple_picks"][str(g.get("holder_ids", [int(next(iter(g["bad_apple_picks"])))])[0])],
                        lambda g: g.get("holder_ids", list(g["bad_apple_picks"])))

    def r2_relabel(d):  # viewers only (exclude the bot and, from game 6, the partner, who sees no relabel)
        c = json.load(open(d / "config.json"))["config"]
        bot, P_ = c["bad_apple_ids"][0], c["relabel_partner"]
        return r2_share(d, holder_item, lambda g: [bot, P_])

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    series = (("baseline", base, r2_baseline, "no change (n=10)"),
              ("swap", swap, r2_swap, "holder swap: new scripted holder,\nold holder becomes an LLM (n=10)"),
              ("relabel", rel, r2_relabel, "relabel: same bot, labels traded\n(viewers only, n=10)"))
    for key, ss, f, lab in series:
        c, mk = COND[key]
        off = {"baseline": -0.12, "swap": 0.0, "relabel": 0.12}[key]
        for ax, rows in ((axs[0], [rounds_by_game(d) for d in ss]), (axs[1], [f(d) for d in ss])):
            rows = np.asarray(rows, float)
            if key == "baseline":
                plot_ci_line(ax, G, rows, c, mk, lab, offset=off)
            else:  # the condition changes at game 6: don't draw a line across the switch
                plot_ci_line(ax, G[:5], rows[:, :5], c, mk, lab, offset=off)
                plot_ci_line(ax, G[5:], rows[:, 5:], c, mk, None, offset=off)
    for ax in axs:
        ax.axvspan(5.5, 10.5, color="#f3f2ee", zorder=0, lw=0)
        ax.set(xticks=G, xlabel="Game in session", xlim=(0.5, 10.5))
    axs[0].text(5.6, 15.2, "after the switch", color=INK2, fontsize=7, va="top")
    axs[0].set(ylabel="Rounds to consensus", ylim=(0, 15.5), title="Convergence time (GLM-5.2)")
    axs[1].set(ylabel="Share on holder's item, round 2", ylim=(-0.03, 1.03), title="Following the current holder")
    panel(axs[0], "a"); panel(axs[1], "b")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.17))
    save(fig, "fig5_swap_relabel")


# ---------------- Figure 6: one-shot Schelling matrix ----------------

def fig6():
    ms = [m for m in MODELS if m != "qwen3.8-27b"]
    picks_ = {m: defaultdict(list) for m in ms}
    for m in ms:
        for r in jl(R / "schelling1" / m / "calls.jsonl"):
            picks_[m][r["board_seed"]].append(r["pick"])
    boards = sorted(set.intersection(*[set(picks_[m]) for m in ms]))

    def dist(m, b):
        ps = picks_[m][b]; c = Counter(x for x in ps if x is not None)
        return {k: v / len(ps) for k, v in c.items()}

    def within(m, b):
        ps = picks_[m][b]; n = len(ps); c = Counter(x for x in ps if x is not None)
        return sum(k * (k - 1) for k in c.values()) / (n * (n - 1))

    def cross(a, b_, bd):
        da, db = dist(a, bd), dist(b_, bd)
        return sum(v * db.get(k, 0) for k, v in da.items())

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axs[0]
    order = sorted(ms, key=lambda m: np.mean([within(m, b) for b in boards]))
    for i, m in enumerate(order):
        mu, lo, hi = boot_ci([within(m, b) for b in boards])
        ax.errorbar(mu, i, xerr=[[mu - lo], [hi - mu]], fmt=MARK[m], color=COLOR[m], ms=6, lw=1.4, capsize=0,
                    mec="white", mew=0.6)
        ax.text(hi + 0.02, i, f"{mu:.2f}", va="center", fontsize=7, color=INK2)
    ax.axvline(0.05, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax.text(0.06, -0.6, "chance (1/20)", color=INK2, fontsize=7, va="center")
    ax.set_yticks(range(len(order)), [LABEL[m] for m in order])
    ax.set(xlim=(0, 0.85), ylim=(-0.9, len(order) - 0.4), xlabel="P(two independent copies pick the same item)",
           title="Blind coordination with one's copy")
    ax.grid(axis="y", visible=False)
    panel(ax, "a")
    ax = axs[1]
    order = order[::-1]  # highest copy-match first, matching panel a top-to-bottom
    M = np.array([[np.mean([within(a, b) if a == b_ else cross(a, b_, b) for b in boards]) for b_ in order] for a in order])
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=0.7)
    for i in range(len(order)):
        for j in range(len(order)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if M[i, j] > 0.42 else INK)
    short = [LABEL[m].replace("DeepSeek-V4-Flash", "DeepSeek").replace("GLM-5.3-Flash", "GLM-5.3F")
             .replace("Gemma-4-26B-A4B", "Gemma-26B").replace("Gemma-4-31B", "Gemma-31B") for m in order]
    ax.set_xticks(range(len(order)), short, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(order)), short, fontsize=7)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Cross-model match probability")
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("P(same item)", fontsize=8); cb.ax.tick_params(labelsize=7); cb.outline.set_visible(False)
    panel(ax, "b")
    save(fig, "fig6_schelling")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()
