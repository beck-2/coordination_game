"""Static learning-curve figures (matplotlib), same data as the viewer's charts.

    python -m coordination_game.plots                 # overnight1 -> results/figures/overnight1_*.png
    python -m coordination_game.plots --experiment overnight1 --out results/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .viewer import learning  # noqa: E402

METRICS = {"rounds": ("Rounds to converge (15 = failed)", (0, 15.5)),
           "converged": ("Share of games converged", (-0.03, 1.03)),
           "holder_r2": ("Share on bad apple's item, round 2", (-0.03, 1.03))}
COLORS = ["#2f5fd0", "#e07a1f", "#2a9d8f", "#c0392b", "#8e44ad", "#7f8c2a", "#d35d9b", "#5d6d7e"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="overnight1")
    p.add_argument("--out", default="results/figures")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    groups = [g for g in learning()["groups"] if g["experiment"] == a.experiment and g["model"] != "mixed"]
    models = sorted({g["model"] for g in groups})
    arms = sorted({g["arm"] for g in groups})
    for metric, (title, ylim) in METRICS.items():
        panels = [(arm, [g for g in groups if g["arm"] == arm and any(v is not None for v in g[metric])]) for arm in arms]
        panels = [x for x in panels if x[1]]
        if not panels:
            continue
        fig, axes = plt.subplots(1, len(panels), figsize=(max(8, 6 * len(panels)), 4.6), sharey=True, squeeze=False)
        for ax, (arm, gs) in zip(axes[0], panels):
            for g in gs:
                xs = [x for x, y in zip(g["games"], g[metric]) if y is not None]
                ys = [y for y in g[metric] if y is not None]
                se = [s for y, s in zip(g[metric], g[metric + "_se"]) if y is not None]
                ax.errorbar(xs, ys, yerr=se, marker="o", ms=4, lw=1.8, capsize=2,
                            color=COLORS[models.index(g["model"]) % len(COLORS)], label=f"{g['model']} (n={g['n_sessions']})")
            ax.set_title(arm); ax.set_xlabel("game"); ax.set_xticks(range(1, 11)); ax.set_ylim(*ylim)
            ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
        axes[0][0].set_ylabel(title)
        h, l = axes[0][-1].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=min(3 if len(panels) == 1 else 4, len(l)), frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0.16, 1, 1))
        f = out / f"{a.experiment}_{metric}.png"
        fig.savefig(f, dpi=150); plt.close(fig)
        print(f)


if __name__ == "__main__":
    main()
