"""Comparison figures for the README, drawn from the saved tables in results/ (no rerun).

    python -m styx.figures
"""

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .experiments import COLORS, FIG, INK, INK2, RES  # noqa: E402

LIGHT = "#a8a6a1"


def tables(path):
    out, cur = [], None
    for line in (RES / path).read_text().splitlines():
        if line.startswith("|") and not line.startswith("|---"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if cur is None:
                cur = (cells, [])
            else:
                extra = len(cells) - len(cur[0])  # a "|" inside a name, like w|w|, in column 1
                cur[1].append(["|".join(cells[:extra + 1])] + cells[extra + 1:] if extra > 0
                              else cells)
        elif cur is not None and not line.startswith("|---"):
            out.append(cur)
            cur = None
    if cur is not None:
        out.append(cur)
    return out


def num(cell):
    return float(re.search(r"[-\d.]+", cell).group())


def bars(ax, groups, series, colors, fmt="{:.0f}"):
    x = np.arange(len(groups))
    w = 0.8 / len(series)
    for i, ((label, vals), color) in enumerate(zip(series.items(), colors)):
        bs = ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, width=w - 0.03, color=color,
                    label=label)
        for b, v in zip(bs, vals):
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 2),
                        textcoords="offset points", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(x, groups)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)


def wormholes_vs_l1():
    w1 = tables("frontier.txt")[0][1]
    w4 = tables("frontier_w4.txt")[0][1]
    ks = ["5", "8", "11"]

    def pick(rows, k, prefix, col):
        return num(next(r for r in rows if r[0] == k and r[1].startswith(prefix))[col])

    series = {
        "u²−v² metric, tuned (400 steps)": [pick(w1, k, "u²−v²", 4) for k in ks],
        "learned wormhole (1600 steps)": [pick(w1, k, "learned wormhole, 1600", 4) for k in ks],
        "log wormhole formula (3000 steps)": [pick(w4, k, "log wormhole, 3000", 5) for k in ks],
        "L1 minimization": [pick(w4, k, "L1 minimization", 5) for k in ks],
        "reweighted L1": [pick(w4, k, "reweighted L1", 5) for k in ks],
    }
    fig, ax = plt.subplots(figsize=(9, 4.4))
    bars(ax, [f"{k} nonzeros" for k in ks], series,
         [COLORS[3], COLORS[4], COLORS[2], INK2, LIGHT])
    ax.set_ylabel("exact recoveries, % of 100 problems")
    ax.set_ylim(0, 112)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Learned wormholes close in on L1 on easy problems, not on hard ones",
                 loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "wormholes_vs_l1.png")
    plt.close(fig)


def kosmos():
    rows = tables("frontier_w5.txt")[0][1]
    steps = sorted({num(r[0]) for r in rows})
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for lat, color, label in (("per-entry", INK2, "per-entry grid (Z⁸)"),
                              ("E8", COLORS[0], "E8 lattice, blocks of 8")):
        y = [num(next(r for r in rows if num(r[0]) == s and r[1].startswith(lat))[2])
             for s in steps]
        ax.plot(steps, y, "-o", color=color, lw=2, ms=5, label=label)
        for s, v in zip(steps, y):
            ax.annotate(f"{v:.1f}", (s, v), xytext=(0, 7 if lat == "E8" else -13),
                        textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax.set_xlabel("grid step (both grids have the same number of points per volume)")
    ax.set_ylabel("test accuracy % after rounding")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title("Rounding onto E8 keeps far more accuracy at coarse steps, no retraining",
                 loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "kosmos.png")
    plt.close(fig)


def network():
    rows = tables("frontier.txt")[1][1]
    names = [r[0] for r in rows]
    means = [num(r[1]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bs = ax.bar(range(len(names)), means, color=[INK2, COLORS[3], COLORS[4]], width=0.6)
    for b, r in zip(bs, rows):
        ax.annotate(r[1], (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(names)), ["plain SGD", "u²−v² metric", "learned wormhole"])
    ax.set_ylim(60, 88)
    ax.set_ylabel("test accuracy %, 10 tasks")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax.set_title("First-layer metric under SGD (5 of 100 inputs matter)", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "network.png")
    plt.close(fig)


def main():
    wormholes_vs_l1()
    kosmos()
    network()


if __name__ == "__main__":
    main()
