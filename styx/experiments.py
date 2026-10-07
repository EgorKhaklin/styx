"""Run the experiments, print the tables, write figures/ and results/.

    python -m styx.experiments
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from .train import digits, run  # noqa: E402
from .transforms import TRANSFORMS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG, RES = ROOT / "figures", ROOT / "results"

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "font.size": 10, "axes.titlesize": 11,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "legend.frameon": False,
    "legend.labelcolor": INK2, "figure.dpi": 140,
})

SEEDS = (0, 1, 2)
LRS = {"sgd": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0],
       "adam": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]}


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    out = "\n".join(lines)
    print(out + "\n")
    return out


def n1_digits():
    print("## N1. Digits (64-128-128-10 MLP, 40 epochs, 3 seeds per cell)\n")
    data = digits()
    sizes = [64, 128, 128, 10]
    acc = {}  # (opt, name) -> array (lrs, seeds), nan = diverged
    for opt, lrs in LRS.items():
        for name, make in TRANSFORMS.items():
            a = np.full((len(lrs), len(SEEDS)), np.nan)
            for i, lr in enumerate(lrs):
                for j, s in enumerate(SEEDS):
                    r, _ = run(make, data, sizes, opt, lr, s)
                    if not r.diverged:
                        a[i, j] = r.test_acc[-1]
            acc[(opt, name)] = a

    rows = []
    for opt, lrs in LRS.items():
        for name in TRANSFORMS:
            a = acc[(opt, name)]
            mean = np.where(np.isnan(a).any(1), 0.0, np.nan_to_num(a).mean(1))
            i = int(np.argmax(mean))
            good = [lrs[k] for k in range(len(lrs)) if mean[k] >= 0.95]
            window = f"{min(good):g} to {max(good):g}" if good else "none"
            rows.append([opt.upper(), name, f"{lrs[i]:g}",
                         f"{100 * mean[i]:.1f} ± {100 * a[i].std():.1f}", window,
                         int(np.isnan(a).any(1).sum())])
    md = table(["optimizer", "transform", "best lr", "test accuracy % at best lr",
                "lrs reaching 95%", "lrs with a diverged seed"], rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (opt, lrs) in zip(axes, LRS.items()):
        for c, name in zip(COLORS, TRANSFORMS):
            a = acc[(opt, name)]
            mean = np.where(np.isnan(a).any(1), np.nan, np.nan_to_num(a).mean(1))
            ax.plot(lrs, 100 * mean, color=c, marker="o", markersize=4, label=name)
        ax.set_xscale("log")
        ax.set_xlabel("learning rate")
        ax.set_title(opt.upper(), loc="left")
    axes[0].set_ylabel("test accuracy % (gaps: a seed diverged)")
    axes[0].set_ylim(0, 100)
    axes[1].legend(loc="lower left", fontsize=8)
    fig.suptitle("Digits: accuracy by learning rate", x=0.02, ha="left", color=INK, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "digits_lr.png")
    plt.close(fig)
    return md


def sparse_task(seed=0, n_train=200, n_val=1000, n_test=2000, d=100, k=5):
    """Binary labels from a random 5-input ReLU teacher; the other 95 inputs are noise."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_train + n_val + n_test, d, generator=g)
    rel = torch.randperm(d, generator=g)[:k]
    W1, W2 = torch.randn(k, 16, generator=g), torch.randn(16, 1, generator=g)
    h = torch.relu(X[:, rel] @ W1)
    z = (h - h.mean(0)) @ W2
    y = (z[:, 0] > z[:, 0].median()).long()
    a, b = n_train, n_train + n_val
    return (X[:a], y[:a]), (X[a:b], y[a:b]), (X[b:], y[b:]), rel


def relevance(model, rel):
    W = next(m for m in model.modules() if isinstance(m, nn.Linear)).weight.detach()
    col = (W**2).sum(0)
    return float(col[rel].sum() / col.sum())


def n2_sparse():
    print("## N2. Few samples, few relevant inputs (200 train, 5 of 100 inputs matter, 10 tasks)\n")
    sizes = [100, 64, 1]
    bce = nn.BCEWithLogitsLoss()
    loss_fn = lambda out, y: bce(out[:, 0], y.float())  # noqa: E731
    lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    rows, out = [], {}
    for init in (1.0, 0.1):
        for name, make in TRANSFORMS.items():
            accs, rels, chosen = [], [], []
            for task in range(10):
                tr, va, te, rel = sparse_task(task)
                best = None
                for lr in lrs:  # pick lr on the validation set, report the test set
                    r, m = run(make, (tr[0], tr[1], va[0], va[1]), sizes, "adam", lr, task,
                               epochs=150, init_scale=init, loss_fn=loss_fn)
                    if not r.diverged and (best is None or r.test_acc[-1] > best[0]):
                        best = (r.test_acc[-1], lr, m)
                _, lr, m = best
                m.eval()
                with torch.no_grad():
                    accs.append(float(((m(te[0])[:, 0] > 0).long() == te[1]).float().mean()))
                rels.append(relevance(m, rel))
                chosen.append(lr)
            out[(init, name)] = (accs, rels)
            diff = 100 * (np.array(accs) - np.array(out[(init, "identity")][0]))
            rows.append([f"{init:g}x", name, f"{100 * np.mean(accs):.1f} ± {100 * np.std(accs):.1f}",
                         "baseline" if name == "identity" else
                         f"{diff.mean():+.1f} (better on {(diff > 0).sum()}/10)",
                         f"{100 * np.mean(rels):.0f}%", ", ".join(f"{c:g}" for c in chosen)])
    md = table(["init scale", "transform", "test accuracy %", "vs identity, same task",
                "first-layer weight on the 5 real inputs",
                "lr picked (per task)"], rows)
    print("Chance level for the weight share is 5%.\n")

    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(TRANSFORMS)
    xs = np.arange(len(names))
    for off, init, alpha in ((-0.18, 1.0, 0.45), (0.18, 0.1, 1.0)):
        ax.bar(xs + off, [100 * np.mean(out[(init, n)][1]) for n in names], width=0.34,
               color=COLORS[0], alpha=alpha, label=f"init scale {init:g}x")
    ax.axhline(5, color=INK2, linestyle="--", linewidth=1)
    ax.annotate("chance (5 of 100 inputs)", (len(names) - 0.5, 5), color=INK2, fontsize=8,
                ha="right", va="bottom")
    ax.set_xticks(xs, names, fontsize=8)
    ax.set_ylabel("% of first-layer weight on real inputs")
    ax.set_title("Which transforms find the inputs that matter", loc="left")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "sparse_relevance.png")
    plt.close(fig)
    return md


def main():
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    out = {"n1": n1_digits(), "n2": n2_sparse()}
    (RES / "summary.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
