"""Learned wormholes (W1, W2) and grid walls (W3).

    python -m styx.frontier       # about 30 minutes on a laptop CPU; writes results/frontier.txt
"""

import itertools
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn.utils import parametrize  # noqa: E402

from .experiments import COLORS, FIG, INK, INK2, RES, relevance, sparse_task, table  # noqa: E402
from .train import digits, mlp  # noqa: E402
from .wormholes import (RoundSTE, Walls, basis_pursuit, descend,  # noqa: E402
                        hadamard_metric, log_wormhole, meta_train, rel_err,
                        reweighted_l1, sparse_tasks)


def w1_learned():
    print("## W1. A learned wormhole on sparse recovery (40 equations, 200 unknowns)\n")
    M = meta_train(k=5, steps=400, iters=400)
    torch.save(M.state_dict(), RES / "learned_metric.pt")
    grid = list(itertools.product([1e-1, 1e-2, 1e-3, 1e-4, 1e-6], [1, 4, 16, 64]))
    rows, curves = [], {}
    for k in (5, 8, 11):
        Xs, ys, _, _, wts = sparse_tasks(24, k, torch.Generator().manual_seed(100 + k))
        X, y, _, _, wt = sparse_tasks(100, k, torch.Generator().manual_seed(200 + k))
        with torch.no_grad():
            a, c = min(grid, key=lambda p: rel_err(descend(Xs, ys, hadamard_metric(*p), 400),
                                                    wts).mean().item())
            res = {"plain GD": rel_err(descend(X, y, lambda w: torch.ones_like(w), 400), wt),
                   f"u²−v² metric, tuned (a={a:g}, c={c:g})":
                       rel_err(descend(X, y, hadamard_metric(a, c), 400), wt),
                   "learned wormhole (trained at k=5)": rel_err(descend(X, y, M, 400), wt),
                   "learned wormhole, 1600 steps": rel_err(descend(X, y, M, 1600), wt)}
        res["L1 minimization (linear program)"] = rel_err(torch.tensor(np.stack(
            [basis_pursuit(X[i].numpy(), y[i].numpy()) for i in range(len(X))])), wt)
        for name, e in res.items():
            rows.append([k, name, f"{e.median():.3f}", f"{e.mean():.3f}",
                         f"{100 * (e < 1e-2).float().mean():.0f}%"])
        curves[k] = res
    md = table(["nonzeros", "method", "median error vs true w", "mean error",
                "exact (error < 0.01)"], rows)
    ws = torch.logspace(-7, 1, 33, dtype=torch.float64)
    with torch.no_grad():
        m = M(ws)
    print("Learned m(|w|): " + ", ".join(f"{x:.0e}: {v:.3f}" for x, v in
                                         zip(ws[::4].tolist(), m[::4].tolist())) + "\n")

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.loglog(ws, m, color=COLORS[0], label="learned m(|w|)")
    ax.loglog(ws, hadamard_metric(1e-6, 4)(ws), color=COLORS[1], linestyle="--",
              label="u²−v² metric, 4|w| (capped)")
    ax.loglog(ws, torch.ones_like(ws), color=INK2, linestyle=":", label="plain GD")
    ax.set_xlabel("|w|")
    ax.set_ylabel("step multiplier m(|w|)")
    ax.set_ylim(1e-6, 10)
    ax.set_title("The wormhole gradient descent learned: a gate", loc="left", color=INK)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "learned_metric.png")
    plt.close(fig)
    return md, M


def fit_metric_sgd(task, metric, lr, init, epochs=150):
    """styx N2's network under SGD; the first layer steps by lr * metric(W) * grad."""
    tr, va, te, rel = sparse_task(task)
    model = mlp([100, 64, 1], task, init)
    first = model[0].weight
    bce = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(task)
    for _ in range(epochs):
        perm = torch.randperm(len(tr[0]), generator=g)
        for i in range(0, len(perm), 32):
            j = perm[i:i + 32]
            model.zero_grad()
            bce(model(tr[0][j])[:, 0], tr[1][j].float()).backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= lr * (metric(p) if p is first else 1.0) * p.grad
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                return None
    with torch.no_grad():
        acc = lambda X, Y: float(((model(X)[:, 0] > 0).long() == Y).float().mean())  # noqa: E731
        return acc(*va), acc(*te), relevance(model, rel)


def w2_network(M):
    print("## W2. The wormholes in a network: N2's task under plain SGD, first layer only\n")
    G = M.float()
    metrics = {"identity": lambda s: (lambda w: 1.0),
               "u²−v² metric, 4|w|/s": lambda s: (lambda w: (4 * w.abs() / s).clamp(max=4)),
               "learned wormhole, m(|w|/s)": lambda s: (lambda w: G(w / s))}
    rows, base = [], None
    for name, make in metrics.items():
        accs, rels = [], []
        scales = [1.0] if name == "identity" else [0.03, 0.1, 0.3]
        for task in range(10):
            best = None  # lr, scale s and init picked on the validation set
            for lr, s, init in itertools.product([0.03, 0.1, 0.3, 1.0], scales, [1.0, 0.1]):
                r = fit_metric_sgd(task, make(s), lr, init)
                if r and (best is None or r[0] > best[0]):
                    best = r
            accs.append(best[1])
            rels.append(best[2])
        accs = np.array(accs)
        base = accs if base is None else base
        diff = 100 * (accs - base)
        rows.append([name, f"{100 * accs.mean():.1f} ± {100 * accs.std():.1f}",
                     "baseline" if name == "identity" else
                     f"{diff.mean():+.1f} (better on {(diff > 0).sum()}/10)",
                     f"{100 * np.mean(rels):.0f}%"])
    return table(["first-layer step", "test accuracy %", "vs identity, same task",
                  "first-layer weight on the 5 real inputs"], rows)


def w3_walls():
    print("## W3. Grid walls: T' falls to eps at multiples of D (digits, Adam, 3 seeds)\n")
    Xtr, ytr, Xte, yte = digits()

    def train(seed, make=None):
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                              nn.Linear(128, 10))
        if make:
            for m in model:
                if isinstance(m, nn.Linear):
                    parametrize.register_parametrization(m, "weight", make())
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        g = torch.Generator().manual_seed(seed)
        for _ in range(60):
            perm = torch.randperm(len(Xtr), generator=g)
            for i in range(0, len(perm), 64):
                j = perm[i:i + 64]
                opt.zero_grad()
                nn.functional.cross_entropy(model(Xtr[j]), ytr[j]).backward()
                opt.step()
        return model

    def evaluate(model, D):
        acc = lambda: float((model(Xte).argmax(1) == yte).float().mean())  # noqa: E731
        with torch.no_grad():
            before = acc()
            Ws = []
            for m in model:
                if isinstance(m, nn.Linear):
                    W = m.weight.detach().clone()
                    if parametrize.is_parametrized(m):
                        parametrize.remove_parametrizations(m, "weight")
                    m.weight.copy_(torch.round(W / D) * D)
                    Ws.append(W.flatten())
            W = torch.cat(Ws)
            dist = float((W / D - torch.round(W / D)).abs().mean())
            return before, acc(), dist, torch.unique(torch.round(W / D)).numel()

    rows = []
    for D in (0.2, 0.35):
        for name, make in [("plain, rounded after training", None),
                           ("STE (quantization-aware)", lambda D=D: RoundSTE(D)),
                           ("walls, eps 0.03", lambda D=D: Walls(D, 0.03)),
                           ("walls, eps 0.3", lambda D=D: Walls(D, 0.3))]:
            r = np.array([evaluate(train(s, make), D) for s in range(3)])
            rows.append([f"{D:g}", name, f"{100 * r[:, 0].mean():.1f}",
                         f"{100 * r[:, 1].mean():.1f} (worst {100 * r[:, 1].min():.1f})",
                         f"{r[:, 2].mean():.3f}", f"{r[:, 3].mean():.0f}"])
    return table(["grid step D", "training", "test acc % before rounding",
                  "test acc % rounded to the grid", "mean distance to grid (0.25 = none)",
                  "grid levels used"], rows)


def w4_log_wormhole():
    print("## W4. The learned gate as a formula: m = f|w|^q + 4w²/(w² + tau²)\n")
    grid = list(itertools.product([1e-4, 1e-3, 1e-2], [0.5, 1.0], [1e-3, 3e-3, 1e-2, 3e-2]))
    rows = []
    for k in (5, 8, 11):
        Xs, ys, _, _, wts = sparse_tasks(24, k, torch.Generator().manual_seed(100 + k))
        X, y, _, _, wt = sparse_tasks(100, k, torch.Generator().manual_seed(200 + k))
        with torch.no_grad():
            for steps in (400, 3000):
                p = min(grid, key=lambda p: rel_err(descend(Xs, ys, log_wormhole(*p), steps),
                                                    wts).median().item())
                e = rel_err(descend(X, y, log_wormhole(*p), steps), wt)
                rows.append([k, f"log wormhole, {steps} steps", f"f={p[0]:g} q={p[1]:g} tau={p[2]:g}",
                             f"{e.median():.4f}", f"{e.mean():.3f}", f"{100 * (e < 1e-2).float().mean():.0f}%"])
        for name, solve in [("L1 minimization (linear program)", basis_pursuit),
                            ("reweighted L1 (5 reweights, eps 0.1)", reweighted_l1)]:
            e = rel_err(torch.tensor(np.stack(
                [solve(X[i].numpy(), y[i].numpy()) for i in range(len(X))])), wt)
            rows.append([k, name, "", f"{e.median():.4f}", f"{e.mean():.3f}",
                         f"{100 * (e < 1e-2).float().mean():.0f}%"])
    return table(["nonzeros", "method", "parameters (picked on 24 other problems)",
                  "median error vs true w", "mean error", "exact (error < 0.01)"], rows)


SECTIONS = ("w1", "w2", "w3", "w4")


def main(argv=None):
    """python -m styx.frontier [w1 w2 w3 w4]; w2 needs w1's metric, so it runs w1 first."""
    import sys
    want = set(argv if argv is not None else sys.argv[1:]) or set(SECTIONS)
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    path = RES / "frontier.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    if want & {"w1", "w2"}:
        out["w1"], M = w1_learned()
        if "w2" in want:
            out["w2"] = w2_network(M)
    if "w3" in want:
        out["w3"] = w3_walls()
    if "w4" in want:
        out["w4"] = w4_log_wormhole()
    path.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
