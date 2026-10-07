"""Learned wormholes: let gradient descent learn its own reparameterization.

Any elementwise transform W = T(theta), trained by gradient descent, moves the
effective weight by

    dW ~= -lr * T'(theta)^2 * dL/dW = -lr * m(W) * dL/dW,   m(W) = T'(T^-1(W))^2

so a transform IS a metric m(W) > 0 (charon's result, and the reparameterization /
mirror-descent equivalence). Instead of picking T by hand, Metric learns m as a
small network of log|W|, and meta_train fits it by differentiating through whole
training runs: the meta-loss is the error on held-out equations after K steps.
The true weights are never shown to the meta-learner.

Walls is a transform with T' ~ eps on a grid, so weights slow down at grid values.
"""

import numpy as np
import torch
from torch import nn

N_TRAIN, N_VAL, D = 40, 40, 200


def sparse_tasks(B, k, gen, d=D):
    """B random problems: n_train + n_val equations, d unknowns, k nonzeros of size 1 to 2."""
    X = torch.randn(B, N_TRAIN + N_VAL, d, generator=gen, dtype=torch.float64)
    w = torch.zeros(B, d, dtype=torch.float64)
    for b in range(B):
        S = torch.randperm(d, generator=gen)[:k]
        w[b, S] = ((torch.randint(0, 2, (k,), generator=gen) * 2 - 1)
                   * (1 + torch.rand(k, generator=gen))).double()
    y = torch.einsum("bnd,bd->bn", X, w)
    return X[:, :N_TRAIN], y[:, :N_TRAIN], X[:, N_TRAIN:], y[:, N_TRAIN:], w


class Metric(nn.Module):
    """m(W) in (0, cap]: a 1-32-32-1 tanh network of log|W|, the learned wormhole."""

    def __init__(self, cap=4.0):
        super().__init__()
        self.cap = cap
        self.f = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(),
                               nn.Linear(32, 1)).double()

    def forward(self, w):
        x = torch.log(w.abs() + 1e-8).unsqueeze(-1) / 5
        return self.cap * torch.sigmoid(self.f(x.to(self.f[0].weight.dtype)).squeeze(-1)).to(w.dtype)


def hadamard_metric(alpha, c, cap=4.0):
    """The metric u^2 - v^2 induces (u*v = alpha^2 / 2 conserved): c*sqrt(W^2 + alpha^2), capped."""
    return lambda w: torch.clamp(c * torch.sqrt(w**2 + alpha**2), max=cap)


def log_wormhole(f, q, tau, cap=4.0):
    """The learned gate, distilled: m(W) = f |W|^q + cap * W^2 / (W^2 + tau^2).

    Away from the small floor term, 1/m = (1 + tau^2 / W^2) / cap, so the mirror
    potential phi (phi'' = 1/m) is W^2 / (2 cap) - (tau^2 / cap) log|W|: a quadratic
    plus a log barrier, the log-sum sparsity penalty's shape. The floor lets W leave 0."""
    return lambda w: torch.clamp(f * (w.abs() + 1e-12) ** q + cap * w * w / (w * w + tau * tau),
                                 max=cap)


def descend(X, y, metric, steps):
    """W <- W - (0.25 / lambda_max) * m(W) * grad, from W = 0, batched over problems.

    With m <= 4 every step stays below the stability limit 1 / lambda_max."""
    n = X.shape[1]
    lam = torch.linalg.eigvalsh(X.transpose(1, 2) @ X / n)[:, -1:]
    w = torch.zeros(X.shape[0], X.shape[2], dtype=X.dtype)
    for _ in range(steps):
        g = torch.einsum("bnd,bn->bd", X, torch.einsum("bnd,bd->bn", X, w) - y) / n
        w = w - (0.25 / lam) * metric(w) * g
    return w


def rel_err(w, w_true):
    return torch.linalg.norm(w - w_true, dim=1) / torch.linalg.norm(w_true, dim=1)


def meta_train(k=5, steps=400, iters=400, batch=16, seed=0, log=print):
    """Fit a Metric so that `steps` of descend() predict held-out equations well."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 1)
    M = Metric()
    opt = torch.optim.Adam(M.parameters(), lr=3e-3)
    for it in range(iters):
        X, y, Xv, yv, _ = sparse_tasks(batch, k, gen)
        w = descend(X, y, M, steps)
        loss = ((torch.einsum("bnd,bd->bn", Xv, w) - yv) ** 2).mean() / (yv**2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 100 == 0:
            log(f"  meta step {it}: held-out relative error {loss.item():.4f}")
    return M


def basis_pursuit(X, y):
    """min ||w||_1 subject to X w = y, as a linear program (the L1 reference)."""
    from scipy.optimize import linprog
    d = X.shape[1]
    r = linprog(np.ones(2 * d), A_eq=np.hstack([X, -X]), b_eq=y, bounds=(0, None),
                method="highs")
    return r.x[:d] - r.x[d:]


class Walls(nn.Module):
    """T(t) = D * (t - (1 - eps) sin(2 pi t) / (2 pi)): T'(t) = D (1 - (1 - eps) cos 2 pi t),
    which falls to eps * D at every integer t, so weights slow down at multiples of D."""

    def __init__(self, D, eps):
        super().__init__()
        self.D, self.eps = D, eps
        self.name = f"walls D={D:g} eps={eps:g}"

    def forward(self, t):
        return self.D * (t - (1 - self.eps) * torch.sin(2 * np.pi * t) / (2 * np.pi))

    def right_inverse(self, W):
        x, t = W / self.D, W / self.D
        for _ in range(60):  # Newton; T is strictly increasing for eps > 0
            f = t - (1 - self.eps) * torch.sin(2 * np.pi * t) / (2 * np.pi) - x
            t = t - f / (1 - (1 - self.eps) * torch.cos(2 * np.pi * t)).clamp(min=1e-3)
        return t


class RoundSTE(nn.Module):
    """Straight-through quantization to multiples of D: the standard QAT baseline."""

    def __init__(self, D):
        super().__init__()
        self.D = D

    def forward(self, w):
        return w + (torch.round(w / self.D) * self.D - w).detach()

    def right_inverse(self, W):
        return W
