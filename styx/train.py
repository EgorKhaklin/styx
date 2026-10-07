"""Data, models and one training run."""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.datasets import load_digits
from torch import nn

from .transforms import apply

torch.set_num_threads(1)  # many small runs in parallel processes beat one wide run


def digits(seed=0):
    """sklearn's 8x8 digits (1797 images, ships with sklearn): standardized, 80/20 split."""
    X, y = load_digits(return_X_y=True)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(0.8 * len(X))
    tr, te = idx[:cut], idx[cut:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    X = (X - mu) / sd
    t = lambda a, dt: torch.tensor(a, dtype=dt)  # noqa: E731
    return (t(X[tr], torch.float32), t(y[tr], torch.long),
            t(X[te], torch.float32), t(y[te], torch.long))


def mlp(sizes, seed, init_scale=1.0):
    torch.manual_seed(seed)
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    model = nn.Sequential(*layers[:-1])
    if init_scale != 1.0:
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    m.weight.mul_(init_scale)
    return model


@dataclass
class Result:
    train_loss: list
    test_acc: list
    diverged: bool


def fit(model, data, opt="sgd", lr=0.1, epochs=40, batch=64, seed=0, loss_fn=None, opt_kw=None):
    Xtr, ytr, Xte, yte = data
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    optim = (torch.optim.SGD if opt == "sgd" else torch.optim.Adam)(model.parameters(), lr=lr,
                                                                 **(opt_kw or {}))
    g = torch.Generator().manual_seed(seed)
    tl, ta = [], []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        total = 0.0
        for i in range(0, len(Xtr), batch):
            j = perm[i:i + batch]
            optim.zero_grad()
            loss = loss_fn(model(Xtr[j]), ytr[j])
            if not torch.isfinite(loss):
                return Result(tl, ta, True)
            loss.backward()
            optim.step()
            total += loss.item() * len(j)
        tl.append(total / len(Xtr))
        ta.append(accuracy(model, Xte, yte))
    return Result(tl, ta, False)


@torch.no_grad()
def accuracy(model, X, y):
    model.eval()
    out = model(X)
    if out.shape[-1] == 1:
        return float(((out[:, 0] > 0).long() == y).float().mean())
    return float((out.argmax(-1) == y).float().mean())


def run(transform, data, sizes, opt, lr, seed, epochs=40, init_scale=1.0, loss_fn=None):
    model = apply(mlp(sizes, seed, init_scale), transform)
    return fit(model, data, opt, lr, epochs, seed=seed, loss_fn=loss_fn), model
