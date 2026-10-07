"""Charon-style weight transforms for torch layers: the layer uses W = T(raw).

Each transform has a right_inverse, so registering it keeps the layer's
initial effective weights exactly as torch drew them: the comparison starts
from the same network and differs only in how the optimizer moves.
"""

import torch
from torch import nn
from torch.nn.utils import parametrize


class Identity(nn.Module):
    name = "identity"

    def forward(self, w):
        return w

    def right_inverse(self, W):
        return W


class SignedSquare(nn.Module):
    """T(w) = w|w|: full range, T'(w) = 2|w| (a dead point at 0)."""

    name = "w|w|"

    def forward(self, w):
        return w * w.abs()

    def right_inverse(self, W):
        return W.sign() * W.abs().sqrt()


class Cube(nn.Module):
    """T(w) = w^3: full range, T'(w) = 3w^2."""

    name = "w^3"

    def forward(self, w):
        return w**3

    def right_inverse(self, W):
        return W.sign() * W.abs().pow(1 / 3)


class Rapidity(nn.Module):
    """T(w) = c tanh(w/c): every weight bounded by c (the relativistic speed limit)."""

    def __init__(self, c=0.2):
        super().__init__()
        self.c = c
        self.name = f"c*tanh(w/c), c={c:g}"

    def forward(self, w):
        return self.c * torch.tanh(w / self.c)

    def right_inverse(self, W):
        return self.c * torch.atanh((W / self.c).clamp(-1 + 1e-6, 1 - 1e-6))


class Hadamard(nn.Module):
    """T(u, v) = u^2 - v^2, two raw tensors per weight.

    right_inverse picks u^2 = max(W, 0) + a^2 and v^2 = max(-W, 0) + a^2, so
    T(u, v) = W exactly. a sets how much of each weight starts "switched on"
    in both halves; the sparsity bias appears when the initial W is small too.
    """

    def __init__(self, a=1e-3):
        super().__init__()
        self.a = a
        self.name = f"u^2-v^2, a={a:g}"

    def forward(self, u, v):
        return u * u - v * v

    def right_inverse(self, W):
        a2 = self.a**2
        return (W.clamp(min=0) + a2).sqrt(), ((-W).clamp(min=0) + a2).sqrt()


class MassEnergy(nn.Module):
    """T(w) = c^2 w: a constant rescale. SGD on it is identity SGD at lr c^4; Adam at lr c^2."""

    def __init__(self, c=7.0):
        super().__init__()
        self.c = c
        self.name = f"c^2*w, c={c:g}"

    def forward(self, w):
        return self.c**2 * w

    def right_inverse(self, W):
        return W / self.c**2


def apply(model: nn.Module, make):
    """Register make() on the weight of every nn.Linear in model. Biases stay plain."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            parametrize.register_parametrization(m, "weight", make())
    return model


TRANSFORMS = {
    "identity": Identity,
    "w|w|": SignedSquare,
    "w^3": Cube,
    "c*tanh(w/c)": lambda: Rapidity(0.2),
    "u^2-v^2": lambda: Hadamard(1e-3),
}
