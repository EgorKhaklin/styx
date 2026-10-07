import pytest
import torch
from torch import nn

from styx.train import digits, fit, mlp
from styx.transforms import TRANSFORMS, MassEnergy, apply

SIZES = [64, 32, 10]


def weights(model):
    return [m.weight.detach().clone() for m in model.modules() if isinstance(m, nn.Linear)]


@pytest.mark.parametrize("name", TRANSFORMS)
def test_registration_keeps_the_initial_network(name):
    """right_inverse makes T(raw) equal torch's own init: same starting network."""
    plain = weights(mlp(SIZES, seed=3))
    wrapped = weights(apply(mlp(SIZES, seed=3), TRANSFORMS[name]))
    for a, b in zip(plain, wrapped):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", TRANSFORMS)
def test_gradient_is_chain_rule(name):
    """autograd's raw-parameter gradient equals dL/dW * T'(raw), checked numerically."""
    torch.manual_seed(0)
    lin = apply(nn.Sequential(nn.Linear(5, 3)), TRANSFORMS[name])[0]
    x, y = torch.randn(7, 5), torch.randn(7, 3)
    params = [p for n, p in lin.named_parameters() if "original" in n]
    loss = ((lin(x) - y) ** 2).mean()
    grads = torch.autograd.grad(loss, params)
    h = 1e-3
    for p, g in zip(params, grads):
        for idx in [(0, 0), (1, 2), (2, 4)]:
            with torch.no_grad():
                p[idx] += h
                up = ((lin(x) - y) ** 2).mean().item()
                p[idx] -= 2 * h
                down = ((lin(x) - y) ** 2).mean().item()
                p[idx] += h
            assert g[idx].item() == pytest.approx((up - down) / (2 * h), rel=2e-2, abs=1e-4)


def _trajectory(make, opt, lr, opt_kw=None):
    Xtr, ytr, Xte, yte = digits()
    model64 = apply(mlp(SIZES, seed=1).double(), make)
    for m in model64.modules():  # biases are not transformed; freeze them so only W differs
        if isinstance(m, nn.Linear):
            m.bias.requires_grad_(False)
    fit(model64, (Xtr[:256].double(), ytr[:256], Xte.double(), yte), opt, lr, epochs=4,
        batch=64, seed=0, opt_kw=opt_kw)
    return weights(model64)


def test_mass_energy_under_sgd_is_identity_at_lr_c4():
    c = 3.0
    a = _trajectory(lambda: MassEnergy(c), "sgd", 1e-3 / c**4)
    b = _trajectory(TRANSFORMS["identity"], "sgd", 1e-3)
    for wa, wb in zip(a, b):
        torch.testing.assert_close(wa, wb, rtol=1e-9, atol=1e-11)


def test_mass_energy_under_adam_is_identity_at_lr_c2():
    """Adam divides out the c^2 in the gradient but not the c^2 in the parameter.
    Exact once Adam's eps is scaled with the gradient (by c^2)."""
    c = 3.0
    a = _trajectory(lambda: MassEnergy(c), "adam", 1e-3 / c**2, {"eps": 1e-8 * c**2})
    b = _trajectory(TRANSFORMS["identity"], "adam", 1e-3, {"eps": 1e-8})
    for wa, wb in zip(a, b):
        torch.testing.assert_close(wa, wb, rtol=1e-6, atol=1e-8)
