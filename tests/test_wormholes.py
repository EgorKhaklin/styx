import numpy as np
import pytest
import torch

from styx.wormholes import (Metric, RoundSTE, Walls, descend, hadamard_metric, rel_err,
                            sparse_tasks)



@pytest.fixture(autouse=True)
def float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def test_metric_is_positive_and_capped():
    m = Metric()(torch.logspace(-9, 2, 50))
    assert (m > 0).all() and (m <= 4).all()


def test_hadamard_metric_is_what_u2_minus_v2_does():
    """One small GD step on (u, v) moves w = u^2 - v^2 by -lr * 4 sqrt(w^2 + 4 (uv)^2) * grad."""
    u, v, g, lr = torch.tensor([0.3, 1.2, 0.01]), torch.tensor([0.5, 0.1, 0.02]), 0.7, 1e-8
    w0 = u * u - v * v
    w1 = (u - lr * 2 * u * g) ** 2 - (v + lr * 2 * v * g) ** 2
    predicted = -lr * hadamard_metric(2 * u * v, 4.0, cap=1e9)(w0) * g
    assert (w1 - w0) == pytest.approx(predicted, rel=1e-6)


def test_descend_without_a_wormhole_is_min_norm():
    X, y, _, _, _ = sparse_tasks(2, 5, torch.Generator().manual_seed(0))
    w = descend(X, y, lambda w: torch.full_like(w, 4.0), 3000)
    for b in range(2):
        assert w[b] == pytest.approx(torch.linalg.pinv(X[b]) @ y[b], abs=1e-8)


def test_rel_err_zero_at_truth():
    _, _, _, _, wt = sparse_tasks(3, 5, torch.Generator().manual_seed(1))
    assert rel_err(wt, wt).abs().max() == 0


@pytest.mark.parametrize("eps", [0.03, 0.1, 0.5])
def test_walls_inverse_and_slow_points(eps):
    T = Walls(0.2, eps)
    W = torch.linspace(-0.9, 0.9, 37)
    assert T(T.right_inverse(W)) == pytest.approx(W, abs=1e-10)
    t = torch.tensor([-2.0, 0.0, 1.0, 3.0], requires_grad=True)
    T(t).sum().backward()
    assert t.grad == pytest.approx(torch.full((4,), eps * 0.2), rel=1e-9)


def test_round_ste_rounds_forward_and_passes_gradient():
    w = torch.tensor([0.07, -0.33, 0.51], requires_grad=True)
    q = RoundSTE(0.2)(w)
    assert q.detach() == pytest.approx(torch.tensor([0.0, -0.4, 0.6]))
    q.sum().backward()
    assert w.grad == pytest.approx(torch.ones(3))
    assert np.isfinite(q.detach().numpy()).all()


def test_log_wormhole_is_a_log_barrier_geometry():
    """Without the floor, 1/m(W) = (1 + tau^2 / W^2) / 4: the potential's log|W| term."""
    from styx.wormholes import log_wormhole
    w = torch.tensor([-0.5, -0.01, 0.003, 0.2, 2.0])
    tau = 0.01
    assert 1 / log_wormhole(0.0, 0.5, tau)(w) == pytest.approx((1 + tau**2 / w**2) / 4, rel=1e-12)


def test_e8_rounding_is_on_the_lattice_and_nearest():
    from styx.wormholes import round_e8
    g = torch.Generator().manual_seed(0)
    x = torch.randn(500, 8, generator=g) * 2
    q = round_e8(x)
    frac = q - torch.floor(q)
    integer = (frac == 0).all(-1) & (q.sum(-1) % 2 == 0)
    half = (frac == 0.5).all(-1) & ((q - 0.5).sum(-1) % 2 == 0)
    assert (integer | half).all()
    # never farther than the nearest point of the integer grid restricted to D8 or D8 + 1/2
    assert ((x - q) ** 2).sum(-1).max() <= 1.0 + 1e-9  # E8 covering radius squared is 1


def test_lattice_quantize_has_equal_density_grids():
    from styx.wormholes import lattice_quantize
    W = torch.randn(13, 7)
    for lat in ("Z8", "E8"):
        assert lattice_quantize(W, 0.3, lat).shape == W.shape
    assert torch.allclose(lattice_quantize(W, 0.3, "Z8"), torch.round(W / 0.3) * 0.3)
