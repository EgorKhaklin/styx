# styx

### Charon's weight transforms, taken into neural networks.

[![Tests](https://github.com/EgorKhaklin/styx/actions/workflows/tests.yml/badge.svg)](https://github.com/EgorKhaklin/styx/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-MIT-2a78d6?style=flat-square)](LICENSE)

[charon](https://github.com/EgorKhaklin/charon) asked what a transform `W = T(raw)` does to
gradient descent on linear regression. The answer was a learning rate that depends on position,
plus a range: the route changes, the destination doesn't, except when many answers fit.
styx is the river Charon crosses. It runs the same transforms through real networks.

Every `nn.Linear` weight goes through `T` with torch's own
[`parametrize`](https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrize.register_parametrization.html).
Each transform has an exact inverse, so the transformed network **starts as the same network**
(tested): the only difference is how the optimizer moves.

| transform | T | what it does to a step |
|---|---|---|
| identity | `w` | the baseline |
| w\|w\| | `w·|w|` | step scales with \|w\|: small weights barely move |
| w³ | `w³` | step scales with w²: even stronger rich-get-richer |
| c·tanh(w/c) | `0.2·tanh(w/0.2)` | every weight bounded by 0.2 |
| u²−v² | `u² − v²`, two tensors | the squared factorization charon's E6 used |

## Results

Every number comes from `python -m styx.experiments` (about 3 minutes on a laptop CPU; full
output in [results/run.txt](results/run.txt)).

**N1. When data is plentiful, no transform helps.** On sklearn's 8×8 digits, a 64-128-128-10
network with a tuned learning rate reaches about 98.5% test accuracy whatever the transform:

| optimizer | transform | best lr | test accuracy % | lrs reaching 95% |
|---|---|---|---|---|
| SGD | identity | 1 | 98.7 ± 0.3 | 0.03 to 1 |
| SGD | w\|w\| | 1 | 98.3 ± 0.2 | 0.1 to 1 |
| SGD | w³ | 0.1 | 97.8 ± 0.4 | 0.1 to 0.3 |
| SGD | c·tanh(w/c) | 1 | 98.7 ± 0.5 | 0.03 to 3 |
| SGD | u²−v² | 1 | 98.2 ± 0.1 | 0.1 to 1 |
| Adam | identity | 0.01 | 98.8 ± 0.3 | 0.0001 to 0.03 |
| Adam | w\|w\| | 0.001 | 98.2 ± 0.1 | 0.0003 to 0.03 |
| Adam | w³ | 0.01 | 98.3 ± 0.2 | 0.0003 to 0.03 |
| Adam | c·tanh(w/c) | 0.03 | 98.5 ± 0.5 | 0.0001 to 0.1 |
| Adam | u²−v² | 0.001 | 98.6 ± 0.4 | 0.0003 to 0.01 |

The one difference is robustness. The bounded `tanh` weights never diverged and kept working
at learning rates where every other network failed (91% under Adam at lr 0.3, against about 10%
for the rest). Charon's prediction carries over: the power transforms need larger learning rates
under SGD, because their steps shrink with the weights.

![digits](figures/digits_lr.png)

**N2. When data is scarce and most inputs are noise, the power transforms win.** 200 training
examples, 100 inputs, labels from a small random network that reads only 5 of them. A
100-64-1 network under Adam, learning rate picked on a separate validation set, scored on 2000
test examples, repeated over 10 independent tasks:

| init scale | transform | test accuracy % | vs identity, same task | first-layer weight on the 5 real inputs |
|---|---|---|---|---|
| 1x | identity | 69.0 ± 4.4 | baseline | 14% |
| 1x | w\|w\| | 72.0 ± 5.2 | +3.0 (better on 10/10) | 21% |
| 1x | w³ | 73.2 ± 5.3 | +4.2 (better on 10/10) | 24% |
| 1x | c·tanh(w/c) | 65.8 ± 3.5 | −3.2 (better on 0/10) | 10% |
| 1x | u²−v² | 72.0 ± 5.2 | +3.0 (better on 10/10) | 23% |
| 0.1x | identity | 69.6 ± 4.9 | baseline | 20% |
| 0.1x | w\|w\| | 74.3 ± 5.2 | +4.7 (better on 10/10) | 37% |
| 0.1x | w³ | 76.1 ± 5.6 | +6.5 (better on 10/10) | 44% |
| 0.1x | c·tanh(w/c) | 66.4 ± 3.8 | −3.2 (better on 0/10) | 19% |
| 0.1x | u²−v² | 73.8 ± 5.5 | +4.2 (better on 10/10) | 38% |

Chance for the weight share is 5%. The transforms whose step shrinks with the weight
(`w|w|`, `w³`, `u²−v²`) put more of the first layer on the inputs that matter, and they beat
the plain network on every one of the 10 tasks. Starting smaller strengthens the effect, as it
did in charon. The bounded `tanh` network is worse on every task: a cap on weights is not a
preference for few of them.

![sparse](figures/sparse_relevance.png)

## What is and isn't new here

The N2 effect is a known one. Powerpropagation (Schwarz et al., NeurIPS 2021) uses the
`w|w|^(α−1)` family for sparse networks, and the `u²−v²` bias toward sparse solutions is
established theory for linear models. styx reproduces it in a small, fully tested setting
and compares the family side by side. It does not claim a new method.

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -e '.[test]'
.venv/bin/python -m pytest -q              # 12 tests
.venv/bin/python -m styx.experiments       # ~3 min; writes figures/ and results/
```

The tests check that registration keeps the initial network exactly, that autograd's gradient
on the raw parameters matches a finite difference for every transform, and that `T = c²·w` is
plain training at learning rate ×c⁴ under SGD and ×c² under Adam (charon's E5 and E7, now in
torch, exact to float64).

## Limits

Small networks, small datasets, CPU only. N1 uses 3 seeds and N2 uses 10 synthetic tasks. The
N2 margins are consistent across tasks but come from one synthetic task family. Nothing here is
a claim about large models or real data.

## License

MIT
