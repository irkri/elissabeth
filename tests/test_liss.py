"""The LISS recursion against brute force over all index tuples.

The first group sets weights by hand, as the thesis tests did, and pins
which projection row feeds which index and pair. The second group checks
the scans of every semiring against an O(T^p) evaluation of the same
factors, including the cases a scan gets wrong most easily: sequences
shorter than p, negative values in max-plus, matrix values, and decays
large enough to need the Hillis-Steele path.
"""
import itertools
import math

import pytest
import torch

from elissabeth import Elissabeth, ElissabethConfig, LISSConfig
from elissabeth.liss import LISSLevel


@pytest.fixture(autouse=True)
def float64():
    """Brute force and scans agree to rounding only in double precision."""
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(dtype)


def build(liss: dict, d: int = 5, context_length: int = 10) -> Elissabeth:
    """A causal one-layer model, so ``mixers.0`` is the LISS itself."""
    config = ElissabethConfig(
        d_hidden=d, n_layers=1, layer_norm=False, residual=False,
        context_length=context_length,
        liss=LISSConfig(**{"bidirectional": False, **liss}),
    )
    return Elissabeth(config, input_dim=d)


def load(model: Elissabeth, weights: dict[str, torch.Tensor]) -> None:
    state = model.state_dict()
    state["embedding.weight"] = torch.eye(5)
    state["unembedding.weight"] = torch.eye(5)
    state["mixers.0.W_H"] = torch.tensor([[1.0]])
    state["mixers.0.W_O"] = torch.eye(5).unsqueeze(1)
    for name, value in weights.items():
        assert state[name].shape == value.shape, name
        state[name] = value
    model.load_state_dict(state)


def values(v: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([vi.T for vi in v], dim=0)


def test_exponential_with_decay() -> None:
    model = build({
        "n_is": 1, "lengths": [3], "d_values": 5,
        "values": {"norm": False},
        "kernels": [{"type": "decay", "alpha_0": 0.5}, {"type": "exponential"}],
    })
    q = [torch.randn(5) for _ in range(3)]
    k = [torch.randn(5) for _ in range(3)]
    v = [torch.randn(5, 5) for _ in range(3)]
    a = torch.tensor([1.0, 3.0, 2.0])
    load(model, {
        "mixers.0.levels.0.values.transform.weight": values(v),
        "mixers.0.levels.0.kernels.0.alpha": a.unsqueeze(0),
        "mixers.0.levels.0.kernels.1.query.transform.weight": torch.stack(q),
        "mixers.0.levels.0.kernels.1.key.transform.weight": torch.stack(k),
    })
    alpha = 0.5 * torch.tanh(a)
    X = torch.randint(0, 5, (10,))
    expected = torch.zeros(10, 5)
    for t in range(10):
        for t1, t2, t3 in itertools.combinations(range(t + 1), 3):
            expected[t] += (
                v[0][X[t1]] * v[1][X[t2]] * v[2][X[t3]]
                * torch.exp(q[0][X[t2]] - k[0][X[t1]])
                * torch.exp(q[1][X[t3]] - k[1][X[t2]])
                * torch.exp(q[2][X[t]] - k[2][X[t3]])
                * torch.exp(-alpha[0] * (t2 - t1 - 1) / 10)
                * torch.exp(-alpha[1] * (t3 - t2 - 1) / 10)
                * torch.exp(-alpha[2] * (t - t3) / 10)
            )
    torch.testing.assert_close(model(X.unsqueeze(0))[0], expected)


def test_cosine_with_decay() -> None:
    model = build({
        "n_is": 1, "lengths": [3], "d_values": 5,
        "values": {"norm": False},
        "kernels": [
            {"type": "decay", "alpha_0": 2},
            {"type": "cosine", "d_qk": 3, "exponent": 2},
        ],
    })
    q = torch.randn(3, 3, 5)
    k = torch.randn(3, 3, 5)
    v = [torch.randn(5, 5) for _ in range(3)]
    a = torch.tensor([1.0, 3.0, 2.0])
    load(model, {
        "mixers.0.levels.0.values.transform.weight": values(v),
        "mixers.0.levels.0.kernels.0.alpha": a.unsqueeze(0),
        "mixers.0.levels.0.kernels.1.query.transform.weight": q.reshape(9, 5),
        "mixers.0.levels.0.kernels.1.key.transform.weight": k.reshape(9, 5),
    })
    alpha = 2 * torch.tanh(a)
    X = torch.randint(0, 5, (10,))

    def cos(l: int, later: int, earlier: int) -> torch.Tensor:
        return torch.prod(torch.cos(q[l, :, later] - k[l, :, earlier]) ** 2)

    expected = torch.zeros(10, 5)
    for t in range(10):
        for t1, t2, t3 in itertools.combinations(range(t + 1), 3):
            expected[t] += (
                v[0][X[t1]] * v[1][X[t2]] * v[2][X[t3]]
                * cos(0, X[t2], X[t1]) * cos(1, X[t3], X[t2])
                * cos(2, X[t], X[t3])
                * torch.exp(-alpha[0] * (t2 - t1 - 1) / 10)
                * torch.exp(-alpha[1] * (t3 - t2 - 1) / 10)
                * torch.exp(-alpha[2] * (t - t3) / 10)
            )
    torch.testing.assert_close(model(X.unsqueeze(0))[0], expected)


def test_cosine_decay_with_cosine() -> None:
    model = build({
        "n_is": 1, "lengths": [3], "d_values": 5,
        "values": {"norm": False},
        "kernels": [
            {"type": "cosine_decay", "alpha_0": 2, "d_alpha": 1,
             "exponent": 2},
            {"type": "cosine", "d_qk": 3, "exponent": 1},
        ],
    })
    q = torch.randn(3, 3, 5)
    k = torch.randn(3, 3, 5)
    v = [torch.randn(5, 5) for _ in range(3)]
    a = torch.tensor([1.0, 3.0, 2.0])
    load(model, {
        "mixers.0.levels.0.values.transform.weight": values(v),
        "mixers.0.levels.0.kernels.0.alpha": a.view(1, 3, 1),
        "mixers.0.levels.0.kernels.1.query.transform.weight": q.reshape(9, 5),
        "mixers.0.levels.0.kernels.1.key.transform.weight": k.reshape(9, 5),
    })
    alpha = 2 * torch.tanh(a)
    X = torch.randint(0, 5, (10,))

    def cos(l: int, later: int, earlier: int) -> torch.Tensor:
        return torch.prod(torch.cos(q[l, :, later] - k[l, :, earlier]))

    expected = torch.zeros(10, 5)
    for t in range(10):
        for t1, t2, t3 in itertools.combinations(range(t + 1), 3):
            expected[t] += (
                v[0][X[t1]] * v[1][X[t2]] * v[2][X[t3]]
                * cos(0, X[t2], X[t1]) * cos(1, X[t3], X[t2])
                * cos(2, X[t], X[t3])
                * torch.cos(alpha[0] * (t2 - t1 - 1) / 10) ** 2
                * torch.cos(alpha[1] * (t3 - t2 - 1) / 10) ** 2
                * torch.cos(alpha[2] * (t - t3) / 10) ** 2
            )
    torch.testing.assert_close(model(X.unsqueeze(0))[0], expected)


@pytest.mark.parametrize("T", [2, 6, 40])
def test_arctic(T: int) -> None:
    """Max-plus with negative values: an index tuple shorter than p must
    never win (the thesis version padded with 0 instead of -inf)."""
    model = build({
        "n_is": 1, "lengths": [3], "d_values": 5, "semiring": "arctic",
        "values": {"norm": False},
    })
    v = [torch.randn(5, 5) - 3 for _ in range(3)]
    load(model, {"mixers.0.levels.0.values.transform.weight": values(v)})
    X = torch.randint(0, 5, (T,))
    expected = torch.zeros(T, 5)
    for t in range(2, T):
        expected[t] = torch.stack([
            v[0][X[t1]] + v[1][X[t2]] + v[2][X[t3]]
            for t1, t2, t3 in itertools.combinations(range(t + 1), 3)
        ]).amax(0)
    torch.testing.assert_close(model(X.unsqueeze(0))[0], expected)


def test_arctic_decay_and_exponential() -> None:
    """In max-plus the kernels are log-factors: -alpha(gap)/T and q - k."""
    model = build({
        "n_is": 1, "lengths": [2], "d_values": 5, "semiring": "arctic",
        "values": {"norm": False},
        "kernels": [{"type": "decay", "alpha_0": 3}, {"type": "exponential"}],
    })
    q = [torch.randn(5) for _ in range(2)]
    k = [torch.randn(5) for _ in range(2)]
    v = [torch.randn(5, 5) for _ in range(2)]
    a = torch.tensor([0.7, -1.2])
    load(model, {
        "mixers.0.levels.0.values.transform.weight": values(v),
        "mixers.0.levels.0.kernels.0.alpha": a.unsqueeze(0),
        "mixers.0.levels.0.kernels.1.query.transform.weight": torch.stack(q),
        "mixers.0.levels.0.kernels.1.key.transform.weight": torch.stack(k),
    })
    alpha = 3 * torch.tanh(a)
    X = torch.randint(0, 5, (10,))
    expected = torch.zeros(10, 5)
    for t in range(1, 10):
        expected[t] = torch.stack([
            v[0][X[t1]] + v[1][X[t2]]
            + q[0][X[t2]] - k[0][X[t1]] + q[1][X[t]] - k[1][X[t2]]
            - alpha[0] * (t2 - t1 - 1) / 10 - alpha[1] * (t - t2) / 10
            for t1, t2 in itertools.combinations(range(t + 1), 2)
        ]).amax(0)
    torch.testing.assert_close(model(X.unsqueeze(0))[0], expected)


def test_bayesian() -> None:
    model = build({
        "n_is": 1, "lengths": [3], "d_values": 5, "semiring": "bayesian",
        "values": {"norm": False},
    })
    v = [torch.rand(5, 5) for _ in range(3)]
    load(model, {"mixers.0.levels.0.values.transform.weight": values(v)})
    X = torch.randint(0, 5, (10,))
    expected = torch.zeros(10, 5)
    for t in range(2, 10):
        expected[t] = torch.stack([
            v[0][X[t1]] * v[1][X[t2]] * v[2][X[t3]]
            for t1, t2, t3 in itertools.combinations(range(t + 1), 3)
        ]).amax(0)
    torch.testing.assert_close(model(X.unsqueeze(0))[0], expected)


# -- the recursion of every semiring against brute force -------------------

SEMIRINGS = {
    "reals": (lambda a, b: a + b, lambda a, b: a * b),
    "bayesian": (torch.maximum, lambda a, b: a * b),
    "arctic": (torch.maximum, lambda a, b: a + b),
    "log": (torch.logaddexp, lambda a, b: a + b),
}


def matmul(a: torch.Tensor, b: torch.Tensor, semiring: str) -> torch.Tensor:
    add, mul = SEMIRINGS[semiring]
    out = mul(a[:, 0:1], b[0:1, :])
    for k in range(1, a.shape[1]):
        out = add(out, mul(a[:, k:k + 1], b[k:k + 1, :]))
    return out


def brute_force(level: LISSLevel, x: torch.Tensor) -> torch.Tensor:
    """``(T, N, d_v, w)`` for a batch of one: the semiring sum over every
    index tuple, with the kernels evaluated densely by ``pair_matrices``."""
    add, mul = SEMIRINGS[level.semiring]
    v = level._values(x)[0]                     # (T, N|1, p|1, d_v, w)
    T = x.shape[1]
    kernel, _ = level.pair_matrices(x)
    out = torch.zeros(T, level.n_is, *v.shape[-2:])
    for t in range(level.p - 1, T):
        for n in range(level.n_is):
            total = None
            for index in itertools.combinations(range(t + 1), level.p):
                chain = index + (t,)
                term = None
                for l in range(level.p):
                    value = v[
                        chain[l],
                        n if v.shape[1] > 1 else 0,
                        l if v.shape[2] > 1 else 0,
                    ]
                    factor = mul(value, kernel[0, n, l, chain[l + 1], chain[l]])
                    if term is None:
                        term = factor
                    elif level.matrix:
                        term = matmul(term, factor, level.semiring)
                    else:
                        term = mul(term, factor)
                total = term if total is None else add(total, term)
            out[t, n] = total
    return out


def make_level(semiring: str, kernels: list[dict], **liss) -> LISSLevel:
    config = LISSConfig(
        d_values=liss.pop("d_values", 3), n_is=liss.pop("n_is", 2),
        lengths=[liss.pop("p", 3)], semiring=semiring, kernels=kernels,
        values={"norm": False}, **liss,
    )
    level = LISSLevel(config, config.lengths[0], d_in=4, context_length=6)
    with torch.no_grad():
        for kernel in level.kernels:
            if hasattr(kernel, "alpha"):
                kernel.alpha.normal_()
    return level


CASES = [
    ("reals", [{"type": "decay", "alpha_0": 3}]),
    ("reals", [{"type": "exponential", "restrict": True},
               {"type": "cosine", "d_qk": 2, "exponent": 2}]),
    ("reals", [{"type": "cosine_decay", "alpha_0": 5, "d_alpha": 2},
               {"type": "decay", "alpha_0": 1, "shared": True}]),
    ("arctic", [{"type": "decay", "alpha_0": 4},
                {"type": "exponential"}]),
    ("log", [{"type": "decay", "alpha_0": 4},
             {"type": "exponential", "share_keys": True}]),
    ("bayesian", [{"type": "decay", "alpha_0": 2}]),
    # exponentials of rank d_qk: the features are contracted with the
    # semiring's own sum, and two kernels multiply to R1*R2 features
    ("reals", [{"type": "exponential", "d_qk": 3, "restrict": True}]),
    ("arctic", [{"type": "decay", "alpha_0": 4},
                {"type": "exponential", "d_qk": 3}]),
    ("log", [{"type": "exponential", "d_qk": 2},
             {"type": "exponential", "d_qk": 3, "share_queries": True}]),
    ("bayesian", [{"type": "exponential", "d_qk": 2, "restrict": True}]),
]


@pytest.mark.parametrize("semiring,kernels", CASES)
@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("T", [2, 6])
def test_recursion(semiring: str, kernels: list[dict], matrix: bool,
                   T: int) -> None:
    level = make_level(semiring, kernels, values_2D=matrix)
    x = torch.randn(1, T, 4)
    if semiring == "bayesian":
        with torch.no_grad():
            level.values.transform.weight.abs_()
            level.values.transform.bias.fill_(0.1)
        x = x.abs()
    out = level(x)[0]
    expected = brute_force(level, x)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("shared", ["share_values", "values_shared"])
def test_shared_values(shared: str) -> None:
    """One value projection for all indices, or for all heads."""
    config = LISSConfig(
        d_values=3, n_is=2, lengths=[3],
        kernels=[{"type": "exponential"}],
        share_values=shared == "share_values",
        values={"norm": False, "shared": shared == "values_shared"},
    )
    level = LISSLevel(config, 3, d_in=4, context_length=6)
    x = torch.randn(1, 6, 4)
    torch.testing.assert_close(level(x)[0], brute_force(level, x))


def test_hillis_fallback_matches_rescaled_cumsum() -> None:
    """A decay past the exp-trick limit takes the Hillis-Steele scan; both
    compute the same sum."""
    from elissabeth.liss import semiring as sr
    u = torch.randn(2, 50, 3, 1, 4, 1)
    rate = torch.tensor([0.5, -0.02, 1.2])
    fast = sr.scan(u, "reals", rate, max_rate=0.0)
    slow = sr.scan(u, "reals", rate, max_rate=1e9)
    torch.testing.assert_close(fast, slow)
    fast = sr.scan(u.abs(), "bayesian", rate.abs(), max_rate=0.0)
    slow = sr.scan(u.abs(), "bayesian", rate.abs(), max_rate=1e9)
    torch.testing.assert_close(fast, slow)


def test_large_decay_does_not_overflow() -> None:
    level = make_level("reals", [{"type": "decay", "alpha_0": 5000}], p=2)
    with torch.no_grad():
        level.kernels[0].alpha.fill_(3.0)
    x = torch.randn(1, 6, 4)
    out = level(x)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[0], brute_force(level, x))


@pytest.mark.parametrize("normalize", ["mean", "sqrt", "learnable"])
def test_normalize(normalize: str) -> None:
    """Every level divides its partial sums by (#positions summed)^gamma."""
    level = make_level("reals", [], p=2, normalize=normalize, n_is=1)
    x = torch.randn(1, 7, 4)
    v = level._values(x)[0, :, 0, :, :, 0]      # (T, p, d_v)
    gamma = {"mean": [1.0, 1.0], "sqrt": [0.5, 0.5]}.get(
        normalize,
        [1 + math.log10(0.25 * math.tanh(5.40988) + 0.75001)] * 2,
    )
    T = x.shape[1]
    s0 = torch.stack([v[: t + 1, 0].sum(0) / (t + 1) ** gamma[0]
                      for t in range(T)])
    expected = torch.zeros(T, v.shape[-1])
    for t in range(1, T):
        expected[t] = sum(v[s, 1] * s0[s - 1] for s in range(1, t + 1)) \
            / t ** gamma[1]
    torch.testing.assert_close(level(x)[0, :, 0, :, 0], expected)


@pytest.mark.parametrize("semiring", ["arctic", "log"])
def test_log_domain_gradients_are_finite(semiring: str) -> None:
    level = make_level(semiring, [{"type": "decay", "alpha_0": 2},
                                  {"type": "exponential"}], p=3)
    x = torch.randn(3, 5, 4, requires_grad=True)
    level(x).sum().backward()
    assert torch.isfinite(x.grad).all()
    for p in level.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()


@pytest.mark.parametrize("semiring,d_qk", [
    ("arctic", 1), ("arctic", 3), ("bayesian", 2),
])
def test_decode_finds_the_argmax_tuple(semiring: str, d_qk: int) -> None:
    """The decoded tuple scores the level's output; with rank ``d_qk > 1``
    that needs the feature each contraction took, not only the scans'
    argmax."""
    level = make_level(semiring, [{"type": "decay", "alpha_0": 4},
                                  {"type": "exponential", "d_qk": d_qk}], p=3)
    x = torch.randn(1, 7, 4)
    if semiring == "bayesian":
        with torch.no_grad():
            level.values.transform.weight.abs_()
            level.values.transform.bias.fill_(0.1)
        x = x.abs()
    add, mul = SEMIRINGS[semiring]
    tuples = level.decode(x)                      # (1, T, N, d_v, p)
    v = level._values(x)[0]
    kernel, _ = level.pair_matrices(x)
    out = level(x)[0]
    for t in range(2, x.shape[1]):
        for n in range(level.n_is):
            for d in range(v.shape[-2]):
                index = tuples[0, t, n, d].tolist()
                chain = index + [t]
                assert index == sorted(set(index))
                score = None
                for l in range(level.p):
                    term = mul(v[chain[l], n, l, d, 0],
                               kernel[0, n, l, chain[l + 1], chain[l]])
                    score = term if score is None else mul(score, term)
                torch.testing.assert_close(score, out[t, n, d, 0])
    assert (tuples[0, :2] == -1).all()


@pytest.mark.parametrize("semiring", ["reals", "arctic", "log", "bayesian"])
def test_exponential_is_the_semiring_sum_over_d_qk(
    semiring: str,
) -> None:
    """``kappa_l(t', t) = (+)_d exp(q_{l,d}(x_t') - k_{l,d}(x_t))`` from the
    projections themselves, independently of the factors the recursion
    (and the brute force above) use."""
    level = make_level(semiring, [{"type": "exponential", "d_qk": 3}], p=2)
    kernel = level.kernels[0]
    x = torch.randn(1, 5, 4)
    q = kernel.query(x)[0]                      # (T, N, p, R)
    k = kernel.key(x)[0]
    score = q.unsqueeze(1) - k.unsqueeze(0)     # (T', T, N, p, R)
    expected = {
        "reals": score.exp().sum(-1),
        "arctic": score.amax(-1),
        "log": score.logsumexp(-1),
        "bayesian": score.exp().amax(-1),
    }[semiring].permute(2, 3, 0, 1)             # (N, p, T', T)
    torch.testing.assert_close(level.pair_matrices(x)[0][0], expected)


def test_cosine_kernels_are_reals_only() -> None:
    for kernel in ("cosine", "cosine_decay"):
        with pytest.raises(ValueError):
            LISSConfig(d_values=2, semiring="arctic",
                       kernels=[{"type": kernel}])
    LISSConfig(d_values=2, semiring="arctic",
               kernels=[{"type": "exponential", "d_qk": 4}])
