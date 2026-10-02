"""``scan: triton`` against the PyTorch path: the same level, the same
weights, the same output and the same gradients.

The PyTorch path is itself checked against brute force (``test_liss.py``),
so agreement in float64 here is agreement with the definition. The cases
cover what a fused scan gets wrong most easily: sequences shorter than one
chunk and lengths across chunk boundaries, features padded to a power of
two, value widths split over several programs, factors broadcast over
the batch or the heads, normalisation, matrix values, the decay rate's
gradient, and the compiled graph with a dynamic batch.
"""
import pytest
import torch

from elissabeth.liss import LISSConfig, LISSLevel, build_liss

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the Triton scans need CUDA",
)


def make_pair(semiring: str, kernels: list[dict], p: int = 3,
              d_in: int = 4, context_length: int = 8,
              **liss) -> tuple[LISSLevel, LISSLevel]:
    """Two copies of one level, on the PyTorch and on the Triton path."""
    config = LISSConfig(
        d_values=liss.pop("d_values", 3), n_is=liss.pop("n_is", 2),
        lengths=[p], semiring=semiring, kernels=kernels, **liss,
    )
    torch.manual_seed(0)
    ref = LISSLevel(config, p, d_in, context_length)
    with torch.no_grad():
        for kernel in ref.kernels:
            if hasattr(kernel, "alpha"):
                kernel.alpha.normal_()
        if ref.beta is not None:
            ref.beta.normal_()
    fused = LISSLevel(config.model_copy(update={"scan": "triton"}), p, d_in,
                      context_length)
    fused.load_state_dict(ref.state_dict())
    return ref.cuda(), fused.cuda()


def gradients(level: LISSLevel, x: torch.Tensor, r: torch.Tensor,
              ) -> dict[str, torch.Tensor]:
    level.zero_grad(set_to_none=True)
    x = x.detach().requires_grad_()
    out = level(x)
    (out * r).sum().backward()
    grads = {"out": out.detach(), "x": x.grad}
    for name, param in level.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad
    return grads


def compare(ref: LISSLevel, fused: LISSLevel, B: int, T: int,
            dtype: torch.dtype = torch.float64, rtol: float = 1e-9,
            atol: float = 1e-9) -> None:
    ref.to(dtype)
    fused.to(dtype)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(B, T, 4, generator=g).to("cuda", dtype)
    r = torch.randn(ref(x).shape, generator=g).to("cuda", dtype)
    assert fused.fused(x) and not ref.fused(x)
    want = gradients(ref, x, r)
    got = gradients(fused, x, r)
    assert want.keys() == got.keys()
    for name in want:
        torch.testing.assert_close(got[name], want[name], rtol=rtol,
                                   atol=atol, msg=name)


CASES = [
    ("reals", [{"type": "decay", "alpha_0": 3}]),
    ("reals", [{"type": "exponential", "restrict": True},
               {"type": "cosine", "d_qk": 2, "exponent": 2}]),
    ("reals", [{"type": "cosine_decay", "alpha_0": 5, "d_alpha": 2},
               {"type": "decay", "alpha_0": 1, "shared": True}]),
    ("reals", [{"type": "decay"}, {"type": "cosine", "d_qk": 3}]),
    ("reals", [{"type": "exponential", "d_qk": 3, "restrict": True}]),
    ("reals", []),
    ("arctic", [{"type": "decay", "alpha_0": 4}, {"type": "exponential"}]),
    ("arctic", [{"type": "decay", "alpha_0": 4},
                {"type": "exponential", "d_qk": 3}]),
    ("arctic", [{"type": "exponential", "d_qk": 2, "share_keys": True}]),
    ("arctic", []),
    ("log", [{"type": "decay", "alpha_0": 4},
             {"type": "exponential", "share_keys": True}]),
    ("log", [{"type": "exponential", "d_qk": 2},
             {"type": "exponential", "d_qk": 3, "share_queries": True}]),
    ("log", [{"type": "decay", "alpha_0": -2}]),
    ("log", []),
]


@pytest.mark.parametrize("semiring,kernels", CASES)
@pytest.mark.parametrize("T", [1, 2, 7, 64, 65, 150])
def test_matches_pytorch(semiring: str, kernels: list[dict], T: int) -> None:
    ref, fused = make_pair(semiring, kernels)
    compare(ref, fused, B=2, T=T)


@pytest.mark.parametrize("semiring", ["reals", "log", "arctic"])
@pytest.mark.parametrize("p", [1, 2, 4])
def test_depths(semiring: str, p: int) -> None:
    ref, fused = make_pair(semiring, [{"type": "decay", "alpha_0": 2},
                                      {"type": "exponential", "d_qk": 2}],
                           p=p)
    compare(ref, fused, B=3, T=130)


@pytest.mark.parametrize("semiring", ["reals", "log", "arctic"])
def test_matrix_values(semiring: str) -> None:
    ref, fused = make_pair(semiring, [{"type": "decay", "alpha_0": 2},
                                      {"type": "exponential"}],
                           values_2D=True)
    compare(ref, fused, B=2, T=70)


@pytest.mark.parametrize("semiring", ["reals", "log", "arctic"])
def test_wide_values_split_over_programs(semiring: str) -> None:
    """``d_v * w`` beyond one tile: the value axis is split over programs
    and ``dk``, ``dq`` are summed over them."""
    kernels = [{"type": "decay", "alpha_0": 2},
               {"type": "exponential", "d_qk": 20}]
    ref, fused = make_pair(semiring, kernels, p=2, d_values=24,
                           values_2D=True, n_is=1)
    compare(ref, fused, B=1, T=40)


@pytest.mark.parametrize("semiring,normalize", [
    ("reals", "mean"), ("reals", "sqrt"), ("reals", "learnable"),
    ("log", "mean"), ("log", "learnable"),
])
def test_normalize(semiring: str, normalize: str) -> None:
    ref, fused = make_pair(semiring, [{"type": "decay", "alpha_0": 2},
                                      {"type": "exponential"}],
                           normalize=normalize)
    compare(ref, fused, B=2, T=90)


@pytest.mark.parametrize("shared", ["share_values", "values_shared"])
@pytest.mark.parametrize("semiring", ["reals", "log", "arctic"])
def test_shared_values(semiring: str, shared: str) -> None:
    ref, fused = make_pair(
        semiring, [{"type": "exponential"}],
        share_values=shared == "share_values",
        values={"shared": shared == "values_shared"},
    )
    compare(ref, fused, B=2, T=70)


@pytest.mark.parametrize("semiring", ["reals", "log", "arctic"])
def test_long_sequences_use_longer_chunks(semiring: str) -> None:
    """Past 8192 positions the reals and log scans take 128-step chunks
    (arctic stays at 64); the float32 result stays within float32 rounding
    of the float64 one."""
    ref, fused = make_pair(semiring, [{"type": "decay", "alpha_0": 40},
                                      {"type": "exponential", "d_qk": 2}],
                           p=2, context_length=10_000)
    compare(ref, fused, B=1, T=10_000, rtol=1e-7, atol=1e-8)
    ref64 = gradients(ref.double(), *_inputs(ref, 1, 10_000))
    got32 = gradients(fused.float(), *[a.float() for a in _inputs(ref, 1,
                                                                10_000)])
    for name, want in ref64.items():
        error = (got32[name].double() - want).norm() / want.norm()
        assert error < 1e-4, (name, error.item())


def _inputs(level: LISSLevel, B: int, T: int) -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(2)
    x = torch.randn(B, T, 4, generator=g).cuda().double()
    r = torch.randn(level(x).shape, generator=g).cuda().double()
    return x, r


def test_bayesian_and_cpu_take_the_pytorch_path() -> None:
    config = LISSConfig(d_values=3, semiring="bayesian", scan="triton")
    level = LISSLevel(config, 2, 4, 8).cuda()
    assert not level.fused(torch.zeros(1, 3, 4, device="cuda"))
    level = LISSLevel(config.model_copy(update={"semiring": "reals"}),
                      2, 4, 8)
    assert not level.fused(torch.zeros(1, 3, 4))
    level(torch.randn(2, 5, 4))


@pytest.mark.parametrize("semiring,kernels", [
    ("reals", [{"type": "decay"}, {"type": "cosine", "d_qk": 3}]),
    ("arctic", [{"type": "decay"}, {"type": "exponential", "d_qk": 2}]),
    ("log", [{"type": "decay"}, {"type": "exponential", "d_qk": 2}]),
])
def test_compiles_with_a_dynamic_batch(semiring: str,
                                       kernels: list[dict]) -> None:
    """``fullgraph`` compile of a bidirectional layer through the custom op,
    one graph for two batch sizes, equal to eager."""
    torch._dynamo.reset()
    config = LISSConfig(d_values=8, n_is=4, lengths=[1, 3],
                        semiring=semiring, kernels=kernels, scan="triton")
    torch.manual_seed(0)
    layer = build_liss(config, 16, context_length=300).cuda()
    compiled = torch.compile(layer, fullgraph=True)
    for B in (3, 5):
        x = torch.randn(B, 300, 16, device="cuda")
        torch._dynamo.mark_dynamic(x, 0)
        want, got = [], []
        for run, store in ((layer, want), (compiled, got)):
            layer.zero_grad(set_to_none=True)
            xi = x.clone().requires_grad_()
            out = run(xi)
            out.square().sum().backward()
            store.extend([out.detach(), xi.grad] + [
                p.grad for p in layer.parameters()])
        for a, b in zip(got, want):
            # float32 sums with cancellation: compare whole tensors
            assert (a - b).norm() <= 1e-5 * b.norm()
