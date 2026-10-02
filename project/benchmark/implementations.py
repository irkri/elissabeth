"""The LISS implementations the benchmarks compare.

An implementation is one way of evaluating the same LISS layer: the same
config and the same weights must give the same output up to rounding,
which ``benchmark.py`` checks against a float64 evaluation of every cell.
The entries run the PyTorch path of ``elissabeth.liss`` and differ in how
it is executed, except ``triton``, the library's own fused Triton scans
(``LISSConfig.scan``). A new kernel becomes one more entry in
:data:`IMPLEMENTATIONS`, and every benchmark and report picks it up by
name.

An entry may

- ``configure`` the layer's config before it is built (where a future
  ``LISSConfig`` backend switch goes),
- ``patch`` the library while a cell runs (a context manager, e.g. which
  scan the level calls), active while the layer is built, compiled and
  timed,
- ``compile`` the layer with ``torch.compile(fullgraph=True,
  dynamic=False)``, the static shapes of a training run,

and says in ``supports`` which configs it can run (``None`` when it can,
otherwise the reason it cannot).
"""
import contextlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from elissabeth.liss import LISSConfig, semiring
from elissabeth.liss.layer import TRITON_SEMIRINGS


@contextlib.contextmanager
def _no_patch() -> Iterator[None]:
    yield


def _always(config: LISSConfig) -> str | None:
    return None


@dataclass(frozen=True)
class Implementation:

    name: str
    description: str
    compile: bool = False
    patch: Callable[[], contextlib.AbstractContextManager] = _no_patch
    configure: Callable[[LISSConfig], LISSConfig] = field(
        default=lambda config: config,
    )
    supports: Callable[[LISSConfig], str | None] = _always


@contextlib.contextmanager
def _native_scans() -> Iterator[None]:
    """Call the ATen scans directly instead of the ``elissabeth::cumulate``
    custom op, so Inductor lowers (and may fuse) them itself."""
    original = semiring._cumulate
    semiring._cumulate = lambda x, s: semiring._cumulate_eager(x, s)
    try:
        yield
    finally:
        semiring._cumulate = original


@contextlib.contextmanager
def _hillis_scans() -> Iterator[None]:
    """Take the Hillis-Steele path for every decayed real/bayesian scan,
    the one the library switches to once ``max_rate * (T-1) > 40``."""
    original = semiring.EXP_TRICK_LIMIT
    semiring.EXP_TRICK_LIMIT = -float("inf")
    try:
        yield
    finally:
        semiring.EXP_TRICK_LIMIT = original


def _has_hillis_path(config: LISSConfig) -> str | None:
    if config.semiring in semiring.LOG_DOMAIN:
        return "the log domain has no Hillis path (its decay is an offset)"
    if not any(k.type == "decay" for k in config.kernels):
        return "no decay kernel, so no decayed scan"
    return None


def _triton(config: LISSConfig) -> LISSConfig:
    return config.model_copy(update={"scan": "triton"})


def _has_triton_kernels(config: LISSConfig) -> str | None:
    if config.semiring not in TRITON_SEMIRINGS:
        return f"no Triton kernels for the {config.semiring!r} semiring"
    return None


IMPLEMENTATIONS: dict[str, Implementation] = {
    impl.name: impl for impl in [
        Implementation(
            "eager",
            "the PyTorch path, uncompiled; the reference every speed-up is"
            " measured against",
        ),
        Implementation(
            "compiled",
            "the PyTorch path under torch.compile(fullgraph=True,"
            " dynamic=False), scans through the elissabeth::cumulate custom"
            " op (what train.py runs)",
            compile=True,
        ),
        Implementation(
            "native",
            "compiled, but with the ATen scans instead of the custom op, so"
            " Inductor generates the scans itself (torch 2.12 crashes on"
            " long split scans, which is why the custom op exists)",
            compile=True,
            patch=_native_scans,
        ),
        Implementation(
            "hillis",
            "compiled, with every decayed real/bayesian scan on the"
            " Hillis-Steele path (O(T log T)), which the library takes"
            " automatically once max_rate * (T-1) > 40",
            compile=True,
            patch=_hillis_scans,
            supports=_has_hillis_path,
        ),
        Implementation(
            "triton",
            "compiled, with scan: triton -- one fused Triton kernel per pair"
            " (key factor, decayed scan, query contraction), which never"
            " stores the R-wide state",
            compile=True,
            configure=_triton,
            supports=_has_triton_kernels,
        ),
    ]
}
