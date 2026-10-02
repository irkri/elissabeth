"""How do LISS levels behave numerically over long sequences?

A level of depth ``p`` combines ``C(t, p)`` index tuples at position ``t``.
In the reals the stored number therefore grows like ``t^p`` when the values
agree in sign over time and like ``t^(p/2)`` when they do not; a decay
toward the distant past grows exponentially; and the exponential kernel's
factors ``e^q``, ``e^-k`` are stored separately, so they can overflow while
the kernel ``e^(q-k)`` itself is moderate. The log domain (``log``,
``arctic``) stores logarithms and cannot overflow, but adds rounding of its
own. This script measures all of it on one LISS level with fixed weights,
evaluated in float64 (the reference) and in every format of ``--dtypes``.

Experiments (``--experiments``):

- ``growth``    semiring x normalisation x depth x input, no kernels: the
                iterated sum itself
- ``decay``     a decay toward the recent past (``alpha > 0``) or the
                distant past (``alpha < 0``), run to 128 context lengths
- ``kernel``    the exponential kernel: a common offset of queries and keys
                (the kernel is unchanged, its factors are not) and a growing
                scale of both, with and without ``restrict``
- ``gradient``  the input gradient of ``sum(y * r)``: size, finiteness and
                error, position by position
- ``primitive`` the PyTorch scans a level is built from, alone: cumsum and
                logcumsumexp with time on a non-innermost dimension (the
                layout of a level's state) and innermost, and the positions
                ``arange(T)`` a decay and a normalisation are computed from

Every run records, at log-spaced positions ``t``, the median and largest
stored magnitude, the fraction of non-finite entries, and the error against
float64: normwise relative in the reals and the bayesian semiring, absolute
in the log domain (where an absolute error of the stored logarithm is the
relative error of the number it stands for). A causal level's output at
``t`` does not depend on the length of the run, so one run to ``T`` gives
the whole curve.

``--normalize ema`` is a prototype that lives only here, not in the
library: each level is divided by its decayed count
``sum_{s=l..t} lambda^(t-s)`` instead of ``t - l + 1``, which equals
``mean`` without a decay and makes a decayed level a weighted mean at
every length.

    cd project
    python benchmark/stability.py --out benchmark/stability01.csv
    python benchmark/plot_stability.py benchmark/stability01.csv
"""
import argparse
import copy
import csv
import json
import math
import platform
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from elissabeth.liss import Decay, Exponential, LISSConfig, LISSLevel
from elissabeth.liss.semiring import LOG_DOMAIN, scan, zero

SEMIRINGS = ["reals", "log", "arctic", "bayesian"]
EXPERIMENTS = ["growth", "decay", "kernel", "gradient", "primitive"]
INPUTS = ["constant", "tokens", "drift", "gaussian", "sparse"]
DTYPES = ["float32", "bfloat16", "float16", "bf16-mixed"]
NORMALIZE = ["none", "mean", "sqrt", "ema"]

B, N, D_V, D_IN = 2, 4, 8, 16
"""Batch, heads, value width and input width of every level here: 64
output channels per position, which the statistics are taken over."""
VOCAB = 8
"""Tokens of the ``tokens`` input."""
DRIFT = 1000
"""Correlation length (positions) of the ``drift`` input."""
MARKS = 0.01
"""Fraction of non-zero positions of the ``sparse`` input."""

FIELDS = [
    "experiment", "semiring", "p", "normalize", "input", "kernels", "alpha",
    "offset", "scale", "restrict", "T", "Tc", "dtype", "quantity", "t",
    "mag_med", "mag_max", "nonfinite", "err",
]


@dataclass(frozen=True)
class Run:

    experiment: str
    semiring: str
    p: int
    T: int
    Tc: int
    normalize: str = "none"
    input: str = "gaussian"
    kernels: str = "none"
    """``none``, ``decay``, ``exp`` or ``decay+exp``."""
    alpha: float = 0.0
    """The decay rate times ``Tc``, set exactly: positive decays toward the
    recent past, negative favours the distant past (``lambda > 1``)."""
    offset: float = 0.0
    """Added to the bias of the exponential kernel's queries and keys
    alike: the kernel ``exp(q - k)`` does not change, ``exp(q)`` does."""
    scale: float = 1.0
    """Multiplies the query and key weights."""
    restrict: bool = False


class EMALevel(LISSLevel):
    """Prototype (benchmark only): normalise every level by its decayed
    count ``sum_{s=l..t} lambda^(t-s)`` instead of ``t - l + 1``. Without
    a decay this is ``normalize: mean``; with one, a level is a weighted
    mean whatever ``t`` is, as the LRU's input normalisation keeps its
    state at the scale of its input."""

    _rate: torch.Tensor | None = None

    def factors(self, x: torch.Tensor):
        rate, query, key = super().factors(x)
        self._rate = rate
        return rate, query, key

    def _normalize(self, state: torch.Tensor, l: int) -> torch.Tensor:
        T, heads = state.shape[1], self.n_is
        rate = None if self._rate is None else self._rate[:, l]
        live = torch.arange(T, device=state.device) >= l
        live = live.view(1, T, 1).expand(1, T, heads)
        shape = (1, T, heads, *([1] * (state.ndim - 3)))
        if self.semiring == "log":
            u = torch.full(live.shape, zero("log", state.dtype),
                           dtype=state.dtype, device=state.device)
            u = u.masked_fill(live, 0.0)
            count = scan(u, "log", rate).clamp_min(0.0)
            return state - count.view(shape)
        count = scan(live.to(state.dtype), "reals", rate, self.max_rate)
        return state / count.clamp_min(1.0).view(shape)


# --------------------------------------------------------------------------
# the runs of each experiment
# --------------------------------------------------------------------------
def _norms(semiring: str, allowed: list[str]) -> list[str]:
    if semiring in ("reals", "log"):
        return allowed
    return ["none"]


def growth_runs(args: argparse.Namespace) -> Iterator[Run]:
    norms = [n for n in args.normalize if n != "ema"]   # ema = mean here
    for semiring in args.semirings:
        for normalize in _norms(semiring, norms):
            for p in args.depths:
                for kind in args.inputs:
                    yield Run("growth", semiring, p, args.T, args.T,
                              normalize, kind)


def decay_runs(args: argparse.Namespace) -> Iterator[Run]:
    norms = [n for n in args.normalize if n in ("none", "mean", "ema")]
    Tc = args.decay_Tc
    T = Tc * args.decay_contexts
    # Past max_rate * (T-1) = 40 the decayed real scan takes the
    # Hillis-Steele path, so these runs measure that path at every t.
    for semiring in args.semirings:
        for normalize in _norms(semiring, norms):
            for alpha in args.alphas:
                for p in (2, 4):
                    for kind in ("constant", "gaussian"):
                        yield Run("decay", semiring, p, T, Tc, normalize,
                                  kind, "decay", alpha)


def kernel_runs(args: argparse.Namespace) -> Iterator[Run]:
    T = args.kernel_T
    for semiring in args.semirings:
        normalize = "mean" if semiring in ("reals", "log") else "none"
        base = Run("kernel", semiring, 2, T, T, normalize, "gaussian", "exp")
        for offset in (0, 20, 40, 60, 80, 100, 120):
            yield replace(base, offset=float(offset))
        for scale in (2, 4, 8, 16, 32):
            for restrict in (False, True):
                yield replace(base, scale=float(scale), restrict=restrict)


def gradient_runs(args: argparse.Namespace) -> Iterator[Run]:
    T = args.gradient_T
    for semiring in args.semirings:
        for normalize in _norms(semiring, ["none", "mean"]):
            for p in (2, 4):
                yield Run("gradient", semiring, p, T, T, normalize,
                          "gaussian", "decay+exp")


PRIMITIVES = {
    "cumsum, time not innermost": "cumsum",
    "cumsum, time innermost": "cumsum",
    "logcumsumexp, time not innermost": "logcumsumexp",
    "positions arange(T)": "arange",
}


def primitive_rows(args: argparse.Namespace, device: torch.device
                   ) -> list[dict]:
    """The scans alone, on random inputs in a level's state layout
    ``(B, T, N, 1, d_v, 1)`` and with time as the last dimension, against
    the same scan in float64; and how far each format puts the positions
    ``0..T-1`` from where they are (``err`` in positions)."""
    T = args.T
    pos = positions(T, args.per_octave)
    g = torch.Generator().manual_seed(3)
    uniform = torch.rand(B, T, N, 1, D_V, 1, generator=g, dtype=torch.float64)
    normal = torch.randn(B, T, N, 1, D_V, 1, generator=g, dtype=torch.float64)
    rows = []
    for name, op in PRIMITIVES.items():
        for dtype in ["float64"] + [d for d in args.dtypes if d != "bf16-mixed"]:
            dt = getattr(torch, dtype)
            if op == "arange":
                exact = torch.arange(T, dtype=torch.float64)
                got = torch.arange(T, device=device, dtype=dt).double().cpu()
                y, ref = got.view(T, 1), exact.view(T, 1)
                # the furthest any position up to t is off (inf past range)
                gap = (got - exact).abs()
                gap = torch.where(torch.isfinite(gap), gap, torch.inf)
                off = torch.cummax(gap, 0).values
            else:
                x = (uniform if op == "cumsum" else normal).to(device)
                inner = "not" not in name
                if inner:
                    x = x.movedim(1, -1).contiguous()
                dim = -1 if inner else 1
                f = torch.cumsum if op == "cumsum" else torch.logcumsumexp
                ref = f(x, dim)
                y = f(x.to(dt), dim).double()
                if inner:
                    y, ref = y.movedim(-1, 1), ref.movedim(-1, 1)
                y, ref = _flat(y), _flat(ref)
            log_domain = op == "logcumsumexp"
            for t in pos:
                mag_med, mag_max, nonfinite, _ = _stats(y[t], ref[t], True)
                if op == "arange":       # in positions, not relative
                    err = off[t].item()
                else:
                    err = _stats(y[t], ref[t], log_domain)[3]
                rows.append({
                    "experiment": "primitive", "semiring": name, "p": 0,
                    "normalize": "", "input": "", "kernels": "", "alpha": 0,
                    "offset": 0, "scale": 1, "restrict": False, "T": T,
                    "Tc": T, "dtype": dtype, "quantity": op, "t": t,
                    "mag_med": mag_med, "mag_max": mag_max,
                    "nonfinite": nonfinite,
                    "err": err if dtype != "float64" else 0.0,
                })
    return rows


# --------------------------------------------------------------------------
# levels and inputs
# --------------------------------------------------------------------------
def build_level(run: Run, device: torch.device) -> LISSLevel:
    kernels: list[dict] = []
    if "decay" in run.kernels:
        kernels.append({"type": "decay", "alpha_0": abs(run.alpha) or 1.0})
    if "exp" in run.kernels:
        kernels.append({"type": "exponential", "restrict": run.restrict})
    config = LISSConfig(
        d_values=D_V, n_is=N, lengths=[run.p], semiring=run.semiring,
        kernels=kernels, bidirectional=False,
        normalize="mean" if run.normalize == "ema" else run.normalize,
    )
    torch.manual_seed(0)
    cls = EMALevel if run.normalize == "ema" else LISSLevel
    level = cls(config, run.p, D_IN, run.Tc).to(device, torch.float64)
    with torch.no_grad():
        for kernel in level.kernels:
            if isinstance(kernel, Decay):
                # tanh(30) is 1 in float64: rate = alpha / Tc exactly.
                kernel.alpha.fill_(math.copysign(30.0, run.alpha)
                                   if run.alpha else 0.0)
            if isinstance(kernel, Exponential):
                for projection in (kernel.query, kernel.key):
                    linear = projection.transform
                    assert isinstance(linear, torch.nn.Linear)
                    linear.weight.mul_(run.scale)
                    linear.bias.fill_(run.offset)
    return level


def make_input(kind: str, T: int, seed: int = 1) -> torch.Tensor:
    """``(B, T, D_IN)`` in float64, LayerNorm'd per position as the
    pre-norm hands it to a mixer (a zero vector stays zero)."""
    g = torch.Generator().manual_seed(seed)
    shape = (B, T, D_IN)
    match kind:
        case "constant":
            x = torch.randn(B, 1, D_IN, generator=g, dtype=torch.float64)
            x = x.expand(shape)
        case "tokens":
            table = torch.randn(VOCAB, D_IN, generator=g, dtype=torch.float64)
            x = table[torch.randint(0, VOCAB, (B, T), generator=g)]
        case "drift":
            # AR(1) with correlation length DRIFT: a slowly turning vector.
            rho = math.exp(-1 / DRIFT)
            eps = torch.randn(shape, generator=g, dtype=torch.float64)
            eps[:, 0] /= math.sqrt(1 - rho**2)
            t = torch.arange(T, dtype=torch.float64).view(1, -1, 1)
            # x_t = sum_s rho^(t-s) eps_s, by the rescaled cumsum in chunks
            # of 100 correlation lengths so rho^-s stays finite.
            x = torch.empty(shape, dtype=torch.float64)
            carry = torch.zeros(B, 1, D_IN, dtype=torch.float64)
            for start in range(0, T, 100 * DRIFT):
                stop = min(T, start + 100 * DRIFT)
                tt = t[:, : stop - start]
                part = torch.cumsum(eps[:, start:stop] * rho ** -tt, 1)
                part = part * rho**tt + carry * rho ** (tt + 1)
                x[:, start:stop] = part
                carry = part[:, -1:]
            x = x * math.sqrt(1 - rho**2)
        case "gaussian":
            x = torch.randn(shape, generator=g, dtype=torch.float64)
        case "sparse":
            x = torch.randn(shape, generator=g, dtype=torch.float64)
            x = x * (torch.rand(B, T, 1, generator=g) < MARKS)
        case _:
            raise ValueError(f"Unknown input {kind!r}.")
    return F.layer_norm(x.contiguous(), (D_IN,))


def positions(T: int, per_octave: int) -> list[int]:
    """Log-spaced positions ``t`` (0-based) up to ``T - 1``."""
    steps = int(math.log2(T) * per_octave)
    pos = {round(2 ** (k / per_octave)) - 1 for k in range(steps + 1)}
    return sorted(p for p in pos | {T - 1} if p < T)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def _cast(level: LISSLevel, name: str) -> tuple[LISSLevel, torch.dtype, bool]:
    """The copy of the float64 level that runs in ``name``: its dtype and
    whether it runs under autocast."""
    if name == "bf16-mixed":
        return copy.deepcopy(level).float(), torch.float32, True
    dtype = getattr(torch, name)
    return copy.deepcopy(level).to(dtype), dtype, False


def _stats(
    y: torch.Tensor,
    ref: torch.Tensor,
    log_domain: bool,
) -> tuple[float, float, float, float]:
    """``(median |y|, max finite |y|, non-finite fraction, error)`` of one
    position's slice."""
    finite = torch.isfinite(y)
    nonfinite = 1.0 - finite.double().mean().item()
    mag = y[finite].abs()
    mag_med = mag.median().item() if mag.numel() else math.nan
    mag_max = mag.max().item() if mag.numel() else math.nan
    if not torch.isfinite(ref).all():
        err = math.nan          # beyond even float64: nothing to compare to
    elif nonfinite > 0:
        err = math.inf
    elif log_domain:
        err = (y - ref).abs().max().item()
    else:
        norm = ref.norm().item()
        err = (y - ref).norm().item() / norm if norm > 0 else math.nan
    return mag_med, mag_max, nonfinite, err


def _rows(run: Run, dtype: str, quantity: str, pos: list[int],
          y: torch.Tensor, ref: torch.Tensor, log_domain: bool) -> list[dict]:
    """``y``, ``ref``: ``(T, entries)`` on the CPU in float64."""
    meta = asdict(run)
    rows = []
    for t in pos:
        mag_med, mag_max, nonfinite, err = _stats(y[t], ref[t], log_domain)
        rows.append({**meta, "dtype": dtype, "quantity": quantity, "t": t,
                     "mag_med": mag_med, "mag_max": mag_max,
                     "nonfinite": nonfinite,
                     "err": err if dtype != "float64" else 0.0})
    return rows


def _flat(y: torch.Tensor) -> torch.Tensor:
    """``(B, T, ...) -> (T, B * ...)`` on the CPU in float64."""
    return y.detach().double().transpose(0, 1).reshape(y.shape[1], -1).cpu()


def evaluate(run: Run, dtypes: list[str], per_octave: int,
             device: torch.device) -> list[dict]:
    level = build_level(run, device)
    x = make_input(run.input, run.T).to(device)
    pos = positions(run.T, per_octave)
    log_domain = run.semiring in LOG_DOMAIN
    gradient = run.experiment == "gradient"
    r = torch.randn((B, run.T, N, D_V, 1),
                    generator=torch.Generator().manual_seed(2),
                    dtype=torch.float64).to(device)

    def forward(module: LISSLevel, dtype: torch.dtype, autocast: bool):
        # detach first: x.to(float64) is x itself, and a leaf that requires
        # grad would make every later cast a non-leaf without a .grad
        xi = x.detach().to(dtype).requires_grad_(gradient)
        context = (torch.autocast(device.type, dtype=torch.bfloat16)
                   if autocast else torch.autocast(device.type, enabled=False))
        with torch.set_grad_enabled(gradient), context:
            y = module(xi)
        if not gradient:
            return _flat(y)
        (y.double() * r).sum().backward()
        assert xi.grad is not None
        return _flat(xi.grad)

    ref = forward(level, torch.float64, False)
    quantity = "grad" if gradient else "output"
    # The gradient of a log-domain level is an ordinary number.
    log_err = log_domain and not gradient
    rows = _rows(run, "float64", quantity, pos, ref, ref, log_err)
    for name in dtypes:
        module, dtype, autocast = _cast(level, name)
        try:
            y = forward(module, dtype, autocast)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
        rows += _rows(run, name, quantity, pos, y, ref, log_err)
        del module, y
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+", default=EXPERIMENTS,
                        choices=EXPERIMENTS)
    parser.add_argument("--semirings", nargs="+", default=SEMIRINGS,
                        choices=SEMIRINGS)
    parser.add_argument("--dtypes", nargs="+", default=DTYPES, choices=DTYPES)
    parser.add_argument("--normalize", nargs="+", default=NORMALIZE,
                        choices=NORMALIZE)
    parser.add_argument("--inputs", nargs="+", default=INPUTS, choices=INPUTS)
    parser.add_argument("--depths", nargs="+", type=int,
                        default=[1, 2, 3, 4, 6, 8])
    parser.add_argument("--T", type=int, default=2**18,
                        help="length of the growth runs")
    parser.add_argument("--decay-Tc", type=int, default=1024,
                        help="context length of the decay runs")
    parser.add_argument("--decay-contexts", type=int, default=128,
                        help="decay runs are this many context lengths long")
    parser.add_argument("--alphas", nargs="+", type=float,
                        default=[1.0, 10.0, -1.0, -10.0],
                        help="decay rates times the context length")
    parser.add_argument("--kernel-T", type=int, default=4096)
    parser.add_argument("--gradient-T", type=int, default=2**16)
    parser.add_argument("--per-octave", type=int, default=2,
                        help="positions recorded per doubling of t")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    makers = {"growth": growth_runs, "decay": decay_runs,
              "kernel": kernel_runs, "gradient": gradient_runs}
    runs = [r for name in args.experiments if name in makers
            for r in makers[name](args)]
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("highest")
    print(f"{len(runs)} runs.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(), "torch": torch.__version__,
        "device": (torch.cuda.get_device_name(device)
                   if device.type == "cuda" else "cpu"),
        "shape": {"B": B, "n_is": N, "d_values": D_V, "d_in": D_IN},
        "args": {k: v for k, v in vars(args).items() if k != "out"},
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        if "primitive" in args.experiments:
            writer.writerows(primitive_rows(args, device))
            f.flush()
            print("primitive scans done.", flush=True)
        for i, run in enumerate(runs, 1):
            rows = evaluate(run, args.dtypes, args.per_octave, device)
            writer.writerows(rows)
            f.flush()
            last = [r for r in rows if r["t"] == run.T - 1]
            summary = " ".join(
                f"{r['dtype']}:{r['err']:.0e}" for r in last
                if r["dtype"] != "float64"
            )
            print(f"[{i}/{len(runs)}] {run.experiment:8s} {run.semiring:8s}"
                  f" p={run.p} {run.normalize:5s} {run.input:8s}"
                  f" {run.kernels:9s} a={run.alpha:+g} off={run.offset:g}"
                  f" s={run.scale:g}{' R' if run.restrict else ''}"
                  f" |ref|={last[0]['mag_max']:.1e} {summary}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
