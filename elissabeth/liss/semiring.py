"""Semiring operations of the LISS recursion.

A LISS level is computed with a sum ``(+)`` and a product ``(x)``:

- ``reals``     ``(+) = sum``,       ``(x) = *``
- ``arctic``    ``(+) = max``,       ``(x) = +``   (max-plus, tropical)
- ``log``       ``(+) = logsumexp``, ``(x) = +``   (smooth arctic)
- ``bayesian``  ``(+) = max``,       ``(x) = *``   (Viterbi; values must be
  non-negative for the recursion to equal the max over index tuples)

Every tensor that goes through a scan has time on dimension 1 and the heads
of the layer on dimension 2: ``(B, T, N, ...)``.
"""
from typing import Literal

import torch

T_Semiring = Literal["reals", "arctic", "log", "bayesian"]

LOG_DOMAIN: tuple[str, ...] = ("arctic", "log")
"""Semirings whose product is ``+``: kernels enter as log-factors."""

EXP_TRICK_LIMIT = 40.0
"""Largest exponent the decayed real scan lets ``exp(rate * t)`` reach
before it switches from one rescaled cumsum to the Hillis-Steele scan."""


def zero(semiring: T_Semiring, dtype: torch.dtype) -> float:
    """The semiring's zero, the value of an empty sum.

    In the log domain this is a finite, very negative number, not ``-inf``:
    ``logcumsumexp`` returns NaN gradients at ``-inf`` entries even when the
    output there is masked afterwards. Every chain of the recursion picks it
    up at most once (through one shift), so it cannot overflow.
    """
    if semiring in LOG_DOMAIN:
        return torch.finfo(dtype).min / 8
    return 0.0


def shift(x: torch.Tensor, semiring: T_Semiring) -> torch.Tensor:
    """``x_{t-1}`` along dimension 1, with the semiring zero at ``t = 0``.
    """
    pad = x.new_full((x.shape[0], 1, *x.shape[2:]), zero(semiring, x.dtype))
    return torch.cat((pad, x[:, :-1]), dim=1)


def add(x: torch.Tensor, semiring: T_Semiring, dim: int) -> torch.Tensor:
    """The semiring sum ``(+)`` over one dimension: ``sum``, ``max`` or
    ``logsumexp``."""
    if semiring == "reals":
        return x.sum(dim)
    if semiring == "log":
        return torch.logsumexp(x, dim=dim)
    return x.amax(dim)


def multiply(
    a: torch.Tensor,
    b: torch.Tensor,
    semiring: T_Semiring,
    matrix: bool,
) -> torch.Tensor:
    """``a (x) b``, elementwise, or as a semiring matrix product over the
    last two dimensions (``values_2D``)."""
    if not matrix:
        return a + b if semiring in LOG_DOMAIN else a * b
    if semiring == "reals":
        return a @ b
    return add(
        multiply(a.unsqueeze(-1), b.unsqueeze(-3), semiring, False),
        semiring, -2,
    )


def _cumulate_eager(x: torch.Tensor, semiring: str) -> torch.Tensor:
    if semiring == "reals":
        return torch.cumsum(x, dim=1)
    if semiring == "log":
        return torch.logcumsumexp(x, dim=1)
    return torch.cummax(x, dim=1).values


# The cumulative (+) is an opaque custom op, not an ATen call inductor may
# lower: torch 2.12's split-scan codegen crashes on a long scan (T ~ 2000)
# whose fused input broadcasts over some of the other dimensions -- which a
# per-head decay does. The op boundary is also where a Triton scan plugs in.
@torch.library.custom_op("elissabeth::cumulate", mutates_args=())
def _cumulate_op(x: torch.Tensor, semiring: str) -> torch.Tensor:
    return _cumulate_eager(x, semiring)


@_cumulate_op.register_fake
def _(x: torch.Tensor, semiring: str) -> torch.Tensor:
    return torch.empty_like(x)


def _reverse_logcumsumexp(x: torch.Tensor) -> torch.Tensor:
    return torch.logcumsumexp(x.flip(1), dim=1).flip(1)


@torch.library.custom_op("elissabeth::cumulate_backward", mutates_args=())
def _cumulate_backward_op(
    grad: torch.Tensor,
    x: torch.Tensor,
    semiring: str,
) -> torch.Tensor:
    """``dL/dx_s = sum_{t >= s} dL/dh_t dh_t/dx_s`` of the cumulative
    (+): a reverse cumsum (reals), the gradient routed to each running
    argmax (max), or ``sum_{t>=s} g_t exp(x_s - h_t)`` computed per sign in
    log space (log; ``-inf`` terms are harmless here, outside autograd).
    """
    if semiring == "reals":
        return grad.flip(1).cumsum(1).flip(1)
    if semiring in ("arctic", "bayesian"):
        index = torch.cummax(x, dim=1).indices
        return torch.zeros_like(x).scatter_add_(1, index, grad)
    out = torch.logcumsumexp(x, dim=1)
    negative_inf = torch.full_like(grad, -torch.inf)
    positive = torch.where(grad > 0, grad, 1).log()
    negative = torch.where(grad < 0, -grad, 1).log()
    positive = torch.where(grad > 0, positive, negative_inf)
    negative = torch.where(grad < 0, negative, negative_inf)
    return (
        torch.exp(_reverse_logcumsumexp(positive - out) + x)
        - torch.exp(_reverse_logcumsumexp(negative - out) + x)
    )


@_cumulate_backward_op.register_fake
def _(grad: torch.Tensor, x: torch.Tensor, semiring: str) -> torch.Tensor:
    return torch.empty_like(x)


def _cumulate_setup(ctx, inputs, output) -> None:
    ctx.semiring = inputs[1]
    ctx.save_for_backward(inputs[0])


def _cumulate_grad(ctx, grad: torch.Tensor):
    (x,) = ctx.saved_tensors
    return _cumulate_backward_op(grad.contiguous(), x, ctx.semiring), None


_cumulate_op.register_autograd(_cumulate_grad, setup_context=_cumulate_setup)


def _cumulate(x: torch.Tensor, semiring: T_Semiring) -> torch.Tensor:
    return _cumulate_op(x.contiguous(), semiring)


def _time(u: torch.Tensor) -> torch.Tensor:
    """Positions ``0..T-1`` shaped to broadcast against ``u``."""
    t = torch.arange(u.shape[1], device=u.device, dtype=u.dtype)
    return t.view(1, -1, *([1] * (u.ndim - 2)))


def _per_head(rate: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return rate.view(1, 1, -1, *([1] * (u.ndim - 3)))


def scan(
    u: torch.Tensor,
    semiring: T_Semiring,
    rate: torch.Tensor | None = None,
    max_rate: float = 0.0,
) -> torch.Tensor:
    """Inclusive semiring scan over dimension 1 with a per-head decay:

    ``h_t = u_t (+) (lambda (x) h_{t-1})``, i.e.
    ``h_t = (+)_{s <= t} lambda^{t-s} (x) u_s``,

    with ``lambda = exp(-rate)`` (reals, bayesian) or ``-rate`` (log
    domain). ``rate`` has one entry per head (dimension 2); ``max_rate`` is
    a static bound on ``|rate|`` that decides how the real scan is done.

    In the log domain the decay is exact as an additive offset,
    ``(+)_s (u_s + rate*s) - rate*t``. In the reals the multiplicative
    version ``cumsum(u_s e^{rate*s}) e^{-rate*t}`` is used while the
    exponent stays below :data:`EXP_TRICK_LIMIT`; past it the scan falls
    back to Hillis-Steele, ``O(T log T)`` but free of overflow.
    """
    if rate is None:
        return _cumulate(u, semiring)
    t = _time(u)
    rate = _per_head(rate.to(u.dtype), u)
    if semiring in LOG_DOMAIN:
        return _cumulate(u + rate * t, semiring) - rate * t
    if max_rate * (u.shape[1] - 1) <= EXP_TRICK_LIMIT:
        return _cumulate(u * torch.exp(rate * t), semiring) \
            * torch.exp(-rate * t)
    return _hillis(u, rate, semiring)


def _hillis(
    u: torch.Tensor,
    rate: torch.Tensor,
    semiring: T_Semiring,
) -> torch.Tensor:
    """Hillis-Steele scan of ``h_t = u_t (+) lambda h_{t-1}``; after the
    step with offset ``2^k`` every ``h_t`` covers ``(t - 2^{k+1}, t]``."""
    decay = torch.exp(-rate)
    offset = 1
    while offset < u.shape[1]:
        carried = decay * u[:, :-offset]
        tail = u[:, offset:]
        tail = (
            tail + carried if semiring == "reals"
            else torch.maximum(tail, carried)
        )
        u = torch.cat((u[:, :offset], tail), dim=1)
        decay = decay * decay
        offset *= 2
    return u


def scan_indices(
    u: torch.Tensor,
    semiring: T_Semiring,
    rate: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The max-scan of :func:`scan` together with its argmax ``s <= t``,
    for the ``arctic`` and ``bayesian`` semirings (decoding)."""
    if semiring not in ("arctic", "bayesian"):
        raise ValueError(f"No argmax in the {semiring!r} semiring.")
    if rate is None:
        values, indices = torch.cummax(u, dim=1)
        return values, indices
    t = _time(u)
    rate = _per_head(rate.to(u.dtype), u)
    if semiring == "arctic":
        values, indices = torch.cummax(u + rate * t, dim=1)
        return values - rate * t, indices
    values, indices = torch.cummax(u * torch.exp(rate * t), dim=1)
    return values * torch.exp(-rate * t), indices
