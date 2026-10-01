"""Kernels between consecutive indices of an iterated sum.

The kernel of pair ``l`` (``l = 0..p-1``) weighs ``t_l`` against the next
index ``t_{l+1}``, where ``t_p`` is the output position ``t``. Every kernel
here is separable, which is what makes the level linear in ``T``:

- a *decay* contributes a per-step rate, ``lambda^{t_{l+1} - t_l - delta_l}``
  with ``delta_l = 1`` for inner pairs and 0 for the last one; the level
  folds it into its scans,
- a *feature* kernel contributes a query factor at ``t_{l+1}`` and a key
  factor at ``t_l`` with ``kappa(t', t) = (+)_r phi_r(t') (x) psi_r(t)``
  over ``R`` features. ``R`` is the kernel's rank: the number of separable
  terms of ``kappa`` as a function of its two arguments (``Kernel.rank``).

Factors are returned in the semiring's product domain: multiplicative for
``reals``/``bayesian``, additive (log-factors) for ``arctic``/``log``. The
level contracts the features with the semiring's own sum, which
distributes over its scans in every semiring. The cosine kernels are
restricted to the reals because their features are signed and only add up
to ``cos`` under an ordinary sum.
"""
import math
from typing import Annotated, Literal

import torch
from pydantic import Field
from torch import nn

from ..config import ModelConfig
from ..hooks import HookedModule
from .projection import Projection, ProjectionConfig
from .semiring import LOG_DOMAIN, T_Semiring


class DecayConfig(ModelConfig):
    """``exp(-alpha_l (t_{l+1} - t_l - delta_l) / context_length)``, in
    every semiring (an additive ``-alpha_l (...)`` in the log domain)."""

    type: Literal["decay"] = "decay"
    alpha_0: float = 1.0
    """``alpha_l = alpha_0 * tanh(a_l)``; negative values favour the
    distant past."""
    shared: bool = False
    """One rate for all pairs."""


class ExponentialConfig(ModelConfig):
    """``(+)_{d=1}^{d_qk} exp(q_{l,d}(x_{t_{l+1}}) - k_{l,d}(x_{t_l}))``,
    in every semiring: ``sum_d exp(q_d - k_d)`` in the reals, the tropical
    polynomial ``max_d (q_d - k_d)`` in the arctic semiring,
    ``logsumexp_d (q_d - k_d)`` in the log semiring.

    The ``d_qk`` query/key components are summed (with the semiring's
    ``(+)``), each term separable, so the rank is ``d_qk``. (The cosine
    kernel multiplies its components instead, rank ``(m+1)^d_qk``; a
    product of exponentials, ``exp(sum_d q_d - sum_d k_d)``, would stay
    rank one.) ``d_qk = 1`` is ``exp(q - k)``; more lets an arctic or log
    level relate the tokens at two indices."""

    type: Literal["exponential"] = "exponential"
    d_qk: int = Field(1, ge=1)
    """Query/key components, summed: the kernel's rank."""
    restrict: bool = False
    """Bound queries and keys by ``tanh``."""
    share_queries: bool = False
    share_keys: bool = False
    projection: ProjectionConfig = ProjectionConfig()


class CosineConfig(ModelConfig):
    """``prod_d cos(q_{l,d}(x_{t_{l+1}}) - k_{l,d}(x_{t_l}))^exponent``,
    reals only. Rank ``(exponent+1)^d_qk``."""

    type: Literal["cosine"] = "cosine"
    d_qk: int = 1
    exponent: int = 1
    restrict: bool = False
    """Bound queries and keys to ``(-pi/4, pi/4)`` by ``tanh``."""
    share_queries: bool = False
    share_keys: bool = False
    projection: ProjectionConfig = ProjectionConfig()


class CosineDecayConfig(ModelConfig):
    """``prod_d cos(alpha_{l,d} (t_{l+1} - t_l - delta_l) /
    context_length)^exponent``, reals only. Rank
    ``(exponent+1)^d_alpha``."""

    type: Literal["cosine_decay"] = "cosine_decay"
    d_alpha: int = 1
    exponent: int = 1
    alpha_0: float = 1.0
    shared: bool = False


T_KernelConfig = Annotated[
    DecayConfig | ExponentialConfig | CosineConfig | CosineDecayConfig,
    Field(discriminator="type"),
]

REALS_ONLY_KERNELS = ("cosine", "cosine_decay")


def cosine_features(angle: torch.Tensor, exponent: int) -> torch.Tensor:
    """Features with ``prod_d cos(a_d - b_d)^m = <phi(a), phi(b)>``.

    ``cos(a - b)^m = sum_j C(m, j) (cos a cos b)^{m-j} (sin a sin b)^j``
    gives ``phi_j(a) = sqrt(C(m, j)) cos^{m-j}(a) sin^j(a)`` per dimension,
    and the product over dimensions is their tensor product:
    ``(..., D) -> (..., (m+1)^D)``. Powers are built by multiplication
    (``pow`` has a NaN gradient at ``0^0``).
    """
    m = exponent
    cos, sin = torch.cos(angle), torch.sin(angle)
    cos_pow, sin_pow = [torch.ones_like(cos)], [torch.ones_like(sin)]
    for _ in range(m):
        cos_pow.append(cos_pow[-1] * cos)
        sin_pow.append(sin_pow[-1] * sin)
    per_dim = torch.stack([
        math.sqrt(math.comb(m, j)) * cos_pow[m - j] * sin_pow[j]
        for j in range(m + 1)
    ], dim=-1)
    features = per_dim[..., 0, :]
    for d in range(1, per_dim.shape[-2]):
        features = (
            features.unsqueeze(-1) * per_dim[..., d, :].unsqueeze(-2)
        ).flatten(-2)
    return features


def _pair_delta(p: int, device: torch.device) -> torch.Tensor:
    """``delta_l``: 1 for the inner pairs, 0 for the last one."""
    delta = torch.ones(p, device=device)
    delta[-1] = 0
    return delta


class Kernel(HookedModule):
    """A kernel of one LISS level, for its ``p`` pairs and ``n_is`` heads.
    """

    rank: int = 1
    """The rank ``R``: separable terms of the kernel, the features its
    factors carry."""
    max_rate: float = 0.0
    """Static bound on the decay rate per step."""

    def __init__(self, n_is: int, p: int, semiring: T_Semiring) -> None:
        super().__init__()
        self.n_is = n_is
        self.p = p
        self.semiring = semiring

    def rates(self) -> torch.Tensor | None:
        """Decay rate per step, ``(n_is, p)``, or ``None``."""
        return None

    def factors(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Query and key factors ``(B|1, T, N|1, p, R)``, or ``None``."""
        return None

    def _pairs(self, a: torch.Tensor, dim: int) -> torch.Tensor:
        """Broadcast a shared (size 1) pair axis to all ``p`` pairs."""
        size = list(a.shape)
        size[dim] = self.p
        return a.expand(size)


class Decay(Kernel):

    def __init__(
        self,
        config: DecayConfig,
        n_is: int,
        p: int,
        semiring: T_Semiring,
        context_length: int,
    ) -> None:
        super().__init__(n_is, p, semiring)
        self.alpha_0 = config.alpha_0
        self.context_length = context_length
        self.max_rate = abs(config.alpha_0) / context_length
        self.alpha = nn.Parameter(torch.zeros(n_is, 1 if config.shared else p))

    def rates(self) -> torch.Tensor:
        rate = self.alpha_0 * torch.tanh(self.alpha) / self.context_length
        return self._pairs(rate, 1)


class Exponential(Kernel):

    def __init__(
        self,
        config: ExponentialConfig,
        n_is: int,
        p: int,
        d_in: int,
        semiring: T_Semiring,
        context_length: int | None,
    ) -> None:
        super().__init__(n_is, p, semiring)
        self.hooks.add_hooks("query", "key")
        self.restrict = config.restrict
        self.rank = config.d_qk
        self.query = Projection(
            config.projection, d_in,
            (n_is, 1 if config.share_queries else p, config.d_qk),
            context_length,
        )
        self.key = Projection(
            config.projection, d_in,
            (n_is, 1 if config.share_keys else p, config.d_qk),
            context_length,
        )

    def factors(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q, k = self.query(x), self.key(x)
        if self.restrict:
            q, k = torch.tanh(q), torch.tanh(k)
        q = self.hook("query", self._pairs(q, 3))
        k = self.hook("key", self._pairs(k, 3))
        if self.semiring in LOG_DOMAIN:
            return q, -k
        return torch.exp(q), torch.exp(-k)


class Cosine(Kernel):

    def __init__(
        self,
        config: CosineConfig,
        n_is: int,
        p: int,
        d_in: int,
        semiring: T_Semiring,
        context_length: int | None,
    ) -> None:
        super().__init__(n_is, p, semiring)
        self.hooks.add_hooks("query", "key")
        self.restrict = config.restrict
        self.exponent = config.exponent
        self.rank = (config.exponent + 1) ** config.d_qk
        self.query = Projection(
            config.projection, d_in,
            (n_is, 1 if config.share_queries else p, config.d_qk),
            context_length,
        )
        self.key = Projection(
            config.projection, d_in,
            (n_is, 1 if config.share_keys else p, config.d_qk),
            context_length,
        )

    def factors(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q, k = self.query(x), self.key(x)
        if self.restrict:
            q = torch.tanh(q) * torch.pi / 4
            k = torch.tanh(k) * torch.pi / 4
        q = self.hook("query", self._pairs(q, 3))
        k = self.hook("key", self._pairs(k, 3))
        return (
            cosine_features(q, self.exponent),
            cosine_features(k, self.exponent),
        )


class CosineDecay(Kernel):

    def __init__(
        self,
        config: CosineDecayConfig,
        n_is: int,
        p: int,
        semiring: T_Semiring,
        context_length: int,
    ) -> None:
        super().__init__(n_is, p, semiring)
        self.alpha_0 = config.alpha_0
        self.exponent = config.exponent
        self.context_length = context_length
        self.rank = (config.exponent + 1) ** config.d_alpha
        self.alpha = nn.Parameter(torch.zeros(
            n_is, 1 if config.shared else p, config.d_alpha,
        ))

    def factors(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # q(t') - k(t) = alpha (t' - t - delta): q at the later index
        # carries the pair's offset, k at the earlier one does not.
        alpha = self.alpha_0 * torch.tanh(self._pairs(self.alpha, 1))
        alpha = alpha / self.context_length
        t = torch.arange(x.shape[1], device=x.device, dtype=alpha.dtype)
        t = t.view(1, -1, 1, 1, 1)
        delta = _pair_delta(self.p, x.device).view(1, 1, 1, -1, 1)
        return (
            cosine_features(alpha * (t - delta), self.exponent),
            cosine_features(alpha * t, self.exponent),
        )


def build_kernel(
    config: T_KernelConfig,
    n_is: int,
    p: int,
    d_in: int,
    semiring: T_Semiring,
    context_length: int | None,
) -> Kernel:
    if config.type in REALS_ONLY_KERNELS and semiring != "reals":
        raise ValueError(
            f"The {config.type!r} kernel only works in the reals, not in"
            f" the {semiring!r} semiring."
        )
    if config.type in ("decay", "cosine_decay") and context_length is None:
        raise ValueError(f"The {config.type!r} kernel needs context_length.")
    match config:
        case DecayConfig():
            assert context_length is not None
            return Decay(config, n_is, p, semiring, context_length)
        case ExponentialConfig():
            return Exponential(
                config, n_is, p, d_in, semiring, context_length,
            )
        case CosineConfig():
            return Cosine(config, n_is, p, d_in, semiring, context_length)
        case CosineDecayConfig():
            assert context_length is not None
            return CosineDecay(config, n_is, p, semiring, context_length)
    raise ValueError(f"Unknown kernel {config!r}.")
