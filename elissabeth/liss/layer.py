"""The LISS layer: learnable iterated sums in a semiring.

A level of depth ``p`` computes, per head ``n`` and position ``t``,

    ISS_t = (+)_{t_1 < ... < t_p <= t} (x)_{l=0}^{p-1}
            v_l(x_{t_l}) (x) kappa_l(t_{l+1}, t_l),      t_p := t,

with one value projection ``v_l`` per index and the product of all
configured kernels ``kappa_l`` per pair. Because every kernel factorises,
``kappa_l(t', t) = lambda_l^{t'-t-delta_l} <phi_l(t'), psi_l(t)>``, the sum
is evaluated by ``p`` scans instead of over all ``O(T^p)`` tuples:

    S_0(t) = scan_{s<=t} lambda_0^{t-s} psi_0(s) v_0(s)
    S_l(t) = scan_{s<=t} lambda_l^{t-s} psi_l(s)
             (<phi_{l-1}(s), S_{l-1}(s-1)> (x) v_l(s))
    ISS_t  = <phi_{p-1}(t), S_{p-1}(t)>

Each ``S_l`` holds ``R_l`` features, so the cost is linear in ``T`` and in
``p``. (The query features are contracted level by level; expanding them
over all pairs at once would carry ``R^p`` terms.)
"""
from typing import Literal

import torch
from pydantic import model_validator
from torch import nn

from ..config import ModelConfig
from ..hooks import HookedModule
from .kernels import RANK_R_KERNELS, Kernel, T_KernelConfig, build_kernel
from .projection import Projection, ValuesConfig
from .semiring import (LOG_DOMAIN, T_Semiring, multiply, scan, scan_indices,
                       shift)


class LISSConfig(ModelConfig):

    d_values: int
    n_is: int = 1
    """Number of iterated sums (heads) per level."""
    lengths: list[int] = [2]
    """Depths ``p`` of the levels of the layer."""
    values_2D: bool = False
    """Matrix-valued values, multiplied as matrices (in the semiring)."""
    semiring: T_Semiring = "reals"
    normalize: Literal["none", "mean", "sqrt", "learnable"] = "none"
    """Divide every level's partial sums by the number of summed positions
    ``c`` to the power ``1`` (mean), ``1/2`` (sqrt) or a learned
    ``gamma_l`` in ``(0.7, 1]``; in the log semiring ``gamma log c`` is
    subtracted instead."""
    share_values: bool = False
    """The same value projection for every index of a level."""
    values: ValuesConfig = ValuesConfig()
    kernels: list[T_KernelConfig] = []

    @model_validator(mode="after")
    def _check(self) -> "LISSConfig":
        if not self.lengths or min(self.lengths) < 1:
            raise ValueError("lengths must be a non-empty list of depths >= 1.")
        if self.normalize != "none" and self.semiring not in ("reals", "log"):
            raise ValueError(
                f"normalize is defined for the reals and the log semiring,"
                f" not for {self.semiring!r}."
            )
        for kernel in self.kernels:
            if kernel.type in RANK_R_KERNELS and self.semiring != "reals":
                raise ValueError(
                    f"The {kernel.type!r} kernel only works in the reals,"
                    f" not in the {self.semiring!r} semiring."
                )
        return self


def _outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Tensor product of the feature axes: ``(..., R1), (..., R2) ->
    (..., R1 R2)``."""
    return (a.unsqueeze(-1) * b.unsqueeze(-2)).flatten(-2)


class LISSLevel(HookedModule):
    """One level (fixed depth ``p``) of a LISS layer.

    Tensors of the recursion are ``(B, T, N, R, d_v, w)``: batch, time,
    heads, kernel features, and the value vector (``w = 1``) or matrix
    (``w = d_v``).
    """

    def __init__(
        self,
        config: LISSConfig,
        p: int,
        d_in: int,
        context_length: int | None,
    ) -> None:
        super().__init__("values", "iss")
        self.p = p
        self.n_is = config.n_is
        self.semiring: T_Semiring = config.semiring
        self.matrix = config.values_2D
        width = config.d_values if config.values_2D else 1
        self.values = Projection(
            config.values,
            d_in,
            (
                1 if config.values.shared else config.n_is,
                1 if config.share_values else p,
                config.d_values,
                width,
            ),
            context_length,
        )
        # Over the flattened (d_v, w): torch 2.12's CUDA LayerNorm backward
        # returns a wrongly shaped weight gradient for a multi-dimensional
        # normalized_shape once there are more than ~1e5 rows.
        self.value_shape = (config.d_values, width)
        self.value_norm = (
            nn.LayerNorm(config.d_values * width) if config.values.norm
            else None
        )
        self.kernels: list[Kernel] = nn.ModuleList([  # type: ignore
            build_kernel(
                kernel, config.n_is, p, d_in, config.semiring, context_length,
            ) for kernel in config.kernels
        ])
        self.max_rate = sum(kernel.max_rate for kernel in self.kernels)
        self.normalize = config.normalize
        self.beta: nn.Parameter | None = None
        if config.normalize == "learnable":
            # gamma = 1 + log10(0.25 tanh(beta) + 0.75001) is ~1 here.
            self.beta = nn.Parameter(torch.full((p,), 5.40988))

    @property
    def rank(self) -> int:
        """Number of kernel features ``R`` carried by the scans."""
        rank = 1
        for kernel in self.kernels:
            rank *= kernel.rank
        return rank

    def factors(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """``(rate, query, key)`` of the product of all kernels: rates
        ``(N, p)`` and factors ``(B|1, T, N|1, p, R)``, each ``None`` when
        no kernel contributes one."""
        rate = query = key = None
        for kernel in self.kernels:
            r = kernel.rates()
            if r is not None:
                rate = r if rate is None else rate + r
            factors = kernel.factors(x)
            if factors is None:
                continue
            q, k = factors
            if query is None or key is None:
                query, key = q, k
            elif self.semiring in LOG_DOMAIN:
                query, key = query + q, key + k
            else:
                query, key = _outer(query, q), _outer(key, k)
        return rate, query, key

    def _values(self, x: torch.Tensor) -> torch.Tensor:
        v = self.values(x)
        if self.value_norm is not None:
            v = self.value_norm(v.flatten(-2)).unflatten(-1, self.value_shape)
        return self.hook("values", v)

    def _expand(
        self,
        u: torch.Tensor,
        key: torch.Tensor | None,
    ) -> torch.Tensor:
        """``psi(s) u_s``: ``(B, T, N, d_v, w) -> (B, T, N, R, d_v, w)``."""
        u = u.unsqueeze(3)
        if key is None:
            return u
        key = key[..., None, None]
        return u + key if self.semiring in LOG_DOMAIN else u * key

    def _contract(
        self,
        state: torch.Tensor,
        query: torch.Tensor | None,
    ) -> torch.Tensor:
        """``<phi(t), S(t)>``: ``(B, T, N, R, d_v, w) -> (B, T, N, d_v, w)``.
        Outside the reals the factors are rank one (``R = 1``)."""
        if query is None:
            return state.squeeze(3)
        query = query[..., None, None]
        if self.semiring in LOG_DOMAIN:
            return (state + query).squeeze(3)
        if self.semiring == "bayesian":
            return (state * query).squeeze(3)
        return (state * query).sum(3)

    def _normalize(self, state: torch.Tensor, l: int) -> torch.Tensor:
        if self.normalize == "none":
            return state
        # Floating point on purpose: in integers inductor turns this into
        # an index expression, and merging loops over it fails (torch 2.12).
        t = torch.arange(state.shape[1], device=state.device, dtype=state.dtype)
        count = (t - (l - 1)).clamp_min(1.0)
        count = count.view(1, -1, *([1] * (state.ndim - 2)))
        if self.normalize == "mean":
            gamma: torch.Tensor | float = 1.0
        elif self.normalize == "sqrt":
            gamma = 0.5
        else:
            assert self.beta is not None
            gamma = 1 + torch.log10(0.25 * torch.tanh(self.beta[l]) + 0.75001)
        if self.semiring == "log":
            return state - gamma * torch.log(count)
        return state / count ** gamma

    def _pair(self, a: torch.Tensor | None, l: int) -> torch.Tensor | None:
        return None if a is None else a[:, :, :, l]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, d_in) -> (B, T, N, d_v, w)``."""
        v = self._values(x)
        rate, query, key = self.factors(x)
        state = torch.empty(0)
        for l in range(self.p):
            u = v[:, :, :, 0 if v.shape[3] == 1 else l]
            if l > 0:
                prev = self._contract(
                    shift(state, self.semiring), self._pair(query, l - 1),
                )
                u = multiply(prev, u, self.semiring, self.matrix)
            state = self._expand(u, self._pair(key, l))
            state = scan(
                state, self.semiring,
                None if rate is None else rate[:, l], self.max_rate,
            )
            state = self._normalize(state, l)
        out = self._contract(state, self._pair(query, self.p - 1))
        if self.semiring in LOG_DOMAIN and self.p > 1:
            # No index tuple fits before t = p-1; the finite stand-in for
            # -inf there must not reach the network.
            t = torch.arange(out.shape[1], device=out.device)
            valid = (t >= self.p - 1).view(1, -1, 1, 1, 1)
            out = torch.where(valid, out, torch.zeros_like(out))
        out = out.expand(-1, -1, self.n_is, -1, -1)
        return self.hook("iss", out)

    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """The maximising index tuple ``(t_1, ..., t_p)`` of every output,
        ``(B, T, N, d_v, p)``, ``-1`` where ``t < p - 1`` (no tuple). This is
        the dynamic-programming read-out of the arctic (and bayesian) level:
        the argmax of each scan, traced back like a Viterbi path.
        """
        if self.semiring not in ("arctic", "bayesian") or self.matrix:
            raise ValueError(
                "decode needs vector values in the arctic or bayesian"
                " semiring."
            )
        v = self._values(x)
        rate, query, key = self.factors(x)
        state = torch.empty(0)
        argmax: list[torch.Tensor] = []
        for l in range(self.p):
            u = v[:, :, :, 0 if v.shape[3] == 1 else l]
            if l > 0:
                prev = self._contract(
                    shift(state, self.semiring), self._pair(query, l - 1),
                )
                u = multiply(prev, u, self.semiring, False)
            state = self._expand(u, self._pair(key, l))
            state, index = scan_indices(
                state, self.semiring, None if rate is None else rate[:, l],
            )
            argmax.append(index[:, :, :, 0, :, 0])
        B, T = x.shape[:2]
        shape = (B, T, self.n_is, argmax[0].shape[-1])
        position = torch.arange(T, device=x.device).view(1, -1, 1, 1)
        position = position.expand(shape)
        path = []
        for index in reversed(argmax):
            t_l = index.expand(shape).gather(1, position)
            path.append(t_l)
            position = (t_l - 1).clamp_min(0)
        tuples = torch.stack(path[::-1], dim=-1)
        valid = (torch.arange(T, device=x.device) >= self.p - 1)
        return torch.where(valid.view(1, -1, 1, 1, 1), tuples, -1)

    @torch.no_grad()
    def pair_matrices(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Every pair kernel evaluated densely, ``kappa_l(t, s)`` as
        ``(B, N, p, T, T)`` in the semiring's product domain (a factor in the
        reals, a log-factor in the arctic/log semirings), and the boolean
        ``(p, T, T)`` support: ``s < t`` for inner pairs, ``s <= t`` for the
        last one. For analysis; this is ``O(T^2)``.
        """
        rate, query, key = self.factors(x)
        B, T = x.shape[:2]
        log = self.semiring in LOG_DOMAIN
        kernel = torch.full(
            (B, self.n_is, self.p, T, T), 0.0 if log else 1.0,
            device=x.device, dtype=x.dtype,
        )
        if query is not None and key is not None:
            shape = (B, T, self.n_is, self.p, query.shape[-1])
            q, k = query.expand(shape), key.expand(shape)
            if log:
                kernel = kernel + (
                    q[..., 0].permute(0, 2, 3, 1).unsqueeze(-1)
                    + k[..., 0].permute(0, 2, 3, 1).unsqueeze(-2)
                )
            else:
                kernel = kernel * torch.einsum("btnpr,bsnpr->bnpts", q, k)
        t = torch.arange(T, device=x.device, dtype=x.dtype)
        delta = torch.ones(self.p, device=x.device, dtype=x.dtype)
        delta[-1] = 0
        gap = t.view(1, -1, 1) - t.view(1, 1, -1) - delta.view(-1, 1, 1)
        if rate is not None:
            decay = -rate.view(1, self.n_is, self.p, 1, 1) * gap
            kernel = kernel + decay if log else kernel * torch.exp(decay)
        return kernel, gap >= 0


class LISS(HookedModule):
    """Learnable iterated sums signature: the levels of ``lengths``, mixed
    over depths and heads by ``W_H`` and projected back by ``W_O``,

        y_t = W_O sum_{p, n} W_H[p, n] ISS^{(p)}_{n, t}.
    """

    def __init__(
        self,
        config: LISSConfig,
        d_in: int,
        context_length: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.levels: list[LISSLevel] = nn.ModuleList([  # type: ignore
            LISSLevel(config, p, d_in, context_length)
            for p in config.lengths
        ])
        n_levels = len(config.lengths)
        self.W_H = nn.Parameter(torch.full(
            (n_levels, config.n_is), 1 / (n_levels * config.n_is),
        ))
        self.W_O = nn.Parameter(torch.empty(
            config.d_values,
            config.d_values if config.values_2D else 1,
            d_in,
        ))
        nn.init.xavier_normal_(self.W_O)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, d_in) -> (B, T, d_in)``."""
        iss = torch.stack([level(x) for level in self.levels])
        mixed = torch.einsum("ln,lbtnvw->btvw", self.W_H, iss)
        return torch.einsum("vwd,btvw->btd", self.W_O, mixed)
