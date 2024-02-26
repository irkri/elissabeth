__all__ = ["LISS", "LISSConfig"]

import warnings
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import nn

from ..base import HookedModule, ModelConfig
from ..hooks import HookCollection
from ..positional import RoPE


@dataclass
class LISSConfig(ModelConfig):
    sum_normalization: Optional[Literal["same", "independent"]] = "independent"
    n_is: int = 1
    length_is: int = 2
    d_values: int = None  # type: ignore
    values_2D: bool = True

    share_queries: bool = False
    share_keys: bool = False
    share_values: bool = False

    pe_key: bool = True
    pe_value: bool = False

    bias_query_key: bool = False
    bias_value: bool = False

    distance_weighting: bool = False
    alpha_multiplier: int = 1

    restrict_query_key: bool = False
    weighting: bool = True
    complex_exponential: bool = False
    normalize_weighting: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.d_values is None:
            self.d_values = self.d_hidden
        if (isinstance(self.d_values, list)
                and len(self.d_values) != self.length_is + 1):
            raise ValueError(
                "'d_values' has to be a list of length 'length_is'+1"
            )


class LISS(HookedModule):
    "Learnable Iterated Sums Signature"

    config: LISSConfig

    def __init__(self, config: LISSConfig) -> None:
        super().__init__(config)
        if config.context_length < config.length_is:
            warnings.warn(
                f"ISS length ({config.length_is}) "
                f"exceeds context length ({config.context_length}), "
                "which probably leads to unsuccessful training",
                RuntimeWarning,
            )

        self.W_Q = self.W_K = None
        self.b_Q = self.b_K = None
        if config.weighting:
            self.W_Q = nn.Parameter(
                torch.empty((
                    config.n_is,
                    1 if config.share_queries else config.length_is,
                    config.d_hidden,
                ))
            )
            nn.init.xavier_normal_(self.W_Q)
            self.W_K = nn.Parameter(
                torch.empty((
                    config.n_is,
                    1 if config.share_keys else config.length_is,
                    config.d_hidden,
                ))
            )
            nn.init.xavier_normal_(self.W_K)
            if config.bias_query_key:
                self.b_Q = nn.Parameter(
                    torch.empty((
                        config.n_is,
                        1 if config.share_queries else config.length_is,
                    ))
                )
                nn.init.zeros_(self.b_Q)
                self.b_K = nn.Parameter(
                    torch.empty((
                        config.n_is,
                        1 if config.share_keys else config.length_is,
                    ))
                )
                nn.init.zeros_(self.b_K)

        self.W_V = nn.Parameter(
            torch.empty((
                config.n_is,
                1 if config.share_values else config.length_is,
                config.d_hidden,
                config.d_values,
                config.d_values if config.values_2D else 1,
            ))
        )
        nn.init.xavier_normal_(self.W_V)
        self.b_V = None
        if config.bias_value:
            self.b_V = nn.Parameter(
                torch.empty((
                    config.n_is,
                    1 if config.share_values else config.length_is,
                    config.d_values,
                    config.d_values if config.values_2D else 1,
                ))
            )
            nn.init.zeros_(self.b_V)

        self.W_O = nn.Parameter(torch.empty((
            config.n_is,
            config.d_values,
            config.d_values if config.values_2D else 1,
            config.d_hidden,
        )))
        nn.init.xavier_normal_(self.W_O)
        if config.complex_exponential:
            self.W_O_i = nn.Parameter(torch.empty((
                config.n_is,
                config.d_values,
                config.d_values if config.values_2D else 1,
                config.d_hidden,
            )))
            nn.init.xavier_normal_(self.W_O_i)

        self.alpha = None
        if config.distance_weighting:
            indices = torch.empty((1, config.context_length, 1, 1))
            indices[0, :, 0, 0] = torch.linspace(
                1/config.context_length, 1, config.context_length
            )
            self.register_buffer("T", indices)
            self.alpha = nn.Parameter(
                torch.empty((1, config.n_is, config.length_is))
            )
            nn.init.ones_(self.alpha)

        self.beta = None
        if config.sum_normalization is not None:
            factor_norm = torch.empty((1, config.context_length, 1, 1, 1))
            factor_norm[0, :, 0, 0, 0] = torch.arange(
                1, config.context_length + 1
            )
            self.register_buffer("norm", factor_norm)
            self.beta = nn.Parameter(torch.empty((
                1 if config.sum_normalization == "same" else config.length_is,
            )))
            nn.init.constant_(self.beta, 5.40988)

        self.pe = nn.Identity()
        if config.pe_key or config.pe_value:
            self.pe = RoPE(T=config.context_length, d=config.d_hidden)

        self.hooks = HookCollection("Q", "K", "V")
        self.hooks.add_hooks(
            [f"iss.{i}" for i in range(1, config.length_is+1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        Q = K = None
        if self.W_Q is not None and self.W_K is not None:
            Q = torch.einsum('hld,btd->bthl', self.W_Q, x)
            if self.b_Q is not None:
                Q = Q + self.b_Q
            K = torch.einsum('hld,btd->bthl', self.W_K, self.pe(x))
            if self.b_K is not None:
                K = K + self.b_K
            if self.config.restrict_query_key:
                Q = torch.tanh(Q)
                K = torch.tanh(K)
        if self.alpha is not None and T > 1:
            alpha = self.config.alpha_multiplier * (1 - 1 / (self.alpha**2+1))
            rel_pos = alpha * self.get_buffer("T")[:, :T, :, :]
            if Q is None and K is None:
                Q = - rel_pos
                K = - rel_pos - alpha/self.config.context_length
            else:
                Q = Q - rel_pos
                K = K - rel_pos - alpha/self.config.context_length
        if Q is not None and K is not None:
            self.hooks("Q", Q)
            self.hooks("K", K)
            Q = Q.unsqueeze(-1).unsqueeze(-1)
            K = K.unsqueeze(-1).unsqueeze(-1)

        V = torch.einsum(
            'hldvw,btd->bthlvw',
            self.W_V,
            self.pe(x) if self.config.pe_value else x,
        )
        if self.b_V is not None:
            V = V + self.b_V
        self.hooks("V", V)

        p = self.config.length_is

        result = V[:, :, :, 0, :, :]
        result = self._weight_is(result, Q, K, 0)
        result = torch.cumsum(result, dim=1)
        if self.config.normalize_weighting:
            denom = torch.ones_like(V[:, :, :, 0, :, :])
            denom = self._weight_is(denom, Q, K, 0)
            denom = torch.cumsum(denom, dim=1)
            result = result / denom

        if self.beta is not None:
            result /= (
                (0.25*torch.tanh(self.beta[0])+0.75001)**(
                    torch.log10(self.get_buffer("norm")[:, :T, :, :, :])
                ) * self.get_buffer("norm")[:, :T, :, :, :]
            )

        for l in range(1, p):
            if self.config.normalize_weighting:
                denom = torch.cumsum(self._weight_is(result, Q, K, l), dim=1)
            result = nn.functional.pad(
                result[:, :-1, :, :, :],
                (0, 0, 0, 0, 0, 0, 1, 0)
            )
            iv = 0 if self.config.share_values else l
            if self.config.values_2D:
                result = V[:, :, :, iv, :, :] @ result
            else:
                result = V[:, :, :, iv, :, :] * result
            result = self._weight_is(result, Q, K, l)
            result = torch.cumsum(result, dim=1)
            if self.config.normalize_weighting:
                result = result / (denom)

            if self.beta is not None:
                result /= nn.functional.pad(
                    (0.25*torch.tanh(self.beta[
                        0 if self.config.sum_normalization == "same" else l
                    ])+0.75001)**(
                        torch.log10(self.get_buffer("norm")[:, :T, :, :, :])
                    ) * self.get_buffer("norm")[:, :T, :, :, :],
                    (0, 0, 0, 0, 0, 0, l, 0),
                    value=1.0,
                )[:, :-l, :, :, :]

        result = self._weight_is(result, Q, K, p)

        if self.hooks.get(f"iss.{p}").is_attached():
            self.hooks(f"iss.{p}", result)
        if self.config.complex_exponential:
            result = (
                torch.einsum("hvwd,bthvw->btd", self.W_O, result.real)
                + torch.einsum("hvwd,bthvw->btd ", self.W_O_i, result.imag)
            )
        else:
            result = torch.einsum("hvwd,bthvw->btd", self.W_O, result)

        return result

    def _weight_is(
        self,
        x: torch.Tensor,
        Q: torch.Tensor | None,
        K: torch.Tensor | None,
        l: int,
    ) -> torch.Tensor:
        p = self.config.length_is
        iq = 0 if self.config.share_queries else l-1
        ik = 0 if self.config.share_keys else l
        if 0 < l < p and self.hooks.get(f"iss.{l}").is_attached():
            temp = x
            if Q is not None:
                temp = x * torch.exp(Q[:, :, :, iq, :, :]
                    * (1j if self.config.complex_exponential else 1)
                )
            self.hooks(f"iss.{l}", temp)
        x = x * torch.exp((
            (0 if l == 0 or Q is None else Q[:, :, :, iq])
            - (0 if l == p or K is None else K[:, :, :, ik])  # type: ignore
        ) * (1j if self.config.complex_exponential else 1))
        return x
