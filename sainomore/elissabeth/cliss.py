__all__ = ["CLISS", "CLISSConfig"]

import itertools
import warnings
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import nn

from ..base import HookedModule, ModelConfig
from ..hooks import HookCollection
from ..positional import RoPE


@dataclass
class CLISSConfig(ModelConfig):
    sum_normalization: Optional[Literal["same", "independent"]] = "independent"
    n_is: int = 1
    length_is: int = 2
    d_query_key: int = 1
    d_values: int = None  # type: ignore
    values_2D: bool = True

    share_queries: bool = False
    share_keys: bool = False
    share_values: bool = False

    pe_query_key: bool = True
    pe_value: bool = False

    bias_query_key: bool = False
    bias_value: bool = False

    distance_weighting: bool = False

    weighting: bool = True
    exponent: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.d_values is None:
            self.d_values = self.d_hidden
        if (isinstance(self.d_values, list)
                and len(self.d_values) != self.length_is + 1):
            raise ValueError(
                "'d_values' has to be a list of length 'length_is'+1"
            )


class CLISS(HookedModule):
    "Learnable Iterated Sums Signature"

    config: CLISSConfig

    def __init__(self, config: CLISSConfig) -> None:
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
                    config.d_query_key,
                ))
            )
            nn.init.xavier_normal_(self.W_Q)
            self.W_K = nn.Parameter(
                torch.empty((
                    config.n_is,
                    1 if config.share_queries else config.length_is,
                    config.d_hidden,
                    config.d_query_key,
                ))
            )
            nn.init.xavier_normal_(self.W_K)
            if config.bias_query_key:
                self.b_Q = nn.Parameter(
                    torch.empty((
                        config.n_is,
                        1 if config.share_queries else config.length_is,
                        config.d_query_key,
                    ))
                )
                nn.init.zeros_(self.b_Q)
                self.b_K = nn.Parameter(
                    torch.empty((
                        config.n_is,
                        1 if config.share_queries else config.length_is,
                        config.d_query_key,
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

        self.alpha = None
        if config.distance_weighting:
            indices = torch.empty((config.context_length, 1, 1))
            indices[:, 0, 0] = torch.linspace(
                1/config.context_length, 1, config.context_length
            )
            self.register_buffer("T", indices)
            self.alpha = nn.Parameter(
                torch.empty((1, config.n_is, config.length_is))
            )
            nn.init.ones_(self.alpha)

        self.beta = None
        if config.sum_normalization is not None:
            factor_norm = torch.empty((1, 1, config.context_length, 1, 1, 1))
            factor_norm[0, 0, :, 0, 0, 0] = torch.arange(
                1, config.context_length + 1
            )
            self.register_buffer("norm", factor_norm)
            self.beta = nn.Parameter(torch.empty((
                1 if config.sum_normalization == "same" else config.length_is,
            )))
            nn.init.constant_(self.beta, 5.40988)

        self.pe = None
        if config.pe_query_key or config.pe_value:
            self.pe = RoPE(T=config.context_length, d=config.d_hidden)

        self._create_weight_signature()

        self.hooks = HookCollection("Q", "K", "V")

    def _create_weight_signature(self) -> None:
        p = self.config.length_is * self.config.d_query_key
        trig_id = []
        trig_exp = [self.config.exponent, 0]
        trig_coeff = 1
        for k in range(self.config.exponent+1):
            trig_id.append(f"{trig_coeff}{trig_exp[0]}{trig_exp[1]}")
            trig_exp[0] -= 1
            trig_exp[1] += 1
            trig_coeff = trig_coeff * (self.config.exponent - k) // (k + 1)
        weightings = torch.zeros(
            ((self.config.exponent+1)**p, 1, 1, 1, 1, 1, 4*p+1),
            dtype=torch.int32,
        )
        weightings[:, ..., 0] = 1
        for c, comb in enumerate(itertools.product(trig_id, repeat=p)):
            for i in range(p):
                weightings[c, ..., 0] *= int(comb[i][0])
                weightings[c, ..., 4*i+1] += int(comb[i][1])
                weightings[c, ..., 4*i+3] += int(comb[i][1])
                weightings[c, ..., 4*i+2] += int(comb[i][2])
                weightings[c, ..., 4*i+4] += int(comb[i][2])
        self.register_buffer("weight_signature", weightings.swapaxes(-1, -3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        sin_K = cos_K = sin_Q = cos_Q = None
        if self.pe is not None:
            x_enc = self.pe(x)
        if self.W_Q is not None and self.W_K is not None:
            Q = torch.einsum(
                'hldi,btd->bthli',
                self.W_Q,
                x_enc if self.config.pe_query_key else x,  # type: ignore
            )
            if self.b_Q is not None:
                Q = Q + self.b_Q
            K = torch.einsum(
                'hldi,btd->bthli',
                self.W_K,
                x_enc if self.config.pe_query_key else x,  # type: ignore
            )
            if self.b_K is not None:
                K = K + self.b_K
            self.hooks("Q", Q)
            self.hooks("K", K)
            Q = Q.unsqueeze(-1).unsqueeze(-1)
            K = K.unsqueeze(-1).unsqueeze(-1)
            sin_Q = torch.sin(Q).unsqueeze(0)
            cos_Q = torch.cos(Q).unsqueeze(0)
            sin_K = torch.sin(K).unsqueeze(0)
            cos_K = torch.cos(K).unsqueeze(0)

        V = torch.einsum(
            'hldvw,btd->bthlvw',
            self.W_V,
            x_enc if self.config.pe_value else x,  # type: ignore
        )
        if self.b_V is not None:
            V = V + self.b_V
        self.hooks("V", V)
        V = V.unsqueeze(0) # add axis for weight signature

        p = self.config.length_is

        result = V[:, :, :, :, 0, :, :]
        result = self._weight_is(result, sin_Q, cos_Q, sin_K, cos_K, 0)
        result = torch.cumsum(result, dim=2)

        if self.beta is not None:
            result /= (
                (0.25*torch.tanh(self.beta[0])+0.75001)**(
                    torch.log10(self.get_buffer("norm")[:, :, :T, :, :, :])
                ) * self.get_buffer("norm")[:, :, :T, :, :, :]
            )

        for l in range(1, p):
            result = nn.functional.pad(
                result[:, :, :-1, :, :, :],
                (0, 0, 0, 0, 0, 0, 1, 0)
            )
            iv = 0 if self.config.share_values else l
            if self.config.values_2D:
                result = V[:, :, :, :, iv, :, :] @ result
            else:
                result = V[:, :, :, :, iv, :, :] * result
            result = self._weight_is(result, sin_Q, cos_Q, sin_K, cos_K, l)
            result = torch.cumsum(result, dim=2)

            if self.beta is not None:
                result /= nn.functional.pad(
                    (0.25*torch.tanh(self.beta[
                        0 if self.config.sum_normalization == "same" else l
                    ])+0.75001)**(
                        torch.log10(self.get_buffer("norm")[:, :, :T, :, :, :])
                    ) * self.get_buffer("norm")[:, :, :T, :, :, :],
                    (0, 0, 0, 0, 0, 0, l, 0),
                    value=1.0,
                )[:, :, :-l, :, :, :]

        result = self._weight_is(result, sin_Q, cos_Q, sin_K, cos_K, p)
        result = (
            result * self.get_buffer("weight_signature")[..., 0, :, :]
        ).sum(dim=0)
        result = torch.einsum("hvwd,bthvw->btd", self.W_O, result)

        return result

    def _weight_is(
        self,
        x: torch.Tensor,
        sin_Q: torch.Tensor | None,
        cos_Q: torch.Tensor | None,
        sin_K: torch.Tensor | None,
        cos_K: torch.Tensor | None,
        l: int,
    ) -> torch.Tensor:
        p = self.config.length_is
        iq = 0 if self.config.share_queries else l-1
        ik = 0 if self.config.share_keys else l
        W = self.get_buffer("weight_signature")
        dqk = self.config.d_query_key
        if cos_Q is not None and sin_Q is not None and l > 0:
            indices = torch.arange(
                W.size(-3)-(4*(l-1)*dqk), W.size(-3)-(4*(l-1)*dqk+dqk*4), -4
            )
            x = x * torch.prod(
                cos_Q[..., iq, :, :, :].pow(W.index_select(-3, indices-4)),
                dim=-3,
            )
            x = x * torch.prod(
                sin_Q[..., iq, :, :, :].pow(W.index_select(-3, indices-3)),
                dim=-3,
            )
        if cos_K is not None and sin_K is not None and l < p:
            indices = torch.arange(
                W.size(-3)-(4*l*dqk), W.size(-3)-(4*l*dqk+dqk*4), -4
            )
            x = x * torch.prod(
                cos_K[..., iq, :, :, :].pow(W.index_select(-3, indices-2)),
                dim=-3,
            )
            x = x * torch.prod(
                sin_K[..., iq, :, :, :].pow(W.index_select(-3, indices-1)),
                dim=-3,
            )
        return x
