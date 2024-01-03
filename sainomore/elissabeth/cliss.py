__all__ = ["CLISS"]

import itertools
import warnings

import torch
from torch import nn

from ..base import HookedModule
from ..hooks import HookCollection
from ..positional import RoPE
from .config import ElissabethConfig


class CLISS(HookedModule):
    "Learnable Iterated Sums Signature"

    config: ElissabethConfig

    def __init__(self, config: ElissabethConfig) -> None:
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
        if config.weighting is not None:
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
                    1 if config.share_queries else config.length_is,
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
                        1 if config.share_queries else config.length_is,
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

        self._weight_signature = self._get_weight_signature()

        self.hooks = HookCollection("Q", "K", "V")

    def _get_weight_signature(self) -> torch.Tensor:
        p = self.config.length_is + 1
        exponent = 2
        trig_id = []
        trig_exp = [exponent, 0]
        trig_coeff = 1
        for k in range(exponent+1):
            trig_id.append(f"{trig_coeff}{trig_exp[0]}{trig_exp[1]}")
            trig_exp[0] -= 1
            trig_exp[1] += 1
            trig_coeff = trig_coeff * (exponent - k) // (k + 1)
        weightings = torch.zeros(
            ((exponent+1)**(p-1), 1, 1, 1, 1, 1, 4*(p-1)+1),
            dtype=torch.int32,
        )
        weightings[:, ..., 0] = 1
        for c, comb in enumerate(itertools.product(trig_id, repeat=p-1)):
            for i in range(p-1):
                weightings[c, ..., 0] *= int(comb[i][0])
                weightings[c, ..., 4*i+1] += int(comb[i][1])
                weightings[c, ..., 4*i+3] += int(comb[i][1])
                weightings[c, ..., 4*i+2] += int(comb[i][2])
                weightings[c, ..., 4*i+4] += int(comb[i][2])
        return weightings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        sin_K = cos_K = sin_Q = cos_Q = None
        if self.pe is not None:
            x_enc = self.pe(x)
        if self.W_Q is not None and self.W_K is not None:
            Q = torch.einsum(
                'hld,btd->bthl',
                self.W_Q,
                x_enc if self.config.pe_query_key else x,  # type: ignore
            )
            if self.b_Q is not None:
                Q = Q + self.b_Q
            K = torch.einsum(
                'hld,btd->bthl',
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
        result = (result * self._weight_signature[..., 0]).sum(dim=0)

        if self.hooks.get(f"iss.{p}").is_attached():
            self.hooks(f"iss.{p}", result)
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
        W = self._weight_signature
        if cos_Q is not None and sin_Q is not None and l > 0:
            x = (x * cos_Q[..., iq, :, :]**W[..., 4*(l-1)+1]
                   * sin_Q[..., iq, :, :]**W[..., 4*(l-1)+2])
        if cos_K is not None and sin_K is not None and l < p:
            x = (x * cos_K[..., ik, :, :]**W[..., 4*(l-1)+3]
                   * sin_K[..., ik, :, :]**W[..., 4*(l-1)+4])
        return x
