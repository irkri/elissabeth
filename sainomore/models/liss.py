__all__ = ["LISS", "Elissabeth", "ElissabethConfig"]

import warnings
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
from torch import nn

from ..hooks import HookCollection
from .base import HookedModule, ModelConfig, SAINoMoreModule
from .positional import RoPE
from .transformer import LearnablePositionalEmbedding, PositionalEncoding


@dataclass
class ElissabethConfig(ModelConfig):
    sum_normalization: Optional[Literal["same", "independent"]] = "independent"
    n_is: int = 1
    length_is: int = 2
    d_values: int = None  # type: ignore
    values_2D: bool = True

    share_queries: bool = False
    share_keys: bool = False
    share_values: bool = False

    d_pe: int = 32
    positional_bias: bool = True
    positional_bias_values: bool = False

    distance_weighting: bool = False

    denominator_is: bool = False

    weighting: Literal["cos", "exp"] | None = "exp"

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
        if config.weighting is not None:
            self.W_Q = nn.Parameter(
                torch.empty((
                    config.n_is,
                    1 if config.share_queries else config.length_is,
                    config.d_hidden,
                ))
            )
            self.b_Q = nn.Parameter(
                torch.empty((
                    config.n_is,
                    1 if config.share_queries else config.length_is,
                ))
            )
            nn.init.xavier_normal_(self.W_Q)
            nn.init.zeros_(self.b_Q)
            self.W_K = nn.Parameter(
                torch.empty((
                    config.n_is,
                    1 if config.share_queries else config.length_is,
                    config.d_hidden,
                ))
            )
            self.b_K = nn.Parameter(
                torch.empty((
                    config.n_is,
                    1 if config.share_queries else config.length_is,
                ))
            )
            nn.init.xavier_normal_(self.W_K)
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
        self.b_V = nn.Parameter(
            torch.empty((
                config.n_is,
                1 if config.share_values else config.length_is,
                config.d_values,
                config.d_values if config.values_2D else 1,
            ))
        )
        nn.init.xavier_normal_(self.W_V)
        nn.init.zeros_(self.b_V)
        self.E_V = None
        if config.positional_bias_values:
            self.E_V = nn.Parameter(
                torch.empty((
                    1 if config.share_values else config.length_is,
                    config.d_hidden,
                    1,
                ))
            )
            nn.init.xavier_normal_(self.E_V)

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
            factor_norm = torch.empty((1, config.context_length, 1, 1, 1))
            factor_norm[0, :, 0, 0, 0] = torch.arange(
                1, config.context_length + 1
            )
            self.register_buffer("norm", factor_norm)
            self.beta = nn.Parameter(torch.empty((
                1 if config.sum_normalization == "same" else config.length_is,
            )))
            nn.init.constant_(self.beta, 5.40988)

        self.E_K = None
        if config.positional_bias or config.positional_bias_values:
            # position = torch.arange(config.context_length).unsqueeze(-1)
            # div_term = torch.exp(
            #     torch.arange(0, config.d_pe, 2)
            #     * (-np.log(config.context_length) / config.d_pe)
            # )
            # pe = torch.zeros(config.context_length, config.d_pe)
            # pe[:, 0::2] = torch.sin(position * div_term * np.pi / 2)
            # if config.d_pe % 2 != 0:
            #     pe[:, 1::2] = torch.cos(position * div_term[:-1] * np.pi / 2)
            # else:
            #     pe[:, 1::2] = torch.cos(position * div_term * np.pi / 2)
            # self.register_buffer("P", pe)

            # self.E_K = nn.Parameter(
            #     torch.empty((
            #         config.n_is,
            #         1 if config.share_keys else config.length_is,
            #         config.d_pe,
            #     ))
            # )
            # nn.init.xavier_normal_(self.E_K)
            self.rope = RoPE(T=config.context_length, d=config.d_hidden)

        if config.weighting == "cos":
            self.mu = nn.Parameter(
                torch.empty((config.length_is, config.n_is))
            )
            nn.init.ones_(self.mu)
            self.nu = nn.Parameter(
                torch.empty((config.length_is, config.n_is))
            )
            nn.init.zeros_(self.nu)

        self.hooks = HookCollection("Q", "K", "V")
        self.hooks.add_hooks(
            [f"iss.{i}" for i in range(1, config.length_is+1)]
        )
        self.hooks.add_hooks(
            [f"weighting.{i}" for i in range(config.length_is)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        Q = K = sin_K = cos_K = sin_Q = cos_Q = None
        if self.W_Q is not None and self.W_K is not None:
            Q = torch.einsum('hld,btd->bthl', self.W_Q, self.rope(x)) + self.b_Q
            # Q = torch.sigmoid(Q)
            K = torch.einsum('hld,btd->bthl', self.W_K, self.rope(x)) + self.b_K
            # K = torch.sigmoid(K)
            if self.alpha is not None and T > 1:
                rel_pos = (
                    torch.sigmoid(self.alpha) * self.get_buffer("T")[:T, :, :]
                )
                Q = Q - rel_pos
                K = K - rel_pos
            if self.E_K is not None:
                K = K + torch.sigmoid(torch.einsum('hld,td->thl', self.E_K,
                    self.get_buffer("P")[:T, :]
                ))
            self.hooks("Q", Q)
            self.hooks("K", K)
            Q = Q.unsqueeze(-1).unsqueeze(-1)
            K = K.unsqueeze(-1).unsqueeze(-1)
            if self.config.weighting == "cos":
                sin_Q = torch.sin(Q)**2
                cos_Q = torch.cos(Q)**2
                sin_K = torch.sin(K)**2
                cos_K = torch.cos(K)**2

        V = torch.einsum('hldvw,btd->bthlvw', self.W_V, x) + self.b_V
        # if self.E_V is not None:
        #     V = V + torch.relu(torch.einsum('ldh,btd->btlh', self.E_V,
        #         self.get_buffer("P")[:, :T, :]
        #     ))
        self.hooks("V", V)

        p = self.config.length_is

        result = V[:, :, :, 0, :, :]
        result = self._weight_is(result, Q, K, sin_Q, cos_Q, sin_K, cos_K, 0)
        result = torch.cumsum(result, dim=1)

        if self.beta is not None:
            result /= (
                (0.25*torch.tanh(self.beta[0])+0.75001)**(
                    torch.log10(self.get_buffer("norm")[:, :T, :, :, :])
                ) * self.get_buffer("norm")[:, :T, :, :, :]
            )

        denom = None
        if self.config.denominator_is:
            denom = torch.ones_like(V[:, :, :, 0, :, :], device=V.device)
            denom = self._weight_is(denom, Q, K, sin_Q, cos_Q, sin_K, cos_K, 0)
            denom = torch.cumsum(denom, dim=1)

        for l in range(1, p):
            self._hook_weighting(Q, K, V, sin_Q, cos_Q, sin_K, cos_K, l)

            result = nn.functional.pad(
                result[:, :-1, :, :, :],
                (0, 0, 0, 0, 0, 0, 1, 0)
            )
            iv = 0 if self.config.share_values else l
            if self.config.values_2D:
                result = V[:, :, :, iv, :, :] @ result
            else:
                result = V[:, :, :, iv, :, :] * result
            result = self._weight_is(
                result, Q, K, sin_Q, cos_Q, sin_K, cos_K, l
            )
            result = torch.cumsum(result, dim=1)

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

            if denom is not None:
                denom = self._weight_is(
                    denom, Q, K, sin_Q, cos_Q, sin_K, cos_K, l
                )
                denom = torch.cumsum(denom, dim=1)

        self._hook_weighting(Q, K, V, sin_Q, cos_Q, sin_K, cos_K, p)

        result = self._weight_is(result, Q, K, sin_Q, cos_Q, sin_K, cos_K, p)

        if denom is not None:
            denom = self._weight_is(denom, Q, K, sin_Q, cos_Q, sin_K, cos_K, p)
            result = result / denom

        result = self.hooks(f"iss.{p}", result)
        result = torch.einsum("hvwd,bthvw->btd", self.W_O, result)

        return result

    def _weight_is(
        self,
        x: torch.Tensor,
        Q: torch.Tensor | None,
        K: torch.Tensor | None,
        sin_Q: torch.Tensor | None,
        cos_Q: torch.Tensor | None,
        sin_K: torch.Tensor | None,
        cos_K: torch.Tensor | None,
        l: int,
    ) -> torch.Tensor:
        p = self.config.length_is
        iq = 0 if self.config.share_queries else l-1
        ik = 0 if self.config.share_keys else l
        if l > 0 and l < p and self.hooks.get(f"iss.{l}").is_attached():
            temp = x
            if (self.config.weighting == "cos"
                    and cos_Q is not None and sin_Q is not None):
                temp = (temp
                    * cos_Q[:, :, iq, :]**(torch.sigmoid(self.nu[l-1])/2)
                    * sin_Q[:, :, iq, :]**(torch.sigmoid(self.mu[l-1])/2)
                )
            elif self.config.weighting == "exp" and Q is not None:
                temp = temp * torch.exp(Q[:, :, :, iq, :, :])
            self.hooks(f"iss.{l}", temp)

        if (self.config.weighting == "exp"
                and Q is not None and K is not None):
            x = x * torch.exp(
                (0 if l == 0 else Q[:, :, :, iq, :, :])
                - (0 if l == p else K[:, :, :, ik, :, :])  # type: ignore
            )

        elif (self.config.weighting == "cos"
                and cos_Q is not None and cos_K is not None
                and sin_K is not None and sin_Q is not None):
            if l > 0:
                x *= (
                    sin_Q[:, :, iq, :]**(torch.sigmoid(self.mu[l-1])/2)
                    * cos_Q[:, :, iq, :]**(torch.sigmoid(self.nu[l-1])/2)
                )
            elif l < p:
                x *= (
                    sin_K[:, :, ik, :]**(torch.sigmoid(self.mu[l])/2)
                    * cos_K[:, :, ik, :]**(torch.sigmoid(self.nu[l])/2)
                )
        return x

    def _hook_weighting(
        self,
        Q: torch.Tensor | None,
        K: torch.Tensor | None,
        V: torch.Tensor,
        sin_Q: torch.Tensor | None,
        cos_Q: torch.Tensor | None,
        sin_K: torch.Tensor | None,
        cos_K: torch.Tensor | None,
        l: int,
    ) -> None:
        if not self.hooks.get(f"weighting.{l-1}").is_attached():
            return
        B = V.size(0)
        T = V.size(1)
        D = self.config.n_is
        matrix = torch.ones((B, D, T, T), device=V.device)
        if self.config.weighting == "exp" and Q is not None and K is not None:
            iq = 0 if self.config.share_queries else l-1
            ik = 0 if self.config.share_keys else l-1
            matrix = matrix * torch.exp(
                Q[:, :, :, iq, 0, 0].repeat(1, T, 1).reshape(B, D, T, T)
                - (K[:, :, :, ik, 0, 0].repeat(1, T, 1).reshape(B, D, T, T)
                    .transpose(1, 2))
            )
        elif (self.config.weighting == "cos"
                and cos_Q is not None and cos_K is not None
                and sin_K is not None and sin_Q is not None):
            iq = 0 if self.config.share_queries else l-1
            ik = 0 if self.config.share_keys else l-1
            matrix = matrix * (
                (cos_Q[:, :, iq, :].repeat(1, T, 1).reshape(B, T, T, D)
                * cos_K[:, :, ik, :].repeat(1, T, 1).reshape(B, T, T, D)
                    .transpose(1, 2))**(torch.sigmoid(self.nu[l-1, :])/2)
                +
                (sin_Q[:, :, iq, :].repeat(1, T, 1).reshape(B, T, T, D)
                * sin_K[:, :, ik, :].repeat(1, T, 1).reshape(B, T, T, D)
                    .transpose(1, 2))**(torch.sigmoid(self.mu[l-1, :])/2)
            )
        self.hooks(f"weighting.{l-1}", matrix)


class Elissabeth(SAINoMoreModule):
    """Extended Learnable Iterated Sums Signature Architecture"""

    config: ElissabethConfig

    def __init__(self, config: ElissabethConfig) -> None:
        super().__init__(config)
        if config.input_type == "token":
            self.embedding = nn.Embedding(
                config.input_vocab_size, config.d_hidden
            )
        if (self.config.positional_encoding is None
                and not config.positional_bias and config.length_is == 1):
            warnings.warn(
                "No positional encoding and ISS length 1. "
                "Elissabeth will be permutation invariant.",
                RuntimeWarning,
            )
        if self.config.positional_encoding == "learnable":
            self.pos_enc = LearnablePositionalEmbedding(config)
        elif self.config.positional_encoding == "sinusoidal":
            self.pos_enc = PositionalEncoding(config)

        self.layers = nn.ModuleList([
            LISS(config) for _ in range(config.n_layers)
        ])
        if config.layer_norm:
            self.layernorms = nn.ModuleList([
                nn.LayerNorm(config.d_hidden) for _ in range(config.n_layers+1)
            ])
        self.unembedding = nn.Linear(
            config.d_hidden, config.output_vocab_size,
            bias=config.bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, V) -> (B, O, T)
        B: batch size
        T: context length
        V: vocabulary size
        O: output dimension
        """
        if self.config.input_type == "token":
            x = self.embedding(x)
        if self.config.positional_encoding is not None:
            x = self.pos_enc(x)

        for i in range(len(self.layers)):
            y = x
            if self.config.layer_norm:
                y = self.layernorms[i](x)
            x = x + self.layers[i](y)

        logits = self.unembedding(x)
        return torch.swapaxes(logits, 1, 2)
