__all__ = ["LISS", "Elissabeth", "ElissabethConfig"]

import warnings
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from ..hooks import HookCollection
from .base import HookedModule, ModelConfig, SAINoMoreModule
from .transformer import LearnablePositionalEmbedding, PositionalEncoding


@dataclass
class ElissabethConfig(ModelConfig):
    normalize_is: bool = True
    length_is: int = 2
    n_is: int = None  # type: ignore

    single_query_key: bool = False
    share_queries: bool = False
    share_keys: bool = False
    positional_bias: bool = True

    share_values: bool = False
    positional_bias_values: bool = False

    distance_weighting: bool = False

    denominator_is: bool = False

    weighting: Literal["cos", "exp"] | None = "exp"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_is is None:
            self.n_is = self.d_hidden


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

        self.W_Q = self.W_K = self.b_K = self.b_Q = None
        if config.weighting is not None:
            self.W_Q = nn.Parameter(
                torch.empty((
                    1 if config.share_queries else config.length_is,
                    config.d_hidden,
                    1 if config.single_query_key else config.n_is,
                ))
            )
            nn.init.xavier_uniform_(self.W_Q)
            self.W_K = nn.Parameter(
                torch.empty((
                    1 if config.share_keys else config.length_is,
                    config.d_hidden,
                    1 if config.single_query_key else config.n_is,
                ))
            )
            nn.init.xavier_uniform_(self.W_K)
            if config.positional_bias:
                self.b_Q = nn.Parameter(
                    torch.empty((
                        1, 1,
                        1 if config.share_queries else config.length_is,
                        1 if config.single_query_key else config.n_is,
                    ))
                )
                nn.init.xavier_uniform_(self.b_Q)
                self.b_K = nn.Parameter(
                    torch.empty((
                        1, 1,
                        1 if config.share_keys else config.length_is,
                        1 if config.single_query_key else config.n_is,
                    ))
                )
                nn.init.xavier_uniform_(self.b_K)

        self.W_V = nn.Parameter(
            torch.empty((
                1 if config.share_values else config.length_is,
                config.d_hidden,
                config.n_is,
            ))
        )
        nn.init.xavier_uniform_(self.W_V)
        self.b_V = None
        if config.positional_bias_values:
            self.b_V = nn.Parameter(
                torch.empty((
                    1, 1,
                    1 if config.share_values else config.length_is,
                    config.n_is,
                ))
            )
            nn.init.xavier_uniform_(self.b_V)

        self.W_O = nn.Parameter(torch.empty((config.n_is, config.d_hidden)))
        nn.init.xavier_uniform_(self.W_O)

        self.alpha = None
        if config.distance_weighting:
            self.alpha = nn.Parameter(
                torch.empty((config.length_is, config.n_is))
            )
            nn.init.ones_(self.alpha)
        if (config.distance_weighting or config.normalize_is
                or config.positional_bias or config.positional_bias_values):
            indices = torch.empty((1, config.context_length, 1))
            indices[0, :, 0] = torch.linspace(
                1/config.context_length, 1, config.context_length
            )
            self.register_buffer("T", indices)

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
        self.hooks.add_hooks([f"iss.{i}" for i in range(config.length_is)])
        self.hooks.add_hooks(
            [f"weighting.{i}" for i in range(config.length_is)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = K = sin_K = cos_K = sin_Q = cos_Q = None
        if self.W_Q is not None and self.W_K is not None:
            Q = torch.einsum('ldh,btd->btlh', self.W_Q, x)
            K = torch.einsum('ldh,btd->btlh', self.W_K, x)
            if self.b_Q is not None and self.b_K is not None:
                Q = Q + (self.b_Q * torch.unsqueeze(self.get_buffer("T"), -1))
                K = K + (self.b_K * torch.unsqueeze(self.get_buffer("T"), -1))
            self.hooks("Q", Q)
            self.hooks("K", K)
            if self.config.weighting == "cos":
                sin_Q = torch.sin(Q)**2
                cos_Q = torch.cos(Q)**2
                sin_K = torch.sin(K)**2
                cos_K = torch.cos(K)**2

        V = torch.einsum('ldh,btd->btlh', self.W_V, x)
        if self.b_V is not None:
            V = V + (self.b_V * torch.unsqueeze(self.get_buffer("T"), -1))
        self.hooks("V", V)

        p = self.config.length_is

        result = V[:, :, 0, :]
        result = self._weight_is(result, Q, K, sin_Q, cos_Q, sin_K, cos_K, 0)
        result = torch.cumsum(result, dim=1)

        denom = None
        if self.config.denominator_is:
            denom = torch.ones_like(V[:, :, 0, :], device=V.device)
            denom = self._weight_is(denom, Q, K, sin_Q, cos_Q, sin_K, cos_K, 0)
            denom = torch.cumsum(denom, dim=1)

        for l in range(1, p):
            if self.config.normalize_is:
                result = result * self.get_buffer("T").flip(1)

            if self.hooks.get(f"iss.{l-1}").is_attached():
                iq = 0 if self.config.share_queries else l-1
                temp = result
                if self.alpha is not None:
                    temp = temp * torch.exp(
                        self.alpha[l, :] * self.get_buffer("T")
                    )
                if (self.config.weighting == "cos"
                        and cos_Q is not None and sin_Q is not None):
                    temp = (temp
                        * cos_Q[:, :, iq, :]**torch.sigmoid(self.nu[l-1, :])
                        * sin_Q[:, :, iq, :]**torch.sigmoid(self.mu[l-1, :])
                    )
                elif self.config.weighting == "exp" and Q is not None:
                    temp = temp * torch.exp(Q[:, :, iq, :])
                self.hooks(f"iss.{l-1}", temp)

            self._hook_weighting(Q, K, V, sin_Q, cos_Q, sin_K, cos_K, l)

            result = nn.functional.pad(
                torch.roll(result, 1, 1)[:, 1:, :], (0, 0, 1, 0),
            )
            result = result * V[:, :, 0 if self.config.share_values else l, :]
            result = self._weight_is(
                result, Q, K, sin_Q, cos_Q, sin_K, cos_K, l
            )
            result = torch.cumsum(result, dim=1)

            if denom is not None:
                denom = self._weight_is(
                    denom, Q, K, sin_Q, cos_Q, sin_K, cos_K, l
                )
                denom = torch.cumsum(denom, dim=1)

        self._hook_weighting(Q, K, V, sin_Q, cos_Q, sin_K, cos_K, p)

        iq = 0 if self.config.share_queries else p-1
        if self.config.normalize_is:
            result = result * self.get_buffer("T").flip(1)
        result = self._weight_is(result, Q, K, sin_Q, cos_Q, sin_K, cos_K, p)

        if denom is not None:
            denom = self._weight_is(denom, Q, K, sin_Q, cos_Q, sin_K, cos_K, p)
            result = result / denom

        result = self.hooks(f"iss.{p-1}", result)
        result = torch.einsum("hd,bth->btd", self.W_O, result)

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
        if l == 0:
            if self.alpha is not None:
                x = x * torch.exp(self.alpha[0, :] * self._indices)
            if self.config.weighting == "exp" and K is not None:
                x = x * torch.exp(-K[:, :, 0, :])
            elif (self.config.weighting == "cos"
                    and sin_K is not None and cos_K is not None):
                x = (x
                    * sin_K[:, :, 0, :]**(torch.sigmoid(self.mu[0, :])/2)
                    * cos_K[:, :, 0, :]**(torch.sigmoid(self.nu[0, :])/2)
                )
        elif l == (p := self.config.length_is):
            iq = 0 if self.config.share_queries else p-1
            if self.alpha is not None:
                x = x * torch.exp(-self.alpha[p-1, :] * self._indices)
            if self.config.weighting == "exp" and Q is not None:
                x = x * torch.exp(Q[:, :, iq, :])
            elif (self.config.weighting == "cos"
                    and cos_Q is not None and sin_Q is not None):
                x = (x
                    * sin_Q[:, :, iq, :]**(torch.sigmoid(self.mu[p-1, :])/2)
                    * cos_Q[:, :, iq, :]**(torch.sigmoid(self.nu[p-1, :])/2)
                )
        else:
            iq = 0 if self.config.share_queries else l-1
            ik = 0 if self.config.share_keys else l
            if self.alpha is not None:
                x = x * torch.exp(
                    (self.alpha[l, :] - self.alpha[l-1, :]) * self._indices
                )
            if (self.config.weighting == "exp"
                    and Q is not None and K is not None):
                x = x * torch.exp(Q[:, :, iq, :] - K[:, :, ik, :])
            elif (self.config.weighting == "cos"
                    and cos_Q is not None and cos_K is not None
                    and sin_K is not None and sin_Q is not None):
                x = (x
                    * cos_Q[:, :, iq, :]**(torch.sigmoid(self.nu[l-1, :])/2)
                    * cos_K[:, :, ik, :]**(torch.sigmoid(self.nu[l, :])/2)
                    * sin_Q[:, :, iq, :]**(torch.sigmoid(self.mu[l-1, :])/2)
                    * sin_K[:, :, ik, :]**(torch.sigmoid(self.mu[l, :])/2)
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
        B = V.shape[0]
        T = self.config.context_length
        D = self.config.n_is if not self.config.single_query_key else 1
        matrix = torch.ones((B, T, T, D), device=V.device)
        if self.config.weighting == "exp" and Q is not None and K is not None:
            iq = 0 if self.config.share_queries else l-1
            ik = 0 if self.config.share_keys else l-1
            matrix = matrix * torch.exp(
                Q[:, :, iq, :].repeat(1, T, 1).reshape(B, T, T, D)
                - K[:, :, ik, :].repeat(1, T, 1).reshape(B, T, T, D)
                    .transpose(1, 2)
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
        if self.alpha is not None:
            T_matrix = self.get_buffer("T").repeat(1, T, 1).reshape(1, T, T, 1)
            matrix = matrix * torch.exp(self.alpha[l-1, :] * (
                T_matrix - T_matrix.transpose(1, 2)
            ))
        self.hooks(f"weighting.{l-1}", matrix)


class Elissabeth(SAINoMoreModule):
    """Extended Learnable Iterated Sums Signature Architecture"""

    config: ElissabethConfig

    def __init__(self, config: ElissabethConfig) -> None:
        super().__init__(config)
        self.embedding = nn.Embedding(config.input_vocab_size, config.d_hidden)
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
        self.normalize = config.layer_norm
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
        x = self.embedding(x)
        if self.config.positional_encoding is not None:
            x = self.pos_enc(x)

        for i in range(len(self.layers)):
            y = x
            if self.normalize:
                y = self.layernorms[i](x)
            x = x + self.layers[i](y)

        logits = self.unembedding(x)
        return torch.swapaxes(logits, 1, 2)
