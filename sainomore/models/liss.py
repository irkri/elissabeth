__all__ = ["LISS", "Elissabeth", "ElissabethConfig"]

import warnings
from dataclasses import dataclass

import torch
from torch import nn

from ..hooks import HookCollection
from .base import HookedModule, ModelConfig, SAINoMoreModule
from .transformer import PositionalEmbedding


@dataclass
class ElissabethConfig(ModelConfig):
    normalize_is: bool = True
    length_is: int = 2
    n_is: int = None  # type: ignore

    query_key_weighting: bool = True
    single_query_key: bool = False
    share_queries: bool = False
    share_keys: bool = False

    share_values: bool = False

    distance_weighting: bool = True
    positional_encoding: bool = False

    denominator_is: bool = False

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

        self.W_Q = self.W_K = None
        if config.query_key_weighting:
            self.W_Q = nn.Parameter(
                torch.empty((
                    config.length_is if not config.share_queries else 1,
                    config.d_hidden,
                    1 if config.single_query_key else config.n_is,
                ))
            )
            self.W_K = nn.Parameter(
                torch.empty((
                    config.length_is if not config.share_keys else 1,
                    config.d_hidden,
                    1 if config.single_query_key else config.n_is,
                ))
            )
        self.W_V = nn.Parameter(
            torch.empty((
                config.length_is if not config.share_values else 1,
                config.d_hidden,
                config.n_is,
            ))
        )
        self.W_O = nn.Parameter(
            torch.empty((config.n_is, config.d_hidden))
        )

        self.alpha = None
        self._indices = None
        if config.distance_weighting:
            self.alpha = nn.Parameter(
                torch.empty((config.length_is, config.n_is))
            )
            nn.init.ones_(self.alpha)
        if config.distance_weighting or config.normalize_is:
            self._indices = nn.Parameter(
                torch.empty((1, config.context_length, 1)),
                requires_grad=False,
            )
            self._indices[0, :, 0] = torch.linspace(
                1/config.context_length, 1, config.context_length
            )

        if self.W_Q is not None and self.W_K is not None:
            nn.init.xavier_uniform_(self.W_Q)
            nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_V)
        nn.init.xavier_uniform_(self.W_O)

        self.hooks = HookCollection("Q", "K", "V")
        self.hooks.add_hooks([f"iss.{i}" for i in range(config.length_is)])
        self.hooks.add_hooks(
            [f"weighting.{i}" for i in range(config.length_is)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = K = None
        if self.W_Q is not None and self.W_K is not None:
            Q = self.hooks("Q",
                torch.sigmoid(torch.einsum('btd,ldh->btlh', x, self.W_Q))
            )
            K = self.hooks("K",
                torch.sigmoid(torch.einsum('btd,ldh->btlh', x, self.W_K))
            )
        V = self.hooks("V", torch.einsum('btd,ldh->btlh', x, self.W_V))

        p = self.config.length_is

        result = V[:, :, 0, :]
        if self.alpha is not None:
            result = result * torch.exp(self.alpha[0, :] * self._indices)
        if K is not None:
            result = result * torch.exp(-K[:, :, 0, :])
        result = torch.cumsum(result, dim=1)

        denom = None
        if self.config.denominator_is:
            denom = torch.ones_like(V[:, :, 0, :], device=V.device)
            if self.alpha is not None:
                denom = denom * torch.exp(self.alpha[0, :] * self._indices)
            if K is not None:
                denom = denom * torch.exp(-K[:, :, 0, :])
            denom = torch.cumsum(denom, dim=1)

        for l in range(1, p):
            iq = 0 if self.config.share_queries else l-1
            ik = 0 if self.config.share_keys else l
            iv = 0 if self.config.share_values else l
            if self._indices is not None and self.config.normalize_is:
                result = result * self._indices.flip(1)

            if self.hooks.get(f"iss.{l-1}").is_attached():
                self.hooks(f"iss.{l-1}",
                    result
                    * (1 if Q is None else torch.exp(Q[:, :, iq, :]))
                    * (1 if self.alpha is None else torch.exp(
                        self.alpha[l, :] * self._indices
                    ))
                )
            self._hook_weighting(Q, K, V, l)

            result = nn.functional.pad(
                torch.roll(result, 1, 1)[:, 1:, :], (0, 0, 1, 0),
            )
            result = result * V [:, :, iv, :]
            if self.alpha is not None:
                result = result * torch.exp(
                    (self.alpha[l, :] - self.alpha[l-1, :]) * self._indices
                )
            if Q is not None and K is not None:
                result = result * torch.exp(Q[:, :, iq, :] - K[:, :, ik, :])
            result = torch.cumsum(result, dim=1)

            if denom is not None:
                if self.alpha is not None:
                    denom = denom * torch.exp(
                        (self.alpha[l, :] - self.alpha[l-1, :]) * self._indices
                    )
                if Q is not None and K is not None:
                    denom = denom * torch.exp(Q[:, :, iq, :] - K[:, :, ik, :])
                denom = torch.cumsum(denom, dim=1)

        self._hook_weighting(Q, K, V, p)
        iq = 0 if self.config.share_queries else p-1
        if self._indices is not None and self.config.normalize_is:
            result = result * self._indices.flip(1)
        if self.alpha is not None:
            result = result * torch.exp(-self.alpha[p-1, :] * self._indices)
        if Q is not None:
            result = result * torch.exp(Q[:, :, iq, :])

        if denom is not None:
            if self.alpha is not None:
                denom = denom * torch.exp(-self.alpha[p-1, :] * self._indices)
            if Q is not None:
                denom = denom * torch.exp(Q[:, :, iq, :])
            result = result / denom

        result = self.hooks(f"iss.{p-1}", result)
        result = torch.einsum("bth,hd->btd", result, self.W_O)

        return result

    def _hook_weighting(
        self,
        Q: torch.Tensor | None,
        K: torch.Tensor | None,
        V: torch.Tensor,
        l: int,
    ) -> None:
        if not self.hooks.get(f"weighting.{l-1}").is_attached():
            return
        B = V.shape[0]
        T = self.config.context_length
        D = self.config.n_is
        matrix = torch.zeros((B, T, T, D), device=V.device)
        if Q is not None and K is not None:
            iq = 0 if self.config.share_queries else l-1
            ik = 0 if self.config.share_keys else l-1
            matrix = matrix + (
                Q[:, :, iq, :].repeat(1, T, 1).reshape(B, T, T, D)
                - K[:, :, ik, :].repeat(1, T, 1).reshape(B, T, T, D)
                    .transpose(1, 2)
            )
        if self.alpha is not None and self._indices is not None:
            index_matrix = self._indices.repeat(1, T, 1).reshape(1, T, T, 1)
            matrix = matrix + self.alpha[l-1, :] * (
                index_matrix - index_matrix.transpose(1, 2)
            )
        self.hooks(f"weighting.{l-1}", torch.exp(matrix))


class CISS(HookedModule):
    "Cosine-weighted Learnable Iterated Sums Signature"

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
        if config.query_key_weighting:
            self.W_Q = nn.Parameter(
                torch.empty((
                    config.length_is if not config.share_queries else 1,
                    config.d_hidden,
                    1 if config.single_query_key else config.n_is,
                ))
            )
            self.W_K = nn.Parameter(
                torch.empty((
                    config.length_is if not config.share_keys else 1,
                    config.d_hidden,
                    1 if config.single_query_key else config.n_is,
                ))
            )
        self.W_V = nn.Parameter(
            torch.empty((
                config.length_is if not config.share_values else 1,
                config.d_hidden,
                config.n_is,
            ))
        )
        self.W_O = nn.Parameter(
            torch.empty((config.n_is, config.d_hidden))
        )

        self.alpha = None
        self._indices = None
        if config.distance_weighting:
            self.alpha = nn.Parameter(
                torch.empty((config.length_is, config.n_is))
            )
            nn.init.ones_(self.alpha)
        if config.distance_weighting or config.normalize_is:
            self._indices = nn.Parameter(
                torch.empty((1, config.context_length, 1)),
                requires_grad=False,
            )
            self._indices[0, :, 0] = torch.linspace(
                1/config.context_length, 1, config.context_length
            )

        self.mu = nn.Parameter(
            torch.empty((config.length_is, config.n_is))
        )
        nn.init.ones_(self.mu)
        self.nu = nn.Parameter(
            torch.empty((config.length_is, config.n_is))
        )
        nn.init.zeros_(self.nu)

        if self.W_Q is not None and self.W_K is not None:
            nn.init.xavier_uniform_(self.W_Q)
            nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_V)
        nn.init.xavier_uniform_(self.W_O)

        self.hooks = HookCollection("Q", "K", "V")
        self.hooks.add_hooks([f"iss.{i}" for i in range(config.length_is)])
        self.hooks.add_hooks(
            [f"weighting.{i}" for i in range(config.length_is)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = K = sin_K = cos_K = sin_Q = cos_Q = None
        if self.W_Q is not None and self.W_K is not None:
            Q = self.hooks("Q",
                torch.sigmoid(torch.einsum('btd,ldh->btlh', x, self.W_Q))
            )
            K = self.hooks("K",
                torch.sigmoid(torch.einsum('btd,ldh->btlh', x, self.W_K))
            )
            sin_Q = torch.sin(Q)**2
            cos_Q = torch.cos(Q)**2
            sin_K = torch.sin(K)**2
            cos_K = torch.cos(K)**2
        V = self.hooks("V", torch.einsum('btd,ldh->btlh', x, self.W_V))

        p = self.config.length_is

        result = V[:, :, 0, :]
        if self.alpha is not None:
            result = result * torch.exp(self.alpha[0, :] * self._indices)
        if sin_K is not None and cos_K is not None:
            result = (result
                * sin_K[:, :, 0, :]**torch.sigmoid(self.mu[0, :])
                * cos_K[:, :, 0, :]**torch.sigmoid(self.nu[0, :])
            )
        result = torch.cumsum(result, dim=1)

        denom = None
        if self.config.denominator_is:
            denom = torch.ones_like(V[:, :, 0, :], device=V.device)
            if self.alpha is not None:
                denom = denom * torch.exp(self.alpha[0, :] * self._indices)
            if sin_K is not None and cos_K is not None:
                denom = (denom
                    * sin_K[:, :, 0, :]**torch.sigmoid(self.mu[0, :])
                    * cos_K[:, :, 0, :]**torch.sigmoid(self.nu[0, :])
                )
            denom = torch.cumsum(denom, dim=1)

        for l in range(1, p):
            iq = 0 if self.config.share_queries else l-1
            ik = 0 if self.config.share_keys else l
            iv = 0 if self.config.share_values else l

            if self._indices is not None and self.config.normalize_is:
                result = result * self._indices.flip(1)

            if self.hooks.get(f"iss.{l-1}").is_attached():
                self.hooks(f"iss.{l-1}",
                    result
                    * (1 if sin_Q is None or cos_Q is None else (
                        cos_Q[:, :, iq, :]**torch.sigmoid(self.nu[l-1, :])
                        * sin_Q[:, :, iq, :]**torch.sigmoid(self.mu[l-1, :])
                    ))
                    * (1 if self.alpha is None else torch.exp(
                        self.alpha[l, :] * self._indices
                    ))
                )
            self._hook_weighting(Q, K, V, l)

            result = nn.functional.pad(
                torch.roll(result, 1, 1)[:, 1:, :], (0, 0, 1, 0),
            )
            result = result * V [:, :, iv, :]
            if self.alpha is not None:
                result = result * torch.exp(
                    (self.alpha[l, :] - self.alpha[l-1, :]) * self._indices
                )
            if (cos_Q is not None and cos_K is not None
                    and sin_K is not None and sin_Q is not None):
                result = (result
                    * cos_Q[:, :, iq, :]**torch.sigmoid(self.nu[l-1, :])
                    * cos_K[:, :, ik, :]**torch.sigmoid(self.nu[l, :])
                    * sin_Q[:, :, iq, :]**torch.sigmoid(self.mu[l-1, :])
                    * sin_K[:, :, ik, :]**torch.sigmoid(self.mu[l, :])
                )
            result = torch.cumsum(result, dim=1)

            if denom is not None:
                if self.alpha is not None:
                    denom = denom * torch.exp(
                        (self.alpha[l, :] - self.alpha[l-1, :]) * self._indices
                    )
                if (cos_Q is not None and cos_K is not None
                        and sin_K is not None and sin_Q is not None):
                    denom = (denom
                        * cos_Q[:, :, iq, :]**torch.sigmoid(self.nu[l-1, :])
                        * cos_K[:, :, ik, :]**torch.sigmoid(self.nu[l, :])
                        * sin_Q[:, :, iq, :]**torch.sigmoid(self.mu[l-1, :])
                        * sin_K[:, :, ik, :]**torch.sigmoid(self.mu[l, :])
                    )
                denom = torch.cumsum(denom, dim=1)

        self._hook_weighting(Q, K, V, p)
        iq = 0 if self.config.share_queries else p-1
        if self._indices is not None and self.config.normalize_is:
            result = result * self._indices.flip(1)
        if self.alpha is not None:
            result = result * torch.exp(-self.alpha[p-1, :] * self._indices)
        if cos_Q is not None and sin_Q is not None:
            result = (result
                * cos_Q[:, :, iq, :]**torch.sigmoid(self.nu[iq, :])
                * sin_Q[:, :, iq, :]**torch.sigmoid(self.mu[iq, :])
            )

        if denom is not None:
            if self.alpha is not None:
                denom = denom * torch.exp(-self.alpha[p-1, :] * self._indices)
            if cos_Q is not None and sin_Q is not None:
                denom = (denom
                    * cos_Q[:, :, iq, :]**torch.sigmoid(self.nu[iq, :])
                    * sin_Q[:, :, iq, :]**torch.sigmoid(self.mu[iq, :])
                )
            result = result / denom

        result = self.hooks(f"iss.{p-1}", result)
        result = torch.einsum("bth,hd->btd", result, self.W_O)

        return result

    def _hook_weighting(
        self,
        Q: torch.Tensor | None,
        K: torch.Tensor | None,
        V: torch.Tensor,
        l: int,
    ) -> None:
        if not self.hooks.get(f"weighting.{l-1}").is_attached():
            return
        B = V.shape[0]
        T = self.config.context_length
        D = self.config.n_is
        matrix = torch.ones((B, T, T, D), device=V.device)
        if Q is not None and K is not None:
            iq = 0 if self.config.share_queries else l-1
            ik = 0 if self.config.share_keys else l-1
            matrix = matrix * torch.cos(
                Q[:, :, iq, :].repeat(1, T, 1).reshape(B, T, T, D)
                - K[:, :, ik, :].repeat(1, T, 1).reshape(B, T, T, D)
                    .transpose(1, 2)
            )
        if self.alpha is not None and self._indices is not None:
            index_matrix = self._indices.repeat(1, T, 1).reshape(B, T, T, D)
            matrix = matrix * torch.exp(self.alpha[l-1, :] * (
                index_matrix - index_matrix.transpose(1, 2)
            ))
        self.hooks(f"weighting.{l-1}", matrix)


class Elissabeth(SAINoMoreModule):
    """Extended Learnable Iterated Sums Signature Architecture"""

    config: ElissabethConfig

    def __init__(self, config: ElissabethConfig) -> None:
        super().__init__(config)
        self.embedding = nn.Embedding(config.input_vocab_size, config.d_hidden)
        self.positional_encoding = config.positional_encoding
        if not self.positional_encoding and config.length_is == 1:
            warnings.warn(
                "No positional encoding and ISS length 1. "
                "Elissabeth will be permutation invariant.",
                RuntimeWarning,
            )
        if self.positional_encoding:
            self.pos_embedding = PositionalEmbedding(config)
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
        if self.positional_encoding:
            x = self.pos_embedding(x)

        for i in range(len(self.layers)):
            y = x
            if self.normalize:
                y = self.layernorms[i](x)
            x = x + self.layers[i](y)

        logits = self.unembedding(x)
        return torch.swapaxes(logits, 1, 2)
