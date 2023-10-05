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
    positional_encoding: bool = False
    normalize_layers: bool = True
    normalize_iss: bool = True

    share_queries: bool = False
    share_keys: bool = False
    share_values: bool = False
    single_query_key: bool = False

    iss_length: int = 2


class LISS(HookedModule):
    "Learnable Iterated Sums Signature"

    def __init__(self, config: ElissabethConfig) -> None:
        super().__init__()
        if config.context_length < config.iss_length:
            warnings.warn(
                f"ISS length ({config.iss_length}) "
                f"exceeds context length ({config.context_length}), "
                "which probably leads to unsuccessful training",
                RuntimeWarning,
            )
        self.share_queries = config.share_queries
        self.share_keys = config.share_keys
        self.share_values = config.share_values

        self.W_Q = nn.Parameter(
            torch.empty((
                config.iss_length if not self.share_queries else 1,
                config.d_hidden,
                1 if config.single_query_key else config.d_head,
            ))
        )
        self.W_K = nn.Parameter(
            torch.empty((
                config.iss_length if not self.share_keys else 1,
                config.d_hidden,
                1 if config.single_query_key else config.d_head,
            ))
        )
        self.W_V = nn.Parameter(
            torch.empty((
                config.iss_length if not self.share_values else 1,
                config.d_hidden,
                config.d_head,
            ))
        )
        self.W_O = nn.Parameter(
            torch.empty((config.d_head, config.d_hidden))
        )

        self.normalize = config.normalize_iss
        self._denom = None
        if self.normalize:
            self._denom = nn.Parameter(
                torch.empty((1, config.context_length, 1)),
                requires_grad=False,
            )
            self._denom[0, :, 0] = torch.arange(1, config.context_length+1)

        self.alpha = nn.Parameter(
            torch.empty((config.iss_length, config.d_head))
        )
        torch.nn.init.ones_(self.alpha)
        self._indices = nn.Parameter(
            torch.empty((1, config.context_length, 1)),
            requires_grad=False,
        )
        self._indices[0, :, 0] = torch.linspace(0, 1, config.context_length)

        torch.nn.init.xavier_uniform_(self.W_Q)
        torch.nn.init.xavier_uniform_(self.W_K)
        torch.nn.init.xavier_uniform_(self.W_V)
        torch.nn.init.xavier_uniform_(self.W_O)

        self.length = config.iss_length

        self.hooks = HookCollection("Q", "K", "V")
        self.hooks.add_hooks([f"iss.{i}" for i in range(self.length)])
        self.hooks.add_hooks([f"weighting.{i}" for i in range(self.length)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.hooks("Q", torch.einsum('btd,ldh->btlh', x, self.W_Q))
        k = self.hooks("K", torch.einsum('btd,ldh->btlh', x, self.W_K))
        V = self.hooks("V", torch.einsum('btd,ldh->btlh', x, self.W_V))

        q = torch.sigmoid(q)
        k = torch.sigmoid(k)

        result = torch.cumsum(
            torch.exp(self.alpha[0, :]*self._indices)
            * torch.exp(-k[:, :, 0, :])
            * V[:, :, 0, :],
        dim=1)

        for l in range(1, self.length):
            iq = 0 if self.share_queries else l-1
            ik = 0 if self.share_keys else l
            iv = 0 if self.share_values else l
            if self.normalize:
                result = result / self._denom
            if self.hooks.get(f"iss.{l-1}").is_attached():
                self.hooks(f"iss.{l-1}", torch.exp(q[:, :, iq, :]) * result)
            if self.hooks.get(f"weighting.{l-1}").is_attached():
                self.hooks(f"weighting.{l-1}",
                    torch.exp(
                        q[:, :, iq, :]
                            .repeat(1, q.size(1), 1)
                            .reshape(q.size(0), q.size(1), q.size(1),
                                     q.size(3))
                        - k[:, :, (0 if self.share_keys else l-1), :]
                            .repeat(1, k.size(1), 1)
                            .reshape(k.size(0), k.size(1), k.size(1),
                                     k.size(3))
                            .transpose(1, 2)
                    )
                )
            result = nn.functional.pad(
                torch.roll(result, 1, 1)[:, 1:, :],
                (0, 0, 1, 0),
            )
            result = torch.cumsum(
                torch.exp(q[:, :, iq, :] - k[:, :, ik, :])
                    * torch.exp((self.alpha[l, :] - self.alpha[l-1, :])
                                * self._indices)
                    * V[:, :, iv, :]
                    * result,
                dim=1,
            )

        if self.hooks.get(f"weighting.{self.length-1}").is_attached():
            self.hooks(f"weighting.{self.length-1}",
                torch.exp(
                    q[:, :, (0 if self.share_queries else self.length-1), :]
                        .repeat(1, q.size(1), 1)
                        .reshape(q.size(0), q.size(1), q.size(1), q.size(3))
                    - k[:, :, (0 if self.share_keys else self.length-1), :]
                        .repeat(1, k.size(1), 1)
                        .reshape(k.size(0), k.size(1), k.size(1), k.size(3))
                        .transpose(1, 2)
                )
            )
        if self.normalize:
            result = result / self._denom
        result = self.hooks(f"iss.{self.length-1}",
            torch.exp(q[:, :, (0 if self.share_queries else self.length-1), :])
            * torch.exp(-self.alpha[self.length-1, :] * self._indices)
            * result,
        )
        result = torch.einsum("bth,hd->btd", result, self.W_O)

        return result


class CISS(HookedModule):
    "Cosine-weighted Learnable Iterated Sums Signature"

    def __init__(self, config: ElissabethConfig) -> None:
        super().__init__()
        if config.context_length < config.iss_length:
            warnings.warn(
                f"ISS length ({config.iss_length}) "
                f"exceeds context length ({config.context_length}), "
                "which probably leads to unsuccessful training",
                RuntimeWarning,
            )
        self.share_queries = config.share_queries
        self.share_keys = config.share_keys
        self.share_values = config.share_values

        self.W_Q = nn.Parameter(
            torch.empty((
                config.iss_length if not self.share_queries else 1,
                config.d_hidden,
                1 if config.single_query_key else config.d_head,
            ))
        )
        self.W_K = nn.Parameter(
            torch.empty((
                config.iss_length if not self.share_keys else 1,
                config.d_hidden,
                1 if config.single_query_key else config.d_head,
            ))
        )
        self.W_V = nn.Parameter(
            torch.empty((
                config.iss_length if not self.share_values else 1,
                config.d_hidden,
                config.d_head,
            ))
        )
        self.W_O = nn.Parameter(
            torch.empty((config.d_head, config.d_hidden))
        )

        self.normalize = config.normalize_iss
        self._denom = None
        if self.normalize:
            self._denom = nn.Parameter(
                torch.empty((1, config.context_length, 1)),
                requires_grad=False,
            )
            self._denom[0, :, 0] = torch.arange(1, config.context_length+1)

        self.alpha = nn.Parameter(
            torch.empty((config.iss_length, config.d_head))
        )
        nn.init.ones_(self.alpha)
        self._indices = nn.Parameter(
            torch.empty((1, config.context_length, 1)),
            requires_grad=False,
        )
        self._indices[0, :, 0] = torch.linspace(0, 1, config.context_length)

        self.mu = nn.Parameter(
            torch.empty((config.iss_length, config.d_head))
        )
        nn.init.ones_(self.mu)
        self.nu = nn.Parameter(
            torch.empty((config.iss_length, config.d_head))
        )
        nn.init.zeros_(self.nu)

        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_V)
        nn.init.xavier_uniform_(self.W_O)

        self.length = config.iss_length

        self.hooks = HookCollection("Q", "K", "V")
        self.hooks.add_hooks([f"iss.{i}" for i in range(self.length)])
        self.hooks.add_hooks([f"weighting.{i}" for i in range(self.length)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.hooks("Q", torch.einsum('btd,ldh->btlh', x, self.W_Q))
        k = self.hooks("K", torch.einsum('btd,ldh->btlh', x, self.W_K))
        V = self.hooks("V", torch.einsum('btd,ldh->btlh', x, self.W_V))

        sin_q = torch.sin(q)**2
        cos_q = torch.cos(q)**2
        sin_k = torch.sin(k)**2
        cos_k = torch.cos(k)**2

        result = torch.cumsum(
            torch.exp(self.alpha[0, :]*self._indices)
            * sin_k[:, :, 0, :]**torch.sigmoid(self.mu[0, :])
            * cos_k[:, :, 0, :]**torch.sigmoid(self.nu[0, :])
            * V[:, :, 0, :],
        dim=1)

        for l in range(1, self.length):
            iq = 0 if self.share_queries else l-1
            ik = 0 if self.share_keys else l
            iv = 0 if self.share_values else l
            if self.normalize:
                result = result / self._denom
            if self.hooks.get(f"iss.{l-1}").is_attached():
                self.hooks(f"iss.{l-1}",
                    sin_q[:, :, iq, :]**torch.sigmoid(self.mu[l-1, :])
                    * cos_q[:, :, iq, :]**torch.sigmoid(self.nu[l-1, :])
                    * result,
                )
            if self.hooks.get(f"weighting.{l-1}").is_attached():
                self.hooks(f"weighting.{l-1}",
                    torch.cos(
                        q[:, :, iq, :]
                            .repeat(1, q.size(1), 1)
                            .reshape(q.size(0), q.size(1), q.size(1),
                                     q.size(3))
                        - k[:, :, (0 if self.share_keys else l-1), :]
                            .repeat(1, k.size(1), 1)
                            .reshape(k.size(0), k.size(1), k.size(1),
                                     k.size(3))
                            .transpose(1, 2)
                    )
                )
            result = nn.functional.pad(
                torch.roll(result, 1, 1)[:, 1:, :],
                (0, 0, 1, 0),
            )
            result = torch.cumsum(
                torch.exp((self.alpha[l, :] - self.alpha[l-1, :]) * self._indices)
                * cos_q[:, :, iq, :]**torch.sigmoid(self.nu[l-1, :])
                * cos_k[:, :, ik, :]**torch.sigmoid(self.nu[l, :])
                * sin_q[:, :, iq, :]**torch.sigmoid(self.mu[l-1, :])
                * sin_k[:, :, ik, :]**torch.sigmoid(self.mu[l, :])
                * V[:, :, iv, :]
                * result,
            dim=1)

        iq = 0 if self.share_queries else self.length-1
        ik = 0 if self.share_keys else self.length-1
        if self.hooks.get(f"weighting.{self.length-1}").is_attached():
            self.hooks(f"weighting.{self.length-1}",
                torch.cos(
                    q[:, :, iq, :]
                        .repeat(1, q.size(1), 1)
                        .reshape(q.size(0), q.size(1), q.size(1), q.size(3))
                    - k[:, :, ik, :]
                        .repeat(1, k.size(1), 1)
                        .reshape(k.size(0), k.size(1), k.size(1), k.size(3))
                        .transpose(1, 2)
                )
            )
        if self.normalize:
            result = result / self._denom
        result = self.hooks(f"iss.{self.length-1}",
            torch.exp(-self.alpha[self.length-1, :] * self._indices)
            * cos_q[:, :, iq, :]**torch.sigmoid(self.nu[self.length-1, :])
            * sin_q[:, :, iq, :]**torch.sigmoid(self.mu[self.length-1, :])
            * result,
        )
        result = torch.einsum("bth,hd->btd", result, self.W_O)

        return result


class Elissabeth(SAINoMoreModule):
    """Extended Learnable Iterated Sums Signature Architecture"""

    def __init__(self, config: ElissabethConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.input_vocab_size, config.d_hidden)
        self.positional_encoding = config.positional_encoding
        if not self.positional_encoding and config.iss_length == 1:
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
        self.normalize = config.normalize_layers
        if config.normalize_layers:
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
