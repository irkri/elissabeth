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
    normalize_iss: bool = False

    separate_qk: bool = True
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
        self.W_Q = nn.Parameter(
            torch.empty((
                config.iss_length,
                config.d_hidden,
                config.d_head if config.separate_qk else 1,
            ))
        )
        self.W_K = nn.Parameter(
            torch.empty((
                config.iss_length,
                config.d_hidden,
                config.d_head if config.separate_qk else 1,
            ))
        )
        self.W_V = nn.Parameter(
            torch.empty((config.iss_length, config.d_hidden, config.d_head))
        )
        self.W_O = nn.Parameter(
            torch.empty((config.d_head, config.d_hidden))
        )

        self.normalize = config.normalize_iss
        if self.normalize:
            self.layernorms = nn.ModuleList([
                nn.LayerNorm(config.d_head) for _ in range(config.iss_length)
            ])

        torch.nn.init.xavier_uniform_(self.W_Q)
        torch.nn.init.xavier_uniform_(self.W_K)
        torch.nn.init.xavier_uniform_(self.W_V)
        torch.nn.init.xavier_uniform_(self.W_O)

        self.length = config.iss_length

        self.hooks = HookCollection("Q", "K", "V", "iss")
        self.hooks.add_hooks([f"iss.{i}" for i in range(self.length)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.hooks("Q", torch.einsum('btd,ldh->btlh', x, self.W_Q))
        k = self.hooks("K", torch.einsum('btd,ldh->btlh', x, self.W_K))
        V = self.hooks("V", torch.einsum('btd,ldh->btlh', x, self.W_V))

        result = self.hooks("iss.0",
            torch.cumsum(torch.exp(-k[:, :, 0, :]) * V[:, :, 0, :], dim=1),
        )

        for l in range(1, self.length):
            result = nn.functional.pad(
                torch.roll(result, 1, 1)[:, 1:, :],
                (0, 0, 1, 0),
            )
            if self.normalize:
                result = self.layernorms[l-1](result)
            result = self.hooks(f"iss.{l}", torch.cumsum(
                torch.exp(q[:, :, l-1, :] - k[:, :, l, :])
                    * V[:, :, l, :]
                    * result,
                dim=1)
            )

        if self.normalize:
            result = self.layernorms[self.length-1](result)
        result = self.hooks("iss",
            torch.exp(q[:, :, self.length-1, :]) * result,
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
            if self.normalize:
                x = self.layernorms[i](x)
            x = x + self.layers[i](x)

        if self.normalize:
            x = self.layernorms[len(self.layers)](x)
        logits = self.unembedding(x)
        return torch.swapaxes(logits, 1, 2)
