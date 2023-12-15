__all__ = ["Elissabeth", "ElissabethConfig"]

import warnings

import torch
from torch import nn

from ..base import SAINoMoreModule
from ..positional import (LearnablePositionalEncoding,
                          SinusoidalPositionalEncoding)
from .cliss import CLISS
from .config import ElissabethConfig
from .liss import LISS


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
                and not config.pe_query_key and not config.pe_value
                and config.length_is == 1):
            warnings.warn(
                "No positional encoding and ISS length 1. "
                "Elissabeth will be permutation invariant.",
                RuntimeWarning,
            )
        if self.config.positional_encoding == "learnable":
            self.pos_enc = LearnablePositionalEncoding(
                config.context_length, config.d_hidden
            )
        elif self.config.positional_encoding == "sinusoidal":
            self.pos_enc = SinusoidalPositionalEncoding(
                config.context_length, config.d_hidden
            )

        self.layers = nn.ModuleList([
            (LISS(config) if config.weighting == "exp" else CLISS(config))
            for _ in range(config.n_layers)
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
