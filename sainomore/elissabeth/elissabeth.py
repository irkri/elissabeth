__all__ = ["Elissabeth"]

import warnings

import torch
from torch import nn

from ..base import SAINoMoreModule
from ..positional import (LearnablePositionalEncoding,
                          SinusoidalPositionalEncoding)
from .cliss import CLISS, CLISSConfig
from .liss import LISS, LISSConfig


class Elissabeth(SAINoMoreModule):
    """Extended Learnable Iterated Sums Signature Architecture"""

    config: LISSConfig | CLISSConfig

    def __init__(self, config: LISSConfig | CLISSConfig) -> None:
        super().__init__(config)
        if config.input_type == "token":
            self.embedding = nn.Embedding(
                config.input_vocab_size, config.d_hidden
            )
        if self.config.positional_encoding == "learnable":
            self.pos_enc = LearnablePositionalEncoding(
                config.context_length, config.d_hidden
            )
        elif self.config.positional_encoding == "sinusoidal":
            self.pos_enc = SinusoidalPositionalEncoding(
                config.context_length, config.d_hidden
            )

        if isinstance(config, LISSConfig):
            self.layers = nn.ModuleList([
                LISS(config) for _ in range(config.n_layers)
            ])
        elif isinstance(config, CLISSConfig):
            self.layers = nn.ModuleList([
                CLISS(config) for _ in range(config.n_layers)
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
