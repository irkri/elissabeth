__all__ = ["Elissabeth"]

from enum import IntFlag
from typing import Any, Literal

import torch
from pydantic import BaseModel, validator
from torch import nn

from ..base import SAINoMoreModule
from ..positional import PositionalEncoding, get_pe
from .liss import LISS
from .weighting import Weighting, get_weighting


class ElissabethConfig(BaseModel):
    context_length: int
    input_vocab_size: int
    input_type: Literal["token", "vector"] = "token"
    positional_encoding: Literal["learnable", "sinusoidal"] | None = None

    d_hidden: int

    n_layers: int = 4
    layer_norm: bool = True

    output_vocab_size: int = -1

    @validator("output_vocab_size", always=True)
    def val_output_size(cls, output_vocab_size, values: dict[str, Any]) -> int:
        if output_vocab_size == -1 and "input_vocab_size" in values:
            return values["input_vocab_size"]
        return output_vocab_size


class Elissabeth(SAINoMoreModule):
    """Extended Learnable Iterated Sums Signature Architecture"""

    _config_class = ElissabethConfig

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.config("input_type") == "token":
            self.embedding = nn.Embedding(
                self.config("input_vocab_size"), self.config("d_hidden")
            )

        self.layers = nn.ModuleList([
            LISS(self, **kwargs) for _ in range(self.config("n_layers"))
        ])
        if self.config("layer_norm"):
            self.layernorms = nn.ModuleList([
                nn.LayerNorm(self.config("d_hidden"))
                for _ in range(self.config("n_layers")+1)
            ])

        self.unembedding = nn.Linear(
            self.config("d_hidden"), self.config("output_vocab_size"),
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, V) -> (B, O, T)
        B: batch size
        T: context length
        V: vocabulary size
        O: output dimension
        """
        if self.config("input_type") == "token":
            x = self.embedding(x)
        for i in range(len(self.layers)):
            y = x
            if self.config("layer_norm"):
                y = self.layernorms[i](x)
            x = x + self.layers[i](y)
        logits = self.unembedding(x)
        return torch.swapaxes(logits, 1, 2)

    @staticmethod
    def build(config: dict[str, Any], *flags: IntFlag) -> "Elissabeth":
        model = Elissabeth(**config)
        weightings = []
        pos_encs = []
        for flag in flags:
            if isinstance(flag, PositionalEncoding):
                pos_encs.extend(get_pe(flag))
            if isinstance(flag, Weighting):
                weightings.extend(get_weighting(flag))
        for layer in model.layers:
            for pe in pos_encs:
                layer.add_pe(pe(**config))
            for weighting in weightings:
                weighting_module = weighting(layer, **config)
                for pe in pos_encs:
                    weighting_module.add_pe(pe(**config))
                layer.add_weighting(weighting_module)
        return model
