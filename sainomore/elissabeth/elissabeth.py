__all__ = ["Elissabeth"]

from enum import IntFlag
from typing import Any, Literal, Self

import torch
from pydantic import BaseModel, validator
from torch import nn

from ..base import SAINoMoreModule
from ..models.mlp import MLP
from ..positional import PositionalEncoding, get_pe
from .liss import LISS
from .weighting import Weighting, get_weighting


class ElissabethConfig(BaseModel):
    context_length: int
    input_vocab_size: int
    input_type: Literal["token", "vector"] = "token"

    d_hidden: int

    n_layers: int = 4
    layer_norm: bool = True
    residual_stream: bool = True

    mlp_size: int | None = None

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
                self.config("input_vocab_size"), self.config("d_hidden"),
            )
        else:
            self.embedding = nn.Linear(
                self.config("input_vocab_size"), self.config("d_hidden"),
                bias=False,
            )
            nn.init.xavier_normal_(self.embedding.weight)

        self.layers = nn.ModuleList([
            LISS(self, **kwargs) for _ in range(self.config("n_layers"))
        ])
        self.layernorms = None
        if self.config("layer_norm"):
            self.layernorms = nn.ModuleList([
                nn.LayerNorm(self.config("d_hidden"))
                for _ in range(self.config("n_layers")+1)
            ])

        self.mlps = None
        self.mlpnorms = None
        if self.config("mlp_size") is not None:
            self.mlps = nn.ModuleList([
                MLP(self, **kwargs) for _ in range(self.config("n_layers"))
            ])
            if self.config("layer_norm"):
                self.mlpnorms = nn.ModuleList([
                    nn.LayerNorm(self.config("d_hidden"))
                    for _ in range(self.config("n_layers"))
                ])

        self.unembedding = nn.Linear(
            self.config("d_hidden"), self.config("output_vocab_size"),
            bias=False,
        )
        nn.init.xavier_normal_(self.unembedding.weight)
        self._residual_stream = self.config("residual_stream")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, V) -> (B, O, T)
        B: batch size
        T: context length
        V: vocabulary size
        O: output dimension
        """
        x = self.embedding(x)
        for i in range(len(self.layers)):
            y = x
            if self.layernorms is not None:
                y = self.layernorms[i](x)
            if self._residual_stream:
                x = x + self.layers[i](y)
            else:
                x = self.layers[i](y)

            if self.mlps is not None:
                y = x
                if self.mlpnorms is not None:
                    y = self.mlpnorms[i](x)
                if self._residual_stream:
                    x = x + self.mlps[i](y)
                else:
                    x = self.mlps[i](y)

        if self.layernorms is not None:
            x = self.layernorms[-1](x)
        return self.unembedding(x)

    @classmethod
    def build(
        cls: type[Self],
        config: dict[str, Any],
        *flags: IntFlag,
    ) -> Self:
        model = cls(**config)
        weightings = []
        pos_encs = []
        for flag in flags:
            if isinstance(flag, PositionalEncoding):
                pos_encs.extend(get_pe(flag))
            if isinstance(flag, Weighting):
                weightings.extend(get_weighting(flag))
        for layer in model.layers:
            for level in layer.levels:
                for pe in pos_encs:
                    level.add_pe(pe(**config))
                for weighting in weightings:
                    weighting_module = weighting(level, **config)
                    for pe in pos_encs:
                        weighting_module.add_pe(pe(**config))
                    level.add_weighting(weighting_module)
        return model
