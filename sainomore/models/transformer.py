from enum import IntFlag
from typing import Any, Literal, Optional

import numpy as np
import torch
from pydantic import BaseModel, validator
from torch import nn

from ..base import HookedModule, SAINoMoreModule
from ..hooks import HookCollection
from ..positional import PositionalEncoding, _PositionalEncoding, get_pe
from .mlp import MLP


class MHAConfig(BaseModel):

    d_head: int
    n_heads: int
    pe_value: bool = False
    bias: bool = True


class MHA(HookedModule):

    _config_class = MHAConfig

    def __init__(
        self,
        parent: Optional["HookedModule"] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent=parent, **kwargs)
        self.d_head = self.config("d_head")
        n_heads = self.config("n_heads")
        d_hidden = self.config("d_hidden")

        self.W_Q = nn.Parameter(torch.empty(n_heads, self.d_head, d_hidden))
        self.W_K = nn.Parameter(torch.empty(n_heads, self.d_head, d_hidden))
        self.W_V = nn.Parameter(torch.empty(n_heads, self.d_head, d_hidden))
        self.W_O = nn.Parameter(torch.empty(d_hidden, self.d_head, n_heads))
        torch.nn.init.xavier_uniform_(self.W_Q)
        torch.nn.init.xavier_uniform_(self.W_K)
        torch.nn.init.xavier_uniform_(self.W_V)
        torch.nn.init.xavier_uniform_(self.W_O)

        self.b_Q = self.b_K = self.b_V = None
        if self.config("bias"):
            self.b_Q = nn.Parameter(torch.empty(1, n_heads, 1, self.d_head))
            self.b_K = nn.Parameter(torch.empty(1, n_heads, 1, self.d_head))
            self.b_V = nn.Parameter(torch.empty(1, n_heads, 1, self.d_head))
            torch.nn.init.zeros_(self.b_Q)
            torch.nn.init.zeros_(self.b_K)
            torch.nn.init.zeros_(self.b_V)

        self.pos_encs = nn.ModuleList()
        T = self.config("context_length")
        self.register_buffer("mask", torch.tril(torch.ones(1, 1, T, T)).bool())

        self.hooks = HookCollection("Q", "K", "V", "A_pre", "A", "heads")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        Q = self.hook("Q", torch.einsum('ihd,btd->bith', self.W_Q, x))
        if self.b_Q is not None:
            Q += self.b_Q
        y = x
        for pe in self.pos_encs:
            y = pe(y)
        K = self.hook("K", torch.einsum('ihd,btd->bith', self.W_K, y))
        if self.b_K is not None:
            K += self.b_K
        if self.config("pe_value"):
            x = y
        V = self.hook("V", torch.einsum('ihd,btd->bith', self.W_V, x))
        if self.b_V is not None:
            V += self.b_V
        attn_scores_pre = self.hook("A_pre",
            torch.einsum('bith,bish->bist', K, Q),
        )
        attn_scores_masked = (attn_scores_pre
            - 1e10 * (~self.get_buffer("mask")[:, :, :T, :T]).float()
        )
        attn_matrix = self.hook("A", nn.functional.softmax(
            attn_scores_masked / np.sqrt(self.d_head),
            dim=-1,
        ))
        z = self.hook("heads", torch.einsum('bith,bist->bish', V, attn_matrix))
        out = torch.einsum('dhi,bish->bsd', self.W_O, z)
        return out

    def add_pe(self, pe: _PositionalEncoding) -> None:
        self.pos_encs.append(pe)


class TransformerLayerConfig(BaseModel):
    pass


class TransformerLayer(HookedModule):

    _config_class = TransformerLayerConfig

    def __init__(
        self,
        parent: Optional["HookedModule"] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent=parent, **kwargs)
        self._normalize = self.config("layer_norm")
        self.mha = MHA(self, **kwargs)
        self.mlp = MLP(self, **kwargs)
        if self._normalize:
            self.layer_norm_att = nn.LayerNorm(self.config("d_hidden"))
            self.layer_norm_mlp = nn.LayerNorm(self.config("d_hidden"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x if not self._normalize else self.layer_norm_att(x)
        x_att = self.mha(y)
        x = x + x_att
        y = x if not self._normalize else self.layer_norm_mlp(x)
        x = x + self.mlp(y)
        return x

    def add_pe(self, pe: _PositionalEncoding) -> None:
        self.mha.add_pe(pe)


class TransformerConfig(BaseModel):
    context_length: int
    input_vocab_size: int
    input_type: Literal["token", "vector"] = "token"

    d_hidden: int

    n_layers: int = 4
    layer_norm: bool = True

    output_vocab_size: int = -1

    @validator("output_vocab_size", always=True)
    def val_output_size(cls, output_vocab_size, values: dict[str, Any]) -> int:
        if output_vocab_size == -1 and "input_vocab_size" in values:
            return values["input_vocab_size"]
        return output_vocab_size


class Transformer(SAINoMoreModule):
    """Decoder-only Transformer"""

    _config_class = TransformerConfig

    def __init__(
        self,
        parent: Optional["HookedModule"] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent=parent, **kwargs)
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
            TransformerLayer(self, **kwargs)
            for _ in range(self.config("n_layers"))
        ])
        self.final_norm = None
        if self.config("layer_norm"):
            self.final_norm = nn.LayerNorm(self.config("d_hidden"))
        self.unembedding = nn.Linear(
            self.config("d_hidden"), self.config("output_vocab_size"),
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        if self.final_norm is not None:
            x = self.final_norm(x)
        return self.unembedding(x)

    @staticmethod
    def build(config: dict[str, Any], *flags: IntFlag) -> "Transformer":
        model = Transformer(**config)
        pos_encs = []
        for flag in flags:
            if isinstance(flag, PositionalEncoding):
                pos_encs.extend(get_pe(flag))
        for layer in model.layers:
            for pe in pos_encs:
                layer.add_pe(pe(**config))
        return model
