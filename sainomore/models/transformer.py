from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn

from ..hooks import HookCollection
from .base import HookedModule, ModelConfig, SAINoMoreModule


@dataclass
class DecoderOnlyTransformerConfig(ModelConfig):
    n_heads: int = 4
    d_head: int = None  # type: ignore
    ffn_units: int = None  # type: ignore

    positional_encoding: Literal["learnable", "sinusoidal"] | None = (
        "sinusoidal"
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.d_head is None:
            self.d_head = self.d_hidden
        if self.ffn_units is None:
            self.ffn_units = self.d_hidden


class LearnablePositionalEmbedding(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.pos_embedding = nn.Parameter(
            torch.empty(config.context_length, config.d_hidden)
        )
        nn.init.xavier_normal_(self.pos_embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embedding[:x.size(1), :]


class PositionalEncoding(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        position = torch.arange(config.context_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, config.d_hidden, 2)
            * (-np.log(10000.0) / config.d_hidden)
        )
        pe = torch.zeros(1, config.context_length, config.d_hidden)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        if config.d_hidden % 2 != 0:
            pe[0, :, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.get_buffer("pe")[:, :x.size(1), :]
        return x


class CausalMultiHeadAttention(HookedModule):

    config: DecoderOnlyTransformerConfig

    def __init__(self, config: DecoderOnlyTransformerConfig) -> None:
        super().__init__(config)
        self.d_head = config.d_head

        self.W_K = nn.Parameter(
            torch.empty(config.n_heads, self.d_head, config.d_hidden)
        )
        self.b_K = nn.Parameter(
            torch.empty(1, config.n_heads, 1, config.d_head)
        )
        self.W_Q = nn.Parameter(
            torch.empty(config.n_heads, self.d_head, config.d_hidden)
        )
        self.b_Q = nn.Parameter(
            torch.empty(1, config.n_heads, 1, config.d_head)
        )
        self.W_V = nn.Parameter(
            torch.empty(config.n_heads, self.d_head, config.d_hidden)
        )
        self.b_V = nn.Parameter(
            torch.empty(1, config.n_heads, 1, config.d_head)
        )
        self.W_O = nn.Parameter(
            torch.empty(config.d_hidden, self.d_head, config.n_heads)
        )
        self.b_O = nn.Parameter(
            torch.empty(config.d_hidden,)
        )

        torch.nn.init.xavier_uniform_(self.W_Q)
        torch.nn.init.xavier_uniform_(self.b_Q)
        torch.nn.init.xavier_uniform_(self.W_K)
        torch.nn.init.xavier_uniform_(self.b_K)
        torch.nn.init.xavier_uniform_(self.W_V)
        torch.nn.init.xavier_uniform_(self.b_V)
        torch.nn.init.xavier_uniform_(self.W_O)
        torch.nn.init.zeros_(self.b_O)

        self.register_buffer("mask",
            torch.tril(
                torch.ones(1, 1, config.context_length, config.context_length)
            ).bool(),
        )

        self.hooks = HookCollection("Q", "K", "V", "A_pre", "A", "heads")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        Q = self.hooks("Q", torch.einsum('ihd,btd->bith', self.W_Q, x)) + self.b_Q
        K = self.hooks("K", torch.einsum('ihd,btd->bith', self.W_K, x)) + self.b_K
        V = self.hooks("V", torch.einsum('ihd,btd->bith', self.W_V, x)) + self.b_V
        attn_scores_pre = self.hooks(
            "A_pre",
            torch.einsum('bith,bish->bist', K, Q),
        )
        # B = Q.size(0)
        # N = Q.size(1)
        # T = Q.size(2)
        # D = Q.size(3)
        # attn_scores_pre = torch.sum(
        #     Q[:, :, :, :].repeat(1, 1, T, 1).reshape(B, N, T, T, D)
        #     - K[:, :, :, :].repeat(1, 1, T, 1).reshape(B, N, T, T, D)
        #         .transpose(2, 3),
        # dim=4)
        attn_scores_masked = (attn_scores_pre
            - 1e10 * (~self.get_buffer("mask")[:, :, :T, :T]).float()
        )
        attn_matrix = self.hooks("A", nn.functional.softmax(
            attn_scores_masked / np.sqrt(self.d_head),
            dim=-1,
        ))
        z = self.hooks(
            "heads",
            torch.einsum('bith,bist->bish', V, attn_matrix),
        )
        out = torch.einsum('dhi,bish->bsd', self.W_O, z) + self.b_O

        return out


class MLP(nn.Module):
    """Two layer feed-forward neural network with a ReLU activation."""

    def __init__(self, config: DecoderOnlyTransformerConfig) -> None:
        super().__init__()

        self.seq = nn.Sequential(
            nn.Linear(config.d_hidden, config.ffn_units),
            nn.ReLU(),
            nn.Linear(config.ffn_units, config.d_hidden)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class DecoderLayer(SAINoMoreModule):

    def __init__(self, config: DecoderOnlyTransformerConfig) -> None:
        super().__init__(config)
        self._normalize = config.layer_norm
        self.causal_self_attention = CausalMultiHeadAttention(config)
        self.mlp = MLP(config)
        if self._normalize:
            self.layer_norm_att = nn.LayerNorm(config.d_hidden)
            self.layer_norm_mlp = nn.LayerNorm(config.d_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x if not self._normalize else self.layer_norm_att(x)
        x_att = self.causal_self_attention(y)
        x = x + x_att
        y = x if not self._normalize else self.layer_norm_mlp(x)
        x = x + self.mlp(y)
        return x


class DecoderOnlyTransformer(SAINoMoreModule):
    def __init__(self, config: DecoderOnlyTransformerConfig) -> None:
        super().__init__(config)
        if config.input_type == "token":
            self.embedding = nn.Embedding(
                config.input_vocab_size, config.d_hidden
            )
        print(f"{self.config.positional_encoding=}")
        if self.config.positional_encoding == "learnable":
            self.pos_enc = LearnablePositionalEmbedding(config)
        elif self.config.positional_encoding == "sinusoidal":
            self.pos_enc = PositionalEncoding(config)
        self.layers = nn.ModuleList([
            DecoderLayer(config)
            for _ in range(config.n_layers)
        ])
        self.final_norm = None
        if config.layer_norm:
            self.final_norm = nn.LayerNorm(config.d_hidden)
        self.unembedding = nn.Linear(
            config.d_hidden, config.output_vocab_size,
            bias=config.bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.input_type == "token":
            x = self.embedding(x)
        if self.config.positional_encoding is not None:
            x = self.pos_enc(x)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        if self.final_norm is not None:
            x = self.final_norm(x)
        logits = self.unembedding(x)

        return torch.swapaxes(logits, 1, 2)
