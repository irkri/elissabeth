from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..hooks import HookCollection
from .base import HookedModule, ModelConfig


@dataclass
class DecoderOnlyTransformerConfig(ModelConfig):
    normalize: bool = True


class PositionalEmbedding(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.pos_embedding = nn.Parameter(
            torch.randn(config.context_length, config.d_hidden)
            / np.sqrt(config.d_hidden)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embedding


class CausalMultiHeadAttention(HookedModule):

    def __init__(self, config: DecoderOnlyTransformerConfig) -> None:
        super().__init__()
        self.d_head = config.d_head

        self.W_K = nn.Parameter(
            torch.empty(config.n_heads, self.d_head, config.d_hidden)
        )
        self.W_Q = nn.Parameter(
            torch.empty(config.n_heads, self.d_head, config.d_hidden)
        )
        self.W_V = nn.Parameter(
            torch.empty(config.n_heads, self.d_head, config.d_hidden)
        )
        self.W_O = nn.Parameter(
            torch.empty(config.d_hidden, self.d_head, config.n_heads)
        )

        torch.nn.init.xavier_uniform_(self.W_Q)
        torch.nn.init.xavier_uniform_(self.W_K)
        torch.nn.init.xavier_uniform_(self.W_V)
        torch.nn.init.xavier_uniform_(self.W_O)

        self.mask = torch.tril(
            torch.ones(1, config.context_length, config.context_length)
        ).bool()

        self.hooks = HookCollection("Q", "K", "V", "A_pre", "A", "heads")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        Q = self.hooks("Q", torch.einsum('ihd,bpd->biph', self.W_Q, x))
        K = self.hooks("K", torch.einsum('ihd,bpd->biph', self.W_K, x))
        V = self.hooks("V", torch.einsum('ihd,bpd->biph', self.W_V, x))
        attn_scores_pre = self.hooks(
            "A_pre",
            torch.einsum('biph,biqh->biqp', K, Q),
        )
        if self.mask.get_device() != x.get_device():
            self.mask = self.mask.to(x.get_device())  # type: ignore
        attn_scores_masked = (
            attn_scores_pre - 1e10 * (~self.mask).float()
        )
        attn_matrix = self.hooks("A", nn.functional.softmax(
            attn_scores_masked / np.sqrt(self.d_head),
            dim=-1,
        ))
        z = self.hooks(
            "heads",
            torch.einsum('biph,biqp->biqh', V, attn_matrix),
        )
        out = torch.einsum('dhi,biqh->bqd', self.W_O, z)

        return out, attn_matrix


class MLP(nn.Module):
    """Two layer feed-forward neural network with a ReLU activation."""

    def __init__(self, config: DecoderOnlyTransformerConfig) -> None:
        super().__init__()

        self.seq = nn.Sequential(
            nn.Linear(config.d_hidden, config.ffn_units),
            nn.ReLU(),
            nn.Linear(config.ffn_units, config.d_hidden)
        )
        self.add = nn.ModuleList([nn.Identity(), nn.Identity()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class DecoderLayer(nn.Module):

    def __init__(self, config: DecoderOnlyTransformerConfig) -> None:
        super().__init__()
        self._normalize = config.normalize
        self.causal_self_attention = CausalMultiHeadAttention(config)
        self.mlp = MLP(config)
        if self._normalize:
            self.layer_norm_att = nn.LayerNorm(config.d_hidden)
            self.layer_norm_mlp = nn.LayerNorm(config.d_hidden)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = x if not self._normalize else self.layer_norm_att(x)
        x_att, att_matrix = self.causal_self_attention(y)
        x = x + x_att
        y = x if not self._normalize else self.layer_norm_mlp(x)
        x = x + self.mlp(y)
        return x, att_matrix


class Decoder(nn.Module):

    def __init__(self, config: DecoderOnlyTransformerConfig) -> None:
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(config)
            for _ in range(config.n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.layers)):
            x, _ = self.layers[i](x)
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: DecoderOnlyTransformerConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            config.input_vocab_size, config.d_hidden
        )
        self.pos_embedding = PositionalEmbedding(config)
        self.decoder = Decoder(config)
        self.unembedding = nn.Linear(config.d_hidden, config.output_vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.pos_embedding(x)
        x = self.decoder(x)
        logits = self.unembedding(x)
        return torch.swapaxes(logits, 1, 2)
