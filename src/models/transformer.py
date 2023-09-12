from dataclasses import dataclass

import torch
from torch import nn
import numpy as np

from .config import ModelConfig


@dataclass
class DecoderOnlyTransformerConfig(ModelConfig):
    normalize: bool = True


class PositionalEmbedding(nn.Module):

    def __init__(self, config: DecoderOnlyTransformerConfig) -> None:
        super().__init__()

        self.pos_embedding = nn.Parameter(
            torch.randn(config.context_length, config.d_hidden)
            / np.sqrt(config.d_hidden)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embedding


class CausalMultiHeadAttention(nn.Module):

    def __init__(self, config: DecoderOnlyTransformerConfig) -> None:
        super().__init__()
        self.d_head = config.d_hidden

        self.W_K = nn.Parameter(
            torch.randn(config.n_heads, self.d_head, config.d_hidden)
            / np.sqrt(config.d_hidden)
        )
        self.W_Q = nn.Parameter(
            torch.randn(config.n_heads, self.d_head, config.d_hidden)
            / np.sqrt(config.d_hidden)
        )
        self.W_V = nn.Parameter(
            torch.randn(config.n_heads, self.d_head, config.d_hidden)
            / np.sqrt(config.d_hidden)
        )
        self.W_O = nn.Parameter(
            torch.randn(config.d_hidden, self.d_head, config.n_heads)
            / np.sqrt(config.d_hidden)
        )
        self.mask = torch.tril(
            torch.ones(1, config.context_length, config.context_length)
        ).bool()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = torch.einsum('ihd,bpd->biph', self.W_K, x)
        q = torch.einsum('ihd,bpd->biph', self.W_Q, x)
        v = torch.einsum('ihd,bpd->biph', self.W_V, x)
        attn_scores_pre = torch.einsum('biph,biqh->biqp', k, q)
        if self.mask.get_device() != x.get_device():
            self.mask = self.mask.to(x.get_device())  # type: ignore
        attn_scores_masked = (
            attn_scores_pre - 1e10 * (~self.mask).float()
        )
        attn_matrix = nn.functional.softmax(
            attn_scores_masked / np.sqrt(self.d_head),
            dim=-1,
        )
        z = torch.einsum('biph,biqp->biqh', v, attn_matrix)
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
        y = x
        if self._normalize:
            y = self.layer_norm_att(x)
        x_att, att_matrix = self.causal_self_attention(y)
        x = x + x_att
        y = x
        if self._normalize:
            y = self.layer_norm_att(x)
        x = x + self.mlp(y)
        return x, att_matrix


class Decoder(nn.Module):

    def __init__(self, config: DecoderOnlyTransformerConfig) -> None:
        super().__init__()

        self.enc_layers = nn.ModuleList([
            DecoderLayer(config)
            for _ in range(config.n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.enc_layers)):
            x, _ = self.enc_layers[i](x)
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
        return logits
