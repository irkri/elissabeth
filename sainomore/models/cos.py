from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from ..hooks import HookCollection
from .base import HookedModule
from .transformer import MLP, DecoderOnlyTransformerConfig, PositionalEmbedding


@dataclass
class CosDecoderOnlyTransformerConfig(DecoderOnlyTransformerConfig):
    use_tanh: bool = False
    epsilon: Optional[float] = None


class CosAttention(HookedModule):

    def __init__(self, config: CosDecoderOnlyTransformerConfig) -> None:
        super().__init__()
        self.W_Q = nn.Parameter(
            torch.empty((config.n_heads, config.d_hidden, 1))
        )
        self.W_K = nn.Parameter(
            torch.empty((config.n_heads, config.d_hidden, 1))
        )
        self.W_V = nn.Parameter(
            torch.empty((config.n_heads, config.d_hidden, config.d_head))
        )
        self.W_O = nn.Parameter(
            torch.empty((config.d_hidden, config.d_head, config.n_heads))
        )

        torch.nn.init.xavier_uniform_(self.W_Q)
        torch.nn.init.xavier_uniform_(self.W_K)
        torch.nn.init.xavier_uniform_(self.W_V)
        torch.nn.init.xavier_uniform_(self.W_O)

        self._use_tanh = config.use_tanh
        self._epsilon = config.epsilon

        self.hooks = HookCollection("Q", "K", "V", "Q_tanh", "K_tanh", "heads",
                                    "denom", "fraction")

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if key is None:
            key = query
        if value is None:
            value = query
        q = self.hooks("Q", torch.einsum('btd,idh->btih', query, self.W_Q))
        k = self.hooks("K", torch.einsum('btd,idh->btih', key, self.W_K))
        V = self.hooks("V", torch.einsum('btd,idh->btih', value, self.W_V))

        if self._use_tanh:
            q = self.hooks("Q_tanh", torch.tanh(q) * torch.pi / 4)
            k = self.hooks("K_tanh", torch.tanh(k) * torch.pi / 4)

        cos_KV = torch.cumsum(torch.cos(k) * V, dim=1)
        sin_KV = torch.cumsum(torch.sin(k) * V, dim=1)
        cos_Q = torch.cos(q)
        sin_Q = torch.sin(q)

        heads = self.hooks("heads", cos_Q * cos_KV + sin_Q * sin_KV)

        if self._epsilon is not None:
            cos_K = torch.cumsum(torch.cos(k), dim=1)
            sin_K = torch.cumsum(torch.sin(k), dim=1)
            heads_denom = self.hooks("denom", cos_Q * cos_K + sin_Q * sin_K)
            heads_denom += self._epsilon
            heads = self.hooks("fraction", heads / heads_denom)

        result = torch.einsum('dhi,btih->btd', self.W_O, heads)
        return result


class CosBlock(nn.Module):

    def __init__(self, config: CosDecoderOnlyTransformerConfig) -> None:
        super().__init__()
        self.cos_attention = CosAttention(config)
        self.mlp = MLP(config)
        self._normalize = config.normalize
        if self._normalize:
            self.layer_norm_att = nn.LayerNorm(config.d_hidden)
            self.layer_norm_mlp = nn.LayerNorm(config.d_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, T, D)
        B: batch size
        T: context length
        D: hidden dimension
        """
        y = x
        if self._normalize:
            y = self.layer_norm_att(x)
        x_att = self.cos_attention(y, y, y)
        x = x + x_att
        y = x
        if self._normalize:
            y = self.layer_norm_mlp(x)
        x = x + self.mlp(y)
        return x


class CosDecoder(nn.Module):

    def __init__(self, config: CosDecoderOnlyTransformerConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            CosBlock(config)
            for _ in range(config.n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, T, D)
        B: batch size
        T: context length
        D: hidden dimension
        """
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x


class CosDecoderOnlyTransformer(nn.Module):

    def __init__(self, config: CosDecoderOnlyTransformerConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.input_vocab_size, config.d_hidden)
        self.pos_embedding = PositionalEmbedding(config)
        self.decoder = CosDecoder(config)
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
        x = self.pos_embedding(x)
        x = self.decoder(x)
        logits = self.unembedding(x)
        return torch.swapaxes(logits, 1, 2)
