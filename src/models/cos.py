from typing import Optional

import torch
from torch import nn

from .config import ModelConfig
from .transformer import MLP


class CosAttention(nn.Module):

    def __init__(
        self,
        config: ModelConfig,
        use_tanh: bool = False,
        use_xavier: bool = False,
        randomize_delta: bool = False,
        epsilon: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.W_Q = nn.Parameter(
            torch.empty((config.n_heads, config.d_hidden))
        )
        self.W_K = nn.Parameter(
            torch.empty((config.n_heads, config.d_hidden))
        )
        self.W_V = nn.Parameter(
            torch.empty((config.n_heads, config.d_hidden, config.d_hidden))
        )
        self.W_O = nn.Parameter(
            torch.ones((config.n_heads, )) / config.n_heads
        )

        if use_xavier:
            nn.init.xavier_uniform_(self.W_Q)
            nn.init.xavier_uniform_(self.W_K)
            nn.init.xavier_uniform_(self.W_V)
        else:
            nn.init.normal_(self.W_Q)
            nn.init.normal_(self.W_K)
            nn.init.normal_(self.W_V)
        if randomize_delta:
            nn.init.normal_(self.W_O)

        self._use_tanh = use_tanh
        self._epsilon = epsilon

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
        q = torch.einsum('bti,pi->btp', query, self.W_Q)
        k = torch.einsum('bti,pi->btp', key, self.W_K)
        V = torch.einsum('bti,pio->btpo', value, self.W_V)

        if self._use_tanh:
            q = torch.tanh(q) * torch.pi / 4
            k = torch.tanh(k) * torch.pi / 4

        cos_KV = torch.cumsum(torch.cos(k).unsqueeze(-1) * V, dim=-3)
        sin_KV = torch.cumsum(torch.sin(k).unsqueeze(-1) * V, dim=-3)
        cos_Q = torch.cos(q).unsqueeze(-1)
        sin_Q = torch.sin(q).unsqueeze(-1)

        heads = cos_Q * cos_KV + sin_Q * sin_KV

        if self._epsilon is not None:
            cos_K = torch.cumsum(torch.cos(k).unsqueeze(-1), dim=-3)
            sin_K = torch.cumsum(torch.sin(k).unsqueeze(-1), dim=-3)
            heads_denom = cos_Q * cos_K + sin_Q * sin_K
            heads_denom += self._epsilon
            heads = heads / heads_denom

        result = torch.einsum('btpo,p->bto', heads, self.W_O)
        return result


class CosBlock(nn.Module):

    def __init__(
        self,
        config: ModelConfig,
        use_tanh: bool = False,
        epsilon: Optional[float] = None,
        normalize: bool = True,
        use_xavier: bool = False,
        randomize_delta: bool = False,
    ) -> None:
        super().__init__()
        self.cos_attn = CosAttention(
            config,
            use_tanh=use_tanh,
            use_xavier=use_xavier,
            randomize_delta=randomize_delta,
            epsilon=epsilon,
        )
        self._normalize = normalize
        self.mlp = MLP(config, normalize=self._normalize)
        if self._normalize:
            self.layer_norm = nn.LayerNorm(config.d_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, T, D)
        B: batch size
        T: context length
        D: hidden dimension
        """
        y = self.cos_attn(x, x, x)
        if self._normalize:
            y = self.layer_norm(y)
        x = x + y
        x = self.mlp(x)
        return x


class CosDecoder(nn.Module):

    def __init__(
        self,
        config: ModelConfig,
        use_tanh: bool = False,
        epsilon: Optional[float] = None,
        normalize: bool = True,
        use_xavier: bool = False,
        randomize_delta: bool = False,
    ) -> None:
        super().__init__()
        self._layers = nn.ModuleList([
            CosBlock(
                config,
                use_tanh=use_tanh,
                epsilon=epsilon,
                normalize=normalize,
                use_xavier=use_xavier,
                randomize_delta=randomize_delta,
            )
            for _ in range(config.n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, T, D)
        B: batch size
        T: context length
        D: hidden dimension
        """
        for i in range(len(self._layers)):
            x = self._layers[i](x)
        return x


class CosDecoderOnlyTransformer(nn.Module):

    def __init__(
        self,
        config: ModelConfig,
        use_tanh: bool = False,
        epsilon: Optional[float] = None,
        normalize: bool = True,
        use_xavier: bool = False,
        randomize_delta: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_embedding)
        self.encoder = CosDecoder(
            config,
            use_tanh=use_tanh,
            epsilon=epsilon,
            normalize=normalize,
            use_xavier=use_xavier,
            randomize_delta=randomize_delta,
        )
        self.unembedding = nn.Linear(config.d_hidden, config.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, V) -> (B, T, O)
        B: batch size
        T: context length
        V: vocabulary size
        O: output dimension
        """
        x = self.embedding(x)
        x = self.encoder(x)
        logits = self.unembedding(x)
        return logits
