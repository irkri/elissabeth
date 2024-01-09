import numpy as np
import torch
from torch import nn


class RoPE(nn.Module):

    def __init__(self, T: int, d: int) -> None:
        super().__init__()
        emb = torch.exp(
            -torch.arange(0, d if d % 2 == 0 else d-1, 2) / d * np.log(T)
        )
        emb = torch.arange(T).unsqueeze(-1) * emb * np.pi / 2
        if d % 2 == 1:
            emb = torch.concat((emb, torch.zeros((T, 1))), dim=1)
        emb = emb.unsqueeze(0)
        self.register_buffer("sin", torch.sin(emb))
        self.register_buffer("cos", torch.cos(emb))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        flag = x.size(-1) % 2 == 1
        if flag:
            x = torch.concat((x, torch.zeros_like(x[:, :, :1])), dim=-1)
        first = (
            x[:, :, ::2] * self.get_buffer("cos")[:T, :]
            - x[:, :, 1::2] * self.get_buffer("sin")[:T, :]
        )
        second = (
            x[:, :, 1::2] * self.get_buffer("cos")[:T, :]
            + x[:, :, ::2] * self.get_buffer("sin")[:T, :]
        )
        result = torch.empty_like(x, dtype=torch.float)
        result[:, :, ::2] = first
        result[:, :, 1::2] = second
        if flag:
            result = result[:, :, :-1]
        return result


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, T: int, d: int) -> None:
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.empty(T, d))
        nn.init.xavier_normal_(self.pos_embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embedding[:x.size(1), :]


class SinusoidalPositionalEncoding(nn.Module):

    def __init__(self, T: int, d: int) -> None:
        super().__init__()
        position = torch.arange(T).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2) * (-np.log(10000.0) / d))
        pe = torch.zeros(1, T, d)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        if d % 2 != 0:
            pe[0, :, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.get_buffer("pe")[:, :x.size(1), :]
        return x


class APES(nn.Module):
    """Additive Positional Encoding with Sinusoids"""

    def __init__(self, T: int, d: int, latent: tuple[int, ...]) -> None:
        super().__init__()
        position = torch.arange(T).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2) * (-np.log(10000.0) / d))
        pe = torch.zeros(T, d)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        self.weight = nn.Parameter(torch.empty(latent + (d, )))
        nn.init.xavier_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoding = torch.einsum(
            "...d,td->t...",
            self.weight,
            self.get_buffer("pe"),
        )
        return x + encoding
