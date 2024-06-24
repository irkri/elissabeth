from abc import ABC
from enum import IntFlag, auto

import numpy as np
import torch
from pydantic import BaseModel
from torch import nn

from .base import HookedModule


class PositionalEncoding(IntFlag):

    RoPE = auto()
    Learnable = auto()
    Sinusoidal = auto()
    APES = auto()


class _PositionalEncoding(ABC, HookedModule):
    pass


class RoPEConfig(BaseModel):

    context_length: int
    d_hidden: int


class RoPE(_PositionalEncoding):

    _config_class = RoPEConfig

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        T = self.config("context_length")
        d = self.config("d_hidden")
        emb = torch.exp(
            -torch.arange(0, d if d % 2 == 0 else d-1, 2) / d * np.log(T)
        )
        emb = torch.arange(T).unsqueeze(-1) * emb * np.pi / 2
        if d % 2 == 1:
            emb = torch.concat((emb, torch.zeros((T, 1))), dim=1)
        self.register_buffer("sin", torch.sin(emb))
        self.register_buffer("cos", torch.cos(emb))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(-2)
        flag = x.size(-1) % 2 == 1
        if flag:
            x = torch.concat((x, torch.zeros_like(x[..., :, :1])), dim=-1)
        first = (
            x[..., :, ::2] * self.get_buffer("cos")[:T, :]
            - x[..., :, 1::2] * self.get_buffer("sin")[:T, :]
        )
        second = (
            x[..., :, 1::2] * self.get_buffer("cos")[:T, :]
            + x[..., :, ::2] * self.get_buffer("sin")[:T, :]
        )
        result = torch.empty_like(x, dtype=torch.float)
        result[..., :, ::2] = first
        result[..., :, 1::2] = second
        if flag:
            result = result[..., :, :-1]
        return result


class LearnableConfig(RoPEConfig):
    pass


class Learnable(_PositionalEncoding):

    _config_class = LearnableConfig

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.pos_embedding = nn.Parameter(
            torch.empty(self.config("context_length"), self.config("d_hidden"))
        )
        nn.init.xavier_normal_(self.pos_embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embedding[:x.size(-2), :]


class SinusoidalConfig(RoPEConfig):
    pass


class Sinusoidal(_PositionalEncoding):

    _config_class = SinusoidalConfig

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        T = self.config("context_length")
        d = self.config("d_hidden")
        position = torch.arange(T).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2) * (-np.log(10000.0) / d))
        pe = torch.zeros(T, d)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.get_buffer("pe")[:x.size(-2), :]
        return x


class APESConfig(RoPEConfig):

    apes_latent: int = 32


class APES(_PositionalEncoding):
    """Additive Positional Encoding with Sinusoids"""

    _config_class = APESConfig

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        T = self.config("context_length")
        d = self.config("d_hidden")
        latent = self.config("apes_latent")
        position = torch.arange(T).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, latent, 2) * (-np.log(10000.0) / latent)
        )
        pe = torch.zeros(T, latent)
        pe[:, 0::2] = torch.sin(position * div_term)
        if latent % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        self.weight = nn.Parameter(torch.empty((latent, d)))
        nn.init.xavier_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoding = torch.einsum(
            "ld,tl->td",
            self.weight,
            self.get_buffer("pe"),
        )
        return x + encoding


dict_pe_id: dict[PositionalEncoding, type[_PositionalEncoding]] = {
    PositionalEncoding.RoPE: RoPE,
    PositionalEncoding.Sinusoidal: Sinusoidal,
    PositionalEncoding.Learnable: Learnable,
    PositionalEncoding.APES: APES,
}


dict_pe_str: dict[str, type[_PositionalEncoding]] = {
    "RoPE": RoPE,
    "Sinusoidal": Sinusoidal,
    "Learnable": Learnable,
    "APES": APES,
}


def get_pe(
        pe: PositionalEncoding | list[str],
) -> list[type[_PositionalEncoding]]:
    pes: list[type[_PositionalEncoding]] = []
    for enc in pe:
        if isinstance(enc, IntFlag):
            pes.append(dict_pe_id[enc])
        elif isinstance(enc, str):
            pes.append(dict_pe_str[enc])
    return pes
