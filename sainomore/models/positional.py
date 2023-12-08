import numpy as np
import torch


class RoPE(torch.nn.Module):

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
