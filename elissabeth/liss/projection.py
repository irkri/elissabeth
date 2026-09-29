import math
from typing import Literal

import torch
from torch import nn

from ..config import ModelConfig


class ProjectionConfig(ModelConfig):
    """How a LISS level maps a position ``x_t`` to its values, queries or
    keys: a linear map, or a two-layer network with the given activation.
    """

    activation: Literal["sin", "relu"] | None = None
    latent: int | None = None
    """Width of the hidden layer when ``activation`` is set (default: the
    input width)."""
    include_time: bool = False
    """Append the relative position ``(t+1) / context_length`` to the
    input."""


class Sin(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


def relative_time(
    x: torch.Tensor,
    context_length: int,
    offset: int = 1,
) -> torch.Tensor:
    """``(t + offset) / context_length`` as a ``(1, T, 1)`` tensor."""
    t = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)
    return ((t + offset) / context_length).view(1, -1, 1)


class Projection(nn.Module):
    """``(B, T, d_in) -> (B, T, *shape)``, one linear map (or network) for
    every head and every index of the iterated sum at once.
    """

    def __init__(
        self,
        config: ProjectionConfig,
        d_in: int,
        shape: tuple[int, ...],
        context_length: int | None,
    ) -> None:
        super().__init__()
        if config.include_time and context_length is None:
            raise ValueError("include_time needs model.context_length.")
        self.shape = shape
        self.context_length = context_length
        self.include_time = config.include_time
        d_in = d_in + int(config.include_time)
        d_out = math.prod(shape)
        if config.activation is None:
            self.transform: nn.Module = nn.Linear(d_in, d_out)
            linears = [self.transform]
        else:
            latent = config.latent if config.latent is not None else d_in
            self.transform = nn.Sequential(
                nn.Linear(d_in, latent),
                Sin() if config.activation == "sin" else nn.ReLU(),
                nn.Linear(latent, d_out),
            )
            linears = [self.transform[0], self.transform[2]]
        for linear in linears:
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        if self.include_time:
            assert self.context_length is not None
            time = relative_time(x, self.context_length)
            y = torch.cat((x, time.expand(x.shape[0], -1, -1)), dim=-1)
        return self.transform(y).unflatten(-1, self.shape)


class ValuesConfig(ProjectionConfig):

    norm: bool = True
    """LayerNorm over each value vector (or matrix, for ``values_2D``)."""
    shared: bool = False
    """One set of values for all ``n_is`` iterated sums."""
