import torch
from pydantic import BaseModel
from torch import nn

from ..base import HookedModule


class MLPConfig(BaseModel):

    mlp_size: int
    d_hidden: int
    mlp_bias: bool = True


class SquaredRelu(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.pow(torch.relu(x), 2)


class MLP(HookedModule):
    """Two layer neural network with a squared ReLU activation."""

    _config_class = MLPConfig

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.seq = nn.Sequential(
            nn.Linear(
                self.config("d_hidden"), self.config("mlp_size"),
                bias=self.config("mlp_bias"),
            ),
            SquaredRelu(),
            nn.Linear(
                self.config("mlp_size"), self.config("d_hidden"),
                bias=self.config("mlp_bias"),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)
