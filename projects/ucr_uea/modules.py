from typing import Optional

import torch
from sainomore.base import HookedModule
from pydantic import BaseModel


class PreparateurConfig(BaseModel):

    max_pow: int


class Preparateur(HookedModule):

    _config_class = PreparateurConfig

    def __init__(
        self,
        parent: Optional["HookedModule"] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, **kwargs)
        self.n_exp = self.config("max_pow") + 1
        self.weight = torch.nn.Parameter(torch.empty(
            (self.config("d_hidden"), 2 * self.n_exp)
        ))
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dx = torch.nn.functional.pad(x[:, 1:]-x[:, :-1], (0, 0, 1, 0))
        powers = torch.cat(
            tuple(x ** i for i in range(self.n_exp))
                + tuple(dx ** i for i in range(self.n_exp)),
            dim=-1,
        )
        return torch.einsum("ij,btj->bti", self.weight, powers)


class SieveConfig(BaseModel):

    classes: int


class Sieve(HookedModule):

    _config_class = SieveConfig

    def __init__(
        self,
        parent: Optional["HookedModule"] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, **kwargs)

        self.weight = torch.nn.Parameter(
            torch.empty((self.config("classes"), 3 * self.config("d_hidden")))
        )
        torch.nn.init.xavier_normal_(self.weight)
        self.norm = torch.nn.LayerNorm(3 * self.config("d_hidden"))
        self.T = self.config("context_length")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dx = torch.nn.functional.pad(x[:, 1:, :]-x[:, :-1, :], (0, 0, 1, 0))
        npi = torch.sum(dx > 0, dim=1)
        mpi = torch.mean(dx * (dx > 0), dim=1)
        end = dx[:, -1, :]
        features = torch.cat(
            (npi, mpi, end),
            dim=-1,
        )
        features = features.unsqueeze(1).repeat(1, self.T, 1)
        features = self.norm(features)
        return torch.einsum("ij,btj->bti", self.weight, features)
