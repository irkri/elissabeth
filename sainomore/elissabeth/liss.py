__all__ = ["LISS", "LISSConfig"]

from typing import Optional, Sequence

import torch
from pydantic import BaseModel
from torch import nn

from ..base import HookedModule
from ..positional import _PositionalEncoding
from .qkv import VGen
from .weighting import _Weighting


class LISSLevelConfig(BaseModel):
    d_values: int
    n_is: int = 1
    length_is: int = 2
    values_2D: bool = True
    sum_normalization: bool = True

    share_values: bool = False
    pe_value: bool = False


class LISSLevel(HookedModule):
    "One level of the learnable iterated sum signature"

    _config_class = LISSLevelConfig

    def __init__(
        self,
        parent: Optional["HookedModule"] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, **kwargs)
        self.P_V = VGen(self, **kwargs)

        self.beta = None
        if self.config("sum_normalization"):
            factor_norm = torch.empty((self.config("context_length"), 1, 1, 1))
            factor_norm[:, 0, 0, 0] = torch.arange(
                1, self.config("context_length") + 1
            )
            self.register_buffer("norm", factor_norm)
            self.beta = nn.Parameter(torch.empty((self.config("length_is"), )))
            nn.init.constant_(self.beta, 5.40988)

        self.weightings = nn.ModuleList()
        self.pos_encs = nn.ModuleList()
        self.hooks.add_hooks("V", "iss")
        self.p = self.config("length_is")

    def add_weighting(self, weighting: _Weighting) -> None:
        self.weightings.append(weighting)

    def add_pe(self, pe: _PositionalEncoding) -> None:
        self.pos_encs.append(pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        for weighting in self.weightings:
            x = weighting.on_forward_start(x)

        if self.config("pe_value"):
            for pe in self.pos_encs:
                x = pe(x)
        V = self.P_V(x)
        self.hook("V", V)

        result = V[..., :, :, :, 0, :, :]
        for weighting in self.weightings:
            result = weighting.on_weighting(result, 0)
        result = torch.cumsum(result, dim=-4)

        if self.beta is not None:
            result /= (
                (0.25*torch.tanh(self.beta[0])+0.75001)**(
                    torch.log10(self.get_buffer("norm")[:T, :, :, :])
                ) * self.get_buffer("norm")[:T, :, :, :]
            )

        for l in range(1, self.p):
            result = nn.functional.pad(
                result[..., :, :-1, :, :, :],
                (0, 0, 0, 0, 0, 0, 1, 0)
            )
            iv = 0 if self.config("share_values") else l
            if V.size(-1) > 1:
                result = result @ V[..., :, :, :, iv, :, :]
            else:
                result = result * V[..., :, :, :, iv, :, :]

            for weighting in self.weightings:
                result = weighting.on_weighting(result, l)
            result = torch.cumsum(result, dim=-4)

            if self.beta is not None:
                result /= nn.functional.pad(
                    (0.25*torch.tanh(self.beta[l])+0.75001)**(
                        torch.log10(self.get_buffer("norm")[:T, :, :, :])
                    ) * self.get_buffer("norm")[:T, :, :, :],
                    (0, 0, 0, 0, 0, 0, l, 0),
                    value=1.0,
                )[:-l, :, :, :]

        for weighting in self.weightings:
            result = weighting.on_weighting(result, self.p)
        for weighting in self.weightings:
            result = weighting.on_forward_end(result)

        self.hook("iss", result)
        return result


class LISSConfig(BaseModel):
    d_values: int
    values_2D: bool = False
    n_is: int = 1
    max_length: Optional[int] = None
    lengths: Optional[Sequence[int]] = None


class LISS(HookedModule):
    "Learnable Iterated Sums Signature"

    _config_class = LISSConfig
    parameter_sorting = {"W_O": (2, 0, 1)}

    def __init__(
        self,
        parent: Optional["HookedModule"] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, **kwargs)
        if self.config("lengths") is not None:
            n_lengths = len(self.config("lengths"))
        elif self.config("max_length") is not None:
            n_lengths = len(self.config("max_length"))
        else:
            raise KeyError("Length of iterated sum has to be specified")
        self.W_H = nn.Parameter(torch.empty((
            n_lengths,
            self.config("n_is"),
        )))
        nn.init.constant_(self.W_H, 1 / self.W_H.nelement())
        self.W_O = nn.Parameter(torch.empty((
            self.config("d_values"),
            self.config("d_values") if self.config("values_2D") else 1,
            self.config("d_hidden"),
        )))
        nn.init.xavier_normal_(self.W_O)

        self.levels = nn.ModuleList()
        if (levels := self.config("lengths")) is not None:
            for p in levels:
                kwargs["length_is"] = p
                self.levels.append(LISSLevel(self, **kwargs))
        else:
            for p in range(self.config("max_length")):
                kwargs["length_is"] = p+1
                self.levels.append(LISSLevel(self, **kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.levels[0](x).unsqueeze(0)
        for k in range(1, len(self.levels)):
            result = torch.concat(
                (result, self.levels[k](x).unsqueeze(0)),
                dim=0,
            )
        result = torch.einsum("ph,pbthvw->btvw", self.W_H, result)
        result = torch.einsum("vwd,btvw->btd", self.W_O, result)
        return result
