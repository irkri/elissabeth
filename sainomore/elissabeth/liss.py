__all__ = ["LISS", "LISSConfig"]

from typing import Optional, Sequence

import torch
from pydantic import BaseModel
from torch import nn

from ..base import HookedModule
from ..positional import _PositionalEncoding
from .weighting import _Weighting


class LISSLevelConfig(BaseModel):
    d_values: int
    n_is: int = 1
    length_is: int = 2
    values_2D: bool = True
    sum_normalization: bool = True

    share_values: bool = False
    pe_value: bool = False

    bias: bool = True


class LISSLevel(HookedModule):
    "One level of the learnable iterated sum signature"

    _config_class = LISSLevelConfig

    def __init__(
        self,
        parent: Optional["HookedModule"] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, **kwargs)
        self.W_V = nn.Parameter(
            torch.empty((
                self.config("n_is"),
                1 if self.config("share_values") else self.config("length_is"),
                self.config("d_hidden"),
                self.config("d_values"),
                self.config("d_values") if self.config("values_2D") else 1,
            ))
        )
        nn.init.xavier_normal_(self.W_V)
        self.b_V = None
        if self.config("bias"):
            self.b_V = nn.Parameter(
                torch.empty((
                    self.config("n_is"),
                    1 if self.config("share_values")
                        else self.config("length_is"),
                    self.config("d_values"),
                    self.config("d_values") if self.config("values_2D") else 1,
                ))
            )
            nn.init.zeros_(self.b_V)

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
        V = torch.einsum('hldvw,...btd->...bthlvw', self.W_V, x)
        if self.b_V is not None:
            V = V + self.b_V
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
    max_length_is: int = 2
    levels: Optional[Sequence[int]] = None


class LISS(HookedModule):
    "Learnable Iterated Sums Signature"

    _config_class = LISSConfig

    def __init__(
        self,
        parent: Optional["HookedModule"] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, **kwargs)
        self.W_O = nn.Parameter(torch.empty((
            self.config("max_length_is") if self.config("levels") is None else
                len(self.config("levels")),
            self.config("n_is"),
            self.config("d_values"),
            self.config("d_values") if self.config("values_2D") else 1,
            self.config("d_hidden"),
        )))
        nn.init.xavier_normal_(self.W_O)

        self.levels = nn.ModuleList()
        if (levels := self.config("levels")) is not None:
            for p in levels:
                kwargs["length_is"] = p
                self.levels.append(LISSLevel(self, **kwargs))
        else:
            for p in range(self.config("max_length_is")):
                kwargs["length_is"] = p+1
                self.levels.append(LISSLevel(self, **kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.levels[0](x).unsqueeze(0)
        for k in range(1, len(self.levels)):
            result = torch.concat(
                (result, self.levels[k](x).unsqueeze(0)),
                dim=0,
            )
        result = torch.einsum("phvwd,pbthvw->btd", self.W_O, result)
        return result
