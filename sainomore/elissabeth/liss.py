__all__ = ["LISS", "LISSConfig"]

import warnings
from typing import Literal, Optional

import torch
from pydantic import BaseModel
from torch import nn

from ..base import HookedModule
from ..positional import RoPE
from .weighting import _Weighting


class LISSConfig(BaseModel):
    d_values: int
    sum_normalization: Optional[Literal["same", "independent"]] = None
    n_is: int = 1
    length_is: int = 2
    values_2D: bool = True

    share_values: bool = False
    pe_value: bool = False

    bias_value: bool = False

    distance_weighting: bool = False
    alpha_multiplier: int = 1

    weighting: bool = True
    complex_exponential: bool = False


class LISS(HookedModule):
    "Learnable Iterated Sums Signature"

    _config_class = LISSConfig

    def __init__(
        self,
        parent: Optional["HookedModule"] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, **kwargs)
        if self.config("context_length") < self.config("length_is"):
            warnings.warn(
                f"ISS length ({self.config('length_is')}) "
                f"exceeds context length ({self.config('context_length')}), "
                "which probably leads to unsuccessful training",
                RuntimeWarning,
            )

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
        if self.config("bias_value"):
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

        self.W_O = nn.Parameter(torch.empty((
            self.config("n_is"),
            self.config("d_values"),
            self.config("d_values") if self.config("values_2D") else 1,
            self.config("d_hidden"),
        )))
        nn.init.xavier_normal_(self.W_O)

        self.beta = None
        if self.config("sum_normalization") is not None:
            factor_norm = torch.empty((self.config("context_length"), 1, 1, 1))
            factor_norm[:, 0, 0, 0] = torch.arange(
                1, self.config("context_length") + 1
            )
            self.register_buffer("norm", factor_norm)
            self.beta = nn.Parameter(torch.empty((
                1 if self.config("sum_normalization") == "same"
                    else self.config("length_is"),
            )))
            nn.init.constant_(self.beta, 5.40988)

        self.pe = nn.Identity()
        if self.config("pe_value"):
            self.pe = RoPE(
                T=self.config("context_length"), d=self.config("d_hidden")
            )

        self.weightings = nn.ModuleList()
        self.hooks.add_hooks("V")
        self.hooks.add_hooks(
            *(f"iss.{i}" for i in range(1, self.config("length_is")+1))
        )

        self.p = self.config("length_is")

    def add_weighting(self, weighting: _Weighting) -> None:
        self.weightings.append(weighting)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        for weighting in self.weightings:
            x = weighting.on_forward_start(x)

        V = torch.einsum('hldvw,btd->bthlvw', self.W_V, self.pe(x))
        if self.b_V is not None:
            V = V + self.b_V
        self.hook("V", V)
        V = V.unsqueeze(0)

        result = V[:, :, :, :, 0, :, :]
        for weighting in self.weightings:
            result = weighting.on_weighting(result, 0)
        result = torch.cumsum(result, dim=2)

        if self.beta is not None:
            result /= (
                (0.25*torch.tanh(self.beta[0])+0.75001)**(
                    torch.log10(self.get_buffer("norm")[:, :T, :, :, :])
                ) * self.get_buffer("norm")[:, :T, :, :, :]
            )

        for l in range(1, self.p):
            result = nn.functional.pad(
                result[:, :, :-1, :, :, :],
                (0, 0, 0, 0, 0, 0, 1, 0)
            )
            iv = 0 if self.config("share_values") else l
            if V.size(-1) > 1:
                result = V[:, :, :, :, iv, :, :] @ result
            else:
                result = V[:, :, :, :, iv, :, :] * result

            for weighting in self.weightings:
                result = weighting.on_weighting(result, l)
            result = torch.cumsum(result, dim=2)

            if self.beta is not None:
                result /= nn.functional.pad(
                    (0.25*torch.tanh(self.beta[
                        0 if self.config("sum_normalization") == "same" else l
                    ])+0.75001)**(
                        torch.log10(self.get_buffer("norm")[:T, :, :, :])
                    ) * self.get_buffer("norm")[:T, :, :, :],
                    (0, 0, 0, 0, 0, 0, l, 0),
                    value=1.0,
                )[:-l, :, :, :]

        for weighting in self.weightings:
            result = weighting.on_weighting(result, self.p)
            result = weighting.on_forward_end(result)

        if self.hooks.get(f"iss.{self.p}").is_attached():
            self.hook(f"iss.{self.p}", result)
        result = torch.einsum("hvwd,bthvw->btd", self.W_O, result[0])

        return result
