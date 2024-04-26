import torch
from torch import nn

from typing import Optional, Literal

from ..base import HookedModule, BaseModel


class Sin(nn.Module):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(input)


class QKGenConfig(BaseModel):

    qk_activation: Optional[Literal["sin", "relu"]] = None
    qk_latent: Optional[int] = None
    qk_include_time: bool = False


class QKGen(HookedModule):

    _config_class = QKGenConfig

    def __init__(
        self,
        parent: Optional[HookedModule] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, **kwargs)

        self._include_time = self.config("qk_include_time")
        T = self.config("context_length")
        if self._include_time:
            indices = torch.empty((1, T, 1))
            indices[0, :, 0] = torch.linspace(1/T, 1, T)
            self.register_buffer("T", indices)

        self._shape = (
            self.config("n_is"),
            self.config("length_is"),
            self.config("d_query_key"),
        )
        out = (
            self.config("n_is")
            * self.config("length_is")
            * self.config("d_query_key")
        )
        in_ = self.config("d_hidden")
        if self._include_time:
            in_ += 1
        latent = self.config("d_hidden")
        activation = self.config("qk_activation")
        if activation is None:
            self.transform = nn.Linear(in_, out)
            torch.nn.init.xavier_normal_(self.transform.weight)
            torch.nn.init.zeros_(self.transform.bias)
        else:
            latent = self.config("qk_latent")
            if latent is None:
                latent = in_
            self.transform = nn.Sequential(
                nn.Linear(in_, latent),
                Sin() if activation == "sin" else nn.ReLU(),
                nn.Linear(latent, out),
            )
            torch.nn.init.xavier_normal_(self.transform[0].weight)
            torch.nn.init.zeros_(self.transform[0].bias)
            torch.nn.init.xavier_normal_(self.transform[2].weight)
            torch.nn.init.zeros_(self.transform[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        T = self.get_buffer("T")
        for _ in range(x.ndim - 3):
            T = T.unsqueeze(0)
        T = T.repeat(*x.shape[:-2], 1, 1)
        if self._include_time:
            y = torch.cat((x, T), dim=-1)
        return self.transform(y).reshape(*x.shape[:-1], *self._shape)


class VGenConfig(BaseModel):

    v_activation: Optional[Literal["sin", "relu"]] = None
    v_latent: Optional[int] = None
    v_include_time: bool = False


class VGen(HookedModule):

    _config_class = VGenConfig

    def __init__(
        self,
        parent: Optional[HookedModule] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, **kwargs)

        self._include_time = self.config("v_include_time")
        T = self.config("context_length")
        if self._include_time:
            indices = torch.empty((1, T, 1))
            indices[0, :, 0] = torch.linspace(1/T, 1, T)
            self.register_buffer("T", indices)

        self._shape = (
            self.config("n_is"),
            self.config("length_is"),
            self.config("d_values"),
            self.config("d_values") if self.config("values_2D") else 1,
        )
        out = (
            self.config("n_is")
            * self.config("length_is")
            * self.config("d_values")
            * (self.config("d_values") if self.config("values_2D") else 1)
        )
        in_ = self.config("d_hidden")
        if self._include_time:
            in_ += 1
        latent = self.config("d_hidden")
        activation = self.config("v_activation")
        if activation is None:
            self.transform = nn.Linear(in_, out)
            torch.nn.init.xavier_normal_(self.transform.weight)
            torch.nn.init.zeros_(self.transform.bias)
        else:
            latent = self.config("v_latent")
            if latent is None:
                latent = in_
            self.transform = nn.Sequential(
                nn.Linear(in_, latent),
                Sin() if activation == "sin" else nn.ReLU(),
                nn.Linear(latent, out),
            )
            torch.nn.init.xavier_normal_(self.transform[0].weight)
            torch.nn.init.zeros_(self.transform[0].bias)
            torch.nn.init.xavier_normal_(self.transform[2].weight)
            torch.nn.init.zeros_(self.transform[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        T = self.get_buffer("T")
        for _ in range(x.ndim - 3):
            T = T.unsqueeze(0)
        T = T.repeat(*x.shape[:-2], 1, 1)
        if self._include_time:
            y = torch.cat((x, T), dim=-1)
        return self.transform(y).reshape(*x.shape[:-1], *self._shape)
