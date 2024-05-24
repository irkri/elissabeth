import itertools
from abc import ABC, abstractmethod
from enum import IntFlag, auto
from typing import Literal, Optional

import torch
from pydantic import BaseModel
from torch import nn

from ..base import HookedModule
from ..positional import _PositionalEncoding
from .qkv import QKGen, Sin


class Weighting(IntFlag):

    ExponentialDecay = auto()
    Exponential = auto()
    ControlledExponential = auto()
    ComplexExponential = auto()
    CosineDecay = auto()
    Cosine = auto()


class _Weighting(ABC, HookedModule):

    def __init__(
        self,
        parent: Optional[HookedModule] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, **kwargs)
        self.hooks.add_hooks("Att", hidden=True)
        self.pos_encs = nn.ModuleList()

    def on_forward_start(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @abstractmethod
    def on_weighting(self, x: torch.Tensor, l: int) -> torch.Tensor:
        ...

    def on_forward_end(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def add_pe(self, pe: _PositionalEncoding) -> None:
        self.pos_encs.append(pe)


class ExponentialDecayConfig(BaseModel):

    share_queries: bool = False
    share_keys: bool = False
    exp_alpha_0: float = 1.0


class ExponentialDecay(_Weighting):

    _config_class = ExponentialDecayConfig

    def __init__(
        self,
        parent: Optional["HookedModule"] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent=parent, **kwargs)
        indices = torch.empty((self.config("context_length"), 1, 1, 1))
        indices[:, 0, 0, 0] = torch.linspace(
            1/self.config("context_length"), 1, self.config("context_length")
        )
        self.register_buffer("T", indices)
        self.alpha = nn.Parameter(torch.empty(
            (1, self.config("n_is"), self.config("length_is"), 1, 1)
        ))
        nn.init.zeros_(self.alpha)
        self._mult = self.config("exp_alpha_0")
        self._p = self.config("length_is")
        self._T = self.config("context_length")

    def on_forward_start(self, x: torch.Tensor) -> torch.Tensor:
        if self.hooks.get("Att").is_attached():
            T = self.config("context_length")
            T_1 = torch.arange(T).unsqueeze(0).to(self.alpha.device)
            T_2 = torch.arange(1, T+1).unsqueeze(0).to(self.alpha.device)
            self.hook("Att", torch.cat((
                torch.exp(
                    ((T_1 - T_2.T) / T).unsqueeze(0).unsqueeze(0)
                    * torch.tanh(self.alpha[0, :, :-1]) * self._mult
                ),
                torch.exp(
                    ((T_1 - T_1.T) / T).unsqueeze(0).unsqueeze(0)
                    * torch.tanh(self.alpha[0, :, -1:]) * self._mult
                ),
            ), dim=1).unsqueeze(0))
        return x

    def on_weighting(self, x: torch.Tensor, l: int) -> torch.Tensor:
        iq = 0 if self.config("share_queries") else l-1
        ik = 0 if self.config("share_keys") else l
        alpha_q = torch.tensor(0) if l == 0 else (
            self._mult * torch.tanh(self.alpha[:, :, iq])
        )
        alpha_k = torch.tensor(0) if l == self._p else (
            self._mult * torch.tanh(self.alpha[:, :, ik])
        )
        x = x * torch.exp(
            (alpha_k - alpha_q) * self.get_buffer("T")[:x.size(-4)]
        )
        if l < self._p - 1:
            x = x * torch.exp(
                self._mult * torch.tanh(self.alpha[:, :, l]) / self._T
            )
        return x

    def add_pe(self, pe: _PositionalEncoding) -> None:
        pass


class ExponentialConfig(BaseModel):

    share_queries: bool = False
    share_keys: bool = False
    restrict_query_key: bool = False
    d_query_key: int = 1


class Exponential(_Weighting):

    _config_class = ExponentialConfig

    def __init__(
        self,
        parent: Optional["HookedModule"] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent=parent, **kwargs)

        self.P_Q = QKGen(self, **kwargs)
        self.P_K = QKGen(self, **kwargs)

        self.hooks.add_hooks("Q", "K")
        self._p = self.config("length_is")

    def on_forward_start(self, x: torch.Tensor) -> torch.Tensor:
        self._Q = self.P_Q(x).squeeze(-1)
        for pe in self.pos_encs:
            x = pe(x)
        self._K = self.P_K(x).squeeze(-1)
        if self.config("restrict_query_key"):
            self._Q = torch.tanh(self._Q)
            self._K = torch.tanh(self._K)
        self._Q = self._Q.unsqueeze(-1).unsqueeze(-1)
        self._K = self._K.unsqueeze(-1).unsqueeze(-1)

        if self.hooks.get("Att").is_attached():
            B, T = x.shape[0], x.shape[1]
            N = self.config("n_is")
            p = self.config("length_is")
            att_mat = torch.empty((B, N, p, T, T)).to(x.device)
            for l in range(p):
                iq = 0 if self.config("share_queries") else l
                ik = 0 if self.config("share_keys") else l
                Q_ = self._Q[..., iq, 0, 0].repeat(1, T, 1).reshape(
                    B, T, T, N,
                ).transpose(1, 2)
                K_ = self._K[..., ik, 0, 0].unsqueeze(1)
                att_mat[:, :, l, :, :] = torch.exp(Q_ - K_).moveaxis(3, 1)
            self.hook("Att", att_mat)

        return x

    def on_weighting(self, x: torch.Tensor, l: int) -> torch.Tensor:
        if self._Q is None or self._K is None:
            raise RuntimeError("Did not calculate query or key")
        iq = 0 if self.config("share_queries") else l-1
        ik = 0 if self.config("share_keys") else l
        x = x * torch.exp(
            (0 if l == 0 else self._Q[:, :, :, iq])
            - (0 if l == self._p else self._K[:, :, :, ik])  # type: ignore
        )
        return x


class ComplexExponential(Exponential):

    def __init__(
        self,
        parent: Optional["HookedModule"] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent=parent, **kwargs)
        self.W_O_real = nn.Parameter(torch.empty((
            self.config("n_is"),
            self.config("d_values"),
            self.config("d_values") if self.config("values_2D") else 1,
        )))
        nn.init.xavier_normal_(self.W_O_real)
        self.W_O_imag = nn.Parameter(torch.empty((
            self.config("n_is"),
            self.config("d_values"),
            self.config("d_values") if self.config("values_2D") else 1,
        )))
        nn.init.xavier_normal_(self.W_O_imag)
        self._p = self.config("length_is")

    def on_forward_start(self, x: torch.Tensor) -> torch.Tensor:
        self._Q = self.P_Q(x)
        for pe in self.pos_encs:
            x = pe(x)
        self._K = self.P_K(x)
        if self.config("restrict_query_key"):
            self._Q = torch.tanh(self._Q)
            self._K = torch.tanh(self._K)
        self._Q = self._Q.unsqueeze(-1).unsqueeze(-1)
        self._K = self._K.unsqueeze(-1).unsqueeze(-1)

        if self.hooks.get("Att").is_attached():
            B, T = x.shape[0], x.shape[1]
            N = self.config("n_is")
            p = self.config("length_is")
            att_mat = torch.empty((B, N, p, T, T)).to(x.device)
            for l in range(p):
                iq = 0 if self.config("share_queries") else l
                ik = 0 if self.config("share_keys") else l
                Q_ = self._Q[..., iq, 0, 0].repeat(1, T, 1).reshape(
                    B, T, T, N,
                ).transpose(1, 2)
                K_ = self._K[..., ik, 0, 0].unsqueeze(1)
                att_mat[:, :, l, :, :] = torch.exp(
                    (Q_ - K_) * 1j
                ).moveaxis(3, 1)
            self.hook("Att", att_mat)

        return x

    def on_weighting(self, x: torch.Tensor, l: int) -> torch.Tensor:
        if self._Q is None or self._K is None:
            raise RuntimeError("Did not calculate query or key")
        iq = 0 if self.config("share_queries") else l-1
        ik = 0 if self.config("share_keys") else l
        x = x * torch.prod(torch.exp((
            (0 if l == 0 else self._Q[:, :, :, iq])
            - (0 if l == self._p else self._K[:, :, :, ik])  # type: ignore
        ) * 1j), dim=-3)
        return x

    def on_forward_end(self, x: torch.Tensor) -> torch.Tensor:
        return x.real * self.W_O_real + x.imag * self.W_O_imag


class ControlledExponentialConfig(BaseModel):

    share_control: bool = False
    activation: Literal["sin", "relu"] = "sin"
    control_mlp_size: int = 16


class ControlledExponential(_Weighting):

    _config_class = ControlledExponentialConfig

    def __init__(
        self,
        parent: Optional["HookedModule"] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent=parent, **kwargs)
        indices = torch.linspace(
            1/self.config("context_length"), 1, self.config("context_length")
        )
        self.register_buffer("T", indices)

        out = self.config("n_is") * (
            1 if self.config("share_control") else self.config("length_is")
        )
        mlp_size = self.config("control_mlp_size")
        self.mlp = nn.Sequential(
            nn.Linear(1, mlp_size),
            Sin() if self.config("activation") == "sin" else nn.ReLU(),
            nn.Linear(mlp_size, out, bias=False),
        )

        self._p = self.config("length_is")
        self._N = self.config("n_is")
        self._T = self.config("context_length")

    def on_forward_start(self, x: torch.Tensor) -> torch.Tensor:
        self._mlp_pass = self.mlp(self.get_buffer("T").unsqueeze(-1)).reshape(
            (self._T, self._p, self._N)
        )
        if self.hooks.get("Att").is_attached():
            T = self._mlp_pass.unsqueeze(0)
            self.hook("Att", torch.exp(
                ((T - T.transpose(0, 1))).unsqueeze(0).unsqueeze(0)
            ).unsqueeze(0))
        return x

    def on_weighting(self, x: torch.Tensor, l: int) -> torch.Tensor:
        iq = 0 if self.config("share_control") else l-1
        ik = 0 if self.config("share_control") else l
        alpha_q = torch.tensor(0) if l == 0 else (
            self._mlp_pass[:x.size(-4), iq, :]
        )
        alpha_k = torch.tensor(0) if l == self._p else (
            self._mlp_pass[:x.size(-4), ik, :]
        )
        x = x * torch.exp(alpha_k - alpha_q).unsqueeze(-1).unsqueeze(-1)
        return x

    def add_pe(self, pe: _PositionalEncoding) -> None:
        pass


class CosineDecayConfig(BaseModel):

    share_queries: bool = False
    share_keys: bool = False
    d_alpha: int = 1
    decay_exponent: int = 1
    cos_alpha_0: float = 1.0


class CosineDecay(_Weighting):

    _config_class = CosineDecayConfig

    def __init__(
        self,
        parent: Optional["HookedModule"] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent=parent, **kwargs)
        self._da = self.config("d_alpha")
        self._dim = 0
        indices = torch.empty((self.config("context_length") + 1, 1, 1, 1, 1))
        indices[:, 0, 0, 0, 0] = torch.linspace(
            0, 1, self.config("context_length") + 1
        )
        self.register_buffer("T", indices)
        self.alpha = nn.Parameter(torch.empty(
            (self.config("length_is"), 1, self.config("n_is"), self._da, 1, 1)
        ))
        nn.init.zeros_(self.alpha)
        self._mult = self.config("cos_alpha_0")
        self._p = self.config("length_is")
        self._T = self.config("context_length")

        P = self.config("length_is") * self._da
        trig_id = []
        exp = self.config("decay_exponent")
        trig_exp = [exp, 0]
        trig_coeff = 1
        for k in range(exp+1):
            trig_id.append(f"{trig_coeff}{trig_exp[0]}{trig_exp[1]}")
            trig_exp[0] -= 1
            trig_exp[1] += 1
            trig_coeff = trig_coeff * (exp - k) // (k + 1)
        weightings = torch.zeros(
            ((exp+1)**P, 1, 1, 1, 1, 1, 4*P+1),
            dtype=torch.int32,
        )
        weightings[:, ..., 0] = 1
        for c, comb in enumerate(itertools.product(trig_id, repeat=P)):
            for i in range(P):
                weightings[c, ..., 0] *= int(comb[i][0])
                weightings[c, ..., 4*i+1] += int(comb[i][1])
                weightings[c, ..., 4*i+3] += int(comb[i][1])
                weightings[c, ..., 4*i+2] += int(comb[i][2])
                weightings[c, ..., 4*i+4] += int(comb[i][2])
        self.register_buffer("weight_signature", weightings.swapaxes(-1, -3))

        self._p = self.config("length_is")

    def on_forward_start(self, x: torch.Tensor) -> torch.Tensor:
        if self.hooks.get("Att").is_attached():
            T = self.config("context_length")
            ind = self.get_buffer("T")[..., 0]
            alpha = torch.tanh(
                self.alpha[:, :, :, :, 0, 0].unsqueeze(1)
            ) * self._mult
            self.hook("Att", torch.cat((
                torch.prod(torch.cos(
                    ((ind[:-1] - ind[1:].transpose(0, 1)) / T).unsqueeze(0)
                    * alpha[:-1]
                )**self.config("decay_exponent"), dim=-1),
                torch.prod(torch.cos(
                    ((ind[:-1] - ind[:-1].transpose(0, 1)) / T).unsqueeze(0)
                    * alpha[-1:]
                )**self.config("decay_exponent"), dim=-1),
            ), dim=0).unsqueeze(0).moveaxis(-1, 1))

        self._dim = - x.ndim - 3
        return x.unsqueeze(0)

    def on_weighting(self, x: torch.Tensor, l: int) -> torch.Tensor:
        iq = 0 if self.config("share_queries") else l-1
        ik = 0 if self.config("share_keys") else l
        W = self.get_buffer("weight_signature")
        for _ in range(-self._dim-6):
            W = W.unsqueeze(1)
        if l > 0:
            T = (
                self._mult * torch.tanh(self.alpha[iq])
                * self.get_buffer("T")[:x.size(-4)]
            )
            ind = torch.arange(
                W.size(-3)-4*(l-1)*self._da, W.size(-3)-4*l*self._da, -4
            ).to(W.device)
            x = x * torch.prod(
                torch.cos(T).pow(W.index_select(-3, ind-4)),
                dim=-3,
            )
            x = x * torch.prod(
                torch.sin(T).pow(W.index_select(-3, ind-3)),
                dim=-3,
            )
        if l < self._p:
            T_slice = slice(1, x.size(-4)+1) if l < self._p - 1 else (
                slice(0, x.size(-4))
            )
            T = (
                self._mult * torch.tanh(self.alpha[ik])
                * self.get_buffer("T")[T_slice]
            )
            ind = torch.arange(
                W.size(-3)-4*l*self._da, W.size(-3)-4*(l+1)*self._da, -4
            ).to(W.device)
            x = x * torch.prod(
                torch.cos(T).pow(W.index_select(-3, ind-2)),
                dim=-3,
            )
            x = x * torch.prod(
                torch.sin(T).pow(W.index_select(-3, ind-1)),
                dim=-3,
            )
        return x

    def on_forward_end(self, x: torch.Tensor) -> torch.Tensor:
        x = (x * self.get_buffer("weight_signature")[..., 0, :, :]).sum(dim=-6)
        return x


class CosineConfig(ExponentialConfig):

    exponent: int = 1


class Cosine(_Weighting):

    _config_class = CosineConfig

    def __init__(
        self,
        parent: Optional["HookedModule"] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent=parent, **kwargs)
        self.P_Q = QKGen(self, **kwargs)
        self.P_K = QKGen(self, **kwargs)

        P = self.config("length_is") * self.config("d_query_key")
        trig_id = []
        trig_exp = [self.config("exponent"), 0]
        trig_coeff = 1
        for k in range(self.config("exponent")+1):
            trig_id.append(f"{trig_coeff}{trig_exp[0]}{trig_exp[1]}")
            trig_exp[0] -= 1
            trig_exp[1] += 1
            trig_coeff = trig_coeff * (self.config("exponent") - k) // (k + 1)
        weightings = torch.zeros(
            ((self.config("exponent")+1)**P, 1, 1, 1, 1, 1, 4*P+1),
            dtype=torch.int32,
        )
        weightings[:, ..., 0] = 1
        for c, comb in enumerate(itertools.product(trig_id, repeat=P)):
            for i in range(P):
                weightings[c, ..., 0] *= int(comb[i][0])
                weightings[c, ..., 4*i+1] += int(comb[i][1])
                weightings[c, ..., 4*i+3] += int(comb[i][1])
                weightings[c, ..., 4*i+2] += int(comb[i][2])
                weightings[c, ..., 4*i+4] += int(comb[i][2])
        self.register_buffer("weight_signature", weightings.swapaxes(-1, -3))

        self.hooks.add_hooks("Q", "K")
        self._p = self.config("length_is")
        self._dqk = self.config("d_query_key")
        self._dim = 0

    def on_forward_start(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.P_Q(x)
        for pe in self.pos_encs:
            x = pe(x)
        K = self.P_K(x)
        if self.config("restrict_query_key"):
            Q = torch.tanh(Q) * torch.pi / 4
            K = torch.tanh(K) * torch.pi / 4
        self.hooks("Q", Q)
        self.hooks("K", K)
        Q = Q.unsqueeze(-1).unsqueeze(-1)
        K = K.unsqueeze(-1).unsqueeze(-1)
        self._sin_Q = torch.sin(Q)
        self._cos_Q = torch.cos(Q)
        self._sin_K = torch.sin(K)
        self._cos_K = torch.cos(K)
        self._dim = - x.ndim - 3

        if self.hooks.get("Att").is_attached():
            B, T = x.size(-3), x.size(-2)
            N = self.config("n_is")
            D = self.config("d_query_key")
            p = self.config("length_is")
            att_mat = torch.empty((B, N, p, T, T)).to(x.device)
            for l in range(p):
                iq = 0 if self.config("share_queries") else l
                ik = 0 if self.config("share_keys") else l
                ind = (0, ) * (x.ndim - 3)
                Q_ = (Q[*ind, ..., iq, :, 0, 0]
                    .repeat(1, T, 1, 1).reshape(B, T, T, N, D)
                    .transpose(1, 2)
                )
                K_ = K[*ind, ..., ik, :, 0, 0].unsqueeze(1)
                att_mat[:, :, l] = torch.prod(
                    torch.cos(Q_ - K_).moveaxis(3, 1),
                    dim=-1,
                )
            self.hook("Att", att_mat)

        return x.unsqueeze(0)

    def on_weighting(self, x: torch.Tensor, l: int) -> torch.Tensor:
        iq = 0 if self.config("share_queries") else l-1
        ik = 0 if self.config("share_keys") else l
        W = self.get_buffer("weight_signature")
        for _ in range(-self._dim-6):
            W = W.unsqueeze(1)
        if l > 0:
            ind = torch.arange(
                W.size(-3)-4*(l-1)*self._dqk, W.size(-3)-4*l*self._dqk, -4
            ).to(W.device)
            x = x * torch.prod(
                self._cos_Q[..., iq, :, :, :].pow(W.index_select(-3, ind-4)),
                dim=-3,
            )
            x = x * torch.prod(
                self._sin_Q[..., iq, :, :, :].pow(W.index_select(-3, ind-3)),
                dim=-3,
            )
        if l < self._p:
            ind = torch.arange(
                W.size(-3)-4*l*self._dqk, W.size(-3)-4*(l+1)*self._dqk, -4
            ).to(W.device)
            x = x * torch.prod(
                self._cos_K[..., ik, :, :, :].pow(W.index_select(-3, ind-2)),
                dim=-3,
            )
            x = x * torch.prod(
                self._sin_K[..., ik, :, :, :].pow(W.index_select(-3, ind-1)),
                dim=-3,
            )
        return x

    def on_forward_end(self, x: torch.Tensor) -> torch.Tensor:
        x = (x * self.get_buffer("weight_signature")[..., 0, :, :]).sum(dim=-6)
        return x


def get_weighting(weighting: Weighting) -> list[type[_Weighting]]:
    weightings: list[type[_Weighting]] = []
    for weight in weighting:
        match weight:
            case Weighting.ExponentialDecay:
                weightings.append(ExponentialDecay)
            case Weighting.Exponential:
                weightings.append(Exponential)
            case Weighting.ComplexExponential:
                weightings.append(ComplexExponential)
            case Weighting.ControlledExponential:
                weightings.append(ControlledExponential)
            case Weighting.CosineDecay:
                weightings.append(CosineDecay)
            case Weighting.Cosine:
                weightings.append(Cosine)
    return weightings
