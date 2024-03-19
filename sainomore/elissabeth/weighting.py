import itertools
from abc import ABC, abstractmethod
from enum import IntFlag, auto
from typing import Optional

import torch
from pydantic import BaseModel
from torch import nn

from ..base import HookedModule
from ..positional import _PositionalEncoding


class Weighting(IntFlag):

    RELATIVE_DISTANCE = auto()
    EXPONENTIAL = auto()
    COMPLEX_EXPONENTIAL = auto()
    COSINE = auto()


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


class RelativeDistanceConfig(BaseModel):

    share_queries: bool = False
    share_keys: bool = False
    alpha_multiplier: int = 1


class RelativeDistance(_Weighting):

    _config_class = RelativeDistanceConfig

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
        self._mult = self.config("alpha_multiplier")
        self._p = self.config("length_is")
        self._T = self.config("context_length")

    def on_weighting(self, x: torch.Tensor, l: int) -> torch.Tensor:
        iq = 0 if self.config("share_queries") else l-1
        ik = 0 if self.config("share_keys") else l
        if 0 < l < self._p and self.get_hook(f"iss.{l}").is_attached():
            temp = x * torch.exp(
                self._mult * torch.tanh(self.alpha[:, :, iq])
            )
            self.hook(f"iss.{l}", temp)
        alpha_q = torch.tensor(0) if l == 0 else (
            self._mult * torch.tanh(self.alpha[:, :, iq])
        )
        alpha_k = torch.tensor(0) if l == self._p else (
            self._mult * torch.tanh(self.alpha[:, :, ik])
        )
        x = x * torch.exp(
            (alpha_k - alpha_q) * self.get_buffer("T")[:x.size(2)]
        )
        if l < self._p - 1:
            x = x * torch.exp(
                self._mult * torch.tanh(self.alpha[:, :, l]) / self._T
            )
        if self.hooks.get("Att").is_attached():
            self.hook("Att",
                ((self.get_buffer("T")[:, :, 0, 0]
                 - self.get_buffer("T")[:, :, 0, 0].T
                 ).unsqueeze(0).unsqueeze(0) * self.alpha[0] * self._mult
                ).unsqueeze(0)
            )
        return x

    def add_pe(self, pe: _PositionalEncoding) -> None:
        pass


class ExponentialConfig(BaseModel):

    share_queries: bool = False
    share_keys: bool = False
    bias: bool = True
    restrict_query_key: bool = False


class Exponential(_Weighting):

    _config_class = ExponentialConfig

    def __init__(
        self,
        parent: Optional["HookedModule"] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent=parent, **kwargs)

        self._Q: torch.Tensor | None = None
        self._K: torch.Tensor | None = None
        self.W_Q = nn.Parameter(torch.empty((
            self.config("n_is"),
            1 if self.config("share_queries") else self.config("length_is"),
            self.config("d_hidden"),
        )))
        nn.init.xavier_normal_(self.W_Q)
        self.W_K = nn.Parameter(torch.empty((
            self.config("n_is"),
            1 if self.config("share_keys") else self.config("length_is"),
            self.config("d_hidden"),
        )))
        nn.init.xavier_normal_(self.W_K)
        self.b_Q = self.b_K = None
        if self.config("bias"):
            self.b_Q = nn.Parameter(torch.empty((
                self.config("n_is"),
                1 if self.config("share_queries")
                    else self.config("length_is"),
            )))
            nn.init.zeros_(self.b_Q)
            self.b_K = nn.Parameter(torch.empty((
                self.config("n_is"),
                1 if self.config("share_keys") else self.config("length_is"),
            )))
            nn.init.zeros_(self.b_K)

        self.hooks.add_hooks("Q", "K")
        self._p = self.config("length_is")

    def on_forward_start(self, x: torch.Tensor) -> torch.Tensor:
        self._Q = torch.einsum('hld,btd->bthl', self.W_Q, x)
        if self.b_Q is not None:
            self._Q = self._Q + self.b_Q
        for pe in self.pos_encs:
            x = pe(x)
        self._K = torch.einsum('hld,btd->bthl', self.W_K, x)
        if self.b_K is not None:
            self._K = self._K + self.b_K
        if self.config("restrict_query_key"):
            self._Q = torch.tanh(self._Q)
            self._K = torch.tanh(self._K)
        self._Q = self._Q.unsqueeze(-1).unsqueeze(-1)
        self._K = self._K.unsqueeze(-1).unsqueeze(-1)

        if self.hooks.get("Att").is_attached():
            B, T = x.shape[0], x.shape[1]
            N = self.config("n_is")
            p = self.config("length_is")
            att_mat = torch.empty((B, N, p, T, T))
            for l in range(p):
                iq = 0 if self.config("share_queries") else l
                ik = 0 if self.config("share_keys") else l
                Q_ = self._Q[..., iq].repeat(1, T, 1).reshape(
                    B, T, T, N,
                ).transpose(1, 2)
                K_ = self._K[..., ik].unsqueeze(1)
                att_mat[:, :, l, :, :] = torch.exp(Q_ - K_).moveaxis(3, 1)
            self.hook("Att", att_mat)

        return x

    def on_weighting(self, x: torch.Tensor, l: int) -> torch.Tensor:
        if self._Q is None or self._K is None:
            raise RuntimeError("Did not calculate query or key")
        iq = 0 if self.config("share_queries") else l-1
        ik = 0 if self.config("share_keys") else l
        if 0 < l < self._p and self.get_hook(f"iss.{l}").is_attached():
            temp = x * torch.exp(self._Q[:, :, :, iq])
            self.hook(f"iss.{l}", temp)
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
            self.config("n_is"), 1,
            self.config("d_values"),
            self.config("d_values") if self.config("values_2D") else 1,
        )))
        nn.init.xavier_normal_(self.W_O_real)
        self.W_O_imag = nn.Parameter(torch.empty((
            self.config("n_is"), 1,
            self.config("d_values"),
            self.config("d_values") if self.config("values_2D") else 1,
        )))
        nn.init.xavier_normal_(self.W_O_imag)
        self._p = self.config("length_is")

    def on_forward_start(self, x: torch.Tensor) -> torch.Tensor:
        self._Q = torch.einsum('hld,btd->bthl', self.W_Q, x)
        if self.b_Q is not None:
            self._Q = self._Q + self.b_Q
        for pe in self.pos_encs:
            x = pe(x)
        self._K = torch.einsum('hld,btd->bthl', self.W_K, x)
        if self.b_K is not None:
            self._K = self._K + self.b_K
        if self.config("restrict_query_key"):
            self._Q = torch.tanh(self._Q)
            self._K = torch.tanh(self._K)
        self._Q = self._Q.unsqueeze(-1).unsqueeze(-1)
        self._K = self._K.unsqueeze(-1).unsqueeze(-1)

        if self.hooks.get("Att").is_attached():
            B, T = x.shape[0], x.shape[1]
            N = self.config("n_is")
            p = self.config("length_is")
            att_mat = torch.empty((B, N, p, T, T))
            for l in range(p):
                iq = 0 if self.config("share_queries") else l
                ik = 0 if self.config("share_keys") else l
                Q_ = self._Q[..., iq].repeat(1, T, 1).reshape(
                    B, T, T, N,
                ).transpose(1, 2)
                K_ = self._K[..., ik].unsqueeze(1)
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
        if 0 < l < self._p and self.hooks.get(f"iss.{l}").is_attached():
            temp = x * torch.exp(self._Q[:, :, :, iq, :, :] * 1j)
            self.hooks(f"iss.{l}", temp)
        x = x * torch.exp((
            (0 if l == 0 else self._Q[:, :, :, iq])
            - (0 if l == self._p else self._K[:, :, :, ik])  # type: ignore
        ) * 1j)
        return x

    def on_forward_end(self, x: torch.Tensor) -> torch.Tensor:
        return x.real * self.W_O_real + x.imag * self.W_O_imag


class CosineConfig(ExponentialConfig):

    d_query_key: int = 1
    exponent: int = 1


class Cosine(_Weighting):

    _config_class = CosineConfig

    def __init__(
        self,
        parent: Optional["HookedModule"] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent=parent, **kwargs)
        self.W_Q = nn.Parameter(torch.empty((
            self.config("n_is"),
            1 if self.config("share_queries") else self.config("length_is"),
            self.config("d_hidden"),
            self.config("d_query_key"),
        )))
        nn.init.xavier_normal_(self.W_Q)
        self.W_K = nn.Parameter(torch.empty((
            self.config("n_is"),
            1 if self.config("share_keys") else self.config("length_is"),
            self.config("d_hidden"),
            self.config("d_query_key"),
        )))
        nn.init.xavier_normal_(self.W_K)
        self.b_Q = self.b_K = None
        if self.config("bias"):
            self.b_Q = nn.Parameter(torch.empty((
                self.config("n_is"),
                1 if self.config("share_queries")
                    else self.config("length_is"),
                self.config("d_query_key"),
            )))
            nn.init.zeros_(self.b_Q)
            self.b_K = nn.Parameter(torch.empty((
                self.config("n_is"),
                1 if self.config("share_keys") else self.config("length_is"),
                self.config("d_query_key"),
            )))
            nn.init.zeros_(self.b_K)

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

    def on_forward_start(self, x: torch.Tensor) -> torch.Tensor:
        Q = torch.einsum('hldi,btd->bthli', self.W_Q, x)
        if self.b_Q is not None:
            Q = Q + self.b_Q
        for pe in self.pos_encs:
            x = pe(x)
        K = torch.einsum('hldi,btd->bthli', self.W_K, x)
        if self.b_K is not None:
            K = K + self.b_K
        if self.config("restrict_query_key"):
            Q = torch.tanh(Q) * torch.pi / 4
            K = torch.tanh(K) * torch.pi / 4
        self.hooks("Q", Q)
        self.hooks("K", K)
        Q = Q.unsqueeze(-1).unsqueeze(-1)
        K = K.unsqueeze(-1).unsqueeze(-1)
        self._sin_Q = torch.sin(Q).unsqueeze(0)
        self._cos_Q = torch.cos(Q).unsqueeze(0)
        self._sin_K = torch.sin(K).unsqueeze(0)
        self._cos_K = torch.cos(K).unsqueeze(0)

        if self.hooks.get("Att").is_attached():
            B, T = x.shape[0], x.shape[1]
            N = self.config("n_is")
            D = self.config("d_query_key")
            p = self.config("length_is")
            att_mat = torch.empty((B, N, p, T, T))
            for l in range(p):
                iq = 0 if self.config("share_queries") else l
                ik = 0 if self.config("share_keys") else l
                Q_ = (Q[..., iq, :]
                    .repeat(1, T, 1, 1).reshape(B, T, T, N, D)
                    .transpose(1, 2)
                )
                K_ = K[..., ik, :].unsqueeze(1)
                att_mat[:, :, l, :, :] = torch.prod(
                    torch.cos(Q_ - K_).moveaxis(3, 1),
                    dim=-1,
                )
            self.hook("Att", att_mat)

        return x

    def on_weighting(self, x: torch.Tensor, l: int) -> torch.Tensor:
        iq = 0 if self.config("share_queries") else l-1
        ik = 0 if self.config("share_keys") else l
        W = self.get_buffer("weight_signature")
        if self._cos_Q is not None and self._sin_Q is not None and l > 0:
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
        if self._cos_K is not None and self._sin_K is not None and l < self._p:
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
        x = (x * self.get_buffer("weight_signature")[..., 0, :, :]).sum(dim=0)
        return x.unsqueeze(0)


def get_weighting(weighting: Weighting) -> list[type[_Weighting]]:
    weightings: list[type[_Weighting]] = []
    for weight in weighting:
        match weight:
            case Weighting.RELATIVE_DISTANCE:
                weightings.append(RelativeDistance)
            case Weighting.EXPONENTIAL:
                weightings.append(Exponential)
            case Weighting.COMPLEX_EXPONENTIAL:
                weightings.append(ComplexExponential)
            case Weighting.COSINE:
                weightings.append(Cosine)
    return weightings
