__all__ = ["LISS"]

import warnings

import torch
from torch import nn

from ..base import HookedModule
from ..hooks import HookCollection
from ..positional import RoPE
from .elissabeth import ElissabethConfig


class LISS(HookedModule):
    "Learnable Iterated Sums Signature"

    config: ElissabethConfig

    def __init__(self, config: ElissabethConfig) -> None:
        super().__init__(config)
        if config.context_length < config.length_is:
            warnings.warn(
                f"ISS length ({config.length_is}) "
                f"exceeds context length ({config.context_length}), "
                "which probably leads to unsuccessful training",
                RuntimeWarning,
            )

        self.W_Q = self.W_K = None
        if config.weighting is not None:
            self.W_Q = nn.Parameter(
                torch.empty((
                    config.n_is,
                    1 if config.share_queries else config.length_is,
                    config.d_hidden,
                ))
            )
            self.b_Q = nn.Parameter(
                torch.empty((
                    config.n_is,
                    1 if config.share_queries else config.length_is,
                ))
            )
            nn.init.xavier_normal_(self.W_Q)
            nn.init.zeros_(self.b_Q)
            self.W_K = nn.Parameter(
                torch.empty((
                    config.n_is,
                    1 if config.share_queries else config.length_is,
                    config.d_hidden,
                ))
            )
            self.b_K = nn.Parameter(
                torch.empty((
                    config.n_is,
                    1 if config.share_queries else config.length_is,
                ))
            )
            nn.init.xavier_normal_(self.W_K)
            nn.init.zeros_(self.b_K)

        self.W_V = nn.Parameter(
            torch.empty((
                config.n_is,
                1 if config.share_values else config.length_is,
                config.d_hidden,
                config.d_values,
                config.d_values if config.values_2D else 1,
            ))
        )
        self.b_V = nn.Parameter(
            torch.empty((
                config.n_is,
                1 if config.share_values else config.length_is,
                config.d_values,
                config.d_values if config.values_2D else 1,
            ))
        )
        nn.init.xavier_normal_(self.W_V)
        nn.init.zeros_(self.b_V)

        self.W_O = nn.Parameter(torch.empty((
            config.n_is,
            config.d_values,
            config.d_values if config.values_2D else 1,
            config.d_hidden,
        )))
        nn.init.xavier_normal_(self.W_O)

        self.alpha = None
        if config.distance_weighting:
            indices = torch.empty((config.context_length, 1, 1))
            indices[:, 0, 0] = torch.linspace(
                1/config.context_length, 1, config.context_length
            )
            self.register_buffer("T", indices)
            self.alpha = nn.Parameter(
                torch.empty((1, config.n_is, config.length_is))
            )
            nn.init.ones_(self.alpha)

        self.beta = None
        if config.sum_normalization is not None:
            factor_norm = torch.empty((1, config.context_length, 1, 1, 1))
            factor_norm[0, :, 0, 0, 0] = torch.arange(
                1, config.context_length + 1
            )
            self.register_buffer("norm", factor_norm)
            self.beta = nn.Parameter(torch.empty((
                1 if config.sum_normalization == "same" else config.length_is,
            )))
            nn.init.constant_(self.beta, 5.40988)

        self.pe = None
        if config.pe_query_key or config.pe_value:
            self.pe = RoPE(T=config.context_length, d=config.d_hidden)

        self.hooks = HookCollection("Q", "K", "V")
        self.hooks.add_hooks(
            [f"iss.{i}" for i in range(1, config.length_is+1)]
        )
        self.hooks.add_hooks(
            [f"weighting.{i}" for i in range(config.length_is)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        Q = K = None
        if self.pe is not None:
            x_enc = self.pe(x)
        if self.W_Q is not None and self.W_K is not None:
            Q = torch.einsum(
                'hld,btd->bthl',
                self.W_Q,
                x_enc if self.config.pe_query_key else x,  # type: ignore
            )# + self.b_Q
            K = torch.einsum(
                'hld,btd->bthl',
                self.W_K,
                x_enc if self.config.pe_query_key else x,  # type: ignore
            )# + self.b_K
        if self.alpha is not None and T > 1:
            rel_pos = (
                torch.sigmoid(self.alpha) * self.get_buffer("T")[:T, :, :]
            )
            if Q is None and K is None:
                Q = - rel_pos
                K = - rel_pos
            else:
                Q = Q - rel_pos
                K = K - rel_pos
        if Q is not None and K is not None:
            self.hooks("Q", Q)
            self.hooks("K", K)
            Q = Q.unsqueeze(-1).unsqueeze(-1)
            K = K.unsqueeze(-1).unsqueeze(-1)

        V = torch.einsum(
            'hldvw,btd->bthlvw',
            self.W_V,
            x_enc if self.config.pe_value else x,  # type: ignore
        )# + self.b_V
        self.hooks("V", V)

        p = self.config.length_is

        result = V[:, :, :, 0, :, :]
        result = self._weight_is(result, Q, K, 0)
        result = torch.cumsum(result, dim=1)

        if self.beta is not None:
            result /= (
                (0.25*torch.tanh(self.beta[0])+0.75001)**(
                    torch.log10(self.get_buffer("norm")[:, :T, :, :, :])
                ) * self.get_buffer("norm")[:, :T, :, :, :]
            )

        for l in range(1, p):
            self._hook_weighting(Q, K, V, l)

            result = nn.functional.pad(
                result[:, :-1, :, :, :],
                (0, 0, 0, 0, 0, 0, 1, 0)
            )
            iv = 0 if self.config.share_values else l
            if self.config.values_2D:
                result = V[:, :, :, iv, :, :] @ result
            else:
                result = V[:, :, :, iv, :, :] * result
            result = self._weight_is(
                result, Q, K, l
            )
            result = torch.cumsum(result, dim=1)

            if self.beta is not None:
                result /= nn.functional.pad(
                    (0.25*torch.tanh(self.beta[
                        0 if self.config.sum_normalization == "same" else l
                    ])+0.75001)**(
                        torch.log10(self.get_buffer("norm")[:, :T, :, :, :])
                    ) * self.get_buffer("norm")[:, :T, :, :, :],
                    (0, 0, 0, 0, 0, 0, l, 0),
                    value=1.0,
                )[:, :-l, :, :, :]

        self._hook_weighting(Q, K, V, p)

        result = self._weight_is(result, Q, K, p)

        if self.hooks.get(f"iss.{p}").is_attached():
            self.hooks(f"iss.{p}", result)
        result = torch.einsum("hvwd,bthvw->btd", self.W_O, result)

        return result

    def _weight_is(
        self,
        x: torch.Tensor,
        Q: torch.Tensor | None,
        K: torch.Tensor | None,
        l: int,
    ) -> torch.Tensor:
        p = self.config.length_is
        iq = 0 if self.config.share_queries else l-1
        ik = 0 if self.config.share_keys else l
        if l > 0 and l < p and self.hooks.get(f"iss.{l}").is_attached():
            temp = x
            if Q is not None:
                temp = x * torch.exp(Q[:, :, :, iq, :, :])
            self.hooks(f"iss.{l}", temp)
        x = x * torch.exp((
                0 if l == 0 or Q is None else Q[:, :, :, iq, :, :])
                - (0 if l == p or K is None else K[:, :, :, ik, :, :]
            )  # type: ignore
        )
        return x

    def _hook_weighting(
        self,
        Q: torch.Tensor | None,
        K: torch.Tensor | None,
        V: torch.Tensor,
        l: int,
    ) -> None:
        if not self.hooks.get(f"weighting.{l-1}").is_attached():
            return
        if Q is None and K is None:
            return
        B = V.size(0)
        T = V.size(1)
        D = self.config.n_is
        matrix = torch.ones((B, D, T, T), device=V.device)
        iq = 0 if self.config.share_queries else l-1
        ik = 0 if self.config.share_keys else l-1
        matrix = matrix * torch.exp(
            (0 if Q is None else
             Q[:, :, :, iq, 0, 0].repeat(1, T, 1).reshape(B, D, T, T)
            ) - (0 if K is None else
               (K[:, :, :, ik, 0, 0].repeat(1, T, 1).reshape(B, D, T, T)
                .transpose(1, 2))
            )  # type: ignore
        )
        self.hooks(f"weighting.{l-1}", matrix)
