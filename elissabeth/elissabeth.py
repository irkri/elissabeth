from typing import Literal

import torch
import torch.nn.functional as F
from pydantic import model_validator
from torch import nn

from .attention import AttentionConfig, CausalSelfAttention
from .config import ModelConfig
from .hooks import HookedModule
from .liss import LISS, LISSConfig


class FFNConfig(ModelConfig):

    units: int
    bias: bool = False


class SwiGLU(nn.Module):
    """Position-wise gated feed-forward ``Z(silu(W x) * V x)``."""

    def __init__(self, config: FFNConfig, d_in: int) -> None:
        super().__init__()
        self.W = nn.Linear(d_in, config.units, bias=config.bias)
        self.V = nn.Linear(d_in, config.units, bias=config.bias)
        self.Z = nn.Linear(config.units, d_in, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Z(F.silu(self.W(x)) * self.V(x))


class ElissabethConfig(ModelConfig):

    d_hidden: int
    n_layers: int = 1
    input_type: Literal["token", "vector"] = "token"
    """Integer tokens (embedded) or real vectors (projected)."""
    context_length: int | None = None
    """Time scale of the LISS kernels: decays and time features are
    measured in ``t / context_length``. A run config fills it in from the
    dataset; it stays fixed when a trained model sees longer sequences."""
    layer_norm: bool = True
    """LayerNorm before every mixer and FFN, and before the read-out."""
    residual: bool = True
    """Add every mixer's and FFN's output to the stream (otherwise it
    replaces the stream)."""

    liss: LISSConfig | None = None
    attention: AttentionConfig | None = None
    """The sequence mixer of every layer: exactly one of ``liss`` (the
    model) and ``attention`` (the transformer baseline)."""
    ffn: FFNConfig | None = None
    """A SwiGLU after every mixer, or none."""

    @model_validator(mode="after")
    def _one_mixer(self) -> "ElissabethConfig":
        if (self.liss is None) == (self.attention is None):
            raise ValueError("Give exactly one of model.liss, model.attention.")
        return self


class Elissabeth(HookedModule):
    """Extended Learnable Iterated Sums Signature Architecture.

    ``embedding -> n_layers x [mixer, SwiGLU] -> norm -> unembedding`` with
    pre-norm residual blocks. Input ``(B, T)`` tokens or ``(B, T, input_dim)``
    vectors, output ``(B, T, output_dim)`` logits.
    """

    def __init__(
        self,
        config: ElissabethConfig,
        input_dim: int,
        output_dim: int | None = None,
    ) -> None:
        layers = [f"layer_{i}" for i in range(config.n_layers)]
        super().__init__("embedding", *layers)
        self.config = config
        d = config.d_hidden
        output_dim = input_dim if output_dim is None else output_dim
        if config.input_type == "token":
            self.embedding: nn.Module = nn.Embedding(input_dim, d)
        else:
            self.embedding = nn.Linear(input_dim, d, bias=False)
            nn.init.xavier_normal_(self.embedding.weight)

        def norm() -> nn.Module:
            return nn.LayerNorm(d) if config.layer_norm else nn.Identity()

        self.mixers = nn.ModuleList([
            LISS(config.liss, d, config.context_length)
            if config.liss is not None
            else CausalSelfAttention(config.attention, d)  # type: ignore
            for _ in range(config.n_layers)
        ])
        self.mixer_norms = nn.ModuleList([norm() for _ in self.mixers])
        self.ffns = nn.ModuleList(
            [SwiGLU(config.ffn, d) for _ in self.mixers]
            if config.ffn is not None else []
        )
        self.ffn_norms = nn.ModuleList([norm() for _ in self.ffns])
        self.final_norm = norm()
        self.unembedding = nn.Linear(d, output_dim, bias=False)
        nn.init.xavier_normal_(self.unembedding.weight)

    def _block(
        self,
        x: torch.Tensor,
        module: nn.Module,
        norm: nn.Module,
    ) -> torch.Tensor:
        y = module(norm(x))
        return x + y if self.config.residual else y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hook("embedding", self.embedding(x))
        for i, (mixer, norm) in enumerate(zip(self.mixers, self.mixer_norms)):
            x = self._block(x, mixer, norm)
            if self.ffns:
                x = self._block(x, self.ffns[i], self.ffn_norms[i])
            x = self.hook(f"layer_{i}", x)
        return self.unembedding(self.final_norm(x))
