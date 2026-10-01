"""Softmax attention: the transformer baseline for Elissabeth.

Used in place of the LISS mixer (``model.attention`` instead of
``model.liss``), so a baseline differs from Elissabeth only in the sequence
mixer: same embedding, norms, SwiGLU and read-out.
"""
import math

import torch
import torch.nn.functional as F
from pydantic import model_validator
from torch import nn

from .config import ModelConfig
from .hooks import HookedModule


class AttentionConfig(ModelConfig):

    n_heads: int = 1
    d_head: int | None = None
    """Width per head (default: ``d_hidden // n_heads``)."""
    rope: bool = True
    """Rotary position embedding on queries and keys."""
    rope_base: float = 10_000.0
    bias: bool = False
    bidirectional: bool = False
    """Attend to every position instead of only the past (an encoder, the
    baseline for set tasks)."""

    @model_validator(mode="after")
    def _check(self) -> "AttentionConfig":
        if self.rope and self.d_head is not None and self.d_head % 2:
            raise ValueError("rope needs an even d_head.")
        return self


def rotate(x: torch.Tensor, base: float) -> torch.Tensor:
    """RoPE on ``(B, H, T, d)``, rotating the pairs ``(x_i, x_{i+d/2})`` by
    ``t * base^{-2i/d}``."""
    T, d = x.shape[-2], x.shape[-1]
    frequency = base ** (
        -torch.arange(0, d // 2, device=x.device, dtype=x.dtype) * 2 / d
    )
    angle = torch.arange(T, device=x.device, dtype=x.dtype)[:, None] * frequency
    cos, sin = torch.cos(angle), torch.sin(angle)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class SelfAttention(HookedModule):

    def __init__(self, config: AttentionConfig, d_in: int) -> None:
        super().__init__("query", "key", "value")
        self.n_heads = config.n_heads
        self.d_head = (
            config.d_head if config.d_head is not None
            else d_in // config.n_heads
        )
        if self.d_head < 1:
            raise ValueError("d_hidden is smaller than n_heads.")
        if config.rope and self.d_head % 2:
            raise ValueError(f"rope needs an even d_head, got {self.d_head}.")
        self.rope = config.rope
        self.rope_base = config.rope_base
        self.causal = not config.bidirectional
        width = self.n_heads * self.d_head
        self.qkv = nn.Linear(d_in, 3 * width, bias=config.bias)
        self.out = nn.Linear(width, d_in, bias=config.bias)

    def _qkv(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head) \
            .permute(2, 0, 3, 1, 4)
        if self.rope:
            q, k = rotate(q, self.rope_base), rotate(k, self.rope_base)
        return self.hook("query", q), self.hook("key", k), \
            self.hook("value", v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self._qkv(x)
        z = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self.out(z.transpose(1, 2).reshape(B, T, -1))

    @torch.no_grad()
    def attention_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """The attention weights ``(B, H, T, T)``, for analysis."""
        q, k, _ = self._qkv(x)
        T = x.shape[1]
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)
        if self.causal:
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            scores = scores.masked_fill(~mask, -torch.inf)
        return scores.softmax(-1)
