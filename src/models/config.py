__all__ = ["ModelConfig"]

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ModelConfig:
    context_length: int
    vocab_size: int
    output_dim: int

    n_layers: int = 4
    n_heads: int = 4

    d_embedding: int = 64
    d_hidden: int = 64
    ffn_units: int = 64

    bias: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
