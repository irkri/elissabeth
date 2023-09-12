__all__ = ["ModelConfig"]

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ModelConfig:
    context_length: int
    input_vocab_size: int
    output_vocab_size: int = None  #type: ignore

    n_layers: int = 4
    n_heads: int = 4

    d_hidden: int = 64
    ffn_units: int = 64

    bias: bool = True

    def __post_init__(self) -> None:
        if self.output_vocab_size is None:
            self.output_vocab_size = self.input_vocab_size

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
