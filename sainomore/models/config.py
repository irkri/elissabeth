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
    d_head: int = None  # type: ignore
    ffn_units: int = None  # type: ignore

    bias: bool = True

    def __post_init__(self) -> None:
        if self.output_vocab_size is None:
            self.output_vocab_size = self.input_vocab_size
        if self.d_head is None:
            self.d_head = self.d_hidden
        if self.ffn_units is None:
            self.ffn_units = self.d_hidden

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
