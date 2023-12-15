from dataclasses import dataclass
from typing import Literal, Optional

from ..base import ModelConfig


@dataclass
class ElissabethConfig(ModelConfig):
    sum_normalization: Optional[Literal["same", "independent"]] = "independent"
    n_is: int = 1
    length_is: int = 2
    d_values: int = None  # type: ignore
    values_2D: bool = True

    share_queries: bool = False
    share_keys: bool = False
    share_values: bool = False

    pe_query_key: bool = True
    pe_value: bool = False

    distance_weighting: bool = False

    weighting: Literal["cos", "exp"] | None = "exp"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.d_values is None:
            self.d_values = self.d_hidden
        if (isinstance(self.d_values, list)
                and len(self.d_values) != self.length_is + 1):
            raise ValueError(
                "'d_values' has to be a list of length 'length_is'+1"
            )
