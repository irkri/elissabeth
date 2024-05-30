from pathlib import Path
from typing import Self, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from ..elissabeth.elissabeth import Elissabeth
from .plotting import plot_attention_matrix, plot_parameter_matrix
from .tools import get_attention_matrices


class ElissabethWatcher:

    def __init__(
        self,
        model: Elissabeth,
    ) -> None:
        self.model = model

    @classmethod
    def load(
        cls: type[Self],
        model: Elissabeth,
        model_id: str,
        on_cpu: bool = False,
    ) -> Self:
        load_path = None
        if model_id is not None:
            directory = Path.cwd()
            for folder in directory.iterdir():
                if not (folder.is_dir() and (folder / model_id).is_dir()):
                    continue
                if (load_path := (folder / model_id / "checkpoints")).exists():
                    load_path = load_path / next(load_path.iterdir())
                    saved_ = torch.load(
                        load_path,
                        map_location=torch.device("cpu") if on_cpu else None,
                    )
                    state_dict = {}
                    for key in saved_["state_dict"]:
                        if key.startswith("model"):
                            state_dict[key[6:]] = saved_["state_dict"][key]
                        else:
                            state_dict[key] = saved_["state_dict"][key]
                    model.load_state_dict(state_dict)
        if model_id is not None and load_path is None:
            raise FileExistsError("Model id does not point to a saved model")
        return cls(model)

    def plot_attention_matrices(
        self,
        example: torch.Tensor,
        show_example: bool = True,
        total: bool = False,
        length_index: int = 0,
        **kwargs,
    ) -> tuple[Figure, np.ndarray]:
        att_mat = get_attention_matrices(
            self.model,
            example,
            total=total,
            length_index=length_index,
        )
        figatt, axatt = plot_attention_matrix(
            att_mat[:, 0],
            example if show_example else None,
            contains_total=total,
            **kwargs,
        )
        return figatt, axatt

    def plot_parameter_matrix(
        self,
        name: str,
        reduce_dims: dict[int, int] | bool = False,
        **kwargs,
    ) -> tuple[Figure, np.ndarray]:
        param = self.model.detach_sorted_parameter(name)
        if isinstance(reduce_dims, dict):
            reduce_dims = dict(sorted(reduce_dims.items()))
            for d, i in reduce_dims.items():
                param = torch.index_select(param, dim=d, index=torch.tensor(i))
                param = param.squeeze(d)
        elif reduce_dims:
            param = param.squeeze()
            while param.ndim > 4:
                param = param[0]
        while param.ndim < 4:
            param = param.unsqueeze(0)
        if param.ndim > 4:
            raise IndexError(
                f"Expected 4 dimensions of parameter {name!r}, "
                f"but got {param.ndim}. Either specify {param.ndim-4} "
                "dimensions in 'reduce_dims' or set this option to 'True'."
            )
        return plot_parameter_matrix(param, **kwargs)
