import json
from pathlib import Path
from typing import Literal, Optional, Self

import numpy as np
import torch
from matplotlib.figure import Figure

from ..elissabeth.elissabeth import Elissabeth
from .plotting import (plot_attention_matrix, plot_parameter_matrix,
                       plot_qkv_probing)
from .tools import get_attention_matrices, probe_qkv_transform


class ElissabethWatcher:

    def __init__(
        self,
        model: Elissabeth,
    ) -> None:
        self.model = model

    @classmethod
    def load(
        cls: type[Self],
        model_id: str,
        model: Optional[Elissabeth] = None,
        on_cpu: bool = False,
    ) -> Self:
        load_path = None
        directory = Path.cwd()
        for folder in directory.iterdir():
            if not (folder.is_dir() and (folder / model_id).is_dir()):
                continue
            if (load_path := (folder / model_id / "checkpoints")).exists():
                config_path = load_path.parent / "config.json"
                if config_path.is_file():
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    model = Elissabeth.build(config)
                if model is None:
                    raise RuntimeError("Model config not found")
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
        if load_path is None or model is None:
            raise FileExistsError("Given id does not point to a saved model")
        return cls(model)

    def plot_attention_matrices(
        self,
        example: torch.Tensor,
        show_example: bool = True,
        total: bool = False,
        layer: int = 0,
        length: int = 0,
        **kwargs,
    ) -> tuple[Figure, np.ndarray]:
        att_mat = get_attention_matrices(
            self.model,
            example,
            layer=layer,
            length=length,
            total=total,
        )
        figatt, axatt = plot_attention_matrix(
            att_mat,
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
                f"but got {param.ndim}: {param.shape}. "
                f"Either specify {param.ndim-4} dimensions in 'reduce_dims' "
                "or set this option to 'True'."
            )
        return plot_parameter_matrix(param, **kwargs)

    def plot_qkv_probing(
        self,
        which: Literal["q", "k", "v"] = "v",
        layer: int = 0,
        length: int = 0,
        weighting: int = 0,
        norm_p: float | str = "fro",
        sharey: bool = True,
        cmap: str = "tab20",
        **kwargs,
    ) ->  tuple[Figure, list[list[np.ndarray]]]:
        prb = probe_qkv_transform(self.model, which, layer, length, weighting)
        return plot_qkv_probing(prb, norm_p, sharey, cmap, **kwargs)
