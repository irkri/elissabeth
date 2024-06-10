import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Optional, Self

import numpy as np
import torch
from matplotlib.figure import Figure

from ..elissabeth.elissabeth import Elissabeth
from .plotting import (plot_attention_matrix, plot_parameter_matrix,
                       plot_qkv_probing, plot_time_parameters)
from .tools import (get_attention_matrices, get_iss, get_values,
                    probe_qkv_transform, reduce_append_dims)


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
        only_kernels: Optional[tuple[int, ...]] = None,
        total: bool = False,
        project_heads: tuple[int, ...] | bool = False,
        layer: int = 0,
        length: int = 0,
        **kwargs,
    ) -> tuple[Figure, np.ndarray]:
        att_mat = get_attention_matrices(
            self.model,
            example,
            layer=layer,
            length=length,
            only_kernels=only_kernels,
            total=total,
            project_heads=project_heads,
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
        append_dims: Sequence[int] | bool = True,
        **kwargs,
    ) -> tuple[Figure, np.ndarray]:
        param = self.model.detach_sorted_parameter(name)
        param = reduce_append_dims(param, 4, reduce_dims, append_dims)
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

    def plot_iss(
        self,
        x: torch.Tensor,
        layer: int = 0,
        length: int = 0,
        project_heads: tuple[int, ...] | bool = False,
        project_values: bool = False,
        reduce_dims: dict[int, int] | bool = False,
        append_dims: Sequence[int] | bool = True,
        **kwargs,
    ) -> tuple[Figure, np.ndarray]:
        iss = get_iss(
            self.model,
            x,
            layer=layer,
            length=length,
            project_heads=project_heads,
            project_values=project_values,
        )
        iss = reduce_append_dims(iss, 4, reduce_dims, append_dims)
        return plot_parameter_matrix(iss, **kwargs)

    def plot_iss_time(
        self,
        x: torch.Tensor,
        layer: int = 0,
        length: int = 0,
        project_heads: tuple[int, ...] | bool = False,
        project_values: bool = False,
        reduce_dims: dict[int, int] | bool = False,
        append_dims: Sequence[int] | bool = True,
        **kwargs,
    ) -> tuple[Figure, np.ndarray]:
        iss = get_iss(
            self.model,
            x,
            layer=layer,
            length=length,
            project_heads=project_heads,
            project_values=project_values,
        )
        iss = reduce_append_dims(iss, 4, reduce_dims, append_dims)
        return plot_time_parameters(iss, **kwargs)

    def plot_values(
        self,
        x: torch.Tensor,
        layer: int = 0,
        length: int = 0,
        project_heads: tuple[int, ...] | bool = False,
        project_values: bool = False,
        reduce_dims: dict[int, int] | bool = False,
        append_dims: Sequence[int] | bool = True,
        **kwargs,
    ) -> tuple[Figure, np.ndarray]:
        v = get_values(
            self.model,
            x,
            layer=layer,
            length=length,
            project_heads=project_heads,
            project_values=project_values,
        )
        v = reduce_append_dims(v, 4, reduce_dims, append_dims)
        return plot_parameter_matrix(v, **kwargs)
