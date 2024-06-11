import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Optional, Self

import numpy as np
import torch
from matplotlib.figure import Figure

from ..elissabeth.elissabeth import Elissabeth
from .plotting import (plot_alphabet_projection, plot_attention_matrix,
                       plot_parameter_matrix, plot_qkv_probing,
                       plot_time_parameters)
from .tools import (get_alphabet_projection, get_attention_matrices, get_iss,
                    get_query_key, get_values, probe_qkv_transform,
                    reduce_append_dims)


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
        reduce_dims: dict[int, int] | bool = False,
        append_dims: Sequence[int] | bool = True,
        **kwargs,
    ) ->  tuple[Figure, list[list[np.ndarray]]]:
        prb = probe_qkv_transform(self.model, which, layer, length, weighting)
        prb = reduce_append_dims(prb, None, reduce_dims, append_dims)
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
        x_axis: Optional[torch.Tensor] = None,
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
        return plot_time_parameters(iss, x_axis=x_axis, **kwargs)

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

    def plot_values_time(
        self,
        x: torch.Tensor,
        x_axis: Optional[torch.Tensor] = None,
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
        return plot_time_parameters(v, x_axis, **kwargs)

    def plot_query_key(
        self,
        x: torch.Tensor,
        which: Literal["Q", "K"],
        layer: int = 0,
        length: int = 0,
        weighting: int = 0,
        project_heads: tuple[int, ...] | bool = False,
        reduce_dims: dict[int, int] | bool = False,
        append_dims: Sequence[int] | bool = True,
        **kwargs,
    ) -> tuple[Figure, np.ndarray]:
        q, k = get_query_key(
            self.model,
            x,
            layer=layer,
            length=length,
            weighting=weighting,
            project_heads=project_heads,
        )
        if which == "Q":
            q = reduce_append_dims(q, 4, reduce_dims, append_dims)
            return plot_parameter_matrix(q, **kwargs)
        elif which == "K":
            k = reduce_append_dims(k, 4, reduce_dims, append_dims)
            return plot_parameter_matrix(k, **kwargs)

    def plot_query_key_time(
        self,
        x: torch.Tensor,
        x_axis: Optional[torch.Tensor] = None,
        layer: int = 0,
        length: int = 0,
        weighting: int = 0,
        names: Optional[tuple[str, ...]] = None,
        project_heads: tuple[int, ...] | bool = False,
        reduce_dims: dict[int, int] | bool = False,
        append_dims: Sequence[int] | bool = True,
        **kwargs,
    ) -> tuple[Figure, np.ndarray]:
        q, k = get_query_key(
            self.model,
            x,
            layer=layer,
            length=length,
            weighting=weighting,
            project_heads=project_heads,
        )
        q = reduce_append_dims(q, 4, reduce_dims, append_dims)
        k = reduce_append_dims(k, 4, reduce_dims, append_dims)
        return plot_time_parameters((q, k), x_axis, names=names, **kwargs)

    def plot_alphabet_projection(
        self,
        layer: int = 0,
        length: int = 0,
        weighting: int = 0,
        n: int = 0,
        p: int = 0,
        q: bool = True,
        k: bool = True,
        v: bool = True,
        transpose: bool = True,
        annotate_axes: bool = True,
        **kwargs,
    ) -> tuple[Figure, np.ndarray]:
        qkv = get_alphabet_projection(
            self.model,
            layer=layer,
            length=length,
            weighting=weighting,
            n=n,
            p=p,
        )
        qkvnew = ()
        labels = []
        if q:
            qkvnew = qkvnew + (qkv[0], )
            labels.append("Q")
        if k:
            qkvnew = qkvnew + (qkv[1], )
            labels.append("K")
        if v:
            qkvnew = qkvnew + (qkv[2], )
            labels.append("V")
        fig, ax = plot_alphabet_projection(
            qkvnew,
            transpose=transpose,
            **kwargs,
        )
        if annotate_axes:
            for i, axis in enumerate(ax[:, 0]):
                axis.set_ylabel(f"{i}" if transpose else f"{labels[i]}")
            for i, axis in enumerate(ax[0, :]):
                axis.set_title(f"{labels[i]}" if transpose else f"{i}")
        return fig, ax
