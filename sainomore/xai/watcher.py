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
    """Class that implements useful methods to analyze trained
    Elissabeth models by combining and wrapping methods of
    `sainomore.xai.tools` and `sainomore.xai.plotting`.

    A convenient way of starting a watcher instance is to use
    `ElissabethWatcher.load` given the string pointing to a trained
    Elissabeth model.

    Args:
        model (Elissabeth): The model this class should watch.
    """

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
        """Returns a new ElissabethWatcher instance for a trained model.

        Args:
            model_id (str): The path of the directory containing
                checkpoints and the configuration json file of the
                trained Elissabeth model. This directory typically is a
                folder named like the id of the model.
            model (Elissabeth, optional): The class that is used to load
                the model. Defaults to the standard Elissabeth class.
            on_cpu (bool, optional): If set to True, forces all
                parameters to be mapped to CPU. Defaults to False.
        """
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

    def get_values(
        self,
        x: torch.Tensor,
        layer: int = 0,
        length: int = 0,
        project_heads: tuple[int, ...] | bool = False,
        project_values: bool = False,
    ) -> torch.Tensor:
        return get_values(
            self.model,
            x,
            layer=layer,
            length=length,
            project_heads=project_heads,
            project_values=project_values,
        )

    def plot_attention_matrices(
        self,
        example: torch.Tensor,
        xlabels: Optional[Sequence[str]] = None,
        show_example: bool = True,
        layer: int = 0,
        length: int = 0,
        only_kernels: Optional[tuple[int, ...]] = None,
        value_direction: Optional[int | tuple[int, int]] = None,
        all_but_first_value: bool = False,
        total: bool = False,
        project_heads: tuple[int, ...] | bool = False,
        reduce_dims: dict[int, int] | bool = False,
        append_dims: Sequence[int] | bool = True,
        index_selection: Optional[Sequence[tuple[int, torch.Tensor]]] = None,
        **kwargs,
    ) -> tuple[Figure, np.ndarray]:
        """Plots the given attention matrix of the given Elissabeth model.

        Args:
            example (torch.Tensor): The example the attention matrix is
                generated for.
            xlabels (sequence of str, optional): Writes the given
                strings on horizontal and vertical axes of each
                attention matrix plot. Defaults to None.
            show_example (bool, optional): Whether the example is shown
                in the plot at the edges of the matrices. Defaults to
                True.
            layer (int, optional): The index of the LISS layer to
                extract the attention matrices from. Defaults to 0.
            length (int, optional): The index of the ISS level to
                extract the attention matrices from. Defaults to 0.
            only_kernels (tuple of int, optional): Restricts the
                attention matrix to kernels with the given indices.
                Defaults to None.
            value_direction (int or tuple of int, optional): If given,
                multiplies the given value dimension of time step ``s``
                to the corresponding kernel ``K(t,s)``.
            all_but_first_value (bool, optional): If set to True, does
                not multiply the value corresponding to the most inner
                index of the iterated sum to the corresponding kernel.
                Defaults to False.
            total (bool, optional): If set to True, also calculates the
                attention matrix for the whole iterated sum
                ``Att_{t_1,t}`` by calculating the iterated sum of all
                weightings. Defaults to False.
            project_heads (tuple of int | bool, optional): Whether the
                ``n_is`` iterated sums of the model should be linearly
                projected using the ``W_H`` matrix in the model. The
                result then is a tensor with first dimension of size 1.
                If a tuple of indices is given, only these iterated sums
                are considered. Defaults to False.
            contains_total (bool, optional): Whether the last attention
                matrix in the ``length_is`` axis actually is the total
                weighting of the iterated sum. Defaults to False.
            cmap (str, optional): Colormap for the attention matrix.
                Defaults to "seismic".
            center_zero (bool, optional): Whether to center the colormap
                such that zero has color white. Defaults to False.
            cmap_example (str, optional): Colormap for the example that
                is plotted besides the matrix plot. Defaults to "Set1".
            causal_mask (bool, optional): Whether to automatically set a
                causal mask for the attention matrix. Defaults to True.
            log_map (bool, optional): Whether to scale the colormap for
                the attention matrix logarithmically. If a tuple of two
                floats is given, these are the arguments for matplotlibs
                ``SymLogNorm`` ``linthresh`` and ``linscale``. This is
                useful for inputs containing negative numbers. Defaults
                to False.
            share_map (bool, optional): Whether to share the colormap
                across different attention matrices in the same layer.
                Defaults to True.
            reduce_dims (dict[int, int] | bool, optional): Reduces the
                dimensionality of the generated attention matrix to the
                given shape or automatically forces the matrix into the
                needed shape for plotting. Defaults to False.
            append_dims (tuple of int | bool, optional): Increases the
                dimensionality of the generated attention matrix to the
                given shape or automatically adds dimensions to the
                matrix for plotting. Defaults to True.
            index_selection (sequence of tuple[int, torch.Tensor],
                optional): Selects only the given indices at the
                corresponding dimension from the attention matrix before
                plotting. Defaults to None.
        Returns:
            tuple[Figure, np.ndarray]: Figure and array of axes from
                matplotlib.
        """
        att = get_attention_matrices(
            self.model,
            example,
            layer=layer,
            length=length,
            only_kernels=only_kernels,
            value_direction=value_direction,
            all_but_first_value=all_but_first_value,
            total=total,
            project_heads=project_heads,
        )
        att = reduce_append_dims(att, 4, reduce_dims, append_dims)
        if index_selection is not None:
            for selection in index_selection:
                att = torch.index_select(att, selection[0], selection[1])
        figatt, axatt = plot_attention_matrix(
            att,
            example if show_example else None,
            xlabels=xlabels,
            contains_total=total,
            **kwargs,
        )
        return figatt, axatt

    def plot_parameter_matrix(
        self,
        name: str,
        reduce_dims: dict[int, int] | bool = False,
        append_dims: Sequence[int] | bool = True,
        index_selection: Optional[Sequence[tuple[int, torch.Tensor]]] = None,
        **kwargs,
    ) -> tuple[Figure, np.ndarray]:
        param = self.model.detach_sorted_parameter(name)
        param = reduce_append_dims(param, 4, reduce_dims, append_dims)
        if index_selection is not None:
            for selection in index_selection:
                param = torch.index_select(param, selection[0], selection[1])
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
        index_selection: Optional[Sequence[tuple[int, torch.Tensor]]] = None,
        **kwargs,
    ) ->  tuple[Figure, list[list[np.ndarray]]]:
        prb = probe_qkv_transform(self.model, which, layer, length, weighting)
        prb = reduce_append_dims(prb, None, reduce_dims, append_dims)
        if index_selection is not None:
            for selection in index_selection:
                prb = torch.index_select(prb, selection[0], selection[1])
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
        index_selection: Optional[Sequence[tuple[int, torch.Tensor]]] = None,
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
        if index_selection is not None:
            for selection in index_selection:
                iss = torch.index_select(iss, selection[0], selection[1])
        return plot_parameter_matrix(iss, **kwargs)

    def plot_iss_time(
        self,
        x: torch.Tensor,
        x_axis: Optional[Sequence[str] | torch.Tensor] = None,
        layer: int = 0,
        length: int = 0,
        project_heads: tuple[int, ...] | bool = False,
        project_values: bool = False,
        reduce_dims: dict[int, int] | bool = False,
        append_dims: Sequence[int] | bool = True,
        index_selection: Optional[Sequence[tuple[int, torch.Tensor]]] = None,
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
        if index_selection is not None:
            for selection in index_selection:
                iss = torch.index_select(iss, selection[0], selection[1])
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
        index_selection: Optional[Sequence[tuple[int, torch.Tensor]]] = None,
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
        if index_selection is not None:
            for selection in index_selection:
                iss = torch.index_select(iss, selection[0], selection[1])
        return plot_parameter_matrix(v, **kwargs)

    def plot_values_time(
        self,
        x: torch.Tensor,
        x_axis: Optional[Sequence[str] | torch.Tensor] = None,
        layer: int = 0,
        length: int = 0,
        project_heads: tuple[int, ...] | bool = False,
        project_values: bool = False,
        reduce_dims: dict[int, int] | bool = False,
        append_dims: Sequence[int] | bool = True,
        index_selection: Optional[Sequence[tuple[int, torch.Tensor]]] = None,
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
        if index_selection is not None:
            for selection in index_selection:
                v = torch.index_select(v, selection[0], selection[1])
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
        index_selection: Optional[Sequence[tuple[int, torch.Tensor]]] = None,
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
        x_axis: Optional[Sequence[str] | torch.Tensor] = None,
        layer: int = 0,
        length: int = 0,
        weighting: int = 0,
        names: Optional[tuple[str, ...]] = None,
        project_heads: tuple[int, ...] | bool = False,
        reduce_dims: dict[int, int] | bool = False,
        append_dims: Sequence[int] | bool = True,
        index_selection: Optional[Sequence[tuple[int, torch.Tensor]]] = None,
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
        tokens: Optional[torch.Tensor] = None,
        positions: Optional[int] = None,
        transpose: bool = True,
        annotate_axes: bool = True,
        reduce_dims: dict[int, int] | bool = False,
        append_dims: Sequence[int] | bool = True,
        **kwargs,
    ) -> tuple[Figure, np.ndarray]:
        qkv = get_alphabet_projection(
            self.model,
            layer=layer,
            length=length,
            weighting=weighting,
            n=n,
            p=p,
            q=q,
            k=k,
            v=v,
            tokens=tokens,
            positions=positions,
        )
        qkv = tuple(map(
            lambda x: reduce_append_dims(x, None, reduce_dims, append_dims),
            qkv
        ))
        labels = []
        if q:
            labels.append("Q")
        if k:
            labels.append("K")
        if v:
            labels.append("V")
        fig, ax = plot_alphabet_projection(
            qkv,
            transpose=transpose,
            **kwargs,
        )
        if annotate_axes:
            for i, axis in enumerate(ax[:, 0]):
                axis.set_ylabel(
                    f"{tokens[i] if tokens is not None else i}"
                    if transpose else f"{labels[i]}"
                )
            for i, axis in enumerate(ax[0, :]):
                axis.set_title(
                    f"{labels[i]}" if transpose else
                    f"{tokens[i] if tokens is not None else i}"
                )
        return fig, ax
