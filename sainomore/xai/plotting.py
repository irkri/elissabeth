from collections.abc import Sequence
from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import CenteredNorm, LogNorm, Normalize, SymLogNorm
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable


def _get_plot_cmap_norm(
    vmin: float,
    vmax: float,
    log: bool | tuple[float, float],
    center_zero: bool,
) -> Normalize:
    if log:
        if isinstance(log, tuple):
            return SymLogNorm(log[0], log[1], vmin=vmin, vmax=vmax)
        if vmin > 0:
            return LogNorm(vmin=vmin, vmax=vmax)
    if center_zero:
        return CenteredNorm(vcenter=0)
    return Normalize(vmin=vmin, vmax=vmax)


def plot_parameter_matrix(
    parameter: torch.Tensor | Sequence[Sequence[torch.Tensor]] |
        Sequence[torch.Tensor],
    cmap: str = "seismic",
    center_zero: bool = True,
    log_cmap: bool | tuple[float, float] = False,
    share_cmap: bool = True,
    **kwargs,
) -> tuple[Figure, np.ndarray]:
    if isinstance(parameter, Sequence):
        parameter = [
            [torch.clone(param) for param in parameter[i]]
            for i in range(len(parameter))
        ]
        rows, cols = len(parameter), len(parameter[0])
    else:
        parameter = torch.clone(parameter)
        rows, cols, *_ = parameter.shape

    fig, ax = plt.subplots(rows, cols, **kwargs)
    if rows == 1:
        ax = np.array([ax])
    if cols == 1:
        ax = np.array(ax)[:, np.newaxis]

    content = []
    for l in range(rows):
        content.append([])
        if share_cmap:
            max_ = np.nanmax(parameter[l])
            min_ = np.nanmin(parameter[l])
        for d in range(cols):
            if not share_cmap:
                max_ = np.nanmax(parameter[l][d])
                min_ = np.nanmin(parameter[l][d])
            content[-1].append(ax[l, d].matshow(
                parameter[l][d],
                cmap=cmap,
                norm=_get_plot_cmap_norm(min_, max_, log_cmap, center_zero),
            ))
            ax[l, d].tick_params(
                top=False, left=False, bottom=False, right=False,
                labeltop=False, labelleft=False, labelbottom=False,
                labelright=False,
            )

    for l in range(rows):
        for d in range(cols):
            divider = make_axes_locatable(ax[l, d])
            axc = divider.append_axes("right", size="5%", pad=0.1)
            if (share_cmap and d == cols-1) or not share_cmap:
                fig.colorbar(content[l][d], cax=axc)
            else:
                axc.remove()
    fig.tight_layout()
    return fig, ax


def plot_time_parameters(
    parameter: torch.Tensor | tuple[torch.Tensor, ...],
    x_axis: Optional[Sequence[str] | torch.Tensor] = None,
    names: Optional[tuple[str, ...]] = None,
    cmap: str = "tab20",
    **kwargs,
) -> tuple[Figure, np.ndarray]:
    if isinstance(parameter, tuple):
        parameter = tuple(torch.clone(p) for p in parameter)
        rows, cols, *_ = parameter[0].shape
        dims = parameter[0].size(2)
    else:
        parameter = torch.clone(parameter)
        rows, cols, *_ = parameter.shape
        dims = parameter.size(2)

    fig, ax = plt.subplots(rows, cols, **kwargs)
    if rows == 1:
        ax = np.array([ax])
    if cols == 1:
        ax = np.array(ax)[:, np.newaxis]

    colors = plt.get_cmap(cmap)
    for l in range(rows):
        for d in range(cols):
            for j in range(dims):
                if isinstance(parameter, tuple):
                    for i, p in enumerate(parameter):
                        if names is not None:
                            label = f"{names[i]} {j}"
                        else:
                            label = f"{j}"
                        ax[l, d].plot(
                            p[l, d, j],
                            label=label if l == 0 and d == 0 else None,
                            color=colors(j),
                            ls="--" if i % 2 == 0 else "-",
                        )
                else:
                    ax[l, d].plot(
                        parameter[l, d, j],
                        label=f"{j}" if l == 0 and d == 0 else None,
                        color=colors(j),
                    )
                if x_axis is not None:
                    ax[l, d].set_xticks(np.arange(len(x_axis)))
                    ax[l, d].set_xticklabels(x_axis)

    fig.tight_layout()
    fig.legend(title="Dimension")
    return fig, ax


def plot_attention_matrix(
    matrix: torch.Tensor,
    example: Optional[torch.Tensor] = None,
    xlabels: Optional[Sequence[str]] = None,
    contains_total: bool = False,
    cmap: str = "seismic",
    center_zero: bool = False,
    cmap_example: str = "Set1",
    causal_mask: bool = True,
    log_cmap: bool | tuple[float, float] = False,
    share_cmap: bool = True,
    **kwargs,
) -> tuple[Figure, np.ndarray]:
    """Plots the given attention matrix from an Elissabeth model.

    Args:
        matrix (torch.Tensor): Attention matrix for one LISS layer of
            shape ``(n_is, length_is, T, T)``.
        example (torch.Tensor, optional): The example the
            attention matrix was generated for. Defaults to None.
        xlabels (sequence of str, optional): Writes the given strings on
            horizontal and vertical axes of each attention matrix plot.
            Defaults to None.
        contains_total (bool, optional): Whether the last attention
            matrix in the ``length_is`` axis actually is the total
            weighting of the iterated sum. Defaults to False.
        cmap (str, optional): Colormap for the attention matrix.
            Defaults to "seismic".
        center_zero (bool, optional): Whether to center the colormap
            such that zero has color white. Defaults to False.
        cmap_example (str, optional): Colormap for the example that is
            plotted besides the matrix plot. Defaults to "Set1".
        causal_mask (bool, optional): Whether to automatically set a
            causal mask for the attention matrix. Defaults to True.
        log_map (bool, optional): Whether to scale the colormap for
            the attention matrix logarithmically. If a tuple of two
            floats is given, these are the arguments for matplotlibs
            ``SymLogNorm`` ``linthresh`` and ``linscale``. This is
            useful for inputs containing negative numbers. Defaults to
            False.
        share_map (bool, optional): Whether to share the colormap across
            different attention matrices in the same layer. Defaults
            to True.

    Returns:
        tuple[Figure, np.ndarray]: Figure and array of axes from
            matplotlib.
    """
    matrix = torch.clone(matrix)
    n_is, iss_length, *_ = matrix.shape

    fig, ax = plt.subplots(n_is, iss_length, **kwargs)
    if n_is == 1:
        ax = np.array([ax])
    if iss_length == 1:
        ax = np.array(ax)[:, np.newaxis]

    mat = None
    if causal_mask:
        triu_indices = np.triu_indices(matrix.shape[2])
        for l in range(n_is):
            for d in range(iss_length):
                matrix[l, d][triu_indices] = np.nan
        if example is not None:
            mat = np.empty_like(matrix[0, 0])
            mat.fill(np.nan)
            for i in range(mat.shape[0]):
                mat[i, i:] = example[i]
    content = []
    for l in range(n_is):
        content.append([])
        max_ = np.nanmax(matrix[l])
        min_ = np.nanmin(matrix[l])
        for d in range(iss_length):
            if not share_cmap:
                max_ = np.nanmax(matrix[l, d])
                min_ = np.nanmin(matrix[l, d])
            if mat is not None:
                ax[l, d].matshow(mat, cmap=cmap_example)
            content[-1].append(ax[l, d].matshow(
                matrix[l, d],
                cmap=cmap,
                norm=_get_plot_cmap_norm(min_, max_, log_cmap, center_zero),
            ))
            labels = (xlabels is not None)
            ax[l, d].tick_params(
                left=False, bottom=False, right=False, top=False,
                labelleft=labels, labelbottom=labels,
                labelright=False, labeltop=False,
            )
            if labels:
                ax[l, d].set_xticks(np.arange(matrix.shape[2]))
                ax[l, d].set_xticklabels(xlabels)
                ax[l, d].set_yticks(np.arange(matrix.shape[2]))
                ax[l, d].set_yticklabels(xlabels)
            if l == 0:
                t_r = "t_{" + f"{d+1}" + "}"
                if d == iss_length-1 and contains_total:
                    t_r = "t_1"
                if d == iss_length-1 or (d == iss_length-2 and contains_total):
                    t_l = "t"
                else:
                    t_l = "t_{" + f"{d+2}" + "}"
                ax[l, d].set_title(f"$K({t_l}, {t_r})$")
            if example is None:
                ax[l, d].set_ylabel(f"${t_l}$")
                ax[l, d].set_xlabel(f"${t_r}$")

    for l in range(n_is):
        for d in range(iss_length):
            divider = make_axes_locatable(ax[l, d])
            axc = divider.append_axes("right", size="5%", pad=0.1)
            if example is not None:
                axb = divider.append_axes(
                    "bottom", size="6%", pad=0.05, sharex=ax[l, d],
                )
                axl = divider.append_axes(
                    "left", size="6%", pad=0.05, sharey=ax[l, d],
                )
                axb.matshow(np.expand_dims(example, 0), cmap=cmap_example)
                axb.tick_params(
                    top=False, left=False, bottom=False, right=False,
                    labeltop=False, labelleft=False, labelbottom=False,
                    labelright=False,
                )
                axl.matshow(np.expand_dims(example, 1), cmap=cmap_example)
                axl.tick_params(
                    top=False, left=False, bottom=False, right=False,
                    labeltop=False, labelleft=False, labelbottom=False,
                    labelright=False,
                )
                t_r = "t_{" + f"{d+1}" + "}"
                if d == iss_length-1 and contains_total:
                    t_r = "t_1"
                if d == iss_length-1 or (d == iss_length-2 and contains_total):
                    t_l = "t"
                else:
                    t_l = "t_{" + f"{d+2}" + "}"
                axl.set_ylabel(f"${t_l}$")
                axb.set_xlabel(f"${t_r}$")
            if (share_cmap and d == iss_length-1) or not share_cmap:
                fig.colorbar(content[l][d], cax=axc)
            else:
                axc.remove()
    fig.tight_layout()
    return fig, ax


def plot_qkv_probing(
    probing: torch.Tensor,
    norm_p: float | str = "fro",
    sharey: bool = True,
    cmap: str = "tab20",
    **kwargs,
) -> tuple[Figure, list[list[np.ndarray]]]:
    """Plots the given qkv probing from an Elissabeth model.

    Args:
        probing (torch.Tensor): Probing of one LISS layer of shape
            ``(d_in, T, n_is, length_is, d_out, ...)``. All additional
            dimension get reduced by taking the norm over them.
        norm_p (float | str, optional): Norm to use for reducing
            additional dimensions in the input. The format is the same
            as the argument ``p`` in the function ``torch.norm``.
            Defaults to 'fro'.
        sharey (bool, optional): If the y-axis of all ``d_in`` plots
            should be shared. Defaults to True.
        cmap (str, optional): Colormap for the different ``d_out`` lines
            of a single plot. Defaults to 'tab20'.

    Returns:
        tuple[Figure, list[list[np.ndarray]]]: Figure and list of list
            of array of axes from matplotlib. The outer two lists
            correspond to subfigures of the main figure.
    """
    in_, T, n_is, lengths, out_, *other = probing.shape
    size_other = int(np.prod(other) if len(other) > 1 else 1)
    fig = plt.figure(**kwargs)
    subfigs = fig.subfigures(n_is, lengths)
    if n_is == 1:
        subfigs = np.array([subfigs])
    if lengths == 1:
        subfigs = np.array(subfigs)[:, np.newaxis]
    colorwheel = plt.get_cmap(cmap)
    ax: list[list[np.ndarray]] = []
    for n in range(n_is):
        ax.append([])
        for p in range(lengths):
            ax[-1].append(subfigs[n, p].subplots(1, in_, sharey=sharey))
            for i in range(in_):
                for j in range(out_):
                    display = probing[i, :, n, p, j]
                    if len(other) > 1 and not (len(other)==1 and other[0]==1):
                        display = torch.norm(
                            display.reshape(T, size_other),
                            p=norm_p,
                            dim=-1,
                        )
                    ax[-1][-1][i].plot(
                        display,
                        label=f"{j}" if i==0 else None,
                        color=colorwheel(j),
                    )
                    ax[-1][-1][i].set_xticks([])
                    ax[-1][-1][i].set_xticklabels([])
            if n == 0 and p == lengths - 1:
                subfigs[n, p].legend(title="Dimension")
    return fig, ax


def plot_alphabet_projection(
    qkv: tuple[torch.Tensor, ...],
    transpose: bool = True,
    cmap: str = "seismic",
    center_zero: bool = True,
    log_cmap: bool | tuple[float, float] = False,
    share_cmap: bool = False,
    **kwargs,
) -> tuple[Figure, np.ndarray]:
    if transpose:
        qkvT = [
            [qkv[i][j] for i in range(len(qkv))]
            for j in range(qkv[0].shape[0])
        ]
    fig, ax = plot_parameter_matrix(
        qkvT if transpose else qkv,
        cmap=cmap,
        center_zero=center_zero,
        log_cmap=log_cmap,
        share_cmap=share_cmap,
        **kwargs,
    )
    return fig, ax
