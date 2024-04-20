from typing import Literal, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .elissabeth import Elissabeth


def get_attention_matrices(
    model: Elissabeth,
    x: torch.Tensor,
    total: bool = False,
    level_index: int = 0,
) -> torch.Tensor:
    """Returns the attention matrices in an Elissabeth model generated
    by the input ``x``.

    Args:
        model (Elissabeth): Trained Elissabeth model.
        x (torch.Tensor): Example to generate the attention matrix for.
            Has to be of shape ``(T, )`` for token input or ``(T, d)``
            for vector input.
        total (bool, optional): If set to True, also calculates the
            attention matrix for the whole iterated sum ``Att_{t_1,t}``
            by calculating the iterated sum of all weightings. Defaults
            to False.
        level_index (int, optional): The index of the ISS level to
            extract the attention matrices from. Defaults to 0.

    Returns:
        torch.Tensor: Attention matrix of shape
            ``(n_is, n_layers, length_is, T, T)``.
    """
    for layer in model.layers:
        for weighting in layer.levels[level_index].weightings:
            weighting.hooks.get("Att").attach()

    model(x.to(next(model.parameters()).device).unsqueeze(0))

    for layer in model.layers:
        for weighting in layer.levels[level_index].weightings:
            weighting.hooks.get("Att").release()

    n_layers = model.config("n_layers")
    if model.layers[0].config("levels") is not None:
        iss_length = model.layers[0].config("levels")[level_index]
    else:
        iss_length = level_index + 1
    N = model.layers[0].config("n_is")
    att_mat = torch.ones((N, n_layers, iss_length, x.size(0), x.size(0)))

    for l in range(n_layers):
        for weighting in model.layers[l].levels[level_index].weightings:
            att_mat[:, l, :, :, :] *= weighting.hooks.get("Att").fwd[0]

    if total:
        total_att = torch.zeros((N, n_layers, x.size(0), x.size(0)))
        ind = torch.triu_indices(x.size(0), x.size(0), offset=0)
        total_att[:, :] = att_mat[:, :, 0]
        total_att[:, :, *ind] = 0
        for p in range(1, iss_length):
            if p == iss_length - 1:
                ind = torch.triu_indices(x.size(0), x.size(0), offset=1)
            mat = torch.clone(att_mat[:, :, p, :, :])
            mat[:, :, *ind] = 0
            total_att[:, :, :, :] = mat @ total_att
        att_mat = torch.cat((att_mat, total_att.unsqueeze(2)), dim=2)

    return att_mat


def _get_plot_cmap_norm(vmin: float, vmax: float, log: bool) -> Normalize:
    if log and vmin > 0:
        return LogNorm(vmin=vmin, vmax=vmax)
    else:
        return Normalize(vmin=vmin, vmax=vmax)


def plot_attention_matrix(
    matrix: torch.Tensor,
    example: Optional[torch.Tensor] = None,
    contains_total: bool = False,
    cmap: str = "seismic",
    cmap_example: str = "Set1",
    causal_mask: bool = True,
    log_cmap: bool = False,
    share_cmap: bool = True,
    **kwargs,
) -> tuple[Figure, np.ndarray]:
    """Plots the given attention matrix from an Elissabeth model.

    Args:
        matrix (np.ndarray): Attention matrix of shape
            ``(n_layers, length_is, T, T)``.
        example (Optional[torch.Tensor], optional): The example the
            attention matrix was generated for. Defaults to None.
        contains_total (bool, optional): Whether the last attention
            matrix in the ``length_is`` axis actually is the total
            weighting of the iterated sum. Defaults to False.
        cmap (str, optional): Colormap for the attention matrix.
            Defaults to "seismic".
        cmap_example (str, optional): Colormap for the example that is
            plotted besides the matrix plot. Defaults to "Set1".
        causal_mask (bool, optional): Whether to automatically set a
            causal mask for the attention matrix. Defaults to True.
        log_map (bool, optional): Whether to scale the colormap for
            the attention matrix logarithmically. Defaults to False.
        share_map (bool, optional): Whether to share the colormap across
            different attention matrices in the same layer. Defaults
            to True.

    Returns:
        tuple[Figure, np.ndarray]: Figure and array of axes from
            matplotlib.
    """
    matrix = torch.clone(matrix)
    n_layers, iss_length, *_ = matrix.shape

    fig, ax = plt.subplots(n_layers, iss_length, **kwargs)
    if n_layers == 1:
        ax = np.array([ax])
    if iss_length == 1:
        ax = np.array(ax)[:, np.newaxis]

    mat = None
    if causal_mask:
        triu_indices = np.triu_indices(matrix.shape[2])
        for l in range(n_layers):
            for d in range(iss_length):
                matrix[l, d][triu_indices] = np.nan
        if example is not None:
            mat = np.empty_like(matrix[0, 0])
            mat.fill(np.nan)
            for i in range(mat.shape[0]):
                mat[i, i:] = example[i]
    content = []
    for l in range(n_layers):
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
                norm=_get_plot_cmap_norm(min_, max_, log_cmap),
            ))
            ax[l, d].tick_params(
                top=False, left=False, bottom=False, right=False,
                labeltop=False, labelleft=False, labelbottom=False,
                labelright=False,
            )
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

    for l in range(n_layers):
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
