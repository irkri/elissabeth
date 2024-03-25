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
    dims: Literal[2, 3] = 2,
) -> torch.Tensor:
    """Returns the attention matrices in an Elissabeth model generated
    by the input ``x``.

    Args:
        model (Elissabeth): Trained Elissabeth model.
        x (torch.Tensor): Example to generate the attention matrix for.
            Has to be of shape ``(T, )`` for token input or ``(T, d)``
            for vector input.
        dims (Literal[2, 3], optional): _description_. Defaults to 2.

    Returns:
        torch.Tensor: Attention matrix of shape
            ``(n_is, n_layers, length_is, T, T)``
    """
    hook_key = "Att" if dims == 2 else "Att 3d"
    for layer in model.layers:
        for weighting in layer.weightings:
            weighting.hooks.get(hook_key).attach()

    model(x.to(next(model.parameters()).device).unsqueeze(0))

    for layer in model.layers:
        for weighting in layer.weightings:
            weighting.hooks.get(hook_key).release()

    n_layers = model.config("n_layers")
    iss_length = model.layers[0].config("length_is")
    N = model.layers[0].config("n_is")
    att_mat = torch.ones(
        (N, n_layers, iss_length, x.size(0), x.size(0)) if dims == 2 else
        (N, n_layers, iss_length, x.size(0), x.size(0), x.size(0))
    )

    for l in range(n_layers):
        for weighting in model.layers[l].weightings:
            att_mat[:, l, ...] *= weighting.hooks.get(hook_key).fwd[0]

    return att_mat


def _get_plot_cmap_norm(vmin: float, vmax: float, log: bool) -> Normalize:
    if log and vmin > 0:
        return LogNorm(vmin=vmin, vmax=vmax)
    else:
        return Normalize(vmin=vmin, vmax=vmax)


def plot_attention_matrix(
    matrix: torch.Tensor,
    example: Optional[torch.Tensor] = None,
    show_product: bool = False,
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
        show_product (bool, optional): Whether to also show the product
            of all attention matrices in one layer. Defaults to False.
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
    n_layers, iss_length, T, *_ = matrix.shape
    is_3d = matrix.ndim == 5

    cols = iss_length+1 if show_product and not is_3d else iss_length
    fig, ax = plt.subplots(
        n_layers, cols,
        subplot_kw=dict(projection='3d') if is_3d else None,
        **kwargs,
    )
    if n_layers == 1:
        ax = np.array([ax])
    if cols == 1:
        ax = np.array(ax)[:, np.newaxis]

    if not is_3d:
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
                content[-1].append(ax[l, d].matshow(
                    matrix[l, d],
                    cmap=cmap,
                    norm=_get_plot_cmap_norm(min_, max_, log_cmap),
                ))
                if mat is not None:
                    ax[l, d].matshow(mat, cmap=cmap_example)
                ax[l, d].tick_params(
                    top=False, left=False, bottom=False, right=False,
                    labeltop=False, labelleft=False, labelbottom=False,
                    labelright=False,
                )
                if l == 0:
                    ax[l, d].set_title(
                        "$K("
                        + "t" + ("_{"+f"{d+2}"+"}" if d+1 < iss_length else "")
                        + ", t_{"+f"{d+1}"+"})$"
                    )
                if example is None:
                    ax[l, d].set_ylabel("$t"
                        + ("_{"+f"{d+2}"+"}$" if d+1 < iss_length else "$")
                    )
                    ax[l, d].set_xlabel("$t_{"+f"{d+1}"+"}$")
            if show_product:
                if not share_cmap:
                    max_ = np.nanmax(matrix[l, d])
                    min_ = np.nanmin(matrix[l, d])
                content[-1].append(ax[l, iss_length].matshow(
                    np.prod(matrix[l], 0),
                    cmap=cmap,
                    norm=_get_plot_cmap_norm(min_, max_, log_cmap),
                ))
                if mat is not None:
                    ax[l, iss_length].matshow(mat, cmap=cmap_example)
                ax[l, iss_length].set_title("Product")
                ax[l, iss_length].tick_params(
                    top=False, left=False, bottom=False, right=False,
                    labeltop=False, labelleft=False, labelbottom=False,
                    labelright=False,
                )

        for l in range(n_layers):
            for d in range(cols):
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
                    axl.set_ylabel("$t"
                        + ("_{"+f"{d+2}"+"}$" if d+1 < iss_length else "$")
                    )
                    axb.set_xlabel("$t_{"+f"{d+1}"+"}$")
                if (share_cmap and d == cols-1) or not share_cmap:
                    fig.colorbar(content[l][d], cax=axc)
                else:
                    axc.remove()
    else:
        def explode(data):
            size = np.array(data.shape)
            size[:3] = 2*size[:3] - 1
            data_e = np.zeros(size, dtype=data.dtype)
            data_e[::2, ::2, ::2] = data
            return data_e

        filled = np.ones((T, T, T))
        if causal_mask:
            triu_indices = np.triu_indices(T)
            filled[triu_indices[0], triu_indices[1], :] = 0
            filled[:, triu_indices[0], triu_indices[1]] = 0
        filled = explode(filled)
        x, y, z = np.indices(
            tuple(np.array(filled.shape) + 1)
        ).astype(float) // 2
        x[0::2, :, :] += 0.01
        y[:, 0::2, :] += 0.01
        z[:, :, 0::2] += 0.01
        x[1::2, :, :] += 0.99
        y[:, 1::2, :] += 0.99
        z[:, :, 1::2] += 0.99
        for l in range(n_layers):
            for d in range(iss_length):
                fc = np.moveaxis(np.array(
                    np.vectorize(plt.get_cmap(cmap))(matrix[l, d])
                ), 0, -1)
                ec = fc.copy()
                fc[..., 3] = 0.8
                ax[l, d].voxels(
                    filled,
                    facecolors=explode(fc),
                    edgecolors=explode(ec),
                )

    fig.tight_layout()

    return fig, ax
