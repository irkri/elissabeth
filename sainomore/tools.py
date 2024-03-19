from typing import Literal, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .elissabeth import Elissabeth, LISSConfig


def get_attention_matrix(
    model: Elissabeth,
    x: torch.Tensor,
    dims: Literal[2, 3] = 2,
) -> np.ndarray:
    x = x.to(next(model.parameters()).device)
    model.attach_all_hooks()
    model(x)
    model.release_all_hooks()
    n_layers = model.config("n_layers")
    iss_length = model.config("length_is")
    B = x.size(0)
    T = x.size(1)
    N = model.config("n_is")
    if dims == 2:
        att_mat = np.empty((B, n_layers, iss_length, T, T, N))
    elif dims == 3:
        att_mat = np.empty((B, n_layers, iss_length-1, T, T, T, N))
    for l in range(n_layers):
        # shape Q/K: (B, T, N, L)
        try:
            Q = model.get_hook(f"layers.{l}", "Q").fwd
        except ValueError:
            Q = None
        try:
            K = model.get_hook(f"layers.{l}", "K").fwd
        except ValueError:
            K = None
        if dims == 2:
            for d in range(iss_length):
                iq = 0 if model.config.share_queries else d
                ik = 0 if model.config.share_keys else d
                if is_liss:
                    Q_ = 0 if Q is None else (Q[..., iq]
                        .repeat(1, T, 1).reshape(B, T, T, N).transpose(1, 2)
                    )
                    K_ = 0 if K is None else K[..., ik].unsqueeze(1)
                    att_mat[:, l, d] = torch.exp(Q_ - K_)  # type: ignore
                elif is_cliss:
                    D = model.config.d_query_key  # type: ignore
                    Q_ = 0 if Q is None else (Q[..., iq, :]
                        .repeat(1, T, 1, 1).reshape(B, T, T, N, D)
                        .transpose(1, 2)
                    )
                    K_ = 0 if K is None else K[..., ik, :].unsqueeze(1)
                    att_mat[:, l, d] = torch.prod(
                        torch.cos(Q_ - K_),  # type: ignore
                        dim=-1,
                    )
        elif dims == 3:
            for d in range(iss_length-1):
                if is_liss:
                    iq1 = 0 if model.config.share_queries else d
                    ik1 = 0 if model.config.share_keys else d
                    iq2 = 0 if model.config.share_queries else d+1
                    ik2 = 0 if model.config.share_keys else d+1
                    Q1 = 0 if Q is None else (Q[..., iq1]
                        .repeat(1, T, 1).reshape((B, T, T, N)).unsqueeze(3)
                        .transpose(1, 2)
                    )
                    K1 = 0 if K is None else (K[..., ik1]
                        .repeat(1, T, 1).reshape((B, T, T, N)).unsqueeze(3)
                    )
                    Q2 = 0 if Q is None else (Q[..., iq2]
                        .repeat(1, T, 1).reshape((B, T, T, N)).unsqueeze(3)
                    )
                    K2 = 0 if K is None else (K[..., ik2]
                        .repeat(1, T, 1).reshape((B, T, T, N)).unsqueeze(1)
                    )
                    att_mat[:, l, d] = torch.exp(
                        Q1 - K1 + Q2 - K2  # type: ignore
                    )
                elif is_cliss:
                    D = model.config.d_query_key  # type: ignore
                    iq1 = 0 if model.config.share_queries else d
                    ik1 = 0 if model.config.share_keys else d
                    iq2 = 0 if model.config.share_queries else d+1
                    ik2 = 0 if model.config.share_keys else d+1
                    Q1 = 0 if Q is None else (Q[..., iq1, :]
                        .repeat(1, T, 1, 1).reshape((B, T, T, N, D))
                        .unsqueeze(3).transpose(1, 2)
                    )
                    K1 = 0 if K is None else (K[..., ik1, :]
                        .repeat(1, T, 1, 1).reshape((B, T, T, N, D))
                        .unsqueeze(3)
                    )
                    Q2 = 0 if Q is None else (Q[..., iq2, :]
                        .repeat(1, T, 1, 1).reshape((B, T, T, N, D))
                        .unsqueeze(3)
                    )
                    K2 = 0 if K is None else (K[..., ik2, :]
                        .repeat(1, T, 1, 1).reshape((B, T, T, N, D))
                        .unsqueeze(1)
                    )
                    att_mat[:, l, d] = torch.prod(
                        torch.cos(Q1 - K1), # type: ignore
                        dim=-1,
                    ) * torch.prod(
                        torch.cos(Q2 - K2), # type: ignore
                        dim=-1,
                    )
    return att_mat


def plot_attention_matrix(
    matrix: np.ndarray,
    example: Optional[np.ndarray] = None,
    show_product: bool = False,
    cmap: str = "seismic",
    cmap_example: str = "Set1",
    causal_mask: bool = True,
    log_colormap: bool = False,
    **kwargs,
) -> tuple[Figure, np.ndarray]:
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
        Norm = LogNorm if log_colormap else Normalize
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
            max_ = np.nanmax(matrix[l])
            min_ = np.nanmin(matrix[l])
            for d in range(iss_length):
                mat_show = ax[l, d].matshow(
                    matrix[l, d],
                    cmap=cmap,
                    norm=Norm(vmin=min_, vmax=max_),
                )
                if not show_product and d == iss_length - 1:
                    content.append(mat_show)
                if mat is not None:
                    ax[l, d].matshow(mat, cmap=cmap_example)
                ax[l, d].tick_params(
                    top=False, left=False, bottom=False, right=False,
                    labeltop=False, labelleft=False, labelbottom=False,
                    labelright=False,
                )
                if l == 0:
                    ax[l, d].set_title(f"Length {d+1}")
            if show_product:
                content.append(ax[l, iss_length].matshow(
                        np.prod(matrix[l], 0),
                        cmap=cmap,
                        norm=Norm(vmin=min_, vmax=max_),
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
                if d == cols-1:
                    fig.colorbar(content[l], cax=axc)
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
