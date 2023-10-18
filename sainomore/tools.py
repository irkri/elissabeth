from typing import Optional

import numpy as np
import torch
from matplotlib.colors import LogNorm, Normalize
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .models import Elissabeth


def get_liss_attention_matrix(
    model: Elissabeth,
    x: torch.Tensor,
) -> np.ndarray:
    x = x.to(next(model.parameters()).device)
    model.attach_all_hooks()
    model(x)
    model.release_all_hooks()
    n_layers = model.config.n_layers
    iss_length = model.config.length_is
    att_mat = np.empty(
        (n_layers, iss_length, model.config.n_is, x.size(1), x.size(1))
    )
    for l in range(n_layers):
        for d in range(iss_length):
            att_mat[l, d] = model.get_hook(
                f"layers.{l}", f"weighting.{d}",
            ).fwd[0, :, :, :].swapaxes(0, 2).swapaxes(1, 2)
    return att_mat


def plot_liss_attention_matrix(
    matrix: np.ndarray,
    example: Optional[np.ndarray] = None,
    cmap: str = "seismic",
    cmap_example: str = "Set1",
    causal_mask: bool = True,
    log_colormap: bool = False,
    **kwargs,
) -> tuple[Figure, np.ndarray]:
    n_layers = matrix.shape[0]
    iss_length = matrix.shape[1]

    fig, ax = plt.subplots(n_layers, iss_length+1, **kwargs)
    if n_layers == 1:
        ax = np.array([ax])

    Norm = LogNorm if log_colormap else Normalize

    indices = np.tril_indices(matrix.shape[2])
    content = []
    for l in range(n_layers):
        max_ = np.max(matrix[l])
        min_ = np.min(matrix[l])
        for d in range(iss_length):
            if causal_mask:
                matrix[l, d][indices] = np.nan
            ax[l, d].matshow(
                matrix[l, d],
                cmap=cmap,
                norm=Norm(vmin=min_, vmax=max_),
            )
            ax[l, d].tick_params(
                top=False, left=False, bottom=False, right=False,
                labeltop=False, labelleft=False, labelbottom=False,
                labelright=False,
            )
            if l == 0:
                ax[l, d].set_title(f"Length {d+1}")
        content.append(ax[l, iss_length].matshow(
                np.prod(matrix[l], 0),
                cmap=cmap,
                norm=Norm(vmin=min_, vmax=max_),
        ))
        ax[l, iss_length].set_title("Product")
        ax[l, iss_length].tick_params(
            top=False, left=False, bottom=False, right=False,
            labeltop=False, labelleft=False, labelbottom=False,
            labelright=False,
        )

    for l in range(n_layers):
        for d in range(iss_length+1):
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
            if d == iss_length:
                fig.colorbar(content[l], cax=axc)
            else:
                axc.remove()

    fig.tight_layout()

    return fig, ax
