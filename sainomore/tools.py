from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_liss_attention_matrix(
    matrix: np.ndarray,
    example: Optional[np.ndarray] = None,
    cmap_example: str = "Set1",
    **kwargs,
) -> tuple[Figure, np.ndarray]:
    n_layers = matrix.shape[0]
    iss_length = matrix.shape[1]

    fig, ax = plt.subplots(n_layers, iss_length+1, **kwargs)
    if n_layers == 1:
        ax = np.array([ax])

    content = []
    for l in range(n_layers):
        max_ = np.max(matrix[l])
        min_ = np.min(matrix[l])
        for d in range(iss_length):
            ax[l, d].matshow(matrix[l, d], vmin=min_, vmax=max_, cmap="hot")
            ax[l, d].tick_params(
                top=False, left=False, bottom=False, right=False,
                labeltop=False, labelleft=False, labelbottom=False,
                labelright=False,
            )
            if l == 0:
                ax[l, d].set_title(f"Length {d+1}")
        content.append(ax[l, iss_length].matshow(
                np.prod(matrix[l], 0), vmin=min_, vmax=max_, cmap="hot",
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
