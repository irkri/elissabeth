import torch
import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from .models.liss import Elissabeth, ElissabethConfig


def plot_liss_attention_matrix(
    model: Elissabeth,
    config: ElissabethConfig,
    x: torch.Tensor,
    qk_index: int = 0,
) -> tuple[Figure, Axes]:
    model.attach_all_hooks()
    _ = model(x)
    model.release_all_hooks()

    fig, ax = plt.subplots(config.n_layers, config.iss_length+1)

    att_mat = np.empty(
        (config.n_layers, config.iss_length, x.size(1), x.size(1))
    )
    for l in range(config.n_layers):
        for d in range(config.iss_length):
            att_mat[l, d] = model.get_hook(
                f"layers.{l}", f"weighting.{d}",
            ).fwd[0, :, :, qk_index]

    content = []
    for l in range(config.n_layers):
        max_ = np.max(att_mat[l])
        min_ = np.min(att_mat[l])
        for d in range(config.iss_length):
            ax[l, d].matshow(model.get_hook(
                f"layers.{l}", f"weighting.{d}",
            ).fwd[0, :, :, 0], vmin=min_, vmax=max_, cmap="hot")
            ax[l, d].set_title(f"Layer {l}, IS {d}")
        content.append(
            ax[l, config.iss_length].matshow(
                np.prod(att_mat[l], 0), vmin=min_, vmax=max_, cmap="hot",
            )
        )
        ax[l, config.iss_length].set_title(f"Product of Layer {l}")

    axis_1 = fig.add_subplot(config.n_layers, config.iss_length+2, 2)
    axis_2 = fig.add_subplot(config.n_layers, config.iss_length+2, config.iss_length+2)
    axis_1.matshow(x.T.repeat(1, 5), cmap="gray")
    axis_1.set_title("input")
    axis_1.xaxis.set_visible(False)
    axis_2.matshow(x.T.repeat(1, 5), cmap="gray")
    axis_2.set_title("input")
    axis_2.xaxis.set_visible(False)
    fig.tight_layout()
    fig.colorbar(content[0])
    fig.colorbar(content[1])

    return fig, ax
