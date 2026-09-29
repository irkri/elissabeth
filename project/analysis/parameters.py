"""Heatmap of a model parameter, or list them all.

    python analysis/parameters.py run_0001                 # list
    python analysis/parameters.py run_0001 mixers.0.W_O    # plot

A parameter with more than two axes is drawn as a grid over its leading
axes (the last two are the image).
"""
import argparse

import matplotlib.pyplot as plt

from common import CHECKPOINTS, finish
from elissabeth import load_model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("name", nargs="?")
    p.add_argument("--weights", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    model, _, _ = load_model(args.run, CHECKPOINTS, args.weights)
    if args.name is None:
        for name, param in model.named_parameters():
            print(f"{name:60s} {tuple(param.shape)}")
        return
    param = model.get_parameter(args.name).detach()
    while param.ndim < 2:
        param = param.unsqueeze(0)
    grid = param.reshape(-1, *param.shape[-2:])
    cols = min(len(grid), 8)
    rows = -(-len(grid) // cols)
    fig, ax = plt.subplots(rows, cols, squeeze=False,
                           figsize=(3 * cols, 3 * rows))
    for i, axis in enumerate(ax.flat):
        axis.axis("off")
        if i < len(grid):
            shown = axis.imshow(grid[i].numpy(), cmap="seismic")
            fig.colorbar(shown, ax=axis, fraction=0.046)
    fig.suptitle(f"{args.name} {tuple(param.shape)}")
    finish(fig, args.out)


if __name__ == "__main__":
    main()
