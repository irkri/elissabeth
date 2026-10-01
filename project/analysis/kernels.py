"""The pair kernels of a LISS level on one input, ``kappa_l(t, s)`` per
head and pair, and optionally the total weight of ``t_1 = s`` for the output
at ``t`` (every path through the pairs, values left out).

    python analysis/kernels.py run_0001 --total --out kernels.pdf
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import CenteredNorm

from common import (example, finish, layer_input, liss_layer, liss_level,
                    load, parser)
from elissabeth.attention import SelfAttention
from elissabeth.liss.semiring import LOG_DOMAIN, multiply, zero


def total_weight(
    kernel: torch.Tensor,
    support: torch.Tensor,
    semiring: str,
) -> torch.Tensor:
    """``(N, p, T, T) -> (N, T, T)``: the semiring matrix product
    ``kappa_{p-1} ... kappa_0`` over the kernels' supports."""
    empty = zero(semiring, kernel.dtype)
    masked = torch.where(support, kernel, torch.full_like(kernel, empty))
    total = masked[:, 0]
    for l in range(1, kernel.shape[1]):
        total = multiply(masked[:, l], total, semiring, matrix=True)
    return total


def show(fig, axis, image: np.ndarray) -> None:
    """Diverging colours centred at 0 for signed data, else viridis."""
    signed = np.nanmin(image) < 0 < np.nanmax(image)
    shown = axis.imshow(
        image,
        cmap="seismic" if signed else "viridis",
        norm=CenteredNorm() if signed else None,
    )
    fig.colorbar(shown, ax=axis, fraction=0.046)


def attention(model, x: torch.Tensor, layer: int, out: str | None) -> None:
    """The transformer baseline: its attention weights per head."""
    weights = model.mixers[layer].attention_matrix(layer_input(model, x, layer))
    heads = weights.shape[1]
    fig, ax = plt.subplots(1, heads, squeeze=False, figsize=(3.2 * heads, 3.2))
    for h in range(heads):
        image = weights[0, h].numpy()
        image[np.triu_indices_from(image, 1)] = np.nan
        show(fig, ax[0, h], image)
        ax[0, h].set_title(f"head {h}", fontsize=8)
    fig.suptitle(f"attention, layer {layer}")
    finish(fig, out)


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--total", action="store_true",
                   help="add the total weight of t_1 for every output t")
    p.add_argument("--project-heads", action="store_true",
                   help="combine the heads with the layer's W_H weights")
    args = p.parse_args()
    model, config = load(args)
    x, target = example(args, config)
    if isinstance(model.mixers[args.layer], SelfAttention):
        attention(model, x, args.layer, args.out)
        return
    level = liss_level(model, args.layer, args.level, args.backward)
    kernel, support = level.pair_matrices(
        layer_input(model, x, args.layer, args.backward),
    )
    kernel = kernel[0]                                   # (N, p, T, T)
    columns = [f"pair {l}: t_{l} -> t_{l + 1}" for l in range(level.p)]
    columns[-1] = f"pair {level.p - 1}: t_{level.p - 1} -> t"
    if args.total:
        kernel = torch.cat(
            (kernel, total_weight(kernel, support, level.semiring)[:, None]),
            dim=1,
        )
        support = torch.cat((support, support[-1:]))
        columns.append("total: t_0 -> t")
    if args.project_heads:
        weights = liss_layer(model, args.layer, args.backward) \
            .W_H[args.level].detach()
        kernel = torch.einsum("n,nltu->ltu", weights, kernel)[None]
    rows = kernel.shape[0]
    fig, ax = plt.subplots(
        rows, len(columns), squeeze=False,
        figsize=(3.2 * len(columns), 3.2 * rows),
    )
    tokens = x[0].tolist()
    for n in range(rows):
        for j, title in enumerate(columns):
            image = kernel[n, j].numpy().copy()
            image[~support[j].numpy()] = np.nan
            if level.semiring in LOG_DOMAIN:
                # no path at all: the finite stand-in for -inf
                image[image < zero(level.semiring, kernel.dtype) / 2] = np.nan
            show(fig, ax[n, j], image)
            ax[n, j].set_title(title if n == 0 else "", fontsize=8)
            ax[n, j].set_ylabel(f"head {n}" if j == 0 else "")
            if len(tokens) <= 40:
                ax[n, j].set_xticks(range(len(tokens)), tokens, fontsize=6)
                ax[n, j].set_yticks(range(len(tokens)), tokens, fontsize=6)
    fig.suptitle(
        f"layer {args.layer}, depth {level.p} ({level.semiring})"
        + (f", target {target}" if target is not None and target.ndim == 0
           else "")
    )
    finish(fig, args.out)


if __name__ == "__main__":
    main()
