"""What a LISS level's projections do with every token: the values,
queries or keys of each token of the vocabulary at one position (the
thesis' ``alphabet projection``). Only for token inputs, and for the first
layer, whose input is the token embedding.

    python analysis/probe.py run_0001 --what values --position 0
"""
import matplotlib.pyplot as plt
import torch

from common import finish, layer_input, liss_level, load, parser


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--what", choices=["values", "query", "key"],
                   default="values")
    p.add_argument("--kernel", type=int, default=0)
    p.add_argument("--position", type=int, default=0,
                   help="position t the tokens are placed at (matters with"
                        " include_time)")
    args = p.parse_args()
    model, config = load(args)
    if args.layer != 0:
        raise SystemExit("Only the first layer reads the embedding directly.")
    level = liss_level(model, 0, args.level, args.backward)
    vocab = config.dataset.input_dim
    # every token at every position up to --position; read the last one
    x = torch.arange(vocab).unsqueeze(1).repeat(1, args.position + 1)
    owner = level if args.what == "values" else level.kernels[args.kernel]
    hook = owner.hooks.get(args.what)
    hook.attach()
    with torch.no_grad():
        level(layer_input(model, x, 0, args.backward))
    data = hook.data[:, -1]                          # (vocab, N, p, ...)
    hook.release()
    data = data.flatten(3) if data.ndim > 3 else data.unsqueeze(-1)
    heads, indices = data.shape[1], data.shape[2]
    fig, ax = plt.subplots(
        heads, indices, squeeze=False,
        figsize=(1 + 0.4 * data.shape[-1] * indices, 0.3 * vocab * heads + 1),
    )
    for n in range(heads):
        for l in range(indices):
            shown = ax[n, l].imshow(data[:, n, l].numpy(), cmap="seismic",
                                    aspect="auto")
            fig.colorbar(shown, ax=ax[n, l])
            ax[n, l].set_yticks(range(vocab))
            ax[n, l].set_title(f"head {n}, index {l}", fontsize=8)
    fig.suptitle(f"{args.what} per token (rows) and channel (columns)")
    finish(fig, args.out)


if __name__ == "__main__":
    main()
