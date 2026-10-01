"""Values, queries/keys or iterated sums of a LISS level over time, for
one input (what the thesis notebooks plotted with ``plot_values_time``,
``plot_query_key_time`` and ``plot_iss_time``).

    python analysis/traces.py run_0001 --what values --out values.pdf
    python analysis/traces.py run_0001 --what query --kernel 0

Values are ``(T, N, p, d_v, w)``, the iterated sums ``(T, N, d_v, w)``,
queries/keys ``(T, N, p[, d_qk])``. One subplot per head and index; the
remaining channels are overlaid.
"""
import matplotlib.pyplot as plt
import torch

from common import example, finish, layer_input, liss_level, load, parser


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--what", choices=["values", "iss", "query", "key"],
                   default="values")
    p.add_argument("--kernel", type=int, default=0,
                   help="kernel index for --what query/key")
    args = p.parse_args()
    model, config = load(args)
    x, _ = example(args, config)
    level = liss_level(model, args.layer, args.level, args.backward)
    owner = level if args.what in ("values", "iss") else level.kernels[args.kernel]
    hook = owner.hooks.get(args.what)
    hook.attach()
    with torch.no_grad():
        level(layer_input(model, x, args.layer, args.backward))
    data = hook.data[0]
    hook.release()
    if args.what == "iss":
        data = data.unsqueeze(2)                     # one "index" axis
    data = data.flatten(3) if data.ndim > 3 else data.unsqueeze(-1)
    T, heads, indices, channels = data.shape
    fig, ax = plt.subplots(
        heads, indices, squeeze=False, sharex=True,
        figsize=(4 * indices, 2 * heads),
    )
    tokens = x[0].tolist()
    for n in range(heads):
        for l in range(indices):
            ax[n, l].plot(data[:, n, l].numpy(), marker=".", linewidth=0.8)
            ax[n, l].set_title(f"head {n}, index {l}", fontsize=8)
    if T <= 60:
        for axis in ax[-1]:
            axis.set_xticks(range(T), tokens, fontsize=6)
    fig.suptitle(f"{args.what}, layer {args.layer}, depth {level.p}")
    finish(fig, args.out)


if __name__ == "__main__":
    main()
