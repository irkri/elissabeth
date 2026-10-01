"""The discrete read-out of an arctic (max-plus) LISS level: for every
output position, head and value channel, the index tuple
``t_1 < ... < t_p <= t`` that attains the maximum, found by tracing the
cumulative maxima back like a Viterbi path (``--backward``: the tuple
``t <= t_p < ... < t_1`` of the backward direction).

    python analysis/arctic_paths.py run_0001 --position -1
"""
import torch

from common import example, layer_input, liss_level, load, parser


def main() -> None:
    p = parser(__doc__)
    p.add_argument("--position", type=int, default=-1,
                   help="output position to explain (default: the last)")
    args = p.parse_args()
    model, config = load(args)
    x, target = example(args, config)
    level = liss_level(model, args.layer, args.level, args.backward)
    stream = layer_input(model, x, args.layer, args.backward)
    tokens = x[0].tolist()
    t = args.position % len(tokens)
    # The backward direction runs on the reversed sequence: its position
    # T-1-t is the output at t, and its indices map back the same way.
    position = len(tokens) - 1 - t if args.backward else t
    tuples = level.decode(stream)[0, position]            # (N, d_v, p)
    if args.backward:
        tuples = torch.where(tuples >= 0, len(tokens) - 1 - tuples, -1)
    with torch.no_grad():
        score = level(stream)[0, position, :, :, 0]       # (N, d_v)
    print(f"input  {tokens}")
    if target is not None:
        print(f"target {target.tolist()}")
    print(f"output position {t}, depth {level.p} ({level.semiring})")
    for n in range(tuples.shape[0]):
        for d in range(tuples.shape[1]):
            index = tuples[n, d].tolist()
            picked = [tokens[i] for i in index]
            print(f"  head {n} channel {d}: t = {index} tokens {picked}"
                  f" score {score[n, d]:.3f}")


if __name__ == "__main__":
    main()
