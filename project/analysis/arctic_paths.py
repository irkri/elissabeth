"""The discrete read-out of an arctic (max-plus) LISS level: for every
output position, head and value channel, the index tuple
``t_1 < ... < t_p <= t`` that attains the maximum, found by tracing the
cumulative maxima back like a Viterbi path.

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
    level = liss_level(model, args.layer, args.level)
    stream = layer_input(model, x, args.layer)
    tuples = level.decode(stream)[0, args.position]       # (N, d_v, p)
    with torch.no_grad():
        score = level(stream)[0, args.position, :, :, 0]  # (N, d_v)
    tokens = x[0].tolist()
    t = args.position % len(tokens)
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
