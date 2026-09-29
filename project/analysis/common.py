"""Shared helpers of the analysis scripts: load a run, draw an example
from its task, and find a LISS level. Run the scripts from ``project/``,
e.g. ``python analysis/kernels.py run_0001``."""
import argparse
from pathlib import Path

import numpy as np
import torch

from elissabeth import Elissabeth, RunConfig, load_model
from elissabeth.liss import LISS, LISSLevel

CHECKPOINTS = Path(__file__).parents[1] / "checkpoints"


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("run", help="run directory (or its name in checkpoints/)")
    p.add_argument("--weights", default=None,
                   help="checkpoint file in the run (default: the latest)")
    p.add_argument("--tokens", default=None,
                   help="comma-separated input tokens instead of a sample")
    p.add_argument("--seed", type=int, default=0,
                   help="seed of the sample drawn from the run's task")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--level", type=int, default=0,
                   help="index into the layer's lengths")
    p.add_argument("--out", default=None,
                   help="save the figure here instead of showing it")
    return p


def load(args: argparse.Namespace) -> tuple[Elissabeth, RunConfig]:
    model, config, _ = load_model(args.run, CHECKPOINTS, args.weights)
    return model, config


def example(
    args: argparse.Namespace,
    config: RunConfig,
) -> tuple[torch.Tensor, np.ndarray | None]:
    """``(1, T)`` input tokens and the task's target (``None`` for given
    tokens)."""
    if args.tokens is not None:
        tokens = [int(t) for t in args.tokens.split(",")]
        return torch.tensor(tokens).unsqueeze(0), None
    x, y = config.dataset.task.generate(1, np.random.default_rng(args.seed))
    return torch.as_tensor(x), y[0]


def liss_level(model: Elissabeth, layer: int, level: int) -> LISSLevel:
    mixer = model.mixers[layer]
    if not isinstance(mixer, LISS):
        raise SystemExit("This model's mixer is not a LISS layer.")
    return mixer.levels[level]


def layer_input(model: Elissabeth, x: torch.Tensor, layer: int) -> torch.Tensor:
    """What the mixer of ``layer`` reads: the normalised stream before it.
    """
    with torch.no_grad():
        stream = model.embedding(x)
        for i in range(layer):
            stream = model._block(stream, model.mixers[i], model.mixer_norms[i])
            if model.ffns:
                stream = model._block(stream, model.ffns[i], model.ffn_norms[i])
        return model.mixer_norms[layer](stream)


def finish(fig, out: str | None) -> None:
    if out is None:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        fig.savefig(out, bbox_inches="tight")
        print(f"Wrote {out}")
