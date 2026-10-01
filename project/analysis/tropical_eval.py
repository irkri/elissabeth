"""A trained Tropical Attention run on the paper's evaluation settings: its
problem in distribution and the length, value and noise out-of-distribution
versions (``elissabeth.tropical.paper_settings``), plus any extra sizes.
Every setting gets fresh data drawn like the reference evaluation: seed 0,
5000 samples, and the 20% held-out part of its split (1000 sequences).

    python analysis/tropical_eval.py <run> [--sizes 16 32 128]

The ``score`` column is what the paper reports: accuracy for the per-set
binary problems and Floyd-Warshall's classes (the reference's micro-F1,
which over all classes is accuracy), F1 of the positive class for the
per-item binary problems, MSE for regression. Writes
``tropical_eval.csv`` into the run directory.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import CHECKPOINTS
from elissabeth import Elissabeth, load_model
from elissabeth.data import DatasetConfig, T_Objective, make_datasets
from elissabeth.lightning import (BEST_NAME, binary_accuracy, binary_counts,
                                  f1_score, find_run_dir, masked_accuracy,
                                  objective_loss)
from elissabeth.tropical import TropicalTask, paper_settings


@torch.no_grad()
def evaluate(
    model: Elissabeth,
    dataset,
    objective: T_Objective,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """The reference's numbers for one test set: the sample-weighted mean of
    the batch losses and their standard deviation, and the metrics."""
    losses, weights = [], []
    counts = torch.zeros(4, dtype=torch.long)
    correct = valid = 0.0
    for x, y in DataLoader(dataset, batch_size=batch_size):
        x, y = x.to(device), y.to(device)
        output = model(x)
        if y.ndim == 1:
            output = model.pool(output)
        losses.append(objective_loss(output, y, objective).item())
        weights.append(len(x))
        if objective == "binary":
            counts += binary_counts(output[..., 0], y).cpu()
        elif objective == "cross_entropy":
            n = (y != -1).sum().item()
            correct += masked_accuracy(output, y).item() * n
            valid += n
    result = {
        "loss": float(np.average(losses, weights=weights)),
        "loss_std": float(np.std(losses)),
        "accuracy": np.nan,
        "f1": np.nan,
    }
    if objective == "binary":
        result["accuracy"] = binary_accuracy(counts.float()).item()
        result["f1"] = f1_score(counts.float()).item()
    elif objective == "cross_entropy":
        result["accuracy"] = correct / max(valid, 1)
    return result


def _pooled(task: TropicalTask) -> bool:
    """Whether the problem has one target per set (its label is a scalar)."""
    _, y = task.model_copy(update={"cache": None}).data(2, 0)
    return y.ndim == 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("run", help="run directory (or its name in checkpoints/)")
    p.add_argument("--weights", default=None,
                   help=f"checkpoint file (default: {BEST_NAME} if there is"
                        " one, else the latest)")
    p.add_argument("--settings", nargs="*", default=None,
                   help="a subset of in_distribution, length, value, noise")
    p.add_argument("--sizes", nargs="*", type=int, default=[],
                   help="more sizes (items, or nodes) to evaluate at")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-samples", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "cpu")
    p.add_argument("--out", default=None,
                   help="CSV path (default: <run>/tropical_eval.csv)")
    args = p.parse_args()

    weights = args.weights
    run_dir = find_run_dir(CHECKPOINTS, args.run)
    if weights is None and (run_dir / BEST_NAME).exists():
        weights = BEST_NAME
    model, config, run_dir = load_model(run_dir, CHECKPOINTS, weights)
    task = config.dataset.task
    if not isinstance(task, TropicalTask):
        raise SystemExit("This run did not train on a tropical task.")
    device = torch.device(args.device)
    model = model.to(device)

    settings = paper_settings(task)
    if args.settings is not None:
        settings = {k: v for k, v in settings.items() if k in args.settings}
    for size in args.sizes:
        settings[f"size_{size}"] = task.model_copy(update={"size": size})
    pooled = _pooled(task)

    rows = []
    print(f"{run_dir.name}: {task.problem}, weights"
          f" {weights or 'latest'}, {task.objective}")
    for name, setting in settings.items():
        data = DatasetConfig(
            task=setting, n_samples=args.n_samples, val_size=0.2,
            seed=args.seed,
        )
        test = make_datasets(data)[1]
        result = evaluate(model, test, task.objective, args.batch_size,
                          device)
        if task.objective == "regression":
            value = result["loss"]
        elif task.objective == "binary" and not pooled:
            value = result["f1"]
        else:
            value = result["accuracy"]
        rows.append({
            "setting": name,
            "size": setting.size,
            "value_range": setting.value_range,
            "noise_prob": setting.noise_prob,
            "adversarial_range": setting.adversarial_range,
            "n": len(test),
            **result,
            "score": value,
        })
        print(f"  {name:16s} size {setting.size:4d}  score {value:8.4f}"
              f"  loss {result['loss']:.4f}  accuracy {result['accuracy']:.4f}"
              f"  f1 {result['f1']:.4f}")

    out = Path(args.out) if args.out else run_dir / "tropical_eval.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
