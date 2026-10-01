"""A sweep over models, problems and learning rates of the Tropical
Attention benchmark: trains every missing run, evaluates it on the paper's
settings (analysis/tropical_eval.py) and collects ``summary.csv``.

    python tropical_sweep.py --root checkpoints/tropical_sweep \\
        --models arctic log cosine_p2 --problems knapsack subset_sum \\
        --lr 1e-4 1e-3 --jobs 8

A run is ``configs/tropical/base.yaml`` with the problem's overlay, then the
model's (``configs/tropical/models/<model>.yaml``), then ``trainer.lr`` and
``trainer.epochs``; it is named ``<model>-<problem>-lr<lr>`` below the root.
Finished runs and evaluations are kept, so the sweep can be extended or
restarted; ``summary.csv`` covers every run below the root.
"""
import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

PROJECT = Path(__file__).parent
CONFIGS = PROJECT / "configs" / "tropical"


def run_name(model: str, problem: str, lr: float) -> str:
    return f"{model}-{problem}-lr{lr:g}"


def train_and_evaluate(
    root: Path,
    model: str,
    problem: str,
    lr: float,
    epochs: int,
    eval_args: list[str],
) -> str:
    name = run_name(model, problem, lr)
    run_dir = root / name
    env = {**os.environ, "ELISSABETH_CHECKPOINTS": str(root)}
    if not (run_dir / f"epoch_{epochs}.ckpt").exists():
        command = [
            sys.executable, "train.py", str(CONFIGS / "base.yaml"),
            "-oc", str(CONFIGS / f"{problem}.yaml"),
            "-oc", str(CONFIGS / "models" / f"{model}.yaml"),
            "-o", f"trainer.lr={lr}", "-o", f"trainer.epochs={epochs}",
            "-o", "trainer.progress_bar=false",
        ]
        command += ["--resume", name] if run_dir.exists() else ["--name", name]
        with open(root / f"{name}.log", "a") as log:
            subprocess.run(command, cwd=PROJECT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    (run_dir / "sweep.json").write_text(json.dumps(
        {"model": model, "problem": problem, "lr": lr},
    ))
    if not (run_dir / "tropical_eval.csv").exists():
        with open(root / f"{name}.log", "a") as log:
            subprocess.run(
                [sys.executable, "analysis/tropical_eval.py", name,
                 *eval_args],
                cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT,
                check=True,
            )
    return name


def summarise(root: Path) -> Path:
    """One row per evaluated run below ``root``: its settings, parameters,
    training time, best epoch and every evaluation score."""
    rows = []
    for record in sorted(root.glob("*/sweep.json")):
        run_dir = record.parent
        if not (run_dir / "tropical_eval.csv").exists():
            continue
        run = json.loads(record.read_text())
        model, problem, lr = run["model"], run["problem"], run["lr"]
        name = run_dir.name
        config = yaml.safe_load((run_dir / "config.yaml").read_text())
        with open(run_dir / "metrics.csv") as f:
            metrics = list(csv.DictReader(f))
        best = min(metrics, key=lambda r: float(r["validation/loss"]))
        with open(run_dir / "tropical_eval.csv") as f:
            scores = {r["setting"]: float(r["score"]) for r in csv.DictReader(f)}
        final = max(run_dir.glob("epoch_*.ckpt"), key=os.path.getmtime)
        seconds = final.stat().st_mtime - (run_dir / "config.yaml").stat().st_mtime
        rows.append({
            "model": model, "problem": problem, "lr": lr,
            "layers": config["model"]["n_layers"],
            "semiring": config["model"]["liss"]["semiring"],
            "lengths": config["model"]["liss"]["lengths"],
            "parameters": _parameters(root / f"{name}.log"),
            "train_seconds": round(seconds),
            "epochs": len(metrics),
            "best_epoch": int(best["epoch"]) + 1,
            "validation_loss": float(best["validation/loss"]),
            **scores,
        })
    out = root / "summary.csv"
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return out


def _parameters(log: Path) -> int | None:
    for line in log.read_text().splitlines():
        if line.startswith("Parameters: "):
            return int(line.split()[1].replace(",", ""))
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--root", type=Path, required=True,
                   help="directory of the sweep's runs")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--problems", nargs="+", required=True)
    p.add_argument("--lr", nargs="+", type=float, required=True)
    p.add_argument("--epochs", type=int, default=100,
                   help="for every problem (Table 5 of the paper: 100)")
    p.add_argument("--jobs", type=int, default=4,
                   help="runs at the same time")
    p.add_argument("--eval-args", nargs=argparse.REMAINDER, default=[],
                   help="passed on to analysis/tropical_eval.py")
    args = p.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        if not (CONFIGS / "models" / f"{model}.yaml").exists():
            p.error(f"no model overlay configs/tropical/models/{model}.yaml")
    for problem in args.problems:
        if not (CONFIGS / f"{problem}.yaml").exists():
            p.error(f"no problem overlay configs/tropical/{problem}.yaml")

    grid = list(itertools.product(args.models, args.problems, args.lr))
    print(f"{len(grid)} runs, {args.jobs} at a time, in {root}")
    failed = []
    with ThreadPoolExecutor(args.jobs) as pool:
        futures = {
            pool.submit(train_and_evaluate, root, *run, args.epochs,
                        args.eval_args): run
            for run in grid
        }
        for future, run in futures.items():
            name = run_name(*run)
            try:
                future.result()
                print(f"done   {name}", flush=True)
            except subprocess.CalledProcessError:
                failed.append(name)
                print(f"FAILED {name} (see {root / name}.log)", flush=True)
    print(f"Wrote {summarise(root)}")
    if failed:
        raise SystemExit(f"{len(failed)} runs failed: {failed}")


if __name__ == "__main__":
    main()
