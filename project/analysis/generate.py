"""Sample text from a model trained on the text task (makemore).

    python analysis/generate.py run_0001 --start "Love " --tokens 60
"""
import argparse

import torch

from common import CHECKPOINTS
from elissabeth import load_model
from elissabeth.data import TextTask


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("--weights", default=None)
    p.add_argument("--start", default="")
    p.add_argument("--tokens", type=int, default=80)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.5)
    args = p.parse_args()
    model, config, _ = load_model(args.run, CHECKPOINTS, args.weights)
    task = config.dataset.task
    if not isinstance(task, TextTask):
        raise SystemExit("This run was not trained on the text task.")
    x = torch.tensor([[0] + task.encode(args.start)] * args.samples)
    with torch.no_grad():
        for _ in range(args.tokens):
            logits = model(x)[:, -1] / args.temperature
            x = torch.cat((x, torch.multinomial(logits.softmax(-1), 1)), 1)
    for row in x.tolist():
        end = row.index(0, 1) if 0 in row[1:] else len(row)
        print(task.decode(row[1:end]))


if __name__ == "__main__":
    main()
