import argparse
import json
from pathlib import Path
from typing import Optional

import lightning.pytorch as L
import numpy as np
import pandas as pd
import torch
import wandb
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from matplotlib import pyplot as plt
from torchmetrics.classification import MulticlassAccuracy

from sainomore.callbacks import (ElissabethISTracker, ElissabethWeighting,
                                 GeneralConfigCallback, WeightHistory)
from sainomore.data import GivenDataModule, occurences
from sainomore.elissabeth import Elissabeth, Weighting
from sainomore.lightning import TokenPredictionModule
from sainomore.positional import PositionalEncoding

torch.set_float32_matmul_precision('high')

SAVE_PATH: Optional[str] = None

config = {
    "n_samples": 500,
    "context_length": 100,
    "characters": 3,
    "occurences": 5,

    "lr": 5e-3,
    "weight_decay": 1e-4,
    "epochs": 5001,

    "batch_size": 64,
    "val_size": 0.2,
}


def build_model(l: int | None = None) -> TokenPredictionModule:
    with open("config.json", "r") as f:
        model_config = json.load(f)

    model_config["context_length"] = (
        config["context_length"] if l is None else l
    )
    model_config["input_vocab_size"] = config["characters"]
    model_config["output_vocab_size"] = 2

    model = Elissabeth.build(
        model_config,
        Weighting.ComplexExponential,
        Weighting.ExponentialDecay,
    )

    lightning_module = TokenPredictionModule(
        model,
        learning_rate=config["lr"],
        weight_decay=config["weight_decay"],
        loss=torch.nn.CrossEntropyLoss(ignore_index=-1),
        accuracy=MulticlassAccuracy(2, ignore_index=-1),
        only_last=True,
    )
    return lightning_module


def train(
    lightning_module: TokenPredictionModule,
    use_wandb: bool = False,
    load_path: Optional[Path] = None,
    progress_bar: bool = False,
    only_test: bool = False,
) -> None:
    data_module = GivenDataModule(
        occurences(
            n_samples=config["n_samples"],
            length=config["context_length"],
            characters=config["characters"],
            occurences=config["occurences"],
        ),
        val_size=config["val_size"],
        batch_size=config["batch_size"],
        num_workers=10,
        # somehow this option is important, atleast on CPU
        # (no more waiting between epochs)
        persistent_workers=True,
    )
    wandb_logger = None
    if use_wandb:
        wandb_logger = WandbLogger(
            project="Elissabeth Occurences",
            checkpoint_name=str(load_path),
            tags=[],
            id=load_path.parts[-3] if load_path is not None else None,
            resume="must" if load_path is not None else False,
        )

        wandb_logger.experiment.config.update(
            lightning_module.model.get_config()
        )
        wandb_logger.experiment.config.update(config, allow_val_change=True)

        wandb_logger.watch(lightning_module, log="all")

    callbacks: list[Callback] = [
        GeneralConfigCallback(max_depth=10),
        ModelCheckpoint(monitor="validation/loss", mode="min"),
    ]

    trainer = L.Trainer(
        max_epochs=config["epochs"],
        accelerator="auto",
        callbacks=callbacks if not only_test else None,
        logger=wandb_logger if not only_test else None,
        default_root_dir=SAVE_PATH,
        enable_progress_bar=progress_bar,
        # overfit_batches=10,
    )

    if only_test:
        trainer.validate(lightning_module, data_module)
    else:
        trainer.fit(lightning_module, data_module, ckpt_path=str(load_path))

    if use_wandb:
        wandb.finish()


def plot(lightning_module: TokenPredictionModule) -> None:
    torch.random.manual_seed(662)
    np.random.seed(662)

    x, y = occurences(
        n_samples=10,
        length=25,
        characters=3,
        occurences=5,
    )
    print(x)
    print(y)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("mode", choices=["train", "test", "plot", "battery"])
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--load", default=None)

    args = parser.parse_args()

    lightning_module = build_model()

    load_path = None
    if args.load is not None:
        directory = Path.cwd()
        for folder in directory.iterdir():
            if not (folder.is_dir() and (folder / args.load).is_dir()):
                continue
            if (load_path := (folder / args.load / "checkpoints")).exists():
                load_path = load_path / next(load_path.iterdir())
                saved_ = torch.load(
                    load_path,
                    # map_location=torch.device("cpu"),
                )
                lightning_module.load_state_dict(saved_["state_dict"])
    if args.load is not None and load_path is None:
        raise FileExistsError(
            "Load specification does not point to a saved model"
        )

    if args.mode == "train":
        train(
            lightning_module,
            load_path=load_path,
            use_wandb=args.online,
            progress_bar=not args.online,
        )
    elif args.mode == "test":
        train(lightning_module, only_test=True)
    elif args.mode == "plot":
        plot(lightning_module)


if __name__ == '__main__':
    main()
