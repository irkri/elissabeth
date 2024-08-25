import argparse
import json
from pathlib import Path
from typing import Optional

import lightning.pytorch as L
import numpy as np
import torch
import wandb
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from torchmetrics.classification import MulticlassAccuracy

from sainomore.callbacks import GeneralConfigCallback
from sainomore.data import GivenDataModule
from sainomore.elissabeth import Elissabeth
from sainomore.lightning import TokenPredictionModule

from data import copying

torch.set_float32_matmul_precision('high')

config = {
    "n_samples": 1000,
    "context_length": 100,
    "n_categories": 10,
    "to_copy": 10,
    "max_dilute": 5,

    "lr": 5e-3,
    "weight_decay": 1e-4,
    "epochs": 1001,

    "batch_size": 64,
    "val_size": 0.2,
}


def build_model(l: int | None = None) -> TokenPredictionModule:
    with open("config.json", "r") as f:
        model_config = json.load(f)

    model_config["context_length"] = (
        config["context_length"] if l is None else l
    )
    model_config["input_vocab_size"] = config["n_categories"]

    model = Elissabeth.build(model_config)
    lightning_module = TokenPredictionModule(
        model,
        learning_rate=config["lr"],
        weight_decay=config["weight_decay"],
        loss=torch.nn.CrossEntropyLoss(ignore_index=-1),
        accuracy=MulticlassAccuracy(config["n_categories"], ignore_index=-1),
        only_last=False,
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
        copying(
            n_samples=config["n_samples"],
            length=config["context_length"],
            n_categories=config["n_categories"],
            to_copy=config["to_copy"],
            max_dilute=config["max_dilute"],
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
            project="Elissabeth Copying",
            checkpoint_name=load_path,
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
        GeneralConfigCallback(),
        ModelCheckpoint(monitor="validation/loss", mode="min"),
    ]

    trainer = L.Trainer(
        max_epochs=config["epochs"],
        accelerator="auto",
        callbacks=callbacks if not only_test else None,
        logger=wandb_logger if not only_test else None,
        enable_progress_bar=progress_bar,
    )

    if only_test:
        trainer.validate(lightning_module, data_module)
    else:
        trainer.fit(lightning_module, data_module, ckpt_path=load_path)

    if use_wandb:
        wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("mode", choices=["train", "test"])
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
                    map_location=torch.device("cpu"),
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


if __name__ == '__main__':
    main()
