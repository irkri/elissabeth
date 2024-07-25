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
from matplotlib import pyplot as plt
from torchmetrics.classification import MulticlassAccuracy
from torchvision.datasets import MNIST
from torchvision import transforms as VT

from sainomore.callbacks import GeneralConfigCallback
from sainomore.data import GivenDatasetModule
from sainomore.elissabeth import Elissabeth
from sainomore.lightning import TokenPredictionModule

torch.set_float32_matmul_precision('high')

SAVE_PATH: Optional[str] = None

def flatten(x: torch.Tensor) -> torch.Tensor:
    return (x.flatten() / 255).float().unsqueeze(-1)

train_set = MNIST(
    "data",
    train=True,
    download=True,
    transform=VT.Compose([VT.PILToTensor(), VT.Lambda(flatten)]),
)
val_set = MNIST(
    "data",
    train=False,
    download=True,
    transform=VT.Compose([VT.PILToTensor(), VT.Lambda(flatten)]),
)

config = {
    "lr": 5e-3,
    "weight_decay": 1e-4,
    "epochs": 1001,

    "batch_size": 64,
    "val_size": 0.2,
}


def build_model(l: int | None = None) -> TokenPredictionModule:
    with open("config.json", "r") as f:
        model_config = json.load(f)

    model = Elissabeth.build(model_config)
    lightning_module = TokenPredictionModule(
        model,
        learning_rate=config["lr"],
        weight_decay=config["weight_decay"],
        loss=torch.nn.CrossEntropyLoss(),
        accuracy=MulticlassAccuracy(10),
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
    data_module = GivenDatasetModule(
        train_set, val_set,
        batch_size=config["batch_size"],
        num_workers=10,
        # somehow this option is important, atleast on CPU
        # (no more waiting between epochs)
        persistent_workers=True,
    )
    wandb_logger = None
    if use_wandb:
        wandb_logger = WandbLogger(
            project="Elissabeth MNIST",
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
        default_root_dir=SAVE_PATH,
        enable_progress_bar=progress_bar,
        # overfit_batches=10,
    )

    if only_test:
        trainer.validate(lightning_module, data_module)
    else:
        trainer.fit(lightning_module, data_module, ckpt_path=load_path)

    if use_wandb:
        wandb.finish()


def plot(lightning_module: TokenPredictionModule) -> None:
    print(next(iter(val_set))[0][0].dtype)
    plt.imshow(next(iter(val_set))[0].reshape(28, 28))
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("mode", choices=["train", "test", "plot"])
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
    elif args.mode == "plot":
        plot(lightning_module)


if __name__ == '__main__':
    main()
