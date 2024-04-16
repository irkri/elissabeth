import argparse
import json
from pathlib import Path
from typing import Optional

import lightning.pytorch as L
import numpy as np
import torch
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.wandb import WandbLogger
from matplotlib import pyplot as plt
from torchmetrics.classification import MulticlassAccuracy

from sainomore.callbacks import (ElissabethISTracker, ElissabethWeighting,
                                 GeneralConfigCallback, WeightHistory)
from sainomore.data import SplitDataModule, long_lookup
from sainomore.elissabeth import Elissabeth, Weighting
from sainomore.lightning import SAILearningModule, TokenPredictionModule
from sainomore.models import Transformer
from sainomore.positional import PositionalEncoding
from sainomore.tools import get_attention_matrices, plot_attention_matrix

from data import UCRLoader

torch.set_float32_matmul_precision('high')

dataloader = UCRLoader("CinCECGTorso")

config = {
    "lr": 5e-3,
    "weight_decay": 1e-2,
    "epochs": 501,

    "batch_size": 16,
}


def build_model() -> SAILearningModule:
    with open("config.json", "r") as f:
        model_config = json.load(f)

    nlabels = dataloader.nlabels
    model_config["context_length"] = dataloader.context_length
    model_config["input_vocab_size"] = 1
    model_config["output_vocab_size"] = nlabels

    model = Elissabeth.build(
        model_config,
        Weighting.ControlledExponential
    )

    # W_V = model.get_parameter("layers.0.W_V")
    # torch.nn.init.ones_(W_V)
    # W_V.requires_grad = False

    lightning_module = TokenPredictionModule(
        model,
        learning_rate=config["lr"],
        weight_decay=config["weight_decay"],
        only_last=True,
        accuracy=MulticlassAccuracy(num_classes=nlabels),
    )
    return lightning_module


def train(
    lightning_module: SAILearningModule,
    use_wandb: bool = False,
    load_path: Optional[str] = None,
    progress_bar: bool = False,
    only_test: bool = False,
) -> None:
    data_module = SplitDataModule(
        dataloader.get_train(),
        dataloader.get_test(),
        batch_size=config["batch_size"],
        num_workers=5,
        persistent_workers=True,
    )

    wandb_logger = None
    if use_wandb:
        wandb_logger = WandbLogger(
            project="Elissabeth UCR",
            checkpoint_name=load_path,
            tags=[config["dataset"]],
            id=load_path.split("/")[1] if load_path is not None else None,
            resume="must" if load_path is not None else False,
        )

        wandb_logger.experiment.config.update(
            lightning_module.model.get_config()
        )
        wandb_logger.experiment.config.update(config, allow_val_change=True)

        wandb_logger.watch(lightning_module, log="all")

    callbacks: list[Callback] = [
        GeneralConfigCallback(max_depth=10),
        WeightHistory((
                "model.embedding.weight",
                "model.unembedding.weight",
                # ("model.layers.0.W_V", (2, 3)),
                ("model.layers.0.W_O", (1, 3)),
            ),
            each_n_epochs=100,
            # save_path=".",
        ),
        # ElissabethISTracker(
        #     example,
        #     each_n_epochs=100,
        #     use_wandb=True,
        #     # save_path=".",
        # ),
    ]

    trainer = L.Trainer(
        max_epochs=config["epochs"],
        accelerator="auto",
        callbacks=callbacks if not only_test else None,
        logger=wandb_logger if not only_test else None,
        enable_progress_bar=progress_bar,
        # overfit_batches=10,
    )

    if only_test:
        trainer.validate(lightning_module, data_module)
    else:
        trainer.fit(lightning_module, data_module, ckpt_path=load_path)

    if use_wandb:
        wandb.finish()


def plot(lightning_module: SAILearningModule) -> None:
    x, y = dataloader.get_test()

    out = lightning_module(x)
    print(out)
    # print(torch.mean((x[0] - out[0])**2))

    # print(out)

    # att = get_attention_matrices(
    #     lightning_module.model,  # type: ignore
    #     example,
    # )
    # # att[:, 0, 0] = torch.log(att[:, 0, 0])
    # # att[:, 0, 1] = torch.log(att[:, 0, 1])
    # figatt, axatt = plot_attention_matrix(
    #     att[:, 0], example[0],
    #     cmap="RdPu",
    #     share_cmap=False,
    #     log_cmap=False,
    #     figsize=(10, 100)
    # )
    # # W_V = lightning_module.get_parameter("model.layers.0.W_V").detach()

    # # fig, ax = plt.subplots(1, 2)

    # # mat1 = ax[0].matshow(W_V[0, 0, :, :, 0])
    # # mat2 = ax[1].matshow(W_V[0, 1, :, :, 0])
    # # fig.colorbar(mat1)
    # # fig.colorbar(mat2)

    # plt.savefig(
    #     Path.cwd() / "plot.png",
    #     bbox_inches="tight",
    #     facecolor=(0, 0, 0, 0),
    # )
    # plt.show()


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
                saved_ = torch.load(
                    next(load_path.iterdir()),
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
