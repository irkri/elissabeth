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
from sainomore.data import GivenDataModule, evencount
from sainomore.elissabeth import Elissabeth, Weighting
from sainomore.lightning import TokenPredictionModule
from sainomore.positional import PositionalEncoding
from sainomore.tools import get_attention_matrices, plot_attention_matrix

torch.set_float32_matmul_precision('high')

SAVE_PATH: Optional[str] = None

config = {
    "n_samples": 500,
    "context_length": 100,
    "n_categories": 3,

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
    model_config["input_vocab_size"] = config["n_categories"]
    model_config["output_vocab_size"] = 2

    model = Elissabeth.build(
        model_config,
        Weighting.ComplexExponential,
        # Weighting.ExponentialDecay,
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
        evencount(
            n_samples=config["n_samples"],
            length=config["context_length"],
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

    x, _ = evencount(
        n_samples=1,
        length=config["context_length"],
    )
    callbacks: list[Callback] = [
        GeneralConfigCallback(max_depth=10),
        ModelCheckpoint(monitor="validation/loss", mode="min"),
        # WeightHistory((
        #         "model.embedding.weight",
        #         "model.unembedding.weight",
        #         "model.layers.0.levels.0.P_V.transform.weight",
        #         "model.layers.0.levels.0.P_V.transform.bias",
        #         "model.layers.0.levels.0.weightings.0.P_Q.transform.0.weight",
        #         "model.layers.0.levels.0.weightings.0.P_Q.transform.0.bias",
        #         "model.layers.0.levels.0.weightings.0.P_Q.transform.1.weight",
        #         "model.layers.0.levels.0.weightings.0.P_Q.transform.1.bias",
        #         "model.layers.0.levels.0.weightings.0.P_K.transform.0.weight",
        #         "model.layers.0.levels.0.weightings.0.P_K.transform.0.bias",
        #         "model.layers.0.levels.0.weightings.0.P_K.transform.1.weight",
        #         "model.layers.0.levels.0.weightings.0.P_K.transform.1.bias",
        #         ("model.layers.0.W_O", (2, 4)),
        #     ),
        #     each_n_epochs=100,
        #     # save_path=".",
        # ),
        # ElissabethWeighting(
        #     example,
        #     each_n_epochs=100,
        #     use_wandb=True,
        #     # save_path=".",
        # ),
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
    torch.random.manual_seed(662)
    np.random.seed(662)

    x, y = evencount(
        n_samples=10,
        length=config["context_length"],
        characters=3,
    )
    print(x)
    print(y)

    # out = lightning_module.predict_step(x)
    # print(out)

    # att = get_attention_matrices(
    #     lightning_module.model,  # type: ignore
    #     x[0],
    #     # total=True,
    # )
    # # idx = att[:, 0, 0] < 0
    # # att[:, 0, 0, *idx] = 1e-20
    # # att[:, 0, 0] = torch.log(att[:, 0, 0])
    # figatt, axatt = plot_attention_matrix(
    #     att[:, 0],
    #     x[0],
    #     cmap="RdPu",
    #     cmap_example="tab20",
    #     share_cmap=False,
    #     log_cmap=False,
    #     causal_mask=True,
    #     # contains_total=True,
    #     figsize=(10, 25)
    # )
    # 0.6296377778053284
    # 0.9977337718009949

    # 0.8785377740859985
    # 0.4517635107040405

    # 0.9316196441650391
    # 0.17080585658550262

    # 0.9936496615409851
    # 0.0214511938393116

    # 0.9998123645782471
    # 0.0037638270296156406


    # x = torch.zeros((16, 16)).unsqueeze(1)
    # x = x.repeat(1, 100, 1)
    # y = torch.empty((16, 100, 17))
    # y[:, :, :16] = x
    # y[:, :, 16] = torch.linspace(1/100, 1, 100)

    # Q_t = lightning_module.model.layers[0].levels[0].weightings[0].P_Q.transform(y).detach()
    # K_t = lightning_module.model.layers[0].levels[0].weightings[0].P_K.transform(y).detach()

    # fig, ax = plt.subplots(1, 2, figsize=(16, 4))

    # for j in range(Q_t.size(0)):
    #     ax[0].plot(torch.norm(Q_t[j, :, :], dim=1), label=f"Dimension {j}")
    #     ax[1].plot(torch.norm(K_t[j, :, :], dim=1), label=f"Dimension {j}")
    # fig.legend()

    # W_V = lightning_module.get_parameter("model.layers.0.W_V").detach()

    # fig, ax = plt.subplots(1, 2)

    # mat1 = ax[0].matshow(W_V[0, 0, :, :, 0])
    # mat2 = ax[1].matshow(W_V[0, 1, :, :, 0])
    # fig.colorbar(mat1)
    # fig.colorbar(mat2)

    # plt.savefig(
    #     Path.cwd() / "plot.pdf",
    #     bbox_inches="tight",
    #     facecolor=(0, 0, 0, 0),
    # )
    # plt.show()


def battery(lightning_module: TokenPredictionModule) -> None:
    lengths = [10, 25, 100, 150, 200, 250, 500, 750, 1000, 10_000]
    samples = 5
    trainit = True
    multiple_keys = False

    accuracy = np.zeros((len(lengths), samples, 2))
    trainer = L.Trainer(max_epochs=500)
    for l in range(len(lengths)):
        print(f"Starting Length {lengths[l]}", end="", flush=True)
        lightning_module = build_model(lengths[l])
        if trainit:
            data_module = GivenDataModule(
                copying(
                    n_samples=1000,
                    length=lengths[l],
                    n_categories=config["n_categories"],
                    to_copy=config["to_copy"],
                ),
                val_size=0.0,
                batch_size=config["batch_size"],
                num_workers=5,
                # somehow this option is important, atleast on CPU
                # (no more waiting between epochs)
                persistent_workers=True,
            )
            trainer.fit(
                lightning_module,
                train_dataloaders=data_module,
            )
        for i in range(samples):
            print(".", end="", flush=True)
            data_module = GivenDataModule(
                copying(
                    n_samples=100,
                    length=lengths[l],
                    n_categories=config["n_categories"],
                    to_copy=config["to_copy"],
                ),
                val_size=1.0,
                batch_size=config["batch_size"],
                num_workers=5,
                # somehow this option is important, atleast on CPU
                # (no more waiting between epochs)
                persistent_workers=True,
            )
            out = trainer.validate(
                lightning_module,
                data_module,
                verbose=False,
            )[0]
            accuracy[l, i, 0] = out["validation/loss"]
            accuracy[l, i, 1] = out["validation/accuracy"]
        pd.DataFrame(
            accuracy[:(l+1), :, 0].T,
            columns=[f"Length {k}" for k in lengths[:(l+1)]],
        ).to_csv("battery_loss.csv")
        pd.DataFrame(
            accuracy[:(l+1), :, 1].T,
            columns=[f"Length {k}" for k in lengths[:(l+1)]],
        ).to_csv("battery_accuracy.csv")
        print()


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
    elif args.mode == "battery":
        battery(lightning_module)


if __name__ == '__main__':
    main()
