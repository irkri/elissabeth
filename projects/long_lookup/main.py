import argparse
import json
import os
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
from sainomore.data import GivenDataModule, long_lookup
from sainomore.elissabeth import Elissabeth, Weighting
from sainomore.lightning import TokenPredictionModule
from sainomore.positional import PositionalEncoding
from sainomore.tools import get_attention_matrix, plot_attention_matrix

torch.set_float32_matmul_precision('high')

SAVE_PATH: Optional[str] = None

config = {
    "n_samples": 1000,
    "context_length": 25,
    "characters": 5,

    "lr": 5e-3,
    "weight_decay": 1e-4,
    "epochs": 501,

    "batch_size": 64,
    "val_size": 0.2,
}


class MyElissabeth(Elissabeth):
    """Elissabeth without residual stream and the output is
    subtracted by a shifted version of the input."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config("input_type") == "token":
            input_ = self.embedding(x)
            x = input_
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        logits = x - torch.nn.functional.pad(input_[:, :-1], (0, 0, 1, 0))
        return torch.swapaxes(logits, 1, 2)


def build_model() -> TokenPredictionModule:
    with open("config.json", "r") as f:
        model_config = json.load(f)

    model_config["context_length"] = config["context_length"]
    model_config["input_vocab_size"] = config["characters"]

    model = MyElissabeth.build(
        model_config,
        Weighting.COSINE | Weighting.RELATIVE_DISTANCE,
        # PositionalEncoding.ROPE,
    )

    state_dict = model.state_dict()

    state_dict["embedding.weight"] = torch.eye(5)
    # state_dict["layers.0.W_V"] = torch.Tensor([[
    #     [[1, 0, 0, 0, 0],
    #     [0, 1, 0, 0, 0],
    #     [0, 0, 1, 0, 0],
    #     [0, 0, 0, 1, 0],
    #     [0, 0, 0, 0, 1]],

    #     [[1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1],
    #     [1, 1, 1, 1, 1]]
    # ]]).unsqueeze(-1)

    state_dict["layers.0.W_O"] = torch.Tensor([[
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ]]).unsqueeze(2)
    state_dict["unembedding.weight"] = torch.eye(5)

    state_dict["layers.0.weightings.0.alpha"] = torch.Tensor([[
        [0, 0]
    ]]).unsqueeze(-1).unsqueeze(-1)

    # d = torch.pi / 2
    # state_dict["layers.0.weightings.1.W_Q"] = torch.Tensor([[
    #     [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    #     [[0, 0, 0], [d, 0, 0], [0, d, 0], [0, 0, d], [d, d, 0]],
    # ]])
    # state_dict["layers.0.weightings.1.W_K"] = torch.Tensor([[
    #     [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    #     [[0, 0, 0], [d, 0, 0], [0, d, 0], [0, 0, d], [d, d, 0]],
    # ]])

    model.load_state_dict(state_dict)
    # model.get_parameter("layers.0.weightings.1.W_Q").requires_grad = False
    # model.get_parameter("layers.0.weightings.1.W_K").requires_grad = False
    # model.get_parameter("layers.0.W_V").requires_grad = False
    model.get_parameter("layers.0.W_O").requires_grad = False
    model.get_parameter("embedding.weight").requires_grad = False
    model.get_parameter("unembedding.weight").requires_grad = False
    # model.get_parameter("layers.0.weightings.0.alpha").requires_grad = False

    lightning_module = TokenPredictionModule(
        model,
        learning_rate=config["lr"],
        weight_decay=config["weight_decay"],
        loss=torch.nn.CrossEntropyLoss(),
        accuracy=MulticlassAccuracy(config["characters"]),
        only_last=True,
    )
    return lightning_module


def train(
    lightning_module: TokenPredictionModule,
    use_wandb: bool = False,
    load_path: Optional[str] = None,
    progress_bar: bool = False,
    only_test: bool = False,
) -> None:
    data_module = GivenDataModule(
        long_lookup(
            n_samples=config["n_samples"],
            length=config["context_length"],
            characters=config["characters"],
            multiple_keys=False,
        ),
        val_size=config["val_size"],
        batch_size=config["batch_size"],
        num_workers=5,
        # somehow this option is important, atleast on CPU
        # (no more waiting between epochs)
        persistent_workers=True,
    )

    wandb_logger = None
    if use_wandb:
        wandb_logger = WandbLogger(
            project="Elissabeth Long Lookup 2",
            checkpoint_name=load_path,
            tags=["one key"],
            id=load_path.split("/")[1] if load_path is not None else None,
            resume="must" if load_path is not None else False,
        )

        wandb_logger.experiment.config.update(
            lightning_module.model.get_config()
        )
        wandb_logger.experiment.config.update(config, allow_val_change=True)

        wandb_logger.watch(lightning_module, log="all")

    example = long_lookup(
        n_samples=1,
        length=config["context_length"],
        characters=config["characters"],
    )[0]
    callbacks: list[Callback] = [
        GeneralConfigCallback(max_depth=10),
        WeightHistory((
                "model.embedding.weight",
                "model.unembedding.weight",
                "model.layers.0.weightings.0.W_Q",
                "model.layers.0.weightings.0.W_K",
                ("model.layers.0.weightings.1.alpha", (2, )),
            ),
            each_n_epochs=100,
        ),
        ElissabethWeighting(example, each_n_epochs=100, use_wandb=True)
        # ElissabethISTracker(
        #     example,
        #     reduce="norm",
        #     each_n_epochs=100,
        #     use_wandb=True,
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
    fig, ax = plt.subplots(2, 2, sharex=True)
    W_Q = lightning_module.model.get_parameter(
        "layers.0.weightings.1.W_Q"
    ).detach().numpy()
    W_K = lightning_module.model.get_parameter(
        "layers.0.weightings.1.W_K"
    ).detach().numpy()

    ax[0, 0].matshow(
        W_Q[0, 0],
        cmap="seismic",
    )
    ax[0, 1].matshow(
        W_Q[0, 1],
        cmap="seismic",
    )
    ax[1, 0].matshow(
        W_K[0, 0],
        cmap="seismic",
    )
    ax[1, 1].matshow(
        W_K[0, 1],
        cmap="seismic",
    )
    # ax[2].matshow(
    #     (b_Q - b_K)[0,0],
    #     cmap="seismic",
    # )
    # W_O = lightning_module.model.get_parameter("embedding.weight").detach().numpy()
    # ax.matshow(
    #     W_O,
    #     cmap="seismic",
    # )
    # print(W_O)
    # mat = get_attention_matrix(
    #     lightning_module.model,  # type: ignore
    #     x,
    #     dims=2,
    # )
    # print(mat.shape)
    # y_hat = lightning_module.predict_step(x, 0)
    # print(f"Input     : {x}\nTarget    : {y}\nPrediction: {y_hat}")
    # plot_attention_matrix(
    #     mat[0, ..., 0],
    #     example=x.numpy()[0],
    #     figsize=(50, 10),
    #     show_product=True,
    #     log_colormap=False,
    #     causal_mask=True,
    # )
    # plt.savefig(
    #     os.path.join(os.path.dirname(__file__), "plot.pdf"),
    #     bbox_inches="tight",
    #     facecolor=(0, 0, 0, 0),
    # )
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("mode", choices=["train", "test", "plot"])
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--load", default=None)

    args = parser.parse_args()

    lightning_module = build_model()

    load = None
    if args.load is not None:
        directory = os.path.dirname(__file__)
        for folder in os.listdir(directory):
            path = os.path.join(directory, folder)
            if not (os.path.isdir(path)
                        and os.path.isdir(os.path.join(path, args.load))):
                continue
            if os.path.join(path, args.load, "checkpoints"):
                chckpt_path = os.path.join(path, args.load, "checkpoints")
                load = os.path.join(chckpt_path, os.listdir(chckpt_path)[0])
                saved_ = torch.load(load)
                lightning_module.load_state_dict(saved_["state_dict"])
    if args.load is not None and load is None:
        raise FileExistsError(
            "Load specification does not point to a saved model"
        )

    if args.mode == "train":
        train(
            lightning_module,
            load_path=load,
            use_wandb=args.online,
            progress_bar=not args.online,
        )
    elif args.mode == "test":
        train(lightning_module, only_test=True)
    elif args.mode == "plot":
        plot(lightning_module)


if __name__ == '__main__':
    main()
