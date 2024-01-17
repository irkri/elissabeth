import argparse
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

from sainomore.elissabeth import Elissabeth, LISSConfig, CLISSConfig
from sainomore.callbacks import (ElissabethISTracker, ElissabethWeighting,
                                 GeneralConfigCallback, WeightHistory)
from sainomore.data import GivenDataModule, long_lookup
from sainomore.lightning import TokenPredictionModule
from sainomore.models import (DecoderOnlyTransformer,
                              DecoderOnlyTransformerConfig, ModelConfig)
from sainomore.tools import (get_liss_attention_matrix,
                             plot_liss_attention_matrix)

torch.set_float32_matmul_precision('high')

SAVE_PATH: Optional[str] = None

config = {
    "n_samples": 5000,
    "context_length": 100,
    "characters": 5,

    "lr": 5e-3,
    "weight_decay": 1e-4,
    "epochs": 501,

    "batch_size": 64,
    "val_size": 0.2,
}


def build_model() -> TokenPredictionModule:
    # model_config = DecoderOnlyTransformerConfig(
    #     context_length=config["context_length"],
    #     input_vocab_size=config["characters"],
    #     n_layers=2,
    #     n_heads=4,
    #     d_hidden=64,
    #     d_head=16,
    #     ffn_units=128,
    #     bias=False,
    # )
    # model = DecoderOnlyTransformer(model_config)

    model_config = LISSConfig(
        context_length=config["context_length"],
        input_vocab_size=config["characters"],
        n_layers=1,
        length_is=2,
        n_is=8,
        d_values=16,
        values_2D=False,
        d_hidden=32,#config["characters"],
        # exponent=1,
        # d_query_key=5,
        bias_query_key=True,
        bias_value=True,
        positional_encoding=None,
        distance_weighting=False,
        pe_key=True,
        pe_value=False,
        share_queries=False,
        share_keys=False,
        share_values=False,
        sum_normalization="same",
    )
    model = Elissabeth(model_config)
    # model.set_eye("embedding.weight")

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
            # multiple_keys=False,
        ),
        val_size=config["val_size"],
        batch_size=config["batch_size"],
        num_workers=3,
        # somehow this option is important, atleast on CPU
        # (no more waiting between epochs)
        persistent_workers=True,
    )
    # ex = next(iter(data_module.train_dataloader()))
    # print(ex[0][0, :])
    # print(ex[1][0, :])
    # exit()

    wandb_logger = None
    if use_wandb:
        wandb_logger = WandbLogger(
            project="Elissabeth Long Lookup",
            checkpoint_name=load_path,
            tags=[lightning_module.model.layers[0].__class__.__name__],
            id=load_path.split("/")[1] if load_path is not None else None,
            resume="must" if load_path is not None else False,
        )

        wandb_logger.experiment.config.update(
            lightning_module.model.config.to_dict()
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
                ("model.layers.0.W_Q", (2, 0)),
                ("model.layers.0.b_Q", (0, )),
                ("model.layers.0.W_K", (2, 0)),
                ("model.layers.0.b_K", (0, )),
                # ("model.layers.0.E_K", (2, 0)),
                # "model.layers.0.W_V",
                # "model.layers.0.b_V",
                # "model.layers.0.E_V",
                # "model.layers.0.W_O",
                # ("model.layers.0.alpha", (1, 2)),
                "model.embedding.weight",
                "model.unembedding.weight",
            ),
            each_n_epochs=100,
        ),
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
    x, y = long_lookup(
        1,
        length=config["context_length"],
        characters=config["characters"],
    )

    # b_Q = lightning_module.model.get_parameter("layers.0.b_Q").detach().numpy()
    # b_K = lightning_module.model.get_parameter("layers.0.b_K").detach().numpy()
    # fig, ax = plt.subplots(4, 1, sharex=True)
    # ax[0].matshow(
    #     b_Q[0,0],
    #     cmap="seismic",
    # )
    # ax[1].matshow(
    #     b_K[0,0],
    #     cmap="seismic",
    # )
    # ax[2].matshow(
    #     (b_Q - b_K)[0,0],
    #     cmap="seismic",
    # )
    # W_O = lightning_module.model.get_parameter("layers.0.W_O").detach().numpy()
    # ax[3].matshow(
    #     W_O.T,
    #     cmap="seismic",
    # )
    # print(W_O)
    mat = get_liss_attention_matrix(lightning_module.model, x)  # type: ignore
    y_hat = lightning_module.predict_step(x, 0)
    print(f"Input     : {x}\nTarget    : {y}\nPrediction: {y_hat}")
    plot_liss_attention_matrix(
        mat[:, :, 0, :, :],
        example=x.numpy()[0],
        figsize=(50, 10),
        log_colormap=False,
        causal_mask=False,
    )
    plt.savefig(
        os.path.join(os.path.dirname(__file__), "plot.pdf"),
        bbox_inches="tight",
        facecolor=(0, 0, 0, 0),
    )
    # plt.show()


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
            else:
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
