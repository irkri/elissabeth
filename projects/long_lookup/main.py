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

from sainomore.callbacks import (ElissabethWeighting, GeneralConfigCallback,
                                 WeightHistory, ElissabethISTracker)
from sainomore.data import long_lookup
from sainomore.data.lightning import GivenDataModule
from sainomore.lightning import TokenPredictionModule
from sainomore.models import (DecoderOnlyTransformer,
                              DecoderOnlyTransformerConfig, Elissabeth,
                              ElissabethConfig, ModelConfig)
from sainomore.tools import (get_liss_attention_matrix,
                             plot_liss_attention_matrix)

torch.set_float32_matmul_precision('high')

SAVE_PATH: Optional[str] = None

config = {
    "n_samples": 5000,
    "context_length": 100,
    "characters": 5,

    "lr": 75e-4,
    "weight_decay": 1e-4,
    "epochs": 1001,

    "batch_size": 32,
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
    #     bias=True,
    # )
    # model = DecoderOnlyTransformer(model_config)

    model_config = ElissabethConfig(
        context_length=config["context_length"],
        input_vocab_size=config["characters"],
        n_layers=1,
        length_is=2,
        n_is=32,
        d_hidden=config["characters"],
        weighting="exp",
        positional_encoding=None,
        denominator_is=False,
        distance_weighting=True,
        positional_bias=True,
        positional_bias_values=False,
        single_query_key=False,
        share_queries=False,
        share_keys=False,
        share_values=False,
        sum_normalization="same",
    )
    model = Elissabeth(model_config)
    model.set_eye("embedding.weight")

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
        ),
        val_size=config["val_size"],
        batch_size=config["batch_size"],
        num_workers=3,
        # somehow this option is important, atleast on CPU
        # (no more waiting between epochs)
        persistent_workers=True,
    )

    wandb_logger = None
    if use_wandb:
        wandb_logger = WandbLogger(
            project="Elissabeth 3",
            checkpoint_name=load_path,
            tags=["Elissabeth", "long lookup"],
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
    # np.save(
    #     os.path.join(
    #         os.path.dirname(__file__), "data", "elissabeth_example.npy",
    #     ),
    #     example.numpy(),
    # )
    callbacks: list[Callback] = [
        GeneralConfigCallback(max_depth=10),
        # Progressbar(),
        WeightHistory((
                "model.layers.0.W_Q",
                "model.layers.0.W_K",
                "model.layers.0.E_K",
                "model.layers.0.W_V",
                # "model.layers.0.E_V",
                "model.layers.0.W_O",
                "model.layers.0.alpha",
                "model.embedding.weight",
                "model.unembedding.weight",
            ),
            reduce_axis=[0, 0, 2, 0, None, None, None, None],
            each_n_epochs=200,
        ),
        ElissabethISTracker(
            example,
            reduce="norm",
            each_n_epochs=200,
            use_wandb=True,
        ),
        # ElissabethWeighting(
        #     example,
        #     each_n_epochs=100,
        #     save_path=os.path.join(os.path.dirname(__file__), "data"),
        # ),
    ]

    trainer = L.Trainer(
        max_epochs=config["epochs"],
        accelerator="auto",
        callbacks=callbacks if not only_test else None,
        logger=wandb_logger if not only_test else None,
        default_root_dir=SAVE_PATH,
        enable_progress_bar=progress_bar,
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
    mat = get_liss_attention_matrix(lightning_module.model, x)
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
