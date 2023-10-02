import os
from typing import Optional

import lightning.pytorch as L
import numpy as np
import torch
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.wandb import WandbLogger
from matplotlib import pyplot as plt

from sainomore.callbacks import (ElissabethWeighting, GeneralConfigCallback,
                                 WeightHistory)
from sainomore.data import long_lookup
from sainomore.data.lightning import GivenDataModule
from sainomore.lightning import TokenPredictionModule
from sainomore.models import Elissabeth, ElissabethConfig, ModelConfig
from sainomore.tools import plot_liss_attention_matrix

USE_WANDB: bool = True
PROGRESS_BAR: bool = False
LOAD_PATH: Optional[str] = None
SAVE_PATH: Optional[str] = None

config = {
    "n_samples": 5000,
    "context_length": 20,
    "characters": 5,

    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 501,

    "batch_size": 32,
    "val_size": 0.2,
}


def build_model() -> tuple[TokenPredictionModule, ElissabethConfig]:
    model_config = ElissabethConfig(
        context_length=config["context_length"],
        input_vocab_size=config["characters"],
        n_layers=1,
        iss_length=5,
        d_hidden=32,
        d_head=32,
        single_query_key=True,
        share_queries=False,
        share_keys=False,
        share_values=False,
        normalize_layers=True,
        normalize_iss=True,
        positional_encoding=True,
    )
    model = Elissabeth(model_config)

    lightning_module = TokenPredictionModule(
        model,
        num_classes=config["characters"],
        learning_rate=config["lr"],
        weight_decay=config["weight_decay"],
        loss=torch.nn.CrossEntropyLoss,
        only_last=True,
    )
    return lightning_module, model_config


def train() -> None:
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
    lightning_module, model_config = build_model()

    if LOAD_PATH is not None:
        saved_ = torch.load(LOAD_PATH)
        lightning_module.load_state_dict(saved_["state_dict"])

    wandb_logger = None
    if USE_WANDB:
        wandb_logger = WandbLogger(
            project="Elissabeth",
            checkpoint_name=LOAD_PATH,
            tags=["Elissabeth", "long_lookup"],
            id=LOAD_PATH.split("/")[1] if LOAD_PATH is not None else None,
            resume="must" if LOAD_PATH is not None else False,
        )

        wandb_logger.experiment.config.update(model_config.to_dict())
        wandb_logger.experiment.config.update(config, allow_val_change=True)

        wandb_logger.watch(lightning_module, log="all")

    example = long_lookup(
        n_samples=1,
        length=config["context_length"],
        characters=config["characters"],
    )[0]
    np.save(
        os.path.join(
            os.path.dirname(__file__), "data", "elissabeth_example.npy",
        ),
        example.numpy(),
    )
    callbacks: list[Callback] = [
        GeneralConfigCallback(max_depth=10),
        WeightHistory((
                "model.layers.0.W_Q",
                "model.layers.0.W_K",
                "model.layers.0.W_V",
                "model.layers.0.W_O",
                "model.embedding.weight",
                "model.unembedding.weight",
            ),
            reduce_axis=[0, 0, 0, None, None, None],
            each_n_epochs=100,
        ),
        ElissabethWeighting(
            example,
            each_n_epochs=100,
            save_path=os.path.join(os.path.dirname(__file__), "data"),
        ),
    ]

    trainer = L.Trainer(
        max_epochs=config["epochs"],
        accelerator="auto",
        callbacks=callbacks,
        logger=wandb_logger,
        default_root_dir=SAVE_PATH,
        enable_progress_bar=PROGRESS_BAR,
    )

    trainer.fit(lightning_module, data_module, ckpt_path=LOAD_PATH)

    if USE_WANDB:
        wandb.finish()


def plot() -> None:
    a = np.load(os.path.join(
            os.path.dirname(__file__), "data", "epoch00400",
            "elissabeth_weighting.npy",
    ))
    ex = np.load(os.path.join(
            os.path.dirname(__file__), "data", "elissabeth_example.npy",
    ))
    plot_liss_attention_matrix(a, example=ex[0], figsize=(20, 5))

    plt.savefig(
        os.path.join(os.path.dirname(__file__), "attention_matrix_1.pdf"),
        bbox_inches="tight",
        facecolor=(0, 0, 0, 0),
    )
    # plt.show()


if __name__ == '__main__':
    plot()
