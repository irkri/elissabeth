from typing import Optional

import lightning.pytorch as L
import numpy as np
import torch
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.wandb import WandbLogger
from matplotlib import pyplot as plt

from sainomore.callbacks import GeneralConfigCallback, WeightHistory
from sainomore.data import streaks
from sainomore.data.lightning import GivenDataModule
from sainomore.lightning import TokenPredictionModule
from sainomore.models import Elissabeth, ElissabethConfig, ModelConfig
from sainomore.tools import plot_liss_attention_matrix

USE_WANDB: bool = False
PROGRESS_BAR: bool = True
LOAD_PATH: Optional[str] = "lightning_logs/version_23/checkpoints/epoch=200-step=5025.ckpt"
SAVE_PATH: Optional[str] = "./"

config = {
    "n_samples": 1000,
    "context_length": 100,
    "signal_length": 10,
    "signal_start": 10,

    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 201,

    "batch_size": 32,
    "val_size": 0.2,
}


def build_model() -> tuple[TokenPredictionModule, ElissabethConfig]:
    model_config = ElissabethConfig(
        context_length=config["context_length"]+1,
        input_vocab_size=2,
        n_layers=2,
        iss_length=2,
        d_hidden=8,
        d_head=8,
        single_query_key=True,
        share_queries=False,
        share_keys=False,
        share_values=False,
        normalize_layers=True,
        normalize_iss=True,
        positional_encoding=False,
    )
    model = Elissabeth(model_config)

    lightning_module = TokenPredictionModule(
        model,
        num_classes=2,
        learning_rate=config["lr"],
        weight_decay=config["weight_decay"],
        loss=torch.nn.CrossEntropyLoss,
        only_last=True,
    )
    return lightning_module, model_config


def train() -> None:
    data_module = GivenDataModule(
        streaks(
            n_samples=config["n_samples"],
            length=config["context_length"],
            signal_length=config["signal_length"],
            signal_start=config["signal_start"],
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
            tags=["Elissabeth", "streaks"],
            id=LOAD_PATH.split("/")[1] if LOAD_PATH is not None else None,
            resume="must" if LOAD_PATH is not None else False,
        )

        wandb_logger.experiment.config.update(model_config.to_dict())
        wandb_logger.experiment.config.update(config, allow_val_change=True)

        wandb_logger.watch(lightning_module, log="all")

    callbacks: list[Callback] = [
        GeneralConfigCallback(max_depth=10),
        # WeightHistory((
        #         "model.layers.0.W_Q",
        #         "model.layers.0.W_K",
        #         "model.layers.0.W_V",
        #         "model.layers.0.W_O",
        #         "model.embedding.weight",
        #         "model.unembedding.weight",
        #     ),
        #     reduce_axis=[0, 0, 0, None, None, None],
        #     each_n_epochs=200,
        # ),
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
    lightning_module, model_config = build_model()

    if LOAD_PATH is not None:
        saved_ = torch.load(LOAD_PATH)
        lightning_module.load_state_dict(saved_["state_dict"])

    x, y = streaks(
            n_samples=1,
            length=config["context_length"],
            signal_length=config["signal_length"],
            signal_start=config["signal_start"],
    )
    plot_liss_attention_matrix(
        lightning_module.model,
        model_config,
        x,
    )

    # plt.savefig("attention_matrix.pdf", bbox_inches="tight")
    plt.show()

    # for name in ["Q", "K", "V"]:
    #     weight = lightning_module.get_parameter(
    #         f"model.decoder.enc_layers.0.causal_self_attention.W_{name}"
    #     ).detach().numpy()

    #     fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    #     for i in range(4):
    #         ax[i].matshow(weight[i, :, :], cmap="hot")
    #     plt.savefig(f"W_{name}.png", bbox_inches="tight")

    # components = ["embedding", "unembedding"]
    # fig, ax = plt.subplots(1, len(components))
    # for i, name in enumerate(components):
    #     weight = lightning_module.get_parameter(
    #         f"model.{name}.weight"
    #     ).detach().numpy()
    #     ax[i].matshow(weight, cmap="hot")
    # plt.savefig(f"{'_'.join(components)}.png", bbox_inches="tight")


if __name__ == '__main__':
    plot()
