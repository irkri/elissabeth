from typing import Optional

import lightning.pytorch as L
import torch
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.wandb import WandbLogger
from matplotlib import pyplot as plt

from sainomore.data import modular_arithmetic
from sainomore.data.lightning import GivenDataModule
from sainomore.models import (CosDecoderOnlyTransformer,
                              CosDecoderOnlyTransformerConfig,
                              DecoderOnlyTransformer,
                              DecoderOnlyTransformerConfig, ModelConfig)
from sainomore.models.callbacks import GeneralConfigCallback
from sainomore.models.lightning import LastTokenPredictionModule

USE_WANDB: bool = True
LOAD_PATH: Optional[str] = None
SAVE_PATH: Optional[str] = None

config = {
    "P": 113,

    "lr": 1e-3,
    "weight_decay": 1.0,
    "betas": (0.9, 0.98),
    "batch_size": 113**2,
    "epochs": 5000,

    "val_size": 0.3,
}


def build_model() -> tuple[L.LightningModule, ModelConfig]:
    model_config = DecoderOnlyTransformerConfig(
        context_length=3,
        input_vocab_size=config["P"]+1,
        output_vocab_size=config["P"],
        n_layers=1,
        n_heads=4,
        d_hidden=32,
        ffn_units=128,
        normalize=True,
    )
    model = DecoderOnlyTransformer(model_config)

    lightning_module = LastTokenPredictionModule(
        model,
        num_classes=config["P"],
        learning_rate=config["lr"],
        weight_decay=config["weight_decay"],
        betas=config["betas"],
    )
    return lightning_module, model_config


def train() -> None:
    data_module = GivenDataModule(
        modular_arithmetic(config["P"]),
        val_size=config["val_size"],
        batch_size=config["batch_size"],
        num_workers=12,
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
            project="cosine_attention",
            checkpoint_name=LOAD_PATH,
            tags=["attention"],
            id=LOAD_PATH.split("/")[1] if LOAD_PATH is not None else None,
            resume="must" if LOAD_PATH is not None else False,
        )

        wandb_logger.experiment.config.update(model_config.to_dict())
        wandb_logger.experiment.config.update(config, allow_val_change=True)

        wandb_logger.watch(lightning_module, log="all")

    callbacks: list[Callback] = [GeneralConfigCallback(max_depth=10)]

    trainer = L.Trainer(
        max_epochs=config["epochs"],
        accelerator="auto",
        callbacks=callbacks,
        logger=wandb_logger,
        default_root_dir=SAVE_PATH,
        enable_progress_bar=False,
    )

    trainer.fit(lightning_module, data_module, ckpt_path=LOAD_PATH)

    if USE_WANDB:
        wandb.finish()


def plot() -> None:
    lightning_module, _ = build_model()

    if LOAD_PATH is not None:
        saved_ = torch.load(LOAD_PATH)
        lightning_module.load_state_dict(saved_["state_dict"])

    for name in ["K", "V", "Q"]:
        weight = lightning_module.get_parameter(
            f"model.decoder.enc_layers.0.causal_self_attention.W_{name}"
        ).detach().numpy()

        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        for i in range(4):
            ax[i].matshow(weight[i, :, :], cmap="hot")
        plt.savefig(f"W_{name}.png", bbox_inches="tight")

    # components = ["embedding", "unembedding"]
    # fig, ax = plt.subplots(1, len(components))
    # for i, name in enumerate(components):
    #     weight = lightning_module.get_parameter(
    #         f"model.{name}.weight"
    #     ).detach().numpy()
    #     ax[i].matshow(weight, cmap="hot")
    # plt.savefig(f"{'_'.join(components)}.png", bbox_inches="tight")


if __name__ == '__main__':
    train()
