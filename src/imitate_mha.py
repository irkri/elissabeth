from typing import Optional

import lightning.pytorch as L
import torch
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.wandb import WandbLogger
from matplotlib import pyplot as plt

from data import imitate_mha
from data.lightning import GivenDataModule
from models import (CosAttention, CosDecoderOnlyTransformerConfig,
                    DecoderOnlyTransformer, DecoderOnlyTransformerConfig,
                    ModelConfig)
from models.callbacks import GeneralConfigCallback
from models.lightning import EveryTokenPredictionModule


USE_WANDB: bool = False
LOAD_PATH: Optional[str] = None
SAVE_PATH: Optional[str] = None

config = {
    "n_samples" : 10_000,
    "length" : 100,
    "embed_dim" : 64,
    "n_heads" : 4,
    "seed" : 1,

    "lr": 1e-3,
    "batch_size": 64,
    "epochs": 50,

    "val_size": 0.3,
}


def build_model() -> tuple[L.LightningModule, ModelConfig]:
    model_config = CosDecoderOnlyTransformerConfig(
        context_length=3,
        input_vocab_size=64,
        output_vocab_size=64,
        n_layers=1,
        n_heads=4,
        d_hidden=64,
        normalize=True,
    )
    model = CosAttention(model_config)

    lightning_module = EveryTokenPredictionModule(
        model,
        learning_rate=config["lr"],
    )
    return lightning_module, model_config


def train() -> None:
    data_module = GivenDataModule(
        imitate_mha(
            n_samples=config["n_samples"],
            length=config["length"],
            embed_dim=config["embed_dim"],
            n_heads=config["n_heads"],
            seed=config["seed"],
        ),
        val_size=config["val_size"],
        batch_size=config["batch_size"],
        num_workers=6,
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
        enable_progress_bar=True,
    )

    trainer.fit(lightning_module, data_module, ckpt_path=LOAD_PATH)

    if USE_WANDB:
        wandb.finish()


def plot() -> None:
    lightning_module, _ = build_model()

    if LOAD_PATH is not None:
        saved_ = torch.load(LOAD_PATH)
        lightning_module.load_state_dict(saved_["state_dict"])

    weight = lightning_module.get_parameter(
        "model.decoder.enc_layers.0.causal_self_attention.W_O"
    ).detach().numpy()

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    for i in range(4):
        ax[i].matshow(weight[i, :, :], cmap="hot")
    plt.show()


if __name__ == '__main__':
    plot()
