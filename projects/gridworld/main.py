from typing import Optional

import lightning.pytorch as L
import torch
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.wandb import WandbLogger
from matplotlib import pyplot as plt

from sainomore.callbacks import GeneralConfigCallback, WeightHistory
from sainomore.data import gridworld
from sainomore.data.lightning import GivenDataModule
from sainomore.lightning import TokenPredictionModule
from sainomore.models import (ELISSABETH, CosDecoderOnlyTransformer,
                              CosDecoderOnlyTransformerConfig,
                              DecoderOnlyTransformer,
                              DecoderOnlyTransformerConfig, ELISSABETHConfig,
                              ModelConfig)

USE_WANDB: bool = False
PROGRESS_BAR: bool = True
LOAD_PATH: Optional[str] = None
SAVE_PATH: Optional[str] = None

config = {
    "n_samples": 5_000,
    "n_steps": 100,
    "S": 9,

    "lr": 5e-4,
    "weight_decay": 1e-4,
    "epochs": 1000,

    "batch_size": 32,
    "val_size": 0.3,
}


def build_model() -> tuple[L.LightningModule, ModelConfig]:
    # model_config = DecoderOnlyTransformerConfig(
    #     context_length=3,
    #     input_vocab_size=config["P"]+1,
    #     output_vocab_size=config["P"],
    #     n_layers=1,
    #     n_heads=4,
    #     d_head=32,
    #     d_hidden=128,
    #     ffn_units=512,
    #     normalize=False,
    # )
    # model = DecoderOnlyTransformer(model_config)
    # model_config = CosDecoderOnlyTransformerConfig(
    #     context_length=config["n_steps"],
    #     input_vocab_size=config["S"],
    #     output_vocab_size=config["S"],
    #     n_layers=1,
    #     n_heads=1,
    #     d_head=32,
    #     d_hidden=32,
    #     ffn_units=128,
    #     normalize=False,
    #     epsilon=None,
    #     use_tanh=False,
    # )
    # model = CosDecoderOnlyTransformer(model_config)
    model_config = ELISSABETHConfig(
        context_length=config["n_steps"],
        input_vocab_size=config["S"],
        output_vocab_size=config["S"],
        n_layers=2,
        iss_length=2,
        d_head=128,
        d_hidden=64,
        separate_qk=True,
    )
    model = ELISSABETH(model_config)

    lightning_module = TokenPredictionModule(
        model,
        num_classes=config["S"],
        learning_rate=config["lr"],
        weight_decay=config["weight_decay"],
        loss=torch.nn.CrossEntropyLoss,
        only_last=False,
    )
    return lightning_module, model_config


def train() -> None:
    data_module = GivenDataModule(
        gridworld(
            n_samples=config["n_samples"],
            n_steps=config["n_steps"],
            S=config["S"],
            cache_path="./data/",
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
            project="cosine_attention",
            checkpoint_name=LOAD_PATH,
            tags=["cos-attention", "gridworld"],
            id=LOAD_PATH.split("/")[1] if LOAD_PATH is not None else None,
            resume="must" if LOAD_PATH is not None else False,
        )

        wandb_logger.experiment.config.update(model_config.to_dict())
        wandb_logger.experiment.config.update(config, allow_val_change=True)

        wandb_logger.watch(lightning_module, log="all")

    callbacks: list[Callback] = [
        GeneralConfigCallback(max_depth=10),
        WeightHistory((
                # "model.decoder.layers.0.cos_attn.W_Q",
                # "model.decoder.layers.0.cos_attn.W_K",
                # "model.decoder.layers.0.cos_attn.W_V",
                # "model.decoder.layers.0.cos_attn.W_O",
                # "model.decoder.layers.0.mlp.seq.0.weight",
                # "model.decoder.layers.0.mlp.seq.2.weight",
                "model.layers.0.W_Q",
                "model.layers.0.W_K",
                "model.layers.0.W_V",
                "model.layers.0.W_O",
                "model.embedding.weight",
                "model.unembedding.weight",
            ),
            reduce_axis=[0, 0, 0, None, None, None],
            each_n_epochs=50,
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
