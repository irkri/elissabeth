from typing import Optional

import lightning.pytorch as L
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.wandb import WandbLogger
import torch

from data import imitate_mha
from data.lightning import GivenDataModule
from models import CosAttention
from models.callbacks import ParameterLoggingCallback
from models.config import ModelConfig
from models.lightning import EveryTokenPredictionModule


USE_WANDB: bool = False
LOAD_PATH: Optional[str] = "lightning_logs/version_1/checkpoints/epoch=9-step=1250.ckpt"


def main() -> None:

    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 64

    data_module = GivenDataModule(
        imitate_mha(
            n_samples=10_000,
            length=100,
            embed_dim=64,
            n_heads=4,
            seed=1,
        ),
        val_size=0.2,
        batch_size=batch_size,
        num_workers=6,
        # somehow this option is important, atleast on CPU
        # (no more waiting between epochs)
        persistent_workers=True,
    )

    model_config = ModelConfig(
        context_length=100,
        vocab_size=64,
        output_dim=64,
        d_hidden=64,
        n_heads=4,
    )

    model = CosAttention(
        model_config,
        use_tanh=False,
        use_xavier=False,
        randomize_delta=False,
        epsilon=None,
    )

    lightning_module = EveryTokenPredictionModule(
        model,
        learning_rate=lr,
    )
    if LOAD_PATH is not None:
        lightning_module.load_state_dict(torch.load(LOAD_PATH))

    wandb_logger = None
    if USE_WANDB:
        wandb_logger = WandbLogger(
            # name="imitate_mha",
            project="cosine_attention",
        )
        wandb_logger.experiment.config.update({
            "n_samples": 10_000,
            "length": 100,
            "embed_dim": 64,
        })
        wandb_logger.experiment.config.update(model_config.to_dict())
        wandb_logger.experiment.config.update({
            "use_tanh": False,
            "use_xavier": False,
            "randomize_delta": False,
            "epsilon": None,
        })
        wandb_logger.experiment.config.update({
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size,
        })

    callbacks: list[Callback] = [ParameterLoggingCallback(max_depth=10)]

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        callbacks=callbacks,
        logger=wandb_logger,
    )

    trainer.fit(lightning_module, data_module)

    if USE_WANDB:
        wandb.finish()


if __name__ == '__main__':
    main()
