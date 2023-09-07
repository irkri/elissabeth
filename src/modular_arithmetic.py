from typing import Optional

import torch
import lightning.pytorch as L
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.wandb import WandbLogger

from data import modular_arithmetic
from data.lightning import GivenDataModule
from models import CosDecoderOnlyTransformer
from models.callbacks import ParameterLoggingCallback
from models.config import ModelConfig
from models.lightning import LastTokenPredictionModule


USE_WANDB: bool = True
LOAD_PATH: Optional[str] = None #"cosine_attention/16g7dpbf/checkpoints/epoch=49-step=3550.ckpt"
SAVE_PATH: Optional[str] = None


def main() -> None:

    epochs = 150
    lr = 1e-3
    P = 53
    batch_size = 32

    data_module = GivenDataModule(
        modular_arithmetic(P),
        val_size=0.3,
        batch_size=batch_size,
        num_workers=6,
        # somehow this option is important, atleast on CPU
        # (no more waiting between epochs)
        persistent_workers=True,
    )

    model_config = ModelConfig(
        context_length=3,
        vocab_size=P+1,
        output_dim=P,
        n_layers=1,
        n_heads=5,
        d_embedding=32,
        d_hidden=32,
        ffn_units=32,
    )

    model = CosDecoderOnlyTransformer(
        model_config,
        use_tanh=False,
        epsilon=None,
        normalize=True,
        use_xavier=False,
        randomize_delta=False,
    )

    lightning_module = LastTokenPredictionModule(
        model,
        num_classes=P,
        learning_rate=lr,
    )
    if LOAD_PATH is not None:
        configs = torch.load(LOAD_PATH)
        lightning_module.load_state_dict(configs["state_dict"])

    wandb_logger = None
    if USE_WANDB:
        wandb_logger = WandbLogger(
            # name="modular_arithmetic",
            project="cosine_attention",
            checkpoint_name=LOAD_PATH,
        )
        wandb_logger.experiment.config.update(model_config.to_dict())
        wandb_logger.experiment.config.update({
            "use_tanh": False,
            "normalize": True,
            "use_xavier": False,
            "randomize_delta": False,
            "epsilon": None,
        })
        wandb_logger.experiment.config.update({
            "P": P,
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size,
        })
        wandb_logger.watch(lightning_module)

    callbacks: list[Callback] = [ParameterLoggingCallback(max_depth=10)]

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        callbacks=callbacks,
        logger=wandb_logger,
        default_root_dir=SAVE_PATH,
    )

    trainer.fit(lightning_module, data_module, ckpt_path=LOAD_PATH)

    if USE_WANDB:
        wandb.finish()


if __name__ == '__main__':
    main()
