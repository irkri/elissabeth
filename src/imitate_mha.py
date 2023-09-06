import lightning.pytorch as L
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.wandb import WandbLogger

from data import GivenDataModule, imitate_mha
from models import CosAttention
from models.callbacks import ParameterLoggingCallback
from models.config import ModelConfig
from models.lightning import EveryTokenPredictionModule


def main(
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 64,
    use_wandb: bool = False,
) -> None:
    data_module = GivenDataModule(
        imitate_mha(
            n_samples=10_000,
            length=7,
            embed_dim=4,
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
        context_length=7,
        vocab_size=4,
        output_dim=4,
        d_hidden=4,
        n_heads=1,
    )

    model = CosAttention(
        model_config,
        use_tanh=False,
        epsilon=None,
        use_xavier=False,
    )

    lightning_module = EveryTokenPredictionModule(
        model,
        learning_rate=lr,
    )

    wandb_logger = None
    if use_wandb:
        wandb_logger = WandbLogger(
            name="imitate_mha",
            project="cosine_attention",
        )
        wandb_logger.experiment.config.update({
            "n_samples": 10_000,
            "length": 7,
            "embed_dim": 4,
        })
        wandb_logger.experiment.config.update(model_config.to_dict())
        wandb_logger.experiment.config.update({
            "use_tanh": False,
            "epsilon": None,
            "use_xavier": False,
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

    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
