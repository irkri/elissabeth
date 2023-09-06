import lightning.pytorch as L
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.wandb import WandbLogger

from data import GivenDataModule, modular_arithmetic
from models import CosDecoderOnlyTransformer
from models.callbacks import ParameterLoggingCallback
from models.config import ModelConfig
from models.lightning import LastTokenPredictionModule


def main(
    epochs: int = 10,
    lr: float = 1e-3,
    P: int = 53,
    batch_size: int = 32,
    use_wandb: bool = False,
) -> None:
    data_module = GivenDataModule(
        modular_arithmetic(P),
        val_size=0.2,
        batch_size=batch_size,
        num_workers=6,
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
    )

    lightning_module = LastTokenPredictionModule(
        model,
        num_classes=P,
        learning_rate=lr,
    )

    wandb_logger = None
    if use_wandb:
        wandb_logger = WandbLogger(
            name="modular_arithmetic",
            project="cosine_attention",
        )
        wandb_logger.experiment.config.update(model_config.to_dict())
        wandb_logger.experiment.config.update({
            "use_tanh": False,
            "epsilon": None,
            "normalize": True,
            "use_xavier": False,
        })
        wandb_logger.experiment.config.update({
            "P": P,
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
