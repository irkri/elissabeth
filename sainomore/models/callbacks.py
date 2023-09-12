__all__ = ["GeneralConfigCallback"]

import lightning.pytorch as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.model_summary.model_summary import summarize


class GeneralConfigCallback(Callback):

    def __init__(self, max_depth: int = 10) -> None:
        super().__init__()
        self._max_depth = max_depth

    def on_train_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        model_summary = summarize(pl_module, max_depth=self._max_depth)

        for logger in trainer.loggers:
            logger.log_hyperparams({
                "total_parameters": model_summary.total_parameters,
                "trainable_parameters": model_summary.trainable_parameters,
                "model_size": model_summary.model_size,
            })
