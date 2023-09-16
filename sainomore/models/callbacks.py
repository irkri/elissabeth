__all__ = ["GeneralConfigCallback"]

from typing import Optional, Sequence

import lightning.pytorch as L
import numpy as np
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.model_summary.model_summary import summarize


class GeneralConfigCallback(Callback):

    def __init__(self, max_depth: int = 10) -> None:
        super().__init__()
        self._max_depth = max_depth

    def on_train_start(
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


class WeightMatrixCallback(Callback):

    def __init__(
        self,
        weight_names: Sequence[str],
        reduce_axis: Optional[Sequence[int | None]] = None,
        each_n_epochs: int = 1,
    ) -> None:
        super().__init__()
        self._weight_names = list(weight_names)
        self._reduce_axis = reduce_axis
        self._each_n_epochs = each_n_epochs
        self._epoch = -1

    def on_train_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        self._epoch = -1

    def on_train_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        self._epoch += 1
        if self._epoch % self._each_n_epochs != 0:
            return
        for k, wname in enumerate(self._weight_names):
            param = pl_module.get_parameter(wname).detach().cpu().numpy()
            if (self._reduce_axis is not None
                    and self._reduce_axis[k] is not None):
                param = [np.take(param, i, self._reduce_axis[k])
                         for i in range(param.shape[self._reduce_axis[k]])]
            else:
                param = [param]
            for logger in trainer.loggers:
                if isinstance(logger, WandbLogger):
                    logger.log_image(wname, param)
