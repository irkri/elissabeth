__all__ = ["TokenPredictionModule"]

from typing import Optional

import lightning.pytorch as L
import torch
import torchmetrics.classification
from torch import nn


class TokenPredictionModule(L.LightningModule):
    """Lightning module that performs a classfication with the given
    model based on data ``(x,y)``, where both ``x`` and ``y`` are
    sequences of tokens.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_classes: Optional[int] = None,
        learning_rate: float = 1e-3,
        only_last: bool = True,
        loss: type[nn.Module] = nn.CrossEntropyLoss,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.learning_rate = learning_rate
        self.model = model
        self.criterion = loss()
        self.acc_metric = torchmetrics.classification.MulticlassAccuracy(
            num_classes,
        ) if num_classes is not None else None
        self._only_last = only_last
        self.optim_kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        x, y = batch
        outputs = self(x)
        if self._only_last:
            y = y[:, -1]
            outputs = outputs[:, :, -1]

        loss = self.criterion(outputs, y)
        self.log(
            'train/loss',
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        metrics = {"loss": loss}

        if self.acc_metric is not None:
            accuracy = self.acc_metric(outputs, y)
            self.log(
                'train/accuracy',
                accuracy,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
            metrics.update({"accuracy": accuracy})

        return metrics

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        x, y = batch
        outputs = self(x)
        if self._only_last:
            y = y[:, -1]
            outputs = outputs[:, :, -1]

        loss = self.criterion(outputs, y)
        self.log(
            'validation/loss',
            loss,
            on_step=False,
            on_epoch=True,
        )
        metrics = {"loss": loss}

        if self.acc_metric is not None:
            accuracy = self.acc_metric(outputs, y)
            self.log(
                'validation/accuracy',
                accuracy,
                on_step=False,
                on_epoch=True,
            )
            metrics.update({"accuracy": accuracy})

        return metrics

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            **self.optim_kwargs,
        )
        return optimizer
