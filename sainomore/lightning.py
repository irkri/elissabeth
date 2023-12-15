__all__ = ["TokenPredictionModule"]

from typing import Any, Optional

import lightning.pytorch as L
import torch
from torch import nn

from .base import SAINoMoreModule


class TokenPredictionModule(L.LightningModule):
    """Lightning module that performs a classfication with the given
    model based on data ``(x,y)``, where both ``x`` and ``y`` are
    sequences of tokens.
    """

    def __init__(
        self,
        model: SAINoMoreModule,
        learning_rate: float = 1e-3,
        only_last: bool = True,
        loss: Optional[nn.Module] = None,
        accuracy: Optional[nn.Module] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss", "accuracy"])
        self.learning_rate = learning_rate
        self.model = model
        self.criterion = loss if loss is not None else nn.CrossEntropyLoss()
        self.acc_metric = accuracy
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
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        metrics = {"loss": loss}

        if self.acc_metric is not None:
            accuracy = self.acc_metric(outputs, y)
            self.log(
                'validation/accuracy',
                accuracy,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
            metrics.update({"accuracy": accuracy})

        return metrics

    def predict_step(
        self,
        batch: Any,
        batch_idx: int = 0,
        dataloader_idx: int = 0,
    ) -> Any:
        output = torch.argmax(self(batch), 1)
        if self._only_last:
            return output[:, -1]
        return output

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            **self.optim_kwargs,
        )
        return optimizer
