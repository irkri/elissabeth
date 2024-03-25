__all__ = ["TokenPredictionModule", "VectorApproximationModule"]

from typing import Any, Optional

import lightning.pytorch as L
import torch
from torch import nn

from .base import SAINoMoreModule


class SAILearningModule(L.LightningModule):

    def __init__(
        self,
        model: SAINoMoreModule,
        loss: nn.Module,
        learning_rate: float,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **optim_kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "loss"])
        self.learning_rate = learning_rate
        self.model = model
        self.criterion = loss
        self._optimizer = optimizer
        self._optim_kwargs = optim_kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self._optimizer is None:
            return torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                **self._optim_kwargs,
            )
        else:
            return self._optimizer


class TokenPredictionModule(SAILearningModule):

    def __init__(
        self,
        model: SAINoMoreModule,
        learning_rate: float = 1e-3,
        only_last: bool = True,
        loss: Optional[nn.Module] = None,
        accuracy: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **optim_kwargs,
    ) -> None:
        super().__init__(
            model=model,
            loss=loss if loss is not None else nn.CrossEntropyLoss(),
            learning_rate=learning_rate,
            optimizer=optimizer,
            **optim_kwargs,
        )
        self.acc_metric = accuracy
        self._only_last = only_last

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        x, y = batch
        outputs = self(x)
        outputs, y = outputs.swapaxes(1, 2), y.swapaxes(1, 2)
        if self._only_last:
            y = y[..., -1]
            outputs = outputs[..., -1]

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
        outputs, y = outputs.swapaxes(1, 2), y.swapaxes(1, 2)
        if self._only_last:
            y = y[..., -1]
            outputs = outputs[..., -1]

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
        output = torch.argmax(self(batch), 2)
        if self._only_last:
            return output[..., -1]
        return output


class VectorApproximationModule(SAILearningModule):

    def __init__(
        self,
        model: SAINoMoreModule,
        learning_rate: float = 1e-3,
        only_last: bool = True,
        loss: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **optim_kwargs,
    ) -> None:
        super().__init__(
            model=model,
            loss=loss if loss is not None else nn.MSELoss(),
            learning_rate=learning_rate,
            optimizer=optimizer,
            **optim_kwargs,
        )
        self._only_last = only_last

    def predict_step(
        self,
        batch: Any,
        batch_idx: int = 0,
        dataloader_idx: int = 0,
    ) -> Any:
        output = self(batch)
        if self._only_last:
            return output[:, -1, :]
        return output
