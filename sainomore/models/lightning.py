__all__ = ["LastTokenPredictionModule", "EveryTokenPredictionModule"]

import lightning.pytorch as L
import torch
import torchmetrics.classification
from torch import nn


class LastTokenPredictionModule(L.LightningModule):
    """Lightning module that performs a classfication with the given
    model based on data ``(x,y)``, where both ``x`` and ``y`` are
    sequences of tokens. This module only compares the models output at
    the last position to the last position in ``y`` with a cross-entropy
    loss function.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_classes: int,
        learning_rate: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.learning_rate = learning_rate
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.acc_metric = torchmetrics.classification.MulticlassAccuracy(
            num_classes,
        )
        self.optim_kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        x, y = batch
        y = y[:, -1]
        outputs = self(x)[:, -1, :]

        loss = self.criterion(outputs, y)
        accuracy = self.acc_metric(outputs, y)

        self.log(
            'train/loss',
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            'train/accuracy',
            accuracy,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return {'loss': loss, 'accuracy': accuracy}

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        x, y = batch
        y = y[:, -1]
        outputs = self(x)[:, -1, :]

        loss = self.criterion(outputs, y)
        accuracy = self.acc_metric(torch.softmax(outputs, dim=-1), y)

        self.log(
            'validation/loss',
            loss,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            'validation/accuracy',
            accuracy,
            on_step=False,
            on_epoch=True,
        )
        return {'validation_loss': loss, 'validation_accuracy': accuracy}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            **self.optim_kwargs,
        )
        return optimizer


class EveryTokenPredictionModule(L.LightningModule):
    """Lightning module that performs a classfication with the given
    model based on data ``(x,y)``, where both ``x`` and ``y`` are
    sequences of tokens. This module compares the models output at every
    position to the corresponding position in ``y`` with a cross-entropy
    loss function.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-3,
        loss: type[nn.Module] = nn.MSELoss,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.learning_rate = learning_rate
        self.model = model
        self.criterion = loss()
        self.optim_kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        x, y = batch
        outputs = self(x)

        loss = self.criterion(outputs, y)
        self.log(
            'train/loss',
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        x, y = batch
        outputs = self(x)

        loss = self.criterion(outputs, y)
        self.log(
            'validation/loss',
            loss,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            **self.optim_kwargs,
        )
        return optimizer
