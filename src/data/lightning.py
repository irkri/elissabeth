__all__ = ["GivenDataModule"]

import lightning.pytorch as L

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


class GivenDataModule(L.LightningDataModule):

    def __init__(
        self,
        xy: tuple[torch.Tensor, torch.Tensor],
        val_size: float = 0.2,
        batch_size: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()
        data = TensorDataset(*xy)
        nt = int(np.ceil((1-val_size) * len(data)))
        nv = int(np.floor(val_size * len(data)))
        self._train_set, self._val_set = random_split(data, [nt, nv])

        self.batch_size = batch_size
        self._kwargs = kwargs

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_set,
            batch_size=self.batch_size,
            shuffle=True,
            **self._kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_set,
            batch_size=self.batch_size,
            **self._kwargs,
        )
