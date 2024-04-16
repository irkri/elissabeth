from pathlib import Path

import torch
import numpy as np


PATH = Path("../../../univariate_arff")


class UCRLoader:

    def __init__(
        self,
        dataset: str,
    ) -> None:
        self.dataset = dataset

    @property
    def context_length(self) -> int:
        data = np.loadtxt(
            PATH / self.dataset / f"{self.dataset}_TRAIN.txt",
            max_rows=2,
        )
        return data.shape[1]

    @property
    def nlabels(self) -> int:
        data = np.loadtxt(
            PATH / self.dataset / f"{self.dataset}_TRAIN.txt",
            usecols=0,
        )
        return len(np.unique(data))

    def get_train(self) -> tuple[torch.Tensor, torch.Tensor]:
        data = np.loadtxt(PATH / self.dataset / f"{self.dataset}_TRAIN.txt")
        y = data[..., 0]
        labels = {l: i for i, l in enumerate(np.unique(y))}
        y = np.vectorize(labels.get)(y)[:, np.newaxis]
        return torch.Tensor(data[:, 1:, np.newaxis]), torch.Tensor(y).long()

    def get_test(self) -> tuple[torch.Tensor, torch.Tensor]:
        data = np.loadtxt(PATH / self.dataset / f"{self.dataset}_TEST.txt")
        y = data[..., 0]
        labels = {l: i for i, l in enumerate(np.unique(y))}
        y = np.vectorize(labels.get)(y)[:, np.newaxis]
        return torch.Tensor(data[:, 1:, np.newaxis]), torch.Tensor(y).long()


if __name__ == "__main__":
    print(UCRLoader("Yoga").nlabels)
