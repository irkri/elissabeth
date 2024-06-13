import os
from typing import Optional
from pathlib import Path

import torch
import numpy as np


class LetterAssembler:

    def __init__(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            lines = list(filter(None, f.read().split("\n")))
        lines = [x.strip() for x in lines]

        self.context_length = max(map(len, lines)) + 1
        self.vocabulary = sorted(list(set("".join(lines))))

        self.stoi = {a: i+1 for i, a in enumerate(self.vocabulary)}
        self.itos = {i: a for a, i in self.stoi.items()}

        self._data = [list(map(self.stoi.get, x)) for x in lines]

    @property
    def vocab_size(self) -> int:
        return len(self.vocabulary) + 1

    def sample(
        self,
        index: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if index is None:
            index = np.random.randint(0, len(self._data))
        x = torch.Tensor(
            [0] + self._data[index]
                + ([0] * (self.context_length - len(self._data[index]) - 1))
        ).long().unsqueeze(0)
        y = torch.Tensor(
            self._data[index] + [0]
                + ([-1] * (self.context_length - len(self._data[index]) - 1))
        ).long().unsqueeze(0)
        return x, y

    def get_dataset(self) -> tuple[torch.Tensor, torch.Tensor]:
        x_set = torch.zeros(len(self._data), self.context_length).long()
        y_set = torch.zeros(len(self._data), self.context_length).long()
        for i in range(len(self._data)):
            x_set[i:i+1, :], y_set[i:i+1, :] = self.sample(i)
        return x_set, y_set

    def translate(self, tensor: torch.Tensor) -> str:
        index = (tensor == 0).nonzero()
        index = index[1] if len(index) > 1 else tensor.size(0)
        return "".join([self.itos[int(i)] for i in tensor[1:index]])

    def to_tensor(self, word: str, fill: bool = True) -> torch.Tensor:
        numerical = [self.stoi[c] for c in word]
        if fill:
            numerical += [self.stoi[" "]] * (
                self.context_length - len(numerical)
            )
        return torch.Tensor(numerical).long()


if __name__ == "__main__":
    print(next(iter(LetterAssembler(
        Path(__file__).parent / "quotes.txt"
    ).get_dataset())))
