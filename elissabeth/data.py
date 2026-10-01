"""The synthetic tasks from the thesis, a character-level text task and the
Tropical Attention benchmark (:mod:`.tropical`), generated in memory and
served by one Lightning data module.

Every task returns inputs ``x`` of shape ``(n, T)`` (int64 tokens) or
``(n, T, F)`` (float vectors) and targets ``y`` of shape ``(n, T)`` (one per
position) or ``(n,)`` (one per sequence, read from the output by the model's
``pooling``). What a target is depends on the task's objective: a class
index (``-1`` = ignored) for ``cross_entropy``, a float 0/1 for ``binary``
or a real number for ``regression`` (``NaN`` = ignored in both).
"""
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import lightning.pytorch as L
import numpy as np
import torch
from pydantic import Field, model_validator
from torch.utils.data import DataLoader, TensorDataset, random_split

from .config import ConfigPath, ModelConfig

T_Data = tuple[np.ndarray, np.ndarray]

T_Objective = Literal["cross_entropy", "binary", "regression"]
"""Cross entropy over ``output_dim`` classes, a binary cross entropy on one
logit, or a squared error on one output."""


class Task(ModelConfig):
    """What the run needs to know about a task besides its fields."""

    @property
    def input_type(self) -> Literal["token", "vector"]:
        return "token"

    @property
    def objective(self) -> T_Objective:
        return "cross_entropy"

    @property
    def causal(self) -> bool:
        """The targets are later inputs (next-token prediction), so a model
        that sees the future sees the answer."""
        return False

    def generate(self, n: int, rng: np.random.Generator) -> T_Data:
        raise NotImplementedError

    def data(self, n: int, seed: int | None) -> T_Data:
        """``n`` samples, the same ones for the same ``seed``."""
        return self.generate(n, np.random.default_rng(seed))


class CopyingTask(Task):
    """Copy the ``to_copy`` data tokens, scattered over the first
    ``to_copy * (1 + max_dilute)`` positions, after a marker token. Tokens
    ``0..n_categories-3`` are data, ``n_categories-2`` is blank and
    ``n_categories-1`` the marker."""

    name: Literal["copying"] = "copying"
    length: int = 100
    n_categories: int = 10
    to_copy: int = 10
    max_dilute: int = 0

    @model_validator(mode="after")
    def _check(self) -> "CopyingTask":
        if self.n_categories < 3:
            raise ValueError("copying needs at least 3 categories.")
        if self.to_copy * (1 + self.max_dilute) > self.length - self.to_copy - 1:
            raise ValueError(
                "copying: the data span to_copy*(1+max_dilute) has to end"
                " before the marker at length - to_copy - 1."
            )
        return self

    @property
    def input_dim(self) -> int:
        return self.n_categories

    @property
    def output_dim(self) -> int:
        return self.n_categories

    def generate(self, n: int, rng: np.random.Generator) -> T_Data:
        span = self.to_copy * (1 + self.max_dilute)
        x = np.full((n, self.length), self.n_categories - 2, dtype=np.int64)
        y = np.full((n, self.length), -1, dtype=np.int64)
        positions = np.sort(
            np.argsort(rng.random((n, span)), axis=1)[:, :self.to_copy],
            axis=1,
        )
        tokens = rng.integers(0, self.n_categories - 2, (n, self.to_copy))
        x[np.arange(n)[:, None], positions] = tokens
        x[:, -self.to_copy - 1] = self.n_categories - 1
        y[:, -self.to_copy:] = tokens
        return x, y


def _is_rotation(marks: np.ndarray) -> np.ndarray:
    """Rows of ``marks`` (a permutation of ``1..m``) that are cyclic
    rotations of ``1, 2, ..., m``."""
    m = marks.shape[1]
    return ((np.roll(marks, -1, axis=1) - marks) % m == 1).all(axis=1)


class CyclicTask(Task):
    """Does the sequence contain ``1, 2, ..., m`` in some cyclic rotation
    (e.g. ``3, 4, ..., m, 1, 2``) as a subsequence? The ``m = characters-1``
    marks are spread over zeros, the last one at the final position; the
    target is binary, one per sequence."""

    name: Literal["cyclic"] = "cyclic"
    length: int = 10
    characters: int = 10

    @model_validator(mode="after")
    def _check(self) -> "CyclicTask":
        if self.characters < 4:
            raise ValueError("cyclic needs characters >= 4 (else every order"
                             " is a rotation).")
        if self.length < self.characters - 1:
            raise ValueError("cyclic needs length >= characters - 1.")
        return self

    @property
    def input_dim(self) -> int:
        return self.characters

    @property
    def output_dim(self) -> int:
        return 2

    def generate(self, n: int, rng: np.random.Generator) -> T_Data:
        m = self.characters - 1
        y = rng.integers(0, 2, n)
        shift = rng.integers(0, m, (n, 1))
        marks = (np.arange(m)[None, :] + shift) % m + 1
        negative = np.flatnonzero(y == 0)
        while negative.size:
            marks[negative] = np.argsort(
                rng.random((negative.size, m)), axis=1,
            ) + 1
            negative = negative[_is_rotation(marks[negative])]
        x = np.zeros((n, self.length), dtype=np.int64)
        positions = np.sort(
            np.argsort(rng.random((n, self.length - 1)), axis=1)[:, :m - 1],
            axis=1,
        )
        x[np.arange(n)[:, None], positions] = marks[:, :-1]
        x[:, -1] = marks[:, -1]
        return x, y.astype(np.int64)


class LookupTask(Task):
    """Find the first occurrence of the last token and answer the token
    right before it (an induction-head task). Without ``multiple_keys`` the
    key occurs exactly once before the end. With ``only_last`` the target
    is the answer alone, otherwise next-token targets at every position
    with the answer at the last one."""

    name: Literal["lookup"] = "lookup"
    length: int = 25
    characters: int = 5
    multiple_keys: bool = True
    only_last: bool = True

    @property
    def input_dim(self) -> int:
        return self.characters

    @property
    def output_dim(self) -> int:
        return self.characters

    @property
    def causal(self) -> bool:
        return not self.only_last

    def generate(self, n: int, rng: np.random.Generator) -> T_Data:
        c, T = self.characters, self.length
        x = rng.integers(0, c, (n, T))
        answers = np.empty(n, dtype=np.int64)
        for i in range(n):
            mark = rng.integers(c)
            if self.multiple_keys:
                indices = np.flatnonzero(x[i] == mark)
                if indices.size and indices[0] == 0:
                    x[i, 0] = (mark + 1) % c
                    indices = indices[1:]
                if indices.size == 0 or (
                    indices.size == 1 and indices[0] == T - 1
                ):
                    indices = np.array([rng.integers(1, T - 1)])
                index = indices[0]
            else:
                index = rng.integers(1, T - 1)
                mask = x[i] == mark
                other = rng.integers(0, c - 1, mask.sum())
                x[i, mask] = other + (other >= mark)
            x[i, index] = mark
            x[i, -1] = mark
            answers[i] = x[i, index - 1]
        if self.only_last:
            return x, answers
        y = np.empty_like(x)
        y[:, :-1] = x[:, 1:]
        y[:, -1] = answers
        return x, y


@lru_cache
def _read_corpus(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    lines = [
        line.strip() for line in path.read_text(encoding="utf-8").split("\n")
    ]
    lines = [line for line in lines if line]
    return tuple(lines), tuple(sorted(set("".join(lines))))


class TextTask(Task):
    """Character-level language modelling on the lines of a text file
    (the makemore replicate). Token 0 starts and ends a line; positions
    after the end are ignored."""

    name: Literal["text"] = "text"
    path: ConfigPath

    @property
    def vocabulary(self) -> tuple[str, ...]:
        return _read_corpus(self.path)[1]

    @property
    def length(self) -> int:
        return max(map(len, _read_corpus(self.path)[0])) + 1

    @property
    def input_dim(self) -> int:
        return len(self.vocabulary) + 1

    @property
    def output_dim(self) -> int:
        return self.input_dim

    @property
    def causal(self) -> bool:
        return True

    def encode(self, text: str) -> list[int]:
        index = {a: i + 1 for i, a in enumerate(self.vocabulary)}
        return [index[a] for a in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.vocabulary[t - 1] for t in tokens if t > 0)

    def generate(self, n: int, rng: np.random.Generator) -> T_Data:
        lines = _read_corpus(self.path)[0]
        x = np.zeros((len(lines), self.length), dtype=np.int64)
        y = np.full((len(lines), self.length), -1, dtype=np.int64)
        for i, line in enumerate(lines):
            tokens = self.encode(line)
            x[i, 1:len(tokens) + 1] = tokens
            y[i, :len(tokens)] = tokens
            y[i, len(tokens)] = 0
        return x, y


# The Tropical Attention bridge subclasses Task, so it is imported once Task
# exists (the same trick as in config.py).
from .tropical import TropicalTask  # noqa: E402

T_Task = Annotated[
    CopyingTask | CyclicTask | LookupTask | TextTask | TropicalTask,
    Field(discriminator="name"),
]


class DatasetConfig(ModelConfig):

    task: T_Task
    n_samples: int = 1000
    """Sequences generated (the text task uses every line instead)."""
    val_size: float = 0.2
    batch_size: int = 64
    num_workers: int = 0
    """Loader workers; the data is in memory, so 0 is usually fastest."""
    seed: int | None = None
    """Seeds generation and the train/validation split."""

    @model_validator(mode="after")
    def _check(self) -> "DatasetConfig":
        if isinstance(self.task, TropicalTask) and self.seed is None:
            raise ValueError(
                "The tropical task needs dataset.seed: it seeds the"
                " reference generator and the split (the reference uses"
                " 999 for training, 0 for evaluation)."
            )
        return self

    @property
    def input_type(self) -> Literal["token", "vector"]:
        return self.task.input_type

    @property
    def objective(self) -> T_Objective:
        return self.task.objective

    @property
    def input_dim(self) -> int:
        return self.task.input_dim

    @property
    def output_dim(self) -> int:
        return self.task.output_dim

    @property
    def length(self) -> int:
        return self.task.length


def make_datasets(
    config: DatasetConfig,
) -> tuple[TensorDataset | torch.utils.data.Subset, ...]:
    """The task's ``(train, validation)`` split. With a seed and a whole
    number of validation samples this is the Tropical Attention driver's
    split (``torch.manual_seed(seed)``, then ``random_split``)."""
    x, y = config.task.data(config.n_samples, config.seed)
    data = TensorDataset(torch.as_tensor(x), torch.as_tensor(y))
    n_val = int(round(config.val_size * len(data)))
    generator = (
        None if config.seed is None
        else torch.Generator().manual_seed(config.seed)
    )
    return tuple(random_split(
        data, [len(data) - n_val, n_val], generator=generator,
    ))


class ElissabethDataModule(L.LightningDataModule):

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__()
        self.config = config
        self._split: tuple | None = None

    def setup(self, stage: str | None = None) -> None:
        if self._split is None:
            self._split = make_datasets(self.config)

    def _loader(self, index: int, shuffle: bool) -> DataLoader:
        self.setup()
        assert self._split is not None
        workers = self.config.num_workers
        return DataLoader(
            self._split[index],
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=workers,
            persistent_workers=workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(0, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(1, shuffle=False)
