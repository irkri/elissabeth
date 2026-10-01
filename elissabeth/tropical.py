"""The Tropical Attention benchmark (Hashemi et al., NeurIPS 2025,
arXiv:2505.17190, github.com/Baran-phys/Tropical-Attention) as a task.

The samples come from the benchmark's own ``dataloaders.py``, imported from a
local clone (``repository``) and never copied or edited, so a seed gives the
reference's own data. What its driver ``experiment.py`` adds on top is
reproduced here and by the data module:

- one seed for the generator and for the 80/20 split (``dataset.seed``; the
  job files use 999 to train and 0 to evaluate);
- the objective of each problem (:data:`PROBLEMS`): binary cross entropy per
  token or per sequence, cross entropy over 33 predecessor classes for
  Floyd-Warshall, squared error for the regression problems;
- optionally its mask of all-zero tokens (``reference_mask``).

The paper's out-of-distribution settings (Appendix F, Table 5) are
:func:`paper_settings`.
"""
import hashlib
import importlib.util
import json
import os
import random
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, NamedTuple, get_args

import numpy as np
import torch
from pydantic import Field, model_validator

from .config import ConfigPath
from .data import T_Data, T_Objective, Task

T_Problem = Literal[
    "subset_sum", "max_subset_sum", "knapsack", "fractional_knapsack",
    "min_coin_change", "quickselect", "balanced_partition", "bin_packing",
    "convex_hull", "three_sum", "floyd_warshall", "scc", "lis",
]

T_Range = tuple[int, int] | tuple[float, float]
"""A generator range. Integers stay integers: the reference draws most
values with ``random.randint``, which rejects floats."""


class Problem(NamedTuple):

    dataset: str
    """The class in the reference ``dataloaders.py``."""
    features: int
    graph: bool
    """Tokens are the ``size^2`` node pairs of a graph, row-major."""
    classification: bool
    """What the reference driver passes as ``classification``."""
    objectives: dict[bool, T_Objective]
    """The objective for each ``classification`` the class supports."""


PROBLEMS: dict[str, Problem] = {
    "subset_sum": Problem(
        "SubsetSumDecisionDataset", 2, False, True, {True: "binary"}),
    "max_subset_sum": Problem(
        "MaxSubsetSumDataset", 1, False, True,
        {True: "binary", False: "regression"}),
    "knapsack": Problem(
        "KnapsackDataset", 3, False, True,
        {True: "binary", False: "regression"}),
    "fractional_knapsack": Problem(
        "FractionalKnapsackDataset", 3, False, False,
        {True: "regression", False: "regression"}),
    "min_coin_change": Problem(
        "MinCoinChangeDataset", 2, False, True,
        {True: "binary", False: "regression"}),
    "quickselect": Problem(
        "QuickselectDataset", 2, False, True,
        {True: "binary", False: "regression"}),
    "balanced_partition": Problem(
        "BalancedPartitionDataset", 1, False, True,
        {True: "binary", False: "regression"}),
    "bin_packing": Problem(
        "BinPackingDataset", 3, False, True,
        {True: "binary", False: "regression"}),
    "convex_hull": Problem(
        "ConvexHullDataset", 2, False, True, {True: "binary"}),
    "three_sum": Problem(
        "ThreeSumDecisionDataset", 2, False, True, {True: "binary"}),
    "floyd_warshall": Problem(
        "FloydWarshallDataset", 3, True, True,
        {True: "cross_entropy", False: "regression"}),
    "scc": Problem("SCCDataset", 3, True, True, {True: "binary"}),
    "lis": Problem("LISDataset", 2, False, False, {False: "regression"}),
}
assert set(PROBLEMS) == set(get_args(T_Problem))

# The paper's evaluation (Appendix F, Table 5). Length: 64 items, 16 nodes.
# Noise: every input perturbed with probability 0.5 by a draw from the
# problem's noise range (SCC flips edges). lis and max_subset_sum are not in
# the paper.
PAPER_LENGTH = {False: 64, True: 16}
PAPER_NOISE_PROB = 0.5
PAPER_VALUE_RANGE: dict[str, T_Range | float] = {
    "subset_sum": (-20, 20),
    "knapsack": (11, 21),
    "fractional_knapsack": (11, 21),
    "min_coin_change": (11, 21),
    "quickselect": (11, 21),
    "balanced_partition": (11, 100),
    "bin_packing": (11, 100),
    "convex_hull": (11, 21),
    "three_sum": (-375, 375),
    "floyd_warshall": (16, 30),
    "scc": 0.1,
}
PAPER_NOISE_RANGE: dict[str, T_Range] = {
    "subset_sum": (10, 30),
    "knapsack": (10, 30),
    "fractional_knapsack": (1, 5),
    "min_coin_change": (1, 5),
    "quickselect": (1, 5),
    "balanced_partition": (10, 30),
    "bin_packing": (10, 30),
    "convex_hull": (1, 5),
    "three_sum": (40, 60),
    "floyd_warshall": (1, 10),
}


def _nth_smallest(items: list, n: int):
    return sorted(items)[n]


@lru_cache
def reference(repository: Path) -> types.ModuleType:
    """The reference ``dataloaders.py``, imported from the clone.

    It imports ``quickselect.hoare`` (a PyPI package that is in neither its
    requirements nor this environment) for the k-th smallest element; when
    that is missing a sort stands in, 0-based as the call site implies. The
    stand-in is removed from ``sys.modules`` again afterwards.
    """
    path = Path(repository) / "dataloaders.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"No dataloaders.py in {repository}: clone"
            " github.com/Baran-phys/Tropical-Attention there."
        )
    stubbed: list[str] = []
    try:
        import quickselect.hoare  # noqa: F401
    except ImportError:
        package = types.ModuleType("quickselect")
        hoare = types.ModuleType("quickselect.hoare")
        hoare.nth_smallest = hoare.select = _nth_smallest  # type: ignore
        package.hoare = hoare  # type: ignore
        sys.modules.update({"quickselect": package, "quickselect.hoare": hoare})
        stubbed = ["quickselect", "quickselect.hoare"]
    try:
        spec = importlib.util.spec_from_file_location(
            "tropical_attention_dataloaders", path,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name in stubbed:
            sys.modules.pop(name, None)
    return module


def generate(
    repository: Path,
    dataset: str,
    kwargs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Samples of one reference class, stacked: ``x`` ``(n, T, F)`` and
    ``y`` ``(n, T)`` or ``(n,)``. The reference seeds Python's and numpy's
    global generators; their states are restored afterwards."""
    cls = getattr(reference(repository), dataset)
    states = random.getstate(), np.random.get_state()
    try:
        samples = cls(**kwargs).data
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
    xs, ys = zip(*samples)
    x = torch.stack(xs).float()
    y = torch.stack(ys)
    if y.ndim == 2 and y.shape[1] == 1 and x.shape[1] > 1:
        y = y[:, 0]          # MaxSubsetSum's optimum comes as a 1-vector
    return x, y


class TropicalTask(Task):
    """One problem of the benchmark. Tokens are the items of a set (most
    problems sort them by value) or the node pairs of a graph, as vectors of
    ``input_dim`` features."""

    name: Literal["tropical"] = "tropical"
    problem: T_Problem
    repository: ConfigPath
    """A clone of github.com/Baran-phys/Tropical-Attention."""
    size: int = Field(8, ge=2)
    """Items per set, or nodes per graph (``size^2`` tokens). The reference
    trains at 8."""
    value_range: T_Range | float | None = None
    target_range: T_Range | None = None
    weight_range: T_Range | None = None
    adversarial_range: T_Range | None = None
    """The generator's ranges; ``None`` keeps the class default, which the
    job files do not rely on. SCC's ``value_range`` is the probability of an
    edge within a community."""
    noise_prob: float = 0.0
    """Probability of perturbing an input by a draw from
    ``adversarial_range``."""
    classification: bool | None = None
    """Per-item labels (``True``) or the problem's optimum (``False``);
    ``None`` is the reference driver's choice. Floyd-Warshall has
    predecessor classes (the code) and distances (what the paper reports as
    MSE)."""
    classes: int = 33
    """Classes of the Floyd-Warshall predecessor labels, ``0..size-1`` and
    ``size`` for none. The reference fixes 33."""
    options: dict[str, Any] = {}
    """Further keyword arguments of the reference class: ``p_range``,
    ``k_range``, ``eps``, ``max_k``, ``normalize``."""
    reference_mask: bool = False
    """Ignore per-token binary targets whose features are all zero, as the
    reference driver does. That drops real tokens: the (0, 0) pair of every
    SCC graph and ConvexHull points at the origin."""
    cache: ConfigPath | None = None
    """Directory for generated data, keyed by everything that decides it
    (including the reference file's content)."""

    @model_validator(mode="after")
    def _check(self) -> "TropicalTask":
        objectives = self.spec.objectives
        if self.classification is not None \
                and self.classification not in objectives:
            raise ValueError(
                f"{self.problem} has no classification={self.classification}"
                " variant."
            )
        if self.objective == "cross_entropy" and self.size >= self.classes:
            raise ValueError(
                f"{self.problem} with {self.size} nodes has {self.size + 1}"
                f" predecessor classes, more than classes={self.classes}."
            )
        return self

    @property
    def spec(self) -> Problem:
        return PROBLEMS[self.problem]

    @property
    def is_classification(self) -> bool:
        if self.classification is None:
            return self.spec.classification
        return self.classification

    @property
    def input_type(self) -> Literal["token", "vector"]:
        return "vector"

    @property
    def objective(self) -> T_Objective:
        return self.spec.objectives[self.is_classification]

    @property
    def input_dim(self) -> int:
        return self.spec.features

    @property
    def output_dim(self) -> int:
        return self.classes if self.objective == "cross_entropy" else 1

    @property
    def length(self) -> int:
        return self.size ** 2 if self.spec.graph else self.size

    def reference_kwargs(self, n: int, seed: int) -> dict[str, Any]:
        """The reference class's keyword arguments for ``n`` samples."""
        kwargs = dict(self.options)
        kwargs.update(
            n_samples=n,
            length_range=(self.size, self.size),
            noise_prob=self.noise_prob,
            classification=self.is_classification,
            seed=seed,
        )
        for name in ("value_range", "target_range", "weight_range",
                     "adversarial_range"):
            value = getattr(self, name)
            if value is not None:
                kwargs[name] = value
        return kwargs

    def _cache_path(self, kwargs: dict[str, Any]) -> Path | None:
        if self.cache is None:
            return None
        source = (Path(self.repository) / "dataloaders.py").read_bytes()
        key = json.dumps({
            "dataset": self.spec.dataset,
            "kwargs": kwargs,
            "source": hashlib.sha256(source).hexdigest(),
        }, sort_keys=True)
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        return Path(self.cache) / f"{self.problem}_{self.size}_{digest}.pt"

    def data(self, n: int, seed: int | None) -> T_Data:
        if seed is None:
            raise ValueError("The tropical task needs a seed.")
        kwargs = self.reference_kwargs(n, seed)
        path = self._cache_path(kwargs)
        if path is not None and path.exists():
            saved = torch.load(path)
            x, y = saved["x"], saved["y"]
        else:
            x, y = generate(self.repository, self.spec.dataset, kwargs)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_suffix(f".{os.getpid()}.part")
                torch.save({"x": x, "y": y}, partial)
                os.replace(partial, path)
        if self.objective == "cross_entropy":
            y = y.long()
        else:
            y = y.float()
            if self.reference_mask and self.objective == "binary" \
                    and y.ndim == 2:
                y = y.masked_fill(x.abs().sum(-1) == 0, torch.nan)
        return x.numpy(), y.numpy()

    def generate(self, n: int, rng: np.random.Generator) -> T_Data:
        return self.data(n, int(rng.integers(2**31)))


def paper_settings(task: TropicalTask) -> dict[str, TropicalTask]:
    """The task and the paper's three out-of-distribution versions of it:
    ``length`` (64 items, 16 nodes), ``value`` (the problem's OOD range) and
    ``noise`` (every input perturbed with probability 0.5 from the
    problem's noise range). ``lis`` and ``max_subset_sum`` are not in the
    paper: they get no ``value`` setting, and their noise uses the task's
    own ``adversarial_range``."""
    settings = {
        "in_distribution": task,
        "length": task.model_copy(
            update={"size": PAPER_LENGTH[task.spec.graph]},
        ),
    }
    if task.problem in PAPER_VALUE_RANGE:
        settings["value"] = task.model_copy(
            update={"value_range": PAPER_VALUE_RANGE[task.problem]},
        )
    noise: dict[str, Any] = {"noise_prob": PAPER_NOISE_PROB}
    if task.problem in PAPER_NOISE_RANGE:
        noise["adversarial_range"] = PAPER_NOISE_RANGE[task.problem]
    settings["noise"] = task.model_copy(update=noise)
    return settings
