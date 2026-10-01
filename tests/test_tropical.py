"""The Tropical Attention bridge: the reference generator is used unchanged
and reproduced exactly (samples, split, masks), the objectives train, and a
schedule-free run checkpoints the weights it evaluates. Skipped without a
clone of the benchmark at ``$TROPICAL_ATTENTION`` or
``~/code_torch/Tropical-Attention``."""
import csv
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from elissabeth import ElissabethDataModule, ElissabethLightningModule
from elissabeth.config import deep_update, load_runconfig, read_mapping
from elissabeth.data import DatasetConfig, make_datasets
from elissabeth.lightning import (BEST_NAME, binary_counts, evaluated_weights,
                                  f1_score)
from elissabeth.tropical import (PAPER_VALUE_RANGE, PROBLEMS, TropicalTask,
                                 generate, paper_settings, reference)

REPOSITORY = Path(os.environ.get(
    "TROPICAL_ATTENTION", "~/code_torch/Tropical-Attention",
)).expanduser()
CONFIGS = Path(__file__).parents[1] / "project" / "configs" / "tropical"

pytestmark = pytest.mark.skipif(
    not (REPOSITORY / "dataloaders.py").is_file(),
    reason="needs a clone of Baran-phys/Tropical-Attention",
)


def task(problem: str, **fields) -> TropicalTask:
    return TropicalTask(problem=problem, repository=REPOSITORY, **fields)


@pytest.mark.parametrize("problem", sorted(PROBLEMS))
def test_problems(problem: str) -> None:
    t = task(problem, size=4)
    x, y = t.data(20, seed=1)
    tokens = 16 if PROBLEMS[problem].graph else 4
    assert x.shape == (20, tokens, t.input_dim) and x.dtype == np.float32
    assert y.shape in ((20,), (20, tokens))
    if t.objective == "cross_entropy":
        assert y.dtype == np.int64 and 0 <= y.min() and y.max() <= 4
    else:
        assert y.dtype == np.float32
    if t.objective == "binary":
        assert set(np.unique(y)) <= {0.0, 1.0}


def test_samples_are_the_references() -> None:
    """The same class, arguments and seed as the reference driver."""
    t = task("knapsack", value_range=(1, 10), weight_range=(1, 10),
             target_range=(10, 20))
    x, y = t.data(50, seed=999)
    cls = reference(REPOSITORY).KnapsackDataset
    samples = cls(n_samples=50, length_range=(8, 8), value_range=(1, 10),
                  weight_range=(1, 10), target_range=(10, 20),
                  noise_prob=0.0, classification=True, seed=999).data
    np.testing.assert_array_equal(x, torch.stack([s[0] for s in samples]))
    np.testing.assert_array_equal(y, torch.stack([s[1] for s in samples]))


def test_global_generators_are_restored() -> None:
    """The reference seeds Python's and numpy's global generators."""
    random.seed(5)
    np.random.seed(5)
    expected = random.random(), np.random.rand()
    random.seed(5)
    np.random.seed(5)
    task("scc").data(5, seed=0)
    assert (random.random(), np.random.rand()) == expected


def test_split_is_the_drivers() -> None:
    """``torch.manual_seed(seed)`` then ``random_split`` into 80/20."""
    config = DatasetConfig(task=task("subset_sum"), n_samples=500, seed=999)
    train, test = make_datasets(config)
    torch.manual_seed(999)
    expected = torch.utils.data.random_split(range(500), [400, 100])
    assert train.indices == expected[0].indices
    assert test.indices == expected[1].indices


def test_seed_is_required() -> None:
    with pytest.raises(ValueError, match="seed"):
        DatasetConfig(task=task("subset_sum"))


def test_cache(tmp_path: Path, monkeypatch) -> None:
    t = task("three_sum", cache=tmp_path)
    x, y = t.data(30, seed=3)
    assert len(list(tmp_path.glob("three_sum_8_*.pt"))) == 1
    monkeypatch.setattr("elissabeth.tropical.generate", None)   # not called
    cached = t.data(30, seed=3)
    np.testing.assert_array_equal(cached[0], x)
    np.testing.assert_array_equal(cached[1], y)
    monkeypatch.undo()
    t.data(30, seed=4)
    assert len(list(tmp_path.glob("three_sum_8_*.pt"))) == 2


def test_reference_mask_drops_all_zero_tokens() -> None:
    """The (0, 0) pair of an SCC graph has features [a_00 = 0, 0, 0]."""
    _, y = task("scc", reference_mask=True).data(10, seed=0)
    assert np.isnan(y[:, 0]).all() and not np.isnan(y[:, 1:]).any()
    _, y = task("scc").data(10, seed=0)
    assert (y[:, 0] == 1).all()


def test_floyd_warshall_variants() -> None:
    assert task("floyd_warshall").objective == "cross_entropy"
    assert task("floyd_warshall").output_dim == 33
    t = task("floyd_warshall", classification=False)
    assert t.objective == "regression" and t.output_dim == 1
    with pytest.raises(ValueError):
        task("scc", classification=False)
    with pytest.raises(ValueError):
        task("floyd_warshall", size=33)


def test_paper_settings() -> None:
    settings = paper_settings(task("knapsack", value_range=(1, 10)))
    assert settings["length"].size == 64
    assert settings["value"].value_range == PAPER_VALUE_RANGE["knapsack"]
    assert settings["noise"].noise_prob == 0.5
    assert settings["noise"].adversarial_range == (10, 30)
    assert paper_settings(task("scc"))["length"].size == 16
    assert "value" not in paper_settings(task("lis"))


@pytest.mark.parametrize("problem", sorted(p.stem for p in CONFIGS.glob("*.yaml")
                                           if p.stem != "base"))
def test_project_configs(problem: str) -> None:
    config = load_runconfig(CONFIGS / "base.yaml",
                            overrides=read_mapping(CONFIGS / f"{problem}.yaml"))
    assert config.dataset.task.problem == problem
    assert config.model.input_type == "vector"
    assert config.model.context_length == config.dataset.length


def test_f1_matches_scikit_learn() -> None:
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    logits = torch.as_tensor(rng.normal(size=500))
    target = torch.as_tensor(rng.integers(0, 2, 500)).double()
    target[rng.random(500) < 0.1] = torch.nan
    valid = ~torch.isnan(target)
    expected = sklearn.f1_score(target[valid].numpy(),
                                (logits[valid] > 0).numpy(), average="binary")
    assert f1_score(binary_counts(logits, target).double()).item() \
        == pytest.approx(expected)


@pytest.mark.parametrize("problem,overrides", [
    ("knapsack", {}),                                        # binary per item
    ("subset_sum", {}),                                      # binary per set
    ("lis", {}),                                             # regression
    ("floyd_warshall", {"model": {"liss": {"semiring": "log"}}}),  # classes
])
def test_training(problem: str, overrides: dict, tmp_path: Path) -> None:
    import lightning.pytorch as L
    small = {"model": {"d_hidden": 16},
             "dataset": {"n_samples": 200, "batch_size": 50,
                         "task": {"cache": None}}}
    config = load_runconfig(CONFIGS / "base.yaml", overrides=deep_update(
        deep_update(read_mapping(CONFIGS / f"{problem}.yaml"), small),
        overrides,
    ))
    module = ElissabethLightningModule(
        config.model, config.trainer, config.dataset.input_dim,
        config.dataset.output_dim, run_dir=tmp_path,
        objective=config.dataset.objective,
    )
    trainer = L.Trainer(max_epochs=2, logger=False, enable_progress_bar=False,
                        enable_checkpointing=False, accelerator="cpu")
    trainer.fit(module, datamodule=ElissabethDataModule(config.dataset))
    assert (tmp_path / BEST_NAME).exists()
    rows = list(csv.DictReader(open(tmp_path / "metrics.csv")))
    assert len(rows) == 2 and np.isfinite(float(rows[-1]["validation/loss"]))
    if config.dataset.objective == "binary":
        assert 0 <= float(rows[-1]["validation/f1"]) <= 1

    # schedule-free: the checkpoint holds the evaluated weights
    optimizer = trainer.optimizers[0]
    optimizer.train()
    stepped = {k: v.clone() for k, v in module.state_dict().items()}
    with evaluated_weights(trainer):
        evaluated = {k: v.clone() for k, v in module.state_dict().items()}
    saved = torch.load(tmp_path / "epoch_2.ckpt", weights_only=False)
    for name, value in saved["state_dict"].items():
        torch.testing.assert_close(value, evaluated[name])
    assert any(not torch.equal(stepped[k], evaluated[k]) for k in stepped)
