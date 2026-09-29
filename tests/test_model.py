"""The model around the LISS layer: configs, causality, the transformer
baseline, the thesis tasks, compiling, and a short training run."""
import itertools
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from elissabeth import (Elissabeth, ElissabethConfig, ElissabethDataModule,
                        ElissabethLightningModule, load_runconfig,
                        parse_overrides, save_runconfig)
from elissabeth.data import CopyingTask, CyclicTask, LookupTask

CONFIGS = Path(__file__).parents[1] / "project" / "configs"


def config_path(name: str, tmp_path: Path) -> Path:
    """A project config; the text task gets a small corpus of its own."""
    path = CONFIGS / name
    if name != "makemore.yaml":
        return path
    corpus = tmp_path / "quotes.txt"
    corpus.write_text("to be or not to be\nall that glitters\n")
    data = yaml.safe_load(path.read_text())
    data["dataset"]["task"]["path"] = str(corpus)
    out = tmp_path / name
    out.write_text(yaml.safe_dump(data))
    return out


@pytest.mark.parametrize("name", sorted(p.name for p in CONFIGS.glob("*.yaml")))
def test_project_configs(name: str, tmp_path: Path) -> None:
    config = load_runconfig(config_path(name, tmp_path))
    assert config.model.context_length == config.dataset.length
    save_runconfig(config, tmp_path / "saved.yaml")
    assert load_runconfig(tmp_path / "saved.yaml") == config
    model = Elissabeth(
        config.model, config.dataset.input_dim, config.dataset.output_dim,
    )
    x = torch.randint(0, config.dataset.input_dim, (2, 17))
    assert model(x).shape == (2, 17, config.dataset.output_dim)


def test_overrides(tmp_path: Path) -> None:
    overrides = parse_overrides([
        "model.d_hidden=7", "model.liss.kernels[1].alpha_0=3",
        "dataset.task.length=40",
    ])
    config = load_runconfig(CONFIGS / "lookup.yaml", overrides=overrides)
    assert config.model.d_hidden == 7
    assert config.model.liss.kernels[1].alpha_0 == 3
    assert config.model.liss.kernels[0].type == "cosine"
    assert config.model.context_length == 40
    with pytest.raises(ValueError):
        load_runconfig(CONFIGS / "lookup.yaml",
                       overrides=parse_overrides(["model.liss.kernels[5].x=1"]))
    with pytest.raises(ValueError):
        load_runconfig(CONFIGS / "lookup.yaml",
                       overrides=parse_overrides(["model.unknown=1"]))


MIXERS = {
    "reals": {"liss": {"d_values": 4, "n_is": 2, "lengths": [1, 3],
                       "normalize": "mean",
                       "kernels": [{"type": "decay", "alpha_0": 5},
                                   {"type": "cosine", "d_qk": 2,
                                    "projection": {"include_time": True}}]}},
    "arctic": {"liss": {"d_values": 4, "n_is": 2, "lengths": [2],
                        "semiring": "arctic", "values_2D": True,
                        "kernels": [{"type": "decay"},
                                    {"type": "exponential"}]}},
    "log": {"liss": {"d_values": 4, "n_is": 2, "lengths": [3],
                     "semiring": "log", "normalize": "learnable",
                     "kernels": [{"type": "exponential", "restrict": True}]}},
    "attention": {"attention": {"n_heads": 2}},
}


def model_for(mixer: str, ffn: bool = True) -> Elissabeth:
    config = ElissabethConfig(
        d_hidden=8, n_layers=2, context_length=12,
        ffn={"units": 16} if ffn else None, **MIXERS[mixer],
    )
    return Elissabeth(config, input_dim=6)


@pytest.mark.parametrize("mixer", sorted(MIXERS))
def test_causal(mixer: str) -> None:
    """The output at t does not depend on anything after t."""
    torch.manual_seed(0)
    model = model_for(mixer)
    x = torch.randint(0, 6, (2, 12))
    y = x.clone()
    y[:, 7:] = torch.randint(0, 6, (2, 5))
    torch.testing.assert_close(model(x)[:, :7], model(y)[:, :7])
    out = model(x)
    out.square().mean().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_longer_than_context_length() -> None:
    """Positions are computed on the fly, so any length works."""
    model = model_for("reals")
    assert model(torch.randint(0, 6, (1, 50))).shape == (1, 50, 6)


@pytest.mark.parametrize("mixer", ["reals", "attention"])
def test_compile_fullgraph_dynamic_batch(mixer: str) -> None:
    torch._dynamo.reset()
    model = model_for(mixer)
    compiled = torch.compile(model, fullgraph=True)
    for batch in (3, 5):
        x = torch.randint(0, 6, (batch, 12))
        torch._dynamo.mark_dynamic(x, 0)
        torch.testing.assert_close(compiled(x), model(x), rtol=1e-4,
                                   atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_cpu(tmp_path: Path) -> None:
    """The copying model at B=64: 10^5 rows of value norm, the size at which
    torch 2.12's CUDA LayerNorm backward broke with a 2D normalized_shape."""
    torch._dynamo.reset()
    config = load_runconfig(CONFIGS / "copying.yaml")
    model = Elissabeth(config.model, 10)
    x = torch.randint(0, 10, (64, 100))
    expected = model(x)
    cuda = Elissabeth(config.model, 10).cuda()
    cuda.load_state_dict(model.state_dict())
    compiled = torch.compile(cuda, fullgraph=True)
    out = compiled(x.cuda())
    out.sum().backward()
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("mixer", ["reals", "arctic", "log"])
def test_cuda_compile_long_sequence(mixer: str) -> None:
    """At T ~ 2000 inductor splits a scan, and its split-scan codegen
    crashed on the per-head decay fused into the cumsum; the scans are an
    opaque custom op for that reason."""
    torch._dynamo.reset()
    model = model_for(mixer).cuda()
    x = torch.randint(0, 6, (4, 2048), device="cuda")
    torch._dynamo.mark_dynamic(x, 0)
    compiled = torch.compile(model, fullgraph=True)
    out = compiled(x)
    out.square().mean().backward()
    torch.testing.assert_close(out, model(x), rtol=1e-3, atol=1e-4)


def test_copying_task() -> None:
    task = CopyingTask(length=30, n_categories=6, to_copy=4, max_dilute=2)
    x, y = task.generate(50, np.random.default_rng(0))
    assert x.shape == y.shape == (50, 30)
    assert (x[:, -5] == 5).all()
    for row, target in zip(x, y):
        data = row[:12][row[:12] < 4]
        assert (target[-4:] == data).all() and (target[:-4] == -1).all()


def test_cyclic_task() -> None:
    task = CyclicTask(length=12, characters=5)
    x, y = task.generate(200, np.random.default_rng(1))
    assert 60 < y.sum() < 140
    rotations = {tuple(np.roll(np.arange(1, 5), s)) for s in range(4)}
    for row, label in zip(x, y):
        marks = tuple(row[row > 0])
        assert sorted(marks) == [1, 2, 3, 4] and row[-1] > 0
        assert (marks in rotations) == bool(label)


@pytest.mark.parametrize("multiple_keys", [False, True])
def test_lookup_task(multiple_keys: bool) -> None:
    task = LookupTask(length=20, characters=4, multiple_keys=multiple_keys)
    x, y = task.generate(100, np.random.default_rng(2))
    for row, answer in zip(x, y):
        first = np.flatnonzero(row == row[-1])[0]
        assert 0 < first < 19 and row[first - 1] == answer
        if not multiple_keys:
            assert (row[:-1] == row[-1]).sum() == 1


def test_lookup_exact_solution() -> None:
    """The thesis' hand-built lookup solution: one depth-2 level, a decay
    that ties t_1 to t_2 - 1 and a cosine kernel matching x_{t_2} with
    x_t. After subtracting the embedding of x_{t-1} the answer wins."""
    config = load_runconfig(CONFIGS / "lookup.yaml")
    model = Elissabeth(config.model, 5)
    d = torch.pi / 2
    qk = torch.tensor([
        [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
        [0, d, 0, 0, d], [0, 0, d, 0, d], [0, 0, 0, d, 0],
    ])
    state = model.state_dict()
    state.update({
        "embedding.weight": torch.eye(5),
        "unembedding.weight": torch.eye(5),
        "mixers.0.W_H": torch.ones(1, 1),
        "mixers.0.W_O": torch.eye(5).unsqueeze(1),
        "mixers.0.levels.0.values.transform.weight":
            torch.cat((torch.eye(5), torch.ones(5, 5))),
        "mixers.0.levels.0.kernels.0.query.transform.weight": qk,
        "mixers.0.levels.0.kernels.0.key.transform.weight": qk,
        "mixers.0.levels.0.kernels.1.alpha": torch.tensor([[100.0, 0.0]]),
    })
    model.load_state_dict(state)
    x, y = config.dataset.task.generate(200, np.random.default_rng(3))
    x = torch.as_tensor(x)
    logits = model(x) - torch.nn.functional.pad(
        model.embedding(x)[:, :-1], (0, 0, 1, 0),
    )
    accuracy = (logits[:, -1].argmax(-1) == torch.as_tensor(y)).float()
    assert accuracy.mean() == 1.0


def test_training_step_runs(tmp_path: Path) -> None:
    import lightning.pytorch as L
    config = load_runconfig(CONFIGS / "copying.yaml", overrides=parse_overrides([
        "dataset.n_samples=64", "dataset.batch_size=16",
        "dataset.task.length=24", "dataset.task.to_copy=3",
        "dataset.task.max_dilute=1", "dataset.seed=0",
    ]))
    module = ElissabethLightningModule(
        config.model, config.trainer, config.dataset.input_dim,
        config.dataset.output_dim, run_dir=tmp_path,
    )
    trainer = L.Trainer(max_epochs=2, logger=False, enable_progress_bar=False,
                        enable_checkpointing=False, accelerator="cpu")
    trainer.fit(module, datamodule=ElissabethDataModule(config.dataset))
    assert (tmp_path / "epoch_2.ckpt").exists()
    assert (tmp_path / "metrics.csv").read_text().count("\n") == 3
