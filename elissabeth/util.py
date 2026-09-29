from pathlib import Path

import torch

from .config import RunConfig, load_runconfig
from .lightning import (CONFIG_NAME, ElissabethLightningModule, find_run_dir,
                        latest_checkpoint)
from .elissabeth import Elissabeth


def load_model(
    run: str | Path,
    model_dir: Path | str | None = None,
    weight_name: str | None = None,
    overrides: dict | None = None,
) -> tuple[Elissabeth, RunConfig, Path]:
    """Rebuild a trained model from a run directory holding a
    ``config.yaml`` and a checkpoint (the latest one unless ``weight_name``
    is given). ``run`` is a path or a directory name below ``model_dir``
    (``./checkpoints`` by default). Returns the model in eval mode on the
    CPU, its config and its run directory.
    """
    root = Path("checkpoints") if model_dir is None else Path(model_dir)
    run_dir = find_run_dir(root, str(run))
    config = load_runconfig(run_dir / CONFIG_NAME, overrides=overrides)
    if weight_name is None:
        latest = latest_checkpoint(run_dir)
        if latest is None:
            raise FileNotFoundError(f"No checkpoint in {run_dir}.")
        weights = latest[1]
    else:
        weights = run_dir / weight_name
    module = ElissabethLightningModule(
        config.model, config.trainer,
        config.dataset.input_dim, config.dataset.output_dim,
    )
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    module.load_state_dict(checkpoint["state_dict"])
    return module.model.eval(), config, run_dir
