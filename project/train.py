import argparse
import os
import secrets
import string
from pathlib import Path

import lightning.pytorch as L
import torch

from elissabeth.config import (deep_update, load_runconfig, parse_overrides,
                               read_mapping, save_runconfig)
from elissabeth.data import ElissabethDataModule
from elissabeth.lightning import (CONFIG_NAME, ElissabethLightningModule,
                                  find_run_dir, latest_checkpoint,
                                  new_run_dir)

CHECKPOINTS_ENV = "ELISSABETH_CHECKPOINTS"
# Where run directories are written; $ELISSABETH_CHECKPOINTS overrides the
# default 'project/checkpoints'.
CHECKPOINTS_PATH = (
    Path(os.path.expandvars(os.environ[CHECKPOINTS_ENV])).expanduser()
    if os.environ.get(CHECKPOINTS_ENV)
    else Path(__file__).parent / "checkpoints"
)


def wandb_id() -> str:
    """A wandb run id: 8 random base-36 digits, as wandb makes them."""
    return "".join(
        secrets.choice(string.ascii_lowercase + string.digits)
        for _ in range(8)
    )


def run(args: argparse.Namespace) -> None:
    overrides: dict = {}
    if args.override:
        overrides = deep_update(overrides, parse_overrides(args.override))
    for path in args.overrideconfig:
        overrides = deep_update(overrides, read_mapping(path))
    if overrides:
        print(f"Applied overrides to config: {overrides}")

    online = args.online is not None
    ckpt_path: Path | None = None
    if args.resume is not None:
        run_dir = find_run_dir(CHECKPOINTS_PATH, args.resume)
        config = load_runconfig(
            args.config or run_dir / CONFIG_NAME, overrides=overrides,
        )
        latest = latest_checkpoint(run_dir)
        if latest is not None:
            ckpt_path = latest[1]
            print(f"Resuming {run_dir.name} after epoch {latest[0]}.")
        run_id = run_dir.name
    else:
        config = load_runconfig(args.config, overrides=overrides)
        run_id = wandb_id() if online else None
        run_dir = new_run_dir(
            CHECKPOINTS_PATH, run_id if online else args.name,
        )
    save_runconfig(config, run_dir / CONFIG_NAME)
    print(f"Run directory: {run_dir}")

    module = ElissabethLightningModule(
        config.model, config.trainer,
        config.dataset.input_dim, config.dataset.output_dim,
        run_dir=run_dir,
    )
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.recompile_limit = 25
    if not args.eager:
        module.model.compile(fullgraph=True)

    logger: object = False
    if online:
        from lightning.pytorch.loggers import WandbLogger
        entity, _, project = args.online.rpartition("/")
        logger = WandbLogger(
            entity=entity or None,
            project=project,
            id=run_id,
            name=args.name,
            resume="allow" if args.resume is not None else None,
            save_dir=str(run_dir),
            config=config.model_dump(mode="json"),
        )

    validate = config.dataset.val_size > 0
    trainer = L.Trainer(
        max_epochs=config.trainer.epochs,
        limit_train_batches=config.trainer.train_steps,
        limit_val_batches=None if validate else 0,
        num_sanity_val_steps=2 if validate else 0,
        check_val_every_n_epoch=config.trainer.val_every,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        gradient_clip_val=config.trainer.gradient_clip_norm,
        precision=config.trainer.precision,
        devices=args.devices,
        logger=logger,
        enable_progress_bar=config.trainer.progress_bar and not online,
        enable_checkpointing=False,
    )
    trainer.fit(
        module,
        datamodule=ElissabethDataModule(config.dataset),
        ckpt_path=None if ckpt_path is None else str(ckpt_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        "Elissabeth - training",
        "Supply a YAML or JSON run config."
        f" Runs are written to '{CHECKPOINTS_PATH}' (${CHECKPOINTS_ENV}).",
    )
    parser.add_argument(
        "config", nargs="?",
        help="path to a run config; optional with --resume",
    )
    parser.add_argument(
        "-o", "--override", action="append", default=[],
        help="override a config value, e.g. -o model.d_hidden=32"
             " -o model.liss.kernels[0].alpha_0=10",
    )
    parser.add_argument(
        "-oc", "--overrideconfig", action="append", default=[],
        help="YAML or JSON file of overrides (repeatable; later files win)",
    )
    parser.add_argument(
        "--online", type=str, default=None,
        help="log to Weights & Biases, as '[team/]project'",
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="run directory for offline runs (default: run_XXXX); the"
             " wandb display name for online runs",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="name of (or path to) a run directory to continue",
    )
    parser.add_argument("--eager", action="store_true",
                        help="skip torch.compile")
    parser.add_argument("--devices", default="auto",
                        help="Lightning 'devices' argument")
    args = parser.parse_args()
    if args.config is None and args.resume is None:
        parser.error("a config is required unless the run is resumed")
    run(args)


if __name__ == "__main__":
    main()
