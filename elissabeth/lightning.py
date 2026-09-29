"""Training: the trainer config, the Lightning module, and the few
callbacks a run needs (checkpoints per epoch, learning rate and parameter
counts)."""
import csv
from pathlib import Path
from typing import Sequence

import lightning.pytorch as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from torch.optim.lr_scheduler import LRScheduler

from .config import ModelConfig
from .elissabeth import Elissabeth, ElissabethConfig

CONFIG_NAME = "config.yaml"


class TrainerConfig(ModelConfig):

    epochs: int = 100
    train_steps: int | None = None
    """Batches per epoch (default: the whole training set)."""

    lr: float = 1e-3
    """The final learning rate, reached after the optional warmup and
    cooldown and held for the rest of the training."""
    start_lr: float | None = None
    warmup_peak_lr: float | None = None
    warmup_epochs: int | None = None
    cooldown_epochs: int | None = None
    """Optional schedule, see :class:`WarmUpCooldownLR`."""
    weight_decay: float = 1e-4
    no_weight_decay: list[str] = []
    """Parameters whose names contain one of these substrings are not
    decayed (e.g. ``alpha`` for the decay rates, ``beta``, ``W_H``)."""
    beta_1: float = 0.9
    beta_2: float = 0.999
    gradient_clip_norm: float | None = 1.0
    accumulate_grad_batches: int = 1
    precision: str = "32-true"

    progress_bar: bool = True
    val_every: int = 1
    """Validate every this many epochs."""
    save_frequent: int = 1
    """Write ``epoch_<n>.ckpt`` every this many epochs, replacing the
    previous one; 0 keeps only ``save_epochs``."""
    save_epochs: list[int] = []
    """Epochs whose checkpoint is kept for good."""


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class WarmUpCooldownLR(LRScheduler):
    """Piecewise linear over epochs: ``start_lr -> warmup_peak_lr`` over
    ``warmup_epochs``, then ``-> lr`` over ``cooldown_epochs``, then ``lr``.
    Without a cooldown the warmup aims at ``lr`` directly.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr: float,
        start_lr: float | None = None,
        warmup_peak_lr: float | None = None,
        warmup_epochs: int | None = None,
        cooldown_epochs: int | None = None,
        last_epoch: int = -1,
    ) -> None:
        self.lr = lr
        self.warmup_epochs = warmup_epochs or 0
        self.cooldown_epochs = cooldown_epochs or 0
        self.start_lr = lr if start_lr is None else start_lr
        self.peak_lr = (
            warmup_peak_lr
            if warmup_peak_lr is not None and self.cooldown_epochs > 0
            else lr
        )
        super().__init__(optimizer, last_epoch)

    def lr_at(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return _lerp(self.start_lr, self.peak_lr, epoch / self.warmup_epochs)
        if epoch < self.warmup_epochs + self.cooldown_epochs:
            return _lerp(
                self.peak_lr, self.lr,
                (epoch - self.warmup_epochs) / self.cooldown_epochs,
            )
        return self.lr

    def get_lr(self) -> list[float]:
        return [self.lr_at(self.last_epoch)] * len(self.optimizer.param_groups)


def masked_accuracy(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -1,
) -> torch.Tensor:
    valid = target != ignore_index
    correct = (logits.argmax(-1) == target) & valid
    return correct.sum() / valid.sum().clamp_min(1)


class ElissabethLightningModule(L.LightningModule):
    """Cross entropy on every target that is not ``-1``. Targets of shape
    ``(B,)`` are compared with the prediction at the last position."""

    def __init__(
        self,
        model_config: ElissabethConfig,
        trainer_config: TrainerConfig,
        input_dim: int,
        output_dim: int | None = None,
        run_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self.model = Elissabeth(model_config, input_dim, output_dim)
        self.trainer_config = trainer_config
        self.run_dir = run_dir

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_callbacks(self) -> list[Callback]:
        callbacks: list[Callback] = [RunStats(self.run_dir)]
        if self.run_dir is not None:
            callbacks.append(EpochCheckpoint(
                self.run_dir,
                self.trainer_config.save_epochs,
                self.trainer_config.save_frequent,
            ))
        return callbacks

    def _step(self, batch: Sequence[torch.Tensor], stage: str) -> torch.Tensor:
        x, y = batch
        # One compiled graph for every batch size (the last, partial batch).
        if x.shape[0] > 1:
            torch._dynamo.mark_dynamic(x, 0)
        logits = self.model(x)
        if y.ndim == 1:
            logits = logits[:, -1]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
            ignore_index=-1,
        )
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True,
                 on_step=False)
        self.log(f"{stage}/accuracy", masked_accuracy(logits, y),
                 prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "validation")

    def configure_optimizers(self) -> torch.optim.Optimizer | dict:
        cfg = self.trainer_config
        decay, exempt = [], []
        for name, p in self.named_parameters():
            if any(key in name for key in cfg.no_weight_decay):
                exempt.append(p)
            else:
                decay.append(p)
        groups = [{"params": decay, "weight_decay": cfg.weight_decay}]
        if exempt:
            groups.append({"params": exempt, "weight_decay": 0.0})
        optimizer = torch.optim.AdamW(
            groups, lr=cfg.lr, betas=(cfg.beta_1, cfg.beta_2),
        )
        if cfg.warmup_epochs is None and cfg.cooldown_epochs is None:
            return optimizer
        scheduler = WarmUpCooldownLR(
            optimizer,
            lr=cfg.lr,
            start_lr=cfg.start_lr,
            warmup_peak_lr=cfg.warmup_peak_lr,
            warmup_epochs=cfg.warmup_epochs,
            cooldown_epochs=cfg.cooldown_epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


def checkpoint_name(epoch: int) -> str:
    return f"epoch_{epoch}.ckpt"


def new_run_dir(root: Path, name: str | None = None) -> Path:
    """A fresh run directory below ``root``, ``run_XXXX`` by default."""
    root.mkdir(parents=True, exist_ok=True)
    if name is None:
        used = [
            int(p.name[4:]) for p in root.glob("run_*")
            if p.is_dir() and p.name[4:].isdigit()
        ]
        name = f"run_{max(used, default=0) + 1:04d}"
    path = root / name
    if path.exists():
        raise FileExistsError(f"Run directory {path} already exists.")
    path.mkdir(parents=True)
    return path


def find_run_dir(root: Path, run: str) -> Path:
    """``run`` as a path, or as a directory name below ``root``."""
    path = Path(run).expanduser()
    if not path.is_dir():
        path = root / run
    if not path.is_dir():
        raise FileNotFoundError(f"No run directory {run!r}.")
    return path


def latest_checkpoint(run_dir: Path) -> tuple[int, Path] | None:
    """``(epoch, path)`` of the most recent ``epoch_<n>.ckpt``."""
    found = {
        int(path.stem[len("epoch_"):]): path
        for path in run_dir.glob("epoch_*.ckpt")
        if path.stem[len("epoch_"):].isdigit()
    }
    if not found:
        return None
    epoch = max(found)
    return epoch, found[epoch]


class EpochCheckpoint(Callback):
    """Writes ``epoch_<n>.ckpt`` after the n-th epoch, every
    ``save_frequent`` epochs (replacing the previous such file) and after
    every epoch in ``save_epochs`` (kept)."""

    def __init__(
        self,
        dirpath: Path,
        save_epochs: Sequence[int] = (),
        save_frequent: int = 1,
    ) -> None:
        super().__init__()
        self.dirpath = Path(dirpath)
        self.save_epochs = set(save_epochs)
        self.save_frequent = save_frequent
        self._frequent: Path | None = None

    def on_train_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        epoch = trainer.current_epoch + 1
        keep = epoch in self.save_epochs
        frequent = self.save_frequent > 0 and epoch % self.save_frequent == 0
        if not (keep or frequent):
            return
        path = self.dirpath / checkpoint_name(epoch)
        if self._frequent is not None and self._frequent != path:
            trainer.strategy.remove_checkpoint(self._frequent)
        self._frequent = None if keep else path
        trainer.save_checkpoint(path)


class RunStats(Callback):
    """Logs the learning rate per epoch, prints the parameter counts (and
    puts them into the wandb summary), and appends the epoch metrics to
    ``metrics.csv`` in the run directory."""

    def __init__(self, run_dir: Path | None = None) -> None:
        super().__init__()
        self.csv_path = None if run_dir is None else run_dir / "metrics.csv"
        self._columns: list[str] | None = None

    def on_train_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        total = sum(p.numel() for p in pl_module.parameters())
        trainable = sum(
            p.numel() for p in pl_module.parameters() if p.requires_grad
        )
        print(f"Parameters: {total:,} total, {trainable:,} trainable.")
        logger = trainer.logger
        experiment = getattr(logger, "experiment", None)
        summary = getattr(experiment, "summary", None)
        if summary is not None:
            summary["params/total"] = total
            summary["params/trainable"] = trainable

    def on_train_epoch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        if trainer.optimizers:
            lr = trainer.optimizers[0].param_groups[0]["lr"]
            pl_module.log("lr", float(lr), on_epoch=True, on_step=False,
                          batch_size=1)

    def on_train_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        if self.csv_path is None:
            return
        row = {"epoch": trainer.current_epoch}
        for key, value in trainer.callback_metrics.items():
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                continue
        new_file = not self.csv_path.exists()
        if self._columns is None:
            if new_file:
                self._columns = list(row)
            else:
                with open(self.csv_path, newline="") as f:
                    self._columns = next(csv.reader(f))
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(self._columns)
            writer.writerow([row.get(c, "") for c in self._columns])
