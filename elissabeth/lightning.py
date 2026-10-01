"""Training: the trainer config, the Lightning module, and the few
callbacks a run needs (checkpoints per epoch and of the best epoch, learning
rate and parameter counts)."""
import csv
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal, Sequence

import lightning.pytorch as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from pydantic import model_validator
from torch.optim.lr_scheduler import LRScheduler

from .config import ModelConfig
from .data import T_Objective
from .elissabeth import Elissabeth, ElissabethConfig

CONFIG_NAME = "config.yaml"
BEST_NAME = "best.ckpt"


class TrainerConfig(ModelConfig):

    epochs: int = 100
    train_steps: int | None = None
    """Batches per epoch (default: the whole training set)."""

    optimizer: Literal["adamw", "radam_schedulefree"] = "adamw"
    """``radam_schedulefree`` (Defazio et al. 2024, package ``schedulefree``)
    is the Tropical Attention protocol's. It needs no schedule, and its
    checkpoints and validation use the weights it evaluates, not the ones
    it steps."""
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
    save_best: str | None = None
    """A logged metric, e.g. ``validation/loss``: ``best.ckpt`` holds the
    epoch where it was lowest. (The Tropical Attention protocol keeps the
    epoch of the lowest loss on its held-out split.)"""

    @model_validator(mode="after")
    def _check(self) -> "TrainerConfig":
        schedule = (self.warmup_epochs, self.cooldown_epochs)
        if self.optimizer == "radam_schedulefree" and any(schedule):
            raise ValueError(
                "radam_schedulefree replaces the schedule: drop"
                " trainer.warmup_epochs and trainer.cooldown_epochs."
            )
        return self


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


def objective_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    objective: T_Objective,
) -> torch.Tensor:
    """The loss of ``output`` ``(..., C)`` against ``target`` ``(...)``,
    averaged over the targets that count: not ``-1`` for the cross entropy,
    not ``NaN`` for the binary cross entropy and the squared error, which
    read the single output channel."""
    if objective == "cross_entropy":
        return F.cross_entropy(
            output.reshape(-1, output.shape[-1]), target.reshape(-1),
            ignore_index=-1,
        )
    output = output[..., 0]
    valid = ~torch.isnan(target)
    target = torch.where(valid, target, 0.0)
    if objective == "binary":
        loss = F.binary_cross_entropy_with_logits(
            output, target, reduction="none",
        )
    else:
        loss = (output - target).square()
    weight = valid.to(loss.dtype)
    return (loss * weight).sum() / weight.sum().clamp_min(1)


def binary_counts(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``[tp, fp, fn, tn]`` of ``logits > 0`` over the targets that are not
    ``NaN``."""
    valid = ~torch.isnan(target)
    predicted = logits > 0
    true = target > 0.5
    return torch.stack([
        (predicted & true & valid).sum(),
        (predicted & ~true & valid).sum(),
        (~predicted & true & valid).sum(),
        (~predicted & ~true & valid).sum(),
    ])


def f1_score(counts: torch.Tensor) -> torch.Tensor:
    """Binary F1 of the positive class from :func:`binary_counts`, 0 when
    there is nothing to score (scikit-learn's ``zero_division=0``)."""
    tp, fp, fn = counts[0], counts[1], counts[2]
    return 2 * tp / (2 * tp + fp + fn).clamp_min(1)


def binary_accuracy(counts: torch.Tensor) -> torch.Tensor:
    return (counts[0] + counts[3]) / counts.sum().clamp_min(1)


@contextmanager
def evaluated_weights(trainer: L.Trainer) -> Iterator[None]:
    """The weights a schedule-free optimizer evaluates, for the duration.
    It steps one sequence of weights and evaluates their average; the
    model holds the stepped ones while in train mode."""
    switched = [
        optimizer for optimizer in trainer.optimizers
        if optimizer.param_groups[0].get("train_mode", False)
    ]
    for optimizer in switched:
        optimizer.eval()  # type: ignore[attr-defined]
    try:
        yield
    finally:
        for optimizer in switched:
            optimizer.train()  # type: ignore[attr-defined]


class ElissabethLightningModule(L.LightningModule):
    """The task's objective on every target that counts. Targets of shape
    ``(B,)`` are compared with the model's pooled output
    (:meth:`Elissabeth.pool`), targets ``(B, T)`` position by position.
    Binary objectives log accuracy and, for validation, the F1 score of
    the positive class."""

    def __init__(
        self,
        model_config: ElissabethConfig,
        trainer_config: TrainerConfig,
        input_dim: int,
        output_dim: int | None = None,
        run_dir: Path | None = None,
        objective: T_Objective = "cross_entropy",
    ) -> None:
        super().__init__()
        self.model = Elissabeth(model_config, input_dim, output_dim)
        self.trainer_config = trainer_config
        self.run_dir = run_dir
        self.objective = objective
        self._counts: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_callbacks(self) -> list[Callback]:
        callbacks: list[Callback] = [RunStats(self.run_dir)]
        if self.run_dir is not None:
            callbacks.append(EpochCheckpoint(
                self.run_dir,
                self.trainer_config.save_epochs,
                self.trainer_config.save_frequent,
                self.trainer_config.save_best,
            ))
        return callbacks

    def _step(self, batch: Sequence[torch.Tensor], stage: str) -> torch.Tensor:
        x, y = batch
        # One compiled graph for every batch size (the last, partial batch).
        if x.shape[0] > 1:
            torch._dynamo.mark_dynamic(x, 0)
        output = self.model(x)
        if y.ndim == 1:
            output = self.model.pool(output)
        loss = objective_loss(output, y, self.objective)
        log = {"prog_bar": True, "on_epoch": True, "on_step": False,
               "batch_size": x.shape[0]}
        self.log(f"{stage}/loss", loss, **log)
        if self.objective == "cross_entropy":
            self.log(f"{stage}/accuracy", masked_accuracy(output, y), **log)
        elif self.objective == "binary":
            counts = binary_counts(output[..., 0], y)
            self.log(f"{stage}/accuracy", binary_accuracy(counts), **log)
            if stage == "validation":
                self._counts = counts if self._counts is None \
                    else self._counts + counts
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "validation")

    def _optimizer_mode(self, train: bool) -> None:
        """Schedule-free optimizers step in train mode and are evaluated in
        eval mode."""
        for optimizer in self.trainer.optimizers:
            if hasattr(optimizer, "train"):
                optimizer.train() if train else optimizer.eval()  # type: ignore

    def on_train_epoch_start(self) -> None:
        self._optimizer_mode(True)

    def on_validation_epoch_start(self) -> None:
        self._optimizer_mode(False)
        self._counts = None

    def on_validation_epoch_end(self) -> None:
        # Exact over the epoch, where the mean of per-batch F1s is not.
        if self._counts is not None:
            self.log("validation/f1", f1_score(self._counts.float()))
            self._counts = None

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
        if cfg.optimizer == "radam_schedulefree":
            import schedulefree
            return schedulefree.RAdamScheduleFree(
                groups, lr=cfg.lr, betas=(cfg.beta_1, cfg.beta_2),
            )
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
    every epoch in ``save_epochs`` (kept), and ``best.ckpt`` whenever the
    metric ``save_best`` reaches a new minimum."""

    def __init__(
        self,
        dirpath: Path,
        save_epochs: Sequence[int] = (),
        save_frequent: int = 1,
        save_best: str | None = None,
    ) -> None:
        super().__init__()
        self.dirpath = Path(dirpath)
        self.save_epochs = set(save_epochs)
        self.save_frequent = save_frequent
        self.save_best = save_best
        self.best = math.inf
        self._frequent: Path | None = None
        self._warned = False

    def state_dict(self) -> dict:
        return {"best": self.best}

    def load_state_dict(self, state_dict: dict) -> None:
        self.best = state_dict["best"]

    def _save(self, trainer: L.Trainer, path: Path) -> None:
        with evaluated_weights(trainer):
            # Explicit, or Lightning logs "`weights_only` was not set".
            trainer.save_checkpoint(path, weights_only=False)

    def on_train_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        epoch = trainer.current_epoch + 1
        if self.save_best is not None:
            value = trainer.callback_metrics.get(self.save_best)
            if value is None and not self._warned:
                print(f"save_best: {self.save_best!r} is not logged (yet);"
                      f" logged are {sorted(trainer.callback_metrics)}.")
                self._warned = True
            elif value is not None and float(value) < self.best:
                self.best = float(value)
                self._save(trainer, self.dirpath / BEST_NAME)
        keep = epoch in self.save_epochs
        frequent = self.save_frequent > 0 and epoch % self.save_frequent == 0
        if not (keep or frequent):
            return
        path = self.dirpath / checkpoint_name(epoch)
        if self._frequent is not None and self._frequent != path:
            trainer.strategy.remove_checkpoint(self._frequent)
        self._frequent = None if keep else path
        self._save(trainer, path)


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
