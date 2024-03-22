from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence

import lightning.pytorch as L
import numpy as np
import torch
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.model_summary.model_summary import summarize
from torch.utils.data import DataLoader

from .base import HookedModule
from .elissabeth import Elissabeth


class GeneralConfigCallback(Callback):

    def __init__(self, max_depth: int = 10) -> None:
        super().__init__()
        self._max_depth = max_depth

    def on_train_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        model_summary = summarize(pl_module, max_depth=self._max_depth)

        for logger in trainer.loggers:
            logger.log_hyperparams({
                "total_parameters": model_summary.total_parameters,
                "trainable_parameters": model_summary.trainable_parameters,
                "model_size": model_summary.model_size,
            })


class WeightHistory(Callback):

    def __init__(
        self,
        weights: Sequence[str | tuple[str, tuple[int, ...]]],
        each_n_epochs: int = 1,
        save_path: Optional[Path | str] = None,
    ) -> None:
        super().__init__()
        self._weights = list(weights)
        self._each_n_epochs = each_n_epochs
        self._epoch = -1
        self._save_path = Path(save_path) if save_path is not None else None

    def on_train_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        self._epoch = -1

    def on_train_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        self._epoch += 1
        if self._epoch % self._each_n_epochs != 0:
            return
        path = None
        if self._save_path is not None:
            path = self._save_path / f"epoch{self._epoch:05}"
            path.mkdir(parents=True, exist_ok=True)
        for weight in self._weights:
            dim = None
            if isinstance(weight, str):
                name = weight
            else:
                name, dim = weight
            try:
                param = pl_module.get_parameter(name).detach().cpu().numpy()
            except AttributeError:
                continue
            if dim is not None:
                param = np.moveaxis(
                    param, dim, tuple(-i for i in range(1, len(dim)+1))
                )
            self._log_image(
                trainer,
                param,
                name,
                (),
                path,
                is_1d=(len(dim) == 1) if dim is not None else False
            )

    def _log_image(
        self,
        trainer: L.Trainer,
        image: np.ndarray,
        name: str,
        dim: tuple[int, ...],
        path: Path | None,
        is_1d: bool,
    ) -> None:
        if (image[dim].ndim == 2 and not is_1d) or image[dim].ndim == 1:
            for logger in trainer.loggers:
                if isinstance(logger, WandbLogger):
                    name_ = f"weights/{name}"
                    if dim:
                        name_ += " ("+",".join(map(str, dim))+")"
                    logger.log_image(name_,
                        [image[dim]] if not is_1d
                            else [image[dim][np.newaxis, :]],
                    )
            if path is not None:
                np.save(path / (name+".npy"), image[dim])
        else:
            for i in range(image[dim].shape[0]):
                self._log_image(
                    trainer,
                    image,
                    name,
                    dim + (i, ),
                    path,
                    is_1d,
                )


class HookHistory(Callback):

    def __init__(
        self,
        model_name: str,
        data: DataLoader,
        hook_names: Optional[Sequence[str]] = None,
        forward: bool = True,
        backward: bool = False,
        each_n_epochs: int = 1,
        transform: Optional[Callable[[list[list[np.ndarray]]], Any]] = None,
        save_path: Optional[Path | str] = None,
    ) -> None:
        super().__init__()
        self._model_name = model_name
        self._data = data
        self._given_hook_names = hook_names
        self._forward = forward
        self._backward = backward
        self._each_n_epochs = each_n_epochs
        self._epoch = -1
        self._transform = transform
        self._save_path = Path(save_path) if save_path is not None else None

    def on_train_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        self._epoch = -1
        model = pl_module.get_submodule("model." + self._model_name)
        if not isinstance(model, HookedModule):
            raise RuntimeError(f"Module {model} is not a HookedModule")
        self._hooks = model.hooks
        self._hook_names = list(
            self._hooks.names if self._given_hook_names is None
            else tuple(self._given_hook_names)
        )

    def on_train_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        self._epoch += 1
        if self._epoch % self._each_n_epochs != 0:
            return

        self._hooks.attach_all(forward=self._forward, backward=self._backward)
        data_fwd: list[list[np.ndarray]] = []
        data_bwd: list[list[np.ndarray]] = []
        columns = ["sample.x", "sample.y"] + self._hook_names

        for bid, batch in enumerate(self._data):
            x, y = batch
            x = x.to(pl_module.device)
            y = y.to(pl_module.device)
            if self._forward:
                data_fwd.append(
                    [x.detach().cpu().numpy(), y.detach().cpu().numpy()]
                )
            if self._backward:
                data_bwd.append(
                    [x.detach().cpu().numpy(), y.detach().cpu().numpy()]
                )
            metrics = pl_module.validation_step([x, y], bid)
            if self._backward:
                metrics["loss"].backward()  # type: ignore
            for hook_name in self._hook_names:
                if self._forward:
                    data_fwd[-1].append(self._hooks.get(hook_name).fwd.numpy())
                if self._backward:
                    data_bwd[-1].append(self._hooks.get(hook_name).bwd.numpy())

        if self._forward and self._transform is not None:
            data_fwd = self._transform(data_fwd)
        if self._backward and self._transform is not None:
            data_bwd = self._transform(data_bwd)

        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                if self._forward:
                    logger.log_table(
                        "hooks/fwd",
                        columns=columns,
                        data=data_fwd,
                        step=self._epoch,
                    )
                if self._backward:
                    logger.log_table(
                        "hooks/bwd",
                        columns=columns,
                        data=data_bwd,
                        step=self._epoch,
                    )

        if self._save_path is not None:
            for s in range(len(data_fwd)):
                path = (
                    self._save_path
                    / f"epoch{self._epoch:05}"
                    / f"sample{s:03}"
                )
                path.mkdir(parents=True, exist_ok=True)
                for i in range(len(data_fwd[s])):
                    np.save(
                        path / (columns[i]+"_fwd.npy"),
                        data_fwd[s][i],
                    )
            for s in range(len(data_bwd)):
                path = (
                    self._save_path
                    / f"epoch{self._epoch:05}"
                    / f"sample{s:03}"
                )
                path.mkdir(parents=True, exist_ok=True)
                for i in range(len(data_bwd[s])):
                    np.save(
                        path / (columns[i]+"_bwd.npy"),
                        data_bwd[s][i],
                    )
        self._hooks.release_all()


class ElissabethWeighting(Callback):

    def __init__(
        self,
        x: torch.Tensor,
        each_n_epochs: int = 1,
        save_path: Optional[Path | str] = None,
        use_wandb: bool = False,
        name: str = "weighting",
    ) -> None:
        super().__init__()
        self._data = x
        self._each_n_epochs = each_n_epochs
        self._epoch = -1
        self._save_path = Path(save_path) if save_path is not None else None
        self._wandb = use_wandb
        self._name = name

    def on_train_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        self._epoch = -1
        self._data = self._data.to(pl_module.device)

    def on_train_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        self._epoch += 1
        if self._epoch % self._each_n_epochs != 0:
            return

        model: Elissabeth = pl_module.model  # type: ignore

        for layer in model.layers:
            for weighting in layer.weightings:
                weighting.hooks.get("Att").attach()

        model(self._data)

        for layer in model.layers:
            for weighting in layer.weightings:
                weighting.hooks.get("Att").release()

        n_layers = model.config("n_layers")
        N = model.layers[0].config("n_is")
        length_is = model.layers[0].config("length_is")
        att_mat = torch.ones(
            (n_layers, N, length_is, self._data.size(1), self._data.size(1))
        )
        for l in range(n_layers):
            for weighting in model.layers[l].weightings:
                att_mat[l] *= weighting.hooks.get("Att").fwd[0]

        if self._wandb:
            for logger in trainer.loggers:
                if isinstance(logger, WandbLogger):
                    columns = [
                        f"N {j} | Length {i}" for i in range(1, length_is+1)
                        for j in range(1, N+1)
                    ]
                    data = [
                        [wandb.Image(att_mat[l, j, d])
                          for d in range(length_is)
                          for j in range(N)]
                        for l in range(n_layers)
                    ]
                    logger.log_table(
                        f"hooks/{self._name}",
                        columns=columns,
                        data=data,
                        step=self._epoch,
                    )

        if self._save_path is not None:
            path = self._save_path / f"epoch{self._epoch:05}"
            path.mkdir(parents=True, exist_ok=True)
            np.save(path / f"{self._name}.npy", att_mat)


class ElissabethISTracker(Callback):

    def __init__(
        self,
        x: torch.Tensor,
        each_n_epochs: int = 1,
        save_path: Optional[Path | str] = None,
        use_wandb: bool = False,
        name: str = "iterated_sums",
    ) -> None:
        super().__init__()
        self._data = x
        self._each_n_epochs = each_n_epochs
        self._epoch = -1
        self._save_path = Path(save_path) if save_path is not None else None
        self._wandb = use_wandb
        self._name = name

    def on_train_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        self._epoch = -1
        self._data = self._data.to(pl_module.device)

    def on_train_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        self._epoch += 1
        if self._epoch % self._each_n_epochs != 0:
            return

        model: Elissabeth = pl_module.model  # type: ignore

        n_layers = model.config("n_layers")
        length_is = model.layers[0].config("length_is")
        n_is = model.layers[0].config("n_is")

        for l in range(n_layers):
            for d in range(1, length_is+1):
                model.get_hook(f"layers.{l}", f"iss.{d}").attach()
        model(self._data)
        model.release_all_hooks()

        values = np.empty((n_layers, length_is, n_is, self._data.size(1)))
        for l in range(n_layers):
            for d in range(length_is):
                values[l, d, :, :] = np.swapaxes(np.mean(np.linalg.norm(
                    model.get_hook(
                        f"layers.{l}", f"iss.{d+1}",
                    ).fwd[0, :, :, :, :, :],
                axis=(3, 4)), axis=0), 0, 1)
        if self._wandb:
            for logger in trainer.loggers:
                if isinstance(logger, WandbLogger):
                    columns = ["Time"] + [
                        f"IS {n}, Layer {i}, Depth {j}"
                        for n in range(1, n_is+1)
                        for i in range(1, n_layers+1)
                        for j in range(1, length_is+1)
                    ]
                    content = [list(range(1, self._data.size(1)))]
                    for n in range(n_is):
                        for l in range(n_layers):
                            for d in range(length_is):
                                content.append(list(values[l, d, n]))
                    content = list(map(list, zip(*content)))
                    logger.log_table(
                        f"hooks/{self._name}",
                        columns=columns,
                        data=content,
                    )

        if self._save_path is not None:
            path = (
                self._save_path
                / f"epoch{self._epoch:05}"
            )
            path.mkdir(parents=True, exist_ok=True)
            np.save(path / f"{self._name}.npy", values)
