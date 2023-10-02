__all__ = ["GeneralConfigCallback", "WeightHistory", "HookHistory"]

import os
from typing import Any, Callable, Optional, Sequence

import lightning.pytorch as L
import numpy as np
import torch
import wandb
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities.model_summary.model_summary import summarize
from torch.utils.data import DataLoader

from .models.base import HookedModule
from .models.liss import Elissabeth


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
        weight_names: Sequence[str],
        reduce_axis: Optional[Sequence[int | None]] = None,
        each_n_epochs: int = 1,
        save_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._weight_names = list(weight_names)
        self._reduce_axis = reduce_axis
        self._each_n_epochs = each_n_epochs
        self._epoch = -1
        self._save_path = save_path

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
            path = os.path.join(self._save_path, f"epoch{self._epoch:05}")
            os.makedirs(path, exist_ok=True)
        for k, wname in enumerate(self._weight_names):
            param = pl_module.get_parameter(wname).detach().cpu().numpy()
            if (self._reduce_axis is not None
                    and self._reduce_axis[k] is not None):
                param = [np.take(param, i, self._reduce_axis[k])
                         for i in range(param.shape[self._reduce_axis[k]])]
            else:
                param = [param]
            for logger in trainer.loggers:
                if isinstance(logger, WandbLogger):
                    logger.log_image("weights/"+wname, param)
            if path is not None:
                np.save(os.path.join(path, wname+".npy"), np.array(param))


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
        save_path: Optional[str] = None,
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
        self._save_path = save_path

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
            if self._forward:
                data_fwd.append(
                    [batch[0].detach().numpy(), batch[1].detach().numpy()]
                )
            if self._backward:
                data_bwd.append(
                    [batch[0].detach().numpy(), batch[1].detach().numpy()]
                )
            metrics = pl_module.validation_step(batch, bid)
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
                path = os.path.join(
                    self._save_path,
                    f"epoch{self._epoch:05}",
                    f"sample{s:03}",
                )
                os.makedirs(path, exist_ok=True)
                for i in range(len(data_fwd[s])):
                    np.save(
                        os.path.join(path, columns[i]+"_fwd.npy"),
                        data_fwd[s][i],
                    )
            for s in range(len(data_bwd)):
                path = os.path.join(
                    self._save_path,
                    f"epoch{self._epoch:05}",
                    f"sample{s:03}",
                )
                os.makedirs(path, exist_ok=True)
                for i in range(len(data_bwd[s])):
                    np.save(
                        os.path.join(path, columns[i]+"_bwd.npy"),
                        data_bwd[s][i],
                    )
        self._hooks.release_all()


class ElissabethWeighting(Callback):

    def __init__(
        self,
        x: torch.Tensor,
        qk_index: int = 0,
        each_n_epochs: int = 1,
        save_path: Optional[str] = None,
        use_wandb: bool = False,
    ) -> None:
        super().__init__()
        self._data = x
        self._qk_index = qk_index
        self._each_n_epochs = each_n_epochs
        self._epoch = -1
        self._save_path = save_path
        self._wandb = use_wandb

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

        model: Elissabeth = pl_module.model  # type: ignore

        model.attach_all_hooks()
        model(self._data)
        model.release_all_hooks()

        n_layers = len(model.layers)
        iss_length: int = model.layers[0].length  # type: ignore
        att_mat = np.empty(
            (n_layers, iss_length, self._data.size(1), self._data.size(1))
        )
        for l in range(n_layers):
            for d in range(iss_length):
                att_mat[l, d] = model.get_hook(
                    f"layers.{l}", f"weighting.{d}",
                ).fwd[0, :, :, self._qk_index]

        if self._wandb:
            for logger in trainer.loggers:
                columns = [
                    f"Length {i}" for i in range(1, iss_length+1)
                ] + ["Product"]
                data = [
                    ([wandb.Image(att_mat[l, d]) for d in range(iss_length)]
                    +[wandb.Image(np.prod(att_mat[l], axis=0))])
                    for l in range(n_layers)
                ]
                if isinstance(logger, WandbLogger):
                    logger.log_table(
                        "hooks/Elissabeth/weighting",
                        columns=columns,
                        data=data,
                        step=self._epoch,
                    )

        if self._save_path is not None:
            path = os.path.join(
                self._save_path,
                f"epoch{self._epoch:05}",
            )
            os.makedirs(path, exist_ok=True)
            np.save(os.path.join(path, "elissabeth_weighting.npy"), att_mat)
