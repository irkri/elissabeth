__all__ = ["HookedModule", "SAINoMoreModule"]

from abc import ABC, abstractmethod
from enum import IntFlag
from typing import Any, Optional

import torch
from pydantic import BaseModel
from torch import nn

from .hooks import Hook, HookCollection


class HookedModule(nn.Module):

    _config_class: type[BaseModel]

    def __init__(
        self,
        parent: Optional["HookedModule"] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self._config = self._config_class(**kwargs)
        self.hooks = HookCollection()
        self._parent: tuple[HookedModule] | None = (
            (parent, ) if parent is not None else None
        )

    @property
    def parent(self) -> Optional["HookedModule"]:
        return self._parent[0] if self._parent is not None else None

    def attach_all_hooks(
        self, *,
        forward: bool = True,
        backward: bool = False,
    ) -> None:
        self.hooks.attach_all(forward=forward, backward=backward)
        for things in self.children():
            if isinstance(things, nn.ModuleList):
                for thing in things:
                    if isinstance(thing, HookedModule):
                        thing.attach_all_hooks(
                            forward=forward,
                            backward=backward,
                        )
            elif isinstance(things, HookedModule):
                things.attach_all_hooks(
                    forward=forward,
                    backward=backward,
                )

    def release_all_hooks(self) -> None:
        self.hooks.release_all()
        for things in self.children():
            if isinstance(things, nn.ModuleList):
                for thing in things:
                    if isinstance(thing, HookedModule):
                        thing.release_all_hooks()
            elif isinstance(things, HookedModule):
                things.release_all_hooks()

    def set_parent(self, parent: "HookedModule") -> None:
        self._parent = (parent, )

    def config(self, key: str) -> Any:
        try:
            return getattr(self._config, key)
        except AttributeError:
            if self.parent is not None:
                return self.parent.config(key)
            else:
                raise AttributeError(f"Attribute {key!r} not found")

    def get_hook(self, name: str) -> Hook:
        try:
            return self.hooks.get(name)
        except KeyError:
            if self.parent is not None:
                return self.parent.get_hook(name)
            else:
                raise KeyError(f"No hook named {name!r} found")

    def hook(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return self.get_hook(name)(x)


class SAINoMoreModule(ABC, HookedModule):

    def attach_all_hooks(
        self, *,
        forward: bool = True,
        backward: bool = False,
    ) -> None:
        for things in self.children():
            if isinstance(things, nn.ModuleList):
                for thing in things:
                    if isinstance(thing, HookedModule):
                        thing.attach_all_hooks(
                            forward=forward,
                            backward=backward,
                        )
            elif isinstance(things, HookedModule):
                things.attach_all_hooks(
                    forward=forward,
                    backward=backward,
                )

    def release_all_hooks(self) -> None:
        for things in self.children():
            if isinstance(things, nn.ModuleList):
                for thing in things:
                    if isinstance(thing, HookedModule):
                        thing.release_all_hooks()
            elif isinstance(things, HookedModule):
                things.release_all_hooks()

    def get_hook(self, module_name: str, name: str) -> Hook:
        module = self.get_submodule(module_name)
        if not isinstance(module, HookedModule):
            raise ValueError(
                f"Module name {module_name!r} does not point to a HookedModule"
            )
        return module.hooks.get(name)

    def set_eye(
        self,
        name: str,
        requires_grad: bool = False,
        dims: Optional[tuple[int, int]] = None,
    ) -> None:
        param = self.get_parameter(name)
        if dims is None:
            matching_dims = []
            for i in range(param.dim()):
                for j in range(i+1, param.dim()):
                    if param.size(i) == param.size(j) and param.size(i) != 1:
                        matching_dims.append((i, j))
            if len(matching_dims) > 1:
                raise ValueError("More than 2 dimensions match in size")
            elif len(matching_dims) == 0:
                raise ValueError("No matching dimensions found")
        else:
            matching_dims = [dims]
        SAINoMoreModule._set_eye(
            param,
            tuple(i for i in range(param.dim()) if i not in matching_dims[0]),
        )
        param.requires_grad = requires_grad

    @staticmethod
    def _set_eye(param: torch.Tensor, dims: tuple[int, ...]) -> None:
        if param.dim() == 2:
            nn.init.eye_(param)
        else:
            dim = dims[-1]
            dims = dims[:-1]
            for i in range(param.size(dim)):
                SAINoMoreModule._set_eye(param.select(dim, i), dims)

    @staticmethod
    @abstractmethod
    def build(
        config: dict[str, Any],
        *flags: IntFlag,
    ) -> "SAINoMoreModule":
        ...
