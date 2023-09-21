import warnings
from typing import Any, Optional, Sequence

import torch
from torch import nn


class Hook(nn.Module):

    def __init__(self, backward: bool = False) -> None:
        super().__init__()
        self._handle = None
        self._handle_backward = None
        self._backward = backward
        self._cache: Optional[torch.Tensor] = None
        self._cache_backward: Optional[torch.Tensor] = None

    def save_backward(self, backward: bool) -> None:
        self._backward = backward

    def hook(
        self,
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
        output: Any,
    ) -> None:
        self._cache = input[0].detach()

    def hook_backward(
        self,
        module: nn.Module,
        input: tuple[torch.Tensor, ...] | torch.Tensor,
        output: tuple[torch.Tensor, ...] | torch.Tensor,
    ) -> None:
        self._cache_backward = input[0].detach()

    @property
    def fwd(self) -> Optional[torch.Tensor]:
        return self._cache

    @property
    def bwd(self) -> Optional[torch.Tensor]:
        return self._cache_backward

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
        if self._handle_backward is not None:
            self._handle_backward.remove()

    def attach(self) -> None:
        self._handle = self.register_forward_hook(self.hook)
        if self._backward:
            self._handle_backward = self.register_full_backward_hook(
                self.hook_backward
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class HookCollection(nn.Module):

    def __init__(self, *names: str, backward: bool = False) -> None:
        super().__init__()
        self._hooks: dict[str, Hook] = {}
        self._backward = backward
        self.add_hooks(names)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._hooks.keys())

    def save_backward(self, backward: bool) -> None:
        self._backward = backward
        for name in self._hooks:
            self._hooks[name].save_backward(self._backward)

    def add_hooks(self, names: str | Sequence[str]) -> None:
        if isinstance(names, str):
            if names in self._hooks:
                warnings.warn(f"Already hooked {names}", RuntimeWarning)
            self._hooks[names] = Hook(backward=self._backward)
        else:
            for name in names:
                if name in self._hooks:
                    warnings.warn(f"Already hooked {name}", RuntimeWarning)
                self._hooks[name] = Hook(backward=self._backward)

    def get(self, name: str) -> Hook:
        if name not in self._hooks:
            raise RuntimeError(f"No hook named {name} found")
        return self._hooks[name]

    def forward(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if name not in self._hooks:
            raise RuntimeError(f"No hook named {name} found")
        return self._hooks[name](x)

    def release_all(self) -> None:
        for name in self._hooks:
            self._hooks[name].remove()

    def attach_all(self) -> None:
        for name in self._hooks:
            self._hooks[name].attach()
