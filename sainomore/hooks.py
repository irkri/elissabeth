import warnings
from typing import Any, Optional, Sequence

import torch
from torch import nn


class Hook(nn.Module):

    def __init__(self, forward: bool = True, backward: bool = False) -> None:
        super().__init__()
        self._handle_fwd = None
        self._handle_bwd = None
        self._forward = forward
        self._backward = backward
        self._cache_fwd: Optional[torch.Tensor] = None
        self._cache_bwd: Optional[torch.Tensor] = None

    def reset(self, forward: bool = True, backward: bool = False) -> None:
        self._forward = forward
        self._backward = backward
        self._cache_fwd = None
        self._cache_bwd = None

    def hook_fwd(
        self,
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
        output: Any,
    ) -> None:
        self._cache_fwd = input[0].detach().cpu()

    def hook_bwd(
        self,
        module: nn.Module,
        input: tuple[torch.Tensor, ...] | torch.Tensor,
        output: tuple[torch.Tensor, ...] | torch.Tensor,
    ) -> None:
        self._cache_bwd = input[0].detach().cpu()

    @property
    def fwd(self) -> torch.Tensor:
        if self._cache_fwd is None:
            raise ValueError("Did not save forward pass of Hook")
        return self._cache_fwd

    @property
    def bwd(self) -> torch.Tensor:
        if self._cache_bwd is None:
            raise ValueError("Did not save backward pass of Hook")
        return self._cache_bwd

    def is_attached(self) -> bool:
        return len(self._forward_hooks) > 0 or len(self._backward_hooks) > 0

    def release(self) -> None:
        if self._handle_fwd is not None:
            self._handle_fwd.remove()
        if self._handle_bwd is not None:
            self._handle_bwd.remove()

    def attach(self) -> None:
        if self._forward:
            self._handle_fwd = self.register_forward_hook(self.hook_fwd)
        if self._backward:
            self._handle_bwd = self.register_full_backward_hook(self.hook_bwd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class HookCollection(nn.Module):

    def __init__(self, *names: str) -> None:
        super().__init__()
        self._hooks: dict[str, Hook] = {}
        self.add_hooks(*names)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._hooks.keys())

    def add_hooks(self, *names: str) -> None:
        for name in names:
            if name in self._hooks:
                warnings.warn(f"Already hooked {name!r}", RuntimeWarning)
            self._hooks[name] = Hook()

    def get(self, name: str) -> Hook:
        if name not in self._hooks:
            raise KeyError(f"No hook named {name!r} found")
        return self._hooks[name]

    def forward(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return self.get(name)(x)

    def release_all(self) -> None:
        for name in self._hooks:
            self._hooks[name].release()

    def attach_all(
        self, *,
        forward: bool = True,
        backward: bool = False,
    ) -> None:
        for name in self._hooks:
            self._hooks[name].reset(forward=forward, backward=backward)
            self._hooks[name].attach()
