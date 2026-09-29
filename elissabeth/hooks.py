from typing import Optional

import torch
from torch import nn


class Hook(nn.Module):
    """Identity module that caches its input while attached. Capturing is
    done through a registered forward hook, so a detached hook is a plain
    identity and costs nothing inside a compiled graph.
    """

    def __init__(self) -> None:
        super().__init__()
        self._handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self._cache: Optional[torch.Tensor] = None

    def _capture(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self._cache = inputs[0].detach().cpu()

    @property
    def data(self) -> torch.Tensor:
        if self._cache is None:
            raise ValueError("Hook has no captured data.")
        return self._cache

    @property
    def is_attached(self) -> bool:
        return self._handle is not None

    def attach(self) -> None:
        self._cache = None
        if self._handle is None:
            self._handle = self.register_forward_hook(self._capture)

    def release(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class HookCollection(nn.Module):
    """A named collection of :class:`Hook` objects owned by one module.
    Hooks are declared in ``__init__``; calling an undeclared name raises,
    so a forward pass never creates modules (which would break compiling).
    """

    def __init__(self, *names: str) -> None:
        super().__init__()
        self._hooks: dict[str, Hook] = {name: Hook() for name in names}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._hooks)

    def add_hooks(self, *names: str) -> None:
        for name in names:
            self._hooks.setdefault(name, Hook())

    def get(self, name: str) -> Hook:
        if name not in self._hooks:
            raise KeyError(f"No hook named {name!r}.")
        return self._hooks[name]

    def forward(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return self._hooks[name](x)

    def attach_all(self) -> None:
        for hook in self._hooks.values():
            hook.attach()

    def release_all(self) -> None:
        for hook in self._hooks.values():
            hook.release()


class HookedModule(nn.Module):
    """Base class giving a module a :class:`HookCollection` and the
    convenience methods to drive every collection in its subtree.
    """

    def __init__(self, *hooks: str) -> None:
        super().__init__()
        self.hooks = HookCollection(*hooks)

    def hook(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return self.hooks(name, x)

    def attach_hooks(self) -> None:
        for module in self.modules():
            if isinstance(module, HookCollection):
                module.attach_all()

    def release_hooks(self) -> None:
        for module in self.modules():
            if isinstance(module, HookCollection):
                module.release_all()

    def collect_hooks(self) -> dict[str, torch.Tensor]:
        """All captured tensors keyed by ``owner/hook_name``."""
        out: dict[str, torch.Tensor] = {}
        for prefix, module in self.named_modules():
            if not isinstance(module, HookCollection):
                continue
            owner = prefix.rsplit(".hooks", 1)[0]
            if owner == "hooks":
                owner = ""
            for name in module.names:
                hook = module.get(name)
                if hook._cache is not None:
                    out[f"{owner}/{name}" if owner else name] = hook.data
        return out
