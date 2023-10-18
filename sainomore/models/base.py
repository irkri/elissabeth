__all__ = ["ModelConfig", "HookedModule", "SAINoMoreModule"]

from dataclasses import asdict, dataclass
from typing import Any, Literal

from torch import nn

from ..hooks import HookCollection, Hook


@dataclass
class ModelConfig:
    context_length: int
    input_vocab_size: int
    output_vocab_size: int = None  #type: ignore

    n_layers: int = 4
    d_hidden: int = 64

    layer_norm: bool = True
    bias: bool = False

    positional_encoding: Literal["learnable", "sinusoidal"] | None = None

    def __post_init__(self) -> None:
        if self.output_vocab_size is None:
            self.output_vocab_size = self.input_vocab_size

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HookedModule(nn.Module):

    hooks: HookCollection

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    def release_all_hooks(self) -> None:
        self.hooks.release_all()

    def attach_all_hooks(
        self, *,
        forward: bool = True,
        backward: bool = False,
    ) -> None:
        self.hooks.attach_all(forward=forward, backward=backward)


class SAINoMoreModule(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

    def attach_all_hooks(
        self, *,
        forward: bool = True,
        backward: bool = False,
    ) -> None:
        for things in self.children():
            if isinstance(things, nn.ModuleList):
                for thing in things:
                    if isinstance(thing, (HookedModule, SAINoMoreModule)):
                        thing.attach_all_hooks(
                            forward=forward,
                            backward=backward,
                        )
            elif isinstance(things, (HookedModule, SAINoMoreModule)):
                things.attach_all_hooks(
                    forward=forward,
                    backward=backward,
                )

    def release_all_hooks(self) -> None:
        for things in self.children():
            if isinstance(things, nn.ModuleList):
                for thing in things:
                    if isinstance(thing, (HookedModule, SAINoMoreModule)):
                        thing.release_all_hooks()
            elif isinstance(things, (HookedModule, SAINoMoreModule)):
                things.release_all_hooks()

    def get_hook(self, module_name: str, name: str) -> Hook:
        module = self.get_submodule(module_name)
        if not isinstance(module, HookedModule):
            raise ValueError(
                f"Module name {module_name} does not point to a HookedModule"
            )
        return module.hooks.get(name)

    def set_eye(self, name: str, requires_grad: bool = False) -> None:
        param = self.get_parameter(name)
        nn.init.eye_(param)
        param.requires_grad = requires_grad
