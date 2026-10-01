import json
import os
import re
from pathlib import Path
from typing import Annotated, Sequence

import yaml
from pydantic import AfterValidator, BaseModel, model_validator


class ModelConfig(BaseModel):

    model_config = {"extra": "forbid"}


_VARIABLE = re.compile(r"\$(\w+|\{[^}]*\})")


def expand(path: Path | str) -> Path:
    """Expand ``$VAR`` / ``${VAR}`` and a leading ``~`` in a configured
    path, the way a shell would. An unset variable raises instead of being
    left in place (``Path.resolve`` would glue ``$VAR`` onto the working
    directory without complaint).
    """
    text = os.fspath(path)
    missing = sorted({
        match.group(0) for match in _VARIABLE.finditer(text)
        if os.environ.get(match.group(1).strip("{}")) is None
    })
    if missing:
        raise ValueError(
            f"{', '.join(missing)} is not set in the environment, so"
            f" {text!r} cannot be expanded. Export it, or write the path out."
        )
    return Path(os.path.expandvars(text)).expanduser()


ConfigPath = Annotated[Path, AfterValidator(expand)]
"""A path field that is expanded once, when the config is loaded."""


# The composite configs live next to their modules and are imported here
# only to assemble ``RunConfig``, after ``ModelConfig`` is defined, which
# breaks the import cycle (each of them subclasses ``ModelConfig``).
from .data import DatasetConfig  # noqa: E402
from .lightning import TrainerConfig  # noqa: E402
from .elissabeth import ElissabethConfig  # noqa: E402


class RunConfig(ModelConfig):

    model: ElissabethConfig
    dataset: DatasetConfig
    trainer: TrainerConfig = TrainerConfig()

    @model_validator(mode="after")
    def _fill_context_length(self) -> "RunConfig":
        """The time scale of the LISS kernels defaults to the task's
        sequence length. It is written into the config here, so the saved
        ``config.yaml`` records it and a model evaluated on longer sequences
        keeps the scale it was trained with.
        """
        if self.model.context_length is None:
            self.model.context_length = self.dataset.length
        return self

    @model_validator(mode="after")
    def _match_task(self) -> "RunConfig":
        """The model reads what the task produces (``model.input_type`` is
        filled in unless given), and a task whose targets are later inputs
        gets a causal model."""
        if "input_type" not in self.model.model_fields_set:
            self.model.input_type = self.dataset.input_type
        elif self.model.input_type != self.dataset.input_type:
            raise ValueError(
                f"model.input_type is {self.model.input_type!r}, but the"
                f" {self.dataset.task.name} task has {self.dataset.input_type}"
                " inputs."
            )
        if self.dataset.task.causal and self.model.bidirectional:
            raise ValueError(
                f"The {self.dataset.task.name} task predicts later inputs, so"
                " a bidirectional model sees its targets: set"
                " model.liss.bidirectional (or model.attention.bidirectional)"
                " to false."
            )
        return self


def deep_update(base: dict, updates: dict, path: str = "") -> dict:
    """Recursively merges ``updates`` into ``base`` in place. An **int**
    key is a position in a list (what ``-o model.liss.kernels[0].alpha_0=2``
    parses to); a string key is a mapping key.
    """
    for key, value in updates.items():
        where = f"{path}.{key}" if path else str(key)
        if isinstance(value, dict) and any(isinstance(k, int) for k in value):
            base[key] = _update_entries(base.get(key), value, where)
        elif isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base[key], value, where)
        else:
            base[key] = value
    return base


def _update_entries(entries: object, updates: dict, where: str) -> list:
    """Apply ``{index: value}`` onto the list at ``where``. An override
    edits what the config has and never appends, so an index outside the
    list is an error.
    """
    if not all(isinstance(k, int) for k in updates):
        raise ValueError(
            f"Override {where!r} mixes list positions with mapping keys."
        )
    if not isinstance(entries, list):
        found = "nothing" if entries is None else type(entries).__name__
        raise ValueError(
            f"Override {where}[...] needs a list at {where!r}, found {found}."
        )
    for index, value in updates.items():
        if not -len(entries) <= index < len(entries):
            raise ValueError(
                f"Override {where}[{index}] is out of range: {where!r} has"
                f" {len(entries)} entries."
            )
        entry = entries[index]
        if isinstance(value, dict) and isinstance(entry, dict):
            entries[index] = deep_update(entry, value, f"{where}[{index}]")
        else:
            entries[index] = value
    return entries


def _parse_value(value: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


_KEY = re.compile(r"^(?P<name>[^\[\]]+)(?P<indices>(?:\[-?\d+\])*)$")
_INDEX = re.compile(r"\[(-?\d+)\]")


def _split_key(part: str, key: str) -> list[str | int]:
    """``"kernels[0]"`` -> ``["kernels", 0]``."""
    match = _KEY.match(part)
    if match is None:
        raise ValueError(
            f"Invalid override key {key!r}: cannot read {part!r}."
        )
    return [
        match["name"],
        *(int(i) for i in _INDEX.findall(match["indices"])),
    ]


def parse_overrides(items: list[str]) -> dict:
    """Turns ``["model.d_hidden=32"]`` into a nested override dict. Values
    are parsed as JSON where possible; a ``[i]`` suffix indexes a list.
    """
    result: dict = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(
                f"Invalid override {item!r}, expected 'key=value'."
            )
        node = result
        parts = [p for part in key.split(".") for p in _split_key(part, key)]
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_value(value)
    return result


def read_mapping(path: str | Path) -> dict:
    """Read a YAML or JSON file holding a mapping."""
    path = Path(path).expanduser()
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    elif path.suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix!r}")
    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a mapping.")
    return data


T_Overrides = dict | Sequence[dict] | None
"""A nested override dictionary, or several applied in order (a later one
wins, and its ``[i]`` indices address the lists the earlier ones left)."""


def apply_overrides(data: dict, overrides: T_Overrides) -> dict:
    if isinstance(overrides, dict):
        overrides = [overrides]
    for layer in overrides or ():
        data = deep_update(data, layer)
    return data


def load_runconfig(path: str | Path, overrides: T_Overrides = None) -> RunConfig:
    """Load a :class:`RunConfig` from a YAML or JSON file, optionally
    applying nested ``overrides`` on top of it."""
    return RunConfig(**apply_overrides(read_mapping(path), overrides))


def load_saved_runconfig(
    path: str | Path,
    overrides: T_Overrides = None,
) -> RunConfig:
    """Load the ``config.yaml`` of a run directory. :func:`save_runconfig`
    writes every field, so a missing one is newer than the run and is set
    to what the run had: ``model.liss.bidirectional`` to false (a LISS was
    causal before the field existed, and is bidirectional by default now).
    """
    data = read_mapping(path)
    liss = data.get("model", {}).get("liss")
    if isinstance(liss, dict) and "bidirectional" not in liss:
        liss["bidirectional"] = False
    return RunConfig(**apply_overrides(data, overrides))


def save_runconfig(config: RunConfig, path: str | Path) -> None:
    """Write a :class:`RunConfig` in a form :func:`load_runconfig` reads."""
    path = Path(path).expanduser()
    data = config.model_dump(mode="json")
    if path.suffix in (".yaml", ".yml"):
        text = yaml.safe_dump(data, sort_keys=False)
    elif path.suffix == ".json":
        text = json.dumps(data, indent=4)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix!r}")
    path.write_text(text)
