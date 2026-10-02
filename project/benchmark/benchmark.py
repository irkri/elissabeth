"""What does a LISS layer cost?

Times one LISS layer -- forward + backward ("train") and the forward under
``no_grad`` ("infer") -- and records the activation memory of a step,
across semirings, kernels, depths ``p``, lengths ``T`` and implementations
(``implementations.py``). Cells a chart compares are timed in turns, one
step each per round (``timing_key``: a configuration's implementations and
tasks, every depth at one length, every rank or setting of a semiring), so
they share whatever load other jobs put on the GPU. Every cell is checked against a
float64 eager evaluation of the same weights, so an implementation that is
fast because it is wrong shows up in the same table. Rows are appended to the CSV as
they are measured (``--resume`` skips the cells already there), the run's
device and versions go to a JSON next to it, and ``plot_benchmark.py``
renders both as an HTML report.

Experiments (``--experiments``):

- ``length``    semiring x depth p x length T; every implementation trains,
                ``--infer-impls`` also run the forward alone
- ``rank``      the kernel rank R (exponential and cosine ``d_qk``) at one
                T and p
- ``settings``  one LISSConfig option changed at a time against a base
                config, in the reals and the arctic semiring
- ``scan``      the two paths of the decayed real/bayesian scan, the
                rescaled cumsum (``compiled``) against Hillis-Steele
                (``hillis``), over the lengths at one depth; kept apart
                because an unrolled Hillis scan takes minutes to compile
- ``attention`` softmax attention (SDPA, the transformer baseline in
                ``elissabeth.attention``) over the same lengths

The layer is causal (one direction; ``bidirectional`` costs twice that and
is one of the settings), reads a LayerNorm'd input as it does inside the
model, and has ``context_length = T``.

    cd project
    python benchmark/benchmark.py --out benchmark/liss01.csv --memory-cap-gb 10
    python benchmark/plot_benchmark.py benchmark/liss01.csv
"""
import argparse
import csv
import gc
import json
import math
import platform
import re
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from pydantic import ValidationError

from elissabeth.attention import AttentionConfig, SelfAttention
from elissabeth.liss import LISSConfig, build_liss
from implementations import IMPLEMENTATIONS, Implementation

SEMIRINGS = ["reals", "log", "arctic", "bayesian"]
TASKS = ["train", "infer"]
EXPERIMENTS = ["length", "rank", "settings", "scan", "attention"]
LENGTHS = [256, 1024, 4096, 16384, 65536, 131072]
DEPTHS = [1, 2, 3, 4, 6]

KERNELS: dict[str, list[dict]] = {
    "decay": [{"type": "decay"}],
    "decay+exp": [{"type": "decay"}, {"type": "exponential"}],
    "decay+cos": [{"type": "decay"}, {"type": "cosine", "d_qk": 3}],
}
"""Named kernel sets; ``exp<d>`` and ``cos<d>`` are a decay times an
exponential or cosine kernel with ``d_qk = d`` (the rank experiment)."""

RANK_EXP = [1, 2, 4, 8, 16]
RANK_COS = [1, 2, 3, 4]

_MLP = {"activation": "relu"}
SETTINGS: list[tuple[str, dict]] = [
    ("base", {}),
    ("decay only (no query/key)", {"kernels": KERNELS["decay"]}),
    ("normalize: mean", {"normalize": "mean"}),
    ("normalize: learnable", {"normalize": "learnable"}),
    ("values_2D", {"values_2D": True}),
    ("share_values", {"share_values": True}),
    ("values.shared", {"values": {"shared": True}}),
    ("values.norm off", {"values": {"norm": False}}),
    ("MLP projections", {
        "values": _MLP,
        "kernels": [{"type": "decay"},
                    {"type": "exponential", "projection": _MLP}],
    }),
    ("include_time", {
        "values": {"include_time": True},
        "kernels": [{"type": "decay"}, {"type": "exponential",
                    "projection": {"include_time": True}}],
    }),
    ("bidirectional", {"bidirectional": True}),
    ("lengths [1, 2, 3]", {"lengths": [1, 2, 3]}),
    ("n_is 2", {"n_is": 2}),
    ("n_is 32", {"n_is": 32}),
    ("d_values 4", {"d_values": 4}),
    ("d_values 64", {"d_values": 64}),
]
"""One change against the base config at a time: ``(label, overrides)``.
Overrides a semiring rejects (normalisation in a max semiring) are
skipped."""

FIELDS = [
    "experiment", "variant", "mixer", "semiring", "kernels", "rank",
    "lengths", "p", "T", "B", "n_is", "d_values", "w", "d_hidden",
    "normalize", "bidirectional", "impl", "task", "status", "compile_s",
    "fwd_ms", "bwd_ms", "total_ms", "fwd_ms_min", "bwd_ms_min",
    "total_ms_min", "iters", "peak_mb",
    "err_out", "err_grad", "work", "params", "gpu_busy", "note",
]
KEY = [
    "experiment", "variant", "mixer", "semiring", "kernels", "lengths", "T",
    "B", "n_is", "d_values", "d_hidden", "impl", "task",
]


@dataclass(frozen=True)
class Cell:

    experiment: str
    semiring: str
    kernels: str
    lengths: str
    T: int
    B: int
    n_is: int
    d_values: int
    d_hidden: int
    impl: str = "eager"
    task: str = "train"
    mixer: str = "liss"
    variant: str = "base"
    overrides: tuple = field(default=(), compare=False)
    """The setting's LISSConfig overrides as ``tuple(dict.items())`` (a
    frozen dataclass cannot hold a dict)."""

    @property
    def group(self) -> "Cell":
        """The cell without its implementation and task: everything that
        shares one set of weights, one input and one float64 reference."""
        return replace(self, impl="", task="")

    def key(self) -> tuple[str, ...]:
        return tuple(str(getattr(self, name)) for name in KEY)


def kernel_list(name: str) -> list[dict]:
    if name in KERNELS:
        return KERNELS[name]
    match = re.fullmatch(r"(exp|cos)(\d+)", name)
    if match is None:
        raise ValueError(f"Unknown kernel set {name!r}.")
    kind = "exponential" if match.group(1) == "exp" else "cosine"
    return [{"type": "decay"}, {"type": kind, "d_qk": int(match.group(2))}]


def _merge(base: dict, update: dict) -> dict:
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def liss_config(cell: Cell) -> LISSConfig:
    config = {
        "d_values": cell.d_values,
        "n_is": cell.n_is,
        "lengths": [int(p) for p in cell.lengths.split(",")],
        "semiring": cell.semiring,
        "kernels": kernel_list(cell.kernels),
        "bidirectional": False,
        "values": {},
    }
    return LISSConfig(**_merge(config, dict(cell.overrides)))


# --------------------------------------------------------------------------
# the cells of each experiment
# --------------------------------------------------------------------------
def _valid(cell: Cell) -> bool:
    try:
        liss_config(cell)
    except ValidationError:
        return False
    return True


def length_cells(args: argparse.Namespace) -> Iterator[Cell]:
    for semiring in args.semirings:
        kernel_sets = ["decay+exp"] + (["decay+cos"] if semiring == "reals"
                                       else [])
        for kernels in kernel_sets:
            for p in args.depths:
                for T in args.lengths:
                    base = Cell("length", semiring, kernels, str(p), T,
                                args.batch, args.heads, args.d_values,
                                args.d_hidden)
                    for impl in args.impls:
                        yield replace(base, impl=impl, task="train")
                        if impl in args.infer_impls and "infer" in args.tasks:
                            yield replace(base, impl=impl, task="infer")


def rank_cells(args: argparse.Namespace) -> Iterator[Cell]:
    sets = [(s, f"exp{d}") for s in args.semirings for d in RANK_EXP]
    if "reals" in args.semirings:
        sets += [("reals", f"cos{d}") for d in RANK_COS]
    for semiring, kernels in sets:
        for impl in args.side_impls:
            yield Cell("rank", semiring, kernels, str(args.side_p),
                       args.side_T, args.batch, args.heads, args.d_values,
                       args.d_hidden, impl=impl)


def settings_cells(args: argparse.Namespace) -> Iterator[Cell]:
    for semiring in [s for s in ("reals", "arctic") if s in args.semirings]:
        for label, overrides in SETTINGS:
            base = Cell("settings", semiring, "decay+exp", str(args.side_p),
                        args.side_T, args.batch, args.heads, args.d_values,
                        args.d_hidden, variant=label,
                        overrides=tuple(overrides.items()))
            if not _valid(base):
                continue
            for impl in args.side_impls:
                yield replace(base, impl=impl)


def scan_cells(args: argparse.Namespace) -> Iterator[Cell]:
    impls = [i for i in ("compiled", "hillis") if i in IMPLEMENTATIONS]
    for semiring in [s for s in ("reals", "bayesian") if s in args.semirings]:
        for T in args.scan_lengths:
            for impl in impls:
                yield Cell("scan", semiring, "decay+exp", str(args.side_p), T,
                           args.batch, args.heads, args.d_values,
                           args.d_hidden, impl=impl)


def attention_cells(args: argparse.Namespace) -> Iterator[Cell]:
    for T in args.lengths:
        for impl in [i for i in ("eager", "compiled") if i in args.impls]:
            for task in args.tasks:
                yield Cell("attention", "softmax", "", "", T, args.batch,
                           args.heads, args.d_values, args.d_hidden,
                           impl=impl, task=task, mixer="attention")


def all_cells(args: argparse.Namespace) -> list[Cell]:
    makers = {
        "length": length_cells, "rank": rank_cells,
        "settings": settings_cells, "scan": scan_cells,
        "attention": attention_cells,
    }
    cells = [
        c for name in args.experiments for c in makers[name](args)
        if c.mixer != "liss"
        or IMPLEMENTATIONS[c.impl].supports(liss_config(c)) is None
    ]
    order: dict[Cell, int] = {}
    for cell in cells:
        order.setdefault(cell.group, len(order))
    return sorted(cells, key=lambda c: order[c.group])


# --------------------------------------------------------------------------
# measuring one cell
# --------------------------------------------------------------------------
def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _first_line(exc: BaseException) -> str:
    lines = [l for l in str(exc).strip().splitlines() if l.strip()]
    return (lines[0] if lines else type(exc).__name__)[:160]


def build(
    cell: Cell,
    device: torch.device,
    impl: Implementation | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.nn.Module:
    torch.manual_seed(0)
    if cell.mixer == "attention":
        module: torch.nn.Module = SelfAttention(
            AttentionConfig(n_heads=cell.n_is), cell.d_hidden,
        )
    else:
        config = liss_config(cell)
        if impl is not None:
            config = impl.configure(config)
        module = build_liss(config, cell.d_hidden, context_length=cell.T)
    return module.to(device=device, dtype=dtype)


def describe(module: torch.nn.Module, cell: Cell) -> dict:
    """The derived columns of a LISS layer: rank, value width, the deepest
    level, and the work, ``sum over levels and directions of
    p * B * T * N * R * d_v * w``: state entries times scans."""
    out: dict = {"params": sum(p.numel() for p in module.parameters())}
    if cell.mixer != "liss":
        return out
    config = liss_config(cell)
    layers = [module.fw, module.bw] if config.bidirectional else [module]
    level = layers[0].levels[0]
    w = config.d_values if config.values_2D else 1
    out.update(
        rank=level.rank, w=w, p=max(config.lengths),
        normalize=config.normalize, bidirectional=config.bidirectional,
        work=sum(
            lv.p * cell.B * cell.T * lv.n_is * lv.rank * config.d_values * w
            for layer in layers for lv in layer.levels
        ),
    )
    return out


def inputs(
    cell: Cell,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A LayerNorm'd Gaussian input, as the pre-norm hands a mixer, and the
    fixed direction ``r`` of the loss ``sum(y * r)``."""
    g = torch.Generator().manual_seed(1)
    shape = (cell.B, cell.T, cell.d_hidden)
    x = F.layer_norm(torch.randn(shape, generator=g), shape[-1:])
    r = torch.randn(shape, generator=g)
    return x.to(device), r.to(device)


def reference(
    cell: Cell,
    state: dict,
    x: torch.Tensor,
    r: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Output and input gradient of the eager layer in float64, on the
    CPU; ``None`` when they do not fit."""
    try:
        module = build(cell, device, dtype=torch.float64)
        module.load_state_dict(state)
        xx = x.double().requires_grad_()
        y = module(xx)
        (y * r.double()).sum().backward()
        assert xx.grad is not None
        return y.detach().cpu(), xx.grad.cpu()
    except torch.OutOfMemoryError:
        return None
    finally:
        _cleanup(device)


def relative_error(a: torch.Tensor, ref: torch.Tensor) -> float:
    a = a.detach().double().cpu()
    if not torch.isfinite(a).all():
        return math.inf
    return ((a - ref).norm() / ref.norm().clamp_min(1e-300)).item()


def gpu_busy(device: torch.device) -> float | None:
    """Utilisation of the GPU while this process is idle: the load other
    jobs put on it, a marker for noisy timings."""
    if device.type != "cuda":
        return None
    time.sleep(0.25)
    props = torch.cuda.get_device_properties(device)
    bus = (f"{props.pci_domain_id:08X}:{props.pci_bus_id:02X}:"
           f"{props.pci_device_id:02X}.0")
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={bus}", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return float(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


class Runner:
    """One cell, built, compiled, checked and warmed up, ready for timed
    steps. The cells of a timing group are timed in turns, one step each
    per round, so all of them see the same load from other jobs on a
    shared GPU, and their ratios do not drift with it."""

    def __init__(self, cell: Cell, impl: Implementation,
                 device: torch.device) -> None:
        self.cell, self.impl, self.device = cell, impl, device
        self.train = cell.task == "train"
        self.row: dict = {"status": "ok"}
        self.fwd: list[float] = []
        self.bwd: list[float] = []
        self.peak: float | None = None
        self.first = 0.0
        self.module: torch.nn.Module | None = None
        self.run = None
        self.xi: torch.Tensor | None = None
        self.r: torch.Tensor | None = None

    def ok(self) -> bool:
        return self.row["status"] == "ok"

    def done(self, args: argparse.Namespace) -> bool:
        n = len(self.fwd)
        return n >= args.max_iters or (
            n >= args.min_iters
            and sum(self.fwd) + sum(self.bwd) >= args.min_time
        )

    def _clear(self) -> None:
        assert self.module is not None and self.xi is not None
        self.xi.grad = None
        self.module.zero_grad(set_to_none=True)

    def step(self) -> tuple[torch.Tensor, float, float]:
        assert self.run is not None and self.xi is not None
        self._clear()
        with self.impl.patch():
            _sync(self.device)
            t0 = time.perf_counter()
            if not self.train:
                with torch.no_grad():
                    y = self.run(self.xi)
                _sync(self.device)
                return y, time.perf_counter() - t0, 0.0
            y = self.run(self.xi)
            _sync(self.device)
            t1 = time.perf_counter()
            (y * self.r).sum().backward()
            _sync(self.device)
            return y.detach(), t1 - t0, time.perf_counter() - t1

    def timed_step(self) -> None:
        cuda = self.device.type == "cuda"
        self._clear()
        if cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            base = torch.cuda.memory_allocated(self.device)
        y, f, b = self.step()
        del y
        if cuda:
            peak = (torch.cuda.max_memory_allocated(self.device) - base) / 2**20
            self.peak = peak if self.peak is None else max(self.peak, peak)
        self.fwd.append(f)
        self.bwd.append(b)

    def fail(self, exc: BaseException) -> None:
        self.row["status"] = ("OOM" if isinstance(exc, torch.OutOfMemoryError)
                              else "FAIL: " + _first_line(exc))
        self.release()

    def release(self) -> None:
        self.module = self.run = self.xi = self.r = None

    def finish(self) -> dict:
        row = self.row
        if self.ok() and self.fwd:
            total = sorted(f + b for f, b in zip(self.fwd, self.bwd))
            median = total[len(total) // 2]
            fwd, bwd = sorted(self.fwd), sorted(self.bwd)
            row.update(
                fwd_ms=fwd[len(fwd) // 2] * 1e3,
                bwd_ms=bwd[len(bwd) // 2] * 1e3 if self.train else None,
                total_ms=median * 1e3,
                fwd_ms_min=fwd[0] * 1e3,
                bwd_ms_min=bwd[0] * 1e3 if self.train else None,
                total_ms_min=total[0] * 1e3,
                iters=len(total),
                compile_s=(max(self.first - median, 0.0) if self.impl.compile
                           else None),
                peak_mb=self.peak,
            )
        self.release()
        return row


def prepare(
    cell: Cell,
    impl: Implementation,
    state: dict,
    x: torch.Tensor,
    r: torch.Tensor,
    ref: tuple[torch.Tensor, torch.Tensor] | None,
    args: argparse.Namespace,
    device: torch.device,
) -> Runner:
    """Build and compile the cell, take its first step (the compile, and the
    float64 check) and the warm-up steps."""
    runner = Runner(cell, impl, device)
    try:
        with impl.patch():
            module = build(cell, device, impl)
            try:
                module.load_state_dict(state)
            except RuntimeError:
                runner.row["note"] = "own weights (state dict does not load)"
            runner.row.update(describe(module, cell))
            runner.module = module
            runner.run = (torch.compile(module, fullgraph=True, dynamic=False)
                          if impl.compile else module)
        runner.xi = x.detach().requires_grad_(runner.train)
        runner.r = r
        start = time.perf_counter()
        y, _, _ = runner.step()
        runner.first = time.perf_counter() - start
        if ref is not None:
            runner.row["err_out"] = relative_error(y, ref[0])
            if runner.train and runner.xi.grad is not None:
                runner.row["err_grad"] = relative_error(runner.xi.grad, ref[1])
        del y
        for _ in range(args.warmup):
            runner.step()
    except Exception as exc:  # noqa: BLE001  (out of memory included)
        runner.fail(exc)
        _cleanup(device)
    return runner


def timing_key(cell: Cell) -> tuple:
    """The cells timed together, in turns: everything a chart of the report
    divides by something else. A configuration's implementations and tasks
    always; in ``length`` every depth at one length (the depth chart), in
    ``rank`` and ``settings`` every variant of a semiring (each bar is a
    ratio to the base)."""
    if cell.experiment == "length":
        return (cell.experiment, cell.semiring, cell.kernels, cell.T)
    if cell.experiment in ("rank", "settings"):
        return (cell.experiment, cell.semiring)
    return tuple(getattr(cell.group, k) for k in KEY)


def run_timing_group(
    cells: list[Cell],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict]:
    """Prepare every cell (per configuration: one set of weights, one input
    and one float64 reference), then time them all in rounds of one step
    each until each has ``--min-iters`` steps and ``--min-time`` seconds of
    them, or ``--max-iters``. Rows come back in the order of ``cells``."""
    torch._dynamo.reset()
    configs: dict[Cell, list[Cell]] = {}
    for cell in cells:
        configs.setdefault(cell.group, []).append(cell)
    runners: dict[Cell, Runner] = {}
    notes: dict[Cell, str] = {}
    for group, members in configs.items():
        x, r = inputs(group, device)
        canonical = build(group, torch.device("cpu"))
        state = {k: v.clone() for k, v in canonical.state_dict().items()}
        check = not args.no_check and not (
            group.mixer == "attention" and group.T > 4096
        )
        ref = reference(group, state, x, r, device) if check else None
        for cell in members:
            runners[cell] = prepare(cell, IMPLEMENTATIONS[cell.impl], state,
                                    x, r, ref, args, device)
            if check and ref is None:
                notes[cell] = "float64 reference OOM"
        del x, r, ref
    while pending := [rn for rn in runners.values()
                      if rn.ok() and not rn.done(args)]:
        for runner in pending:
            try:
                runner.timed_step()
            except Exception as exc:  # noqa: BLE001
                runner.fail(exc)
                _cleanup(device)
    rows = []
    for cell in cells:
        row = runners[cell].finish()
        row["note"] = row.get("note") or notes.get(cell, "")
        rows.append(row)
    del runners
    torch._dynamo.reset()
    _cleanup(device)
    return rows


# --------------------------------------------------------------------------
# bookkeeping
# --------------------------------------------------------------------------
def _git(path: Path, *command: str) -> str:
    try:
        return subprocess.run(
            ["git", *command], cwd=path, capture_output=True, text=True,
            timeout=10,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def write_meta(path: Path, args: argparse.Namespace, device: torch.device,
               n_cells: int) -> None:
    import elissabeth
    root = Path(elissabeth.__file__).resolve().parents[1]
    meta = json.loads(path.read_text()) if path.exists() else {"sessions": []}
    session = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "elissabeth_commit": _git(root, "rev-parse", "--short", "HEAD"),
        "elissabeth_dirty": bool(_git(root, "status", "--porcelain",
                                      "elissabeth")),
        "cells": n_cells,
        # what is timed in turns; plot_benchmark.py words its method note on it
        "timing": "groups",
        "args": {k: v for k, v in vars(args).items() if k != "out"},
    }
    try:
        import triton
        session["triton"] = triton.__version__
    except ImportError:
        pass
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        session.update(device=props.name,
                       device_memory_gb=props.total_memory / 2**30,
                       cuda=torch.version.cuda)
    else:
        session["device"] = platform.processor() or "cpu"
    meta["sessions"].append(session)
    meta["implementations"] = {
        name: impl.description for name, impl in IMPLEMENTATIONS.items()
    }
    path.write_text(json.dumps(meta, indent=2))


def done_keys(path: Path) -> set[tuple[str, ...]]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {tuple(row[k] for k in KEY) for row in csv.DictReader(f)}


def append(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in FIELDS})


def _fmt(row: dict) -> str:
    if row["status"] != "ok":
        return row["status"]
    parts = [f"{row['total_ms']:.2f} ms"]
    if row.get("peak_mb") is not None:
        parts.append(f"{row['peak_mb']:.0f} MiB")
    if row.get("compile_s") is not None:
        parts.append(f"compile {row['compile_s']:.1f} s")
    if row.get("err_out") is not None:
        parts.append(f"err {row['err_out']:.1e}")
    return ", ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--out", type=Path, required=True,
                        help="CSV to append to (its .json gets the device)")
    parser.add_argument("--experiments", nargs="+", default=EXPERIMENTS,
                        choices=EXPERIMENTS)
    parser.add_argument("--semirings", nargs="+", default=SEMIRINGS,
                        choices=SEMIRINGS)
    parser.add_argument("--impls", nargs="+",
                        default=[i for i in IMPLEMENTATIONS if i != "hillis"],
                        choices=list(IMPLEMENTATIONS),
                        help="implementations of the length experiment (the"
                             " Hillis path has its own, the scan experiment)")
    parser.add_argument("--infer-impls", nargs="+",
                        default=["eager", "compiled"],
                        help="implementations whose forward alone is timed too"
                             " (length experiment)")
    parser.add_argument("--side-impls", nargs="+",
                        default=["eager", "compiled"],
                        choices=list(IMPLEMENTATIONS),
                        help="implementations of the rank and settings"
                             " experiments")
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    parser.add_argument("--depths", nargs="+", type=int, default=DEPTHS)
    parser.add_argument("--lengths", nargs="+", type=int, default=LENGTHS)
    parser.add_argument("--scan-lengths", nargs="+", type=int,
                        default=[1024, 16384, 131072],
                        help="lengths of the scan experiment (an unrolled"
                             " Hillis scan compiles for minutes)")
    parser.add_argument("--side-T", type=int, default=16384,
                        help="length of the rank and settings experiments")
    parser.add_argument("--side-p", type=int, default=3,
                        help="depth of the rank and settings experiments")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8, help="n_is")
    parser.add_argument("--d-values", type=int, default=16)
    parser.add_argument("--d-hidden", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--min-iters", type=int, default=3)
    parser.add_argument("--max-iters", type=int, default=20)
    parser.add_argument("--min-time", type=float, default=1.0,
                        help="seconds of timed steps to collect per cell")
    parser.add_argument("--no-check", action="store_true",
                        help="skip the float64 reference")
    parser.add_argument("--memory-cap-gb", type=float, default=None,
                        help="cap this process's CUDA memory (a shared GPU);"
                             " cells beyond it report OOM")
    parser.add_argument("--matmul-precision", default="highest",
                        choices=["highest", "high", "medium"],
                        help="'highest' keeps TF32 out of the error check;"
                             " train.py uses 'high'")
    parser.add_argument("--cold-compile", action="store_true",
                        help="disable Inductor's caches, so compile_s is a"
                             " cold compile")
    parser.add_argument("--resume", action="store_true",
                        help="skip the cells already in --out")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the cells and exit")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.side_impls = [i for i in args.side_impls if i in IMPLEMENTATIONS]
    args.infer_impls = [i for i in args.infer_impls if i in args.impls]

    cells = all_cells(args)
    done = done_keys(args.out) if args.resume else set()
    todo = [c for c in cells if c.key() not in done]
    if args.out.exists() and not args.resume and not args.dry_run:
        parser.error(f"{args.out} exists; pass --resume to add to it")
    print(f"{len(cells)} cells, {len(todo)} to run, "
          f"{len({c.group for c in todo})} configurations.")
    if args.dry_run:
        for cell in todo:
            print("  ", cell.experiment, cell.variant, cell.semiring,
                  cell.kernels, f"p={cell.lengths}", f"T={cell.T}", cell.impl,
                  cell.task)
        return

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.set_float32_matmul_precision(args.matmul_precision)
    # Up to (implementations x tasks) compiled copies of one forward live
    # side by side in a group; torch < 2.8 calls the limit cache_size_limit.
    import torch._dynamo.config as dynamo_config
    try:
        dynamo_config.recompile_limit = 64
    except AttributeError:
        dynamo_config.cache_size_limit = 64
    if args.cold_compile:
        import torch._inductor.config as inductor_config
        inductor_config.force_disable_caches = True
    if args.memory_cap_gb is not None and device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, args.memory_cap_gb * 2**30 / total), device,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_meta(args.out.with_suffix(".json"), args, device, len(todo))

    groups: dict[tuple, list[Cell]] = {}
    for cell in todo:
        groups.setdefault(timing_key(cell), []).append(cell)
    i = 0
    for members in groups.values():
        busy = gpu_busy(device)
        rows = run_timing_group(members, args, device)
        for cell, row in zip(members, rows):
            i += 1
            row.update({k: getattr(cell, k) for k in KEY}, gpu_busy=busy)
            append(args.out, row)
            print(f"[{i}/{len(todo)}] {cell.experiment:9s} {cell.variant[:20]:20s}"
                  f" {cell.semiring:8s} {cell.kernels:9s} p={cell.lengths:5s}"
                  f" T={cell.T:<6d} {cell.impl:8s} {cell.task:5s} {_fmt(row)}",
                  flush=True)

if __name__ == "__main__":
    main()
