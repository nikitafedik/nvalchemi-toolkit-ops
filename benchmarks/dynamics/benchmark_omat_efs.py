# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark direct OMat model calls on OMat batches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.config import load_yaml_config  # noqa: E402


def _apply_nvalchemiops_path_override() -> str:
    override = os.environ.get("NVALCHEMIOPS_PATH")
    if not override:
        return ""
    override_path = Path(override).expanduser().resolve()
    if not override_path.exists():
        raise FileNotFoundError(f"NVALCHEMIOPS_PATH does not exist: {override_path}")
    sys.path[:] = [path for path in sys.path if path != str(override_path)]
    sys.path.insert(0, str(override_path))
    for module_name in list(sys.modules):
        if module_name == "nvalchemiops" or module_name.startswith("nvalchemiops."):
            sys.modules.pop(module_name, None)
    return str(override_path)


NVALCHEMIOPS_PATH_OVERRIDE = _apply_nvalchemiops_path_override()

from benchmarks.dynamics.model_stacks import (  # noqa: E402
    _build_d3,
    _build_mace,
    _build_model_stack,
    _build_tensornet,
    _dtype_from_config,
    _stack_config,
    _sync_work,
)
from benchmarks.dynamics.direct_tensornet import DirectSingleGradTensorNetModel  # noqa: E402

TIMING_SCOPE = "model_full_forward"
AUTO_TARGET_START = (1_000, 10_000)
AUTO_TARGET_STEP = 10_000

METHOD_ALIASES = {
    "mace_ef": ("mace", {"energy", "forces"}),
    "mace_efs": ("mace", {"energy", "forces", "stress"}),
    "mace_d3_ef": ("mace_d3", {"energy", "forces"}),
    "mace_d3_efs": ("mace_d3", {"energy", "forces", "stress"}),
    "tnet_ef": ("tnet", {"energy", "forces"}),
    "tnet_efs": ("tnet", {"energy", "forces", "stress"}),
    "tnet_d3_ef": ("tnet_d3", {"energy", "forces"}),
    "tnet_d3_efs": ("tnet_d3", {"energy", "forces", "stress"}),
}


def _configure_torch_acceleration() -> None:
    precision = os.environ.get("OMAT_FLOAT32_MATMUL_PRECISION", "high")
    if precision:
        torch.set_float32_matmul_precision(precision)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = os.environ.get(
            "OMAT_ALLOW_TF32",
            "1",
        ).lower() in {"1", "true", "yes", "on"}
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32


def _load_archive(path: Path) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    archive = torch.load(path, map_location="cpu", weights_only=False)
    data = archive["data"]
    segments = archive["segment_lengths"]["atoms"].to(dtype=torch.long)
    starts = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(segments, 0)])
    return data, segments, starts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return ""


def _module_facts(module_name: str) -> dict[str, str]:
    facts = {
        f"{module_name}_module_version": "",
        f"{module_name}_module_file": "",
        f"{module_name}_module_error": "",
    }
    try:
        module = __import__(module_name)
        facts[f"{module_name}_module_version"] = str(
            getattr(module, "__version__", ""),
        )
        module_file = getattr(module, "__file__", "")
        facts[f"{module_name}_module_file"] = str(module_file or "")
    except Exception as exc:
        facts[f"{module_name}_module_error"] = f"{type(exc).__name__}: {exc}"
    return facts


def _choose_segments(
    segments: torch.Tensor,
    target_atoms: int,
    *,
    seed: int,
    sampling_mode: str,
) -> list[int]:
    if sampling_mode == "repeat_largest":
        idx = int(torch.argmax(segments).item())
        n_atoms = int(segments[idx])
        repeats = max(1, math.ceil(target_atoms / n_atoms))
        return [idx] * repeats
    if sampling_mode != "random":
        raise ValueError(f"Unsupported sampling mode: {sampling_mode}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + target_atoms * 17)
    order = torch.randperm(int(segments.numel()), generator=generator).tolist()
    chosen: list[int] = []
    total = 0
    cursor = 0
    while total < target_atoms and cursor < len(order):
        idx = int(order[cursor])
        cursor += 1
        n_atoms = int(segments[idx])
        if total and total + n_atoms > target_atoms:
            continue
        chosen.append(idx)
        total += n_atoms
    if not chosen:
        smallest = int(torch.argmin(segments).item())
        chosen.append(smallest)
    return chosen


def _make_atomic_data(
    data: dict[str, torch.Tensor],
    starts: torch.Tensor,
    idx: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
):
    from nvalchemi.data import AtomicData

    start = int(starts[idx])
    end = int(starts[idx + 1])
    n_atoms = end - start
    charge = data.get("charge")
    return AtomicData(
        positions=data["coord"][start:end].to(device=device, dtype=dtype).clone(),
        atomic_numbers=data["numbers"][start:end].to(device=device, dtype=torch.long),
        charge=(
            charge[idx].to(device=device, dtype=dtype).reshape(1, 1)
            if isinstance(charge, torch.Tensor)
            else None
        ),
        forces=torch.zeros(n_atoms, 3, dtype=dtype, device=device),
        energy=torch.zeros(1, 1, dtype=dtype, device=device),
        velocities=torch.zeros(n_atoms, 3, dtype=dtype, device=device),
        cell=data["cell"][idx].to(device=device, dtype=dtype).reshape(1, 3, 3),
        pbc=torch.ones(1, 3, dtype=torch.bool, device=device),
    )


def _make_batch(
    data: dict[str, torch.Tensor],
    segments: torch.Tensor,
    starts: torch.Tensor,
    target_atoms: int,
    *,
    seed: int,
    sampling_mode: str,
    device: torch.device,
    dtype: torch.dtype,
):
    from nvalchemi.data import Batch

    indices = _choose_segments(
        segments,
        target_atoms,
        seed=seed,
        sampling_mode=sampling_mode,
    )
    data_list = [
        _make_atomic_data(data, starts, idx, device=device, dtype=dtype)
        for idx in indices
    ]
    batch = Batch.from_data_list(data_list, device=device)
    return batch, indices, int(sum(int(segments[idx]) for idx in indices))


def _make_source_batch(
    data: dict[str, torch.Tensor],
    segments: torch.Tensor,
    starts: torch.Tensor,
    *,
    dtype: torch.dtype,
):
    from nvalchemi.data import Batch

    cpu = torch.device("cpu")
    data_list = [
        _make_atomic_data(data, starts, idx, device=cpu, dtype=dtype)
        for idx in range(int(segments.numel()))
    ]
    return Batch.from_data_list(data_list, device=cpu)


def _target_prefix_count(segments: torch.Tensor, target_atoms: int) -> tuple[int, int]:
    total_available = int(segments.sum().item())
    repeats = max(1, math.ceil(target_atoms / total_available))
    systems_to_include = 0
    cumulative_atoms = 0

    for atom_count in segments.tolist() * repeats:
        atom_count = int(atom_count)
        if cumulative_atoms + atom_count <= target_atoms:
            cumulative_atoms += atom_count
            systems_to_include += 1
            continue
        if target_atoms - cumulative_atoms > cumulative_atoms + atom_count - target_atoms:
            cumulative_atoms += atom_count
            systems_to_include += 1
        break

    return max(systems_to_include, 1), cumulative_atoms


def _make_prefix_rollover_batch(
    source_batch,
    segments: torch.Tensor,
    target_atoms: int,
    *,
    device: torch.device,
):
    systems_to_include, total_atoms = _target_prefix_count(segments, target_atoms)
    repeats = max(1, math.ceil(systems_to_include / int(segments.numel())))
    batch = source_batch
    if repeats > 1:
        batch = source_batch.clone()
        for _ in range(repeats - 1):
            batch.append(source_batch.clone())
    subbatch = batch.index_select(slice(0, systems_to_include)).to(device)
    indices = [i % int(segments.numel()) for i in range(systems_to_include)]
    return subbatch, indices, int(total_atoms)


def _target_atom_sequence(explicit_targets: list[int] | None):
    if explicit_targets is not None:
        yield from explicit_targets
        return
    yield from AUTO_TARGET_START
    target = AUTO_TARGET_START[-1] + AUTO_TARGET_STEP
    while True:
        yield target
        target += AUTO_TARGET_STEP


def _build_config(config_path: Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    config.setdefault("model_stacks", {})
    config["model_stacks"]["position_perturbation"] = 0.0
    mace_cfg = config["model_stacks"].setdefault("mace", {})
    if "OMAT_MACE_COMPILE_MODEL" in os.environ:
        mace_cfg["compile_model"] = os.environ["OMAT_MACE_COMPILE_MODEL"].lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    tnet_cfg = config["model_stacks"].setdefault("tensornet", {})
    if "OMAT_TNET_COMPILE_MODEL" in os.environ:
        tnet_cfg["compile_model"] = os.environ["OMAT_TNET_COMPILE_MODEL"].lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    return config


def _parse_outputs(value: str) -> set[str]:
    outputs = {item.strip() for item in value.split(",") if item.strip()}
    allowed = {"energy", "forces", "stress"}
    unknown = outputs - allowed
    if unknown:
        raise ValueError(f"Unsupported outputs {sorted(unknown)}; expected subset of {sorted(allowed)}.")
    if "energy" not in outputs or "forces" not in outputs:
        raise ValueError("OMat benchmark outputs must include energy and forces.")
    return outputs


def _resolve_methods_and_outputs(
    methods: list[str],
    cli_outputs: set[str] | None,
) -> tuple[list[str], set[str], dict[str, str]]:
    resolved: list[str] = []
    alias_outputs: set[str] | None = None
    labels: dict[str, str] = {}
    for method in methods:
        if method in METHOD_ALIASES:
            resolved_method, method_outputs = METHOD_ALIASES[method]
            resolved.append(resolved_method)
            labels[resolved_method] = method
            if alias_outputs is None:
                alias_outputs = set(method_outputs)
            elif alias_outputs != method_outputs:
                raise ValueError("Cannot mix EF and EFS method aliases in one benchmark invocation.")
        else:
            resolved.append(method)
            labels[method] = method

    outputs = alias_outputs if alias_outputs is not None else cli_outputs
    if alias_outputs is not None and cli_outputs is not None and cli_outputs != alias_outputs:
        raise ValueError(
            f"Method alias requests outputs={','.join(sorted(alias_outputs))}, "
            f"but --outputs requested {','.join(sorted(cli_outputs))}."
        )
    if outputs is None:
        outputs = {"energy", "forces", "stress"}
    return resolved, set(outputs), labels


def _model_config(model: Any) -> Any | None:
    return getattr(model, "model_config", None)


def _supported_outputs(model: Any) -> set[str]:
    config = _model_config(model)
    value = getattr(config, "outputs", None) if config is not None else None
    return set(value) if value is not None else set()


def _active_outputs(model: Any) -> set[str]:
    config = _model_config(model)
    value = getattr(config, "active_outputs", None) if config is not None else None
    return set(value) if value is not None else set()


def _iter_pipeline_inner_models(model: torch.nn.Module):
    inner_models = getattr(model, "benchmark_inner_models", None)
    if inner_models is not None:
        yield from inner_models
        return
    for group in getattr(model, "groups", []):
        for step in getattr(group, "steps", []):
            yield getattr(step, "model", step)


def _set_active_outputs(model: Any, requested: set[str]) -> None:
    config = _model_config(model)
    if config is None:
        return
    supported = _supported_outputs(model)
    active = requested if not supported else requested & supported
    if active:
        config.active_outputs = set(active)


def _validate_active_outputs(
    model: torch.nn.Module,
    requested: set[str],
) -> dict[str, set[str]]:
    inner = list(_iter_pipeline_inner_models(model))
    if not inner:
        inner = [model]

    active_by_type: dict[str, set[str]] = {}
    failures: list[str] = []
    for inner_model in inner:
        supported = _supported_outputs(inner_model)
        if not supported:
            continue
        required = requested & supported
        active = _active_outputs(inner_model)
        active_by_type[type(inner_model).__name__] = active
        missing = required - active
        if missing:
            failures.append(
                f"{type(inner_model).__name__} missing active outputs {sorted(missing)} "
                f"from requested {sorted(requested)}"
            )
    if failures:
        raise RuntimeError("; ".join(failures))
    return active_by_type


def _set_outputs(model: torch.nn.Module, outputs: set[str]) -> tuple[str, str, bool]:
    requested = set(outputs)
    _set_active_outputs(model, requested)
    for inner_model in _iter_pipeline_inner_models(model):
        _set_active_outputs(inner_model, requested)
    active_by_type = _validate_active_outputs(model, requested)
    inner_outputs = ";".join(
        f"{name}:{','.join(sorted(active))}" for name, active in sorted(active_by_type.items())
    )
    return ",".join(sorted(requested)), inner_outputs, "stress" in requested


def _tensornet_compile_enabled(model: torch.nn.Module) -> bool:
    if bool(getattr(model, "tnet_compile_enabled", False)):
        return True
    for inner_model in _iter_pipeline_inner_models(model):
        if bool(getattr(inner_model, "tnet_compile_enabled", False)):
            return True
        wrapped = getattr(inner_model, "model", None)
        if bool(getattr(wrapped, "tnet_compile_enabled", False)):
            return True
    primary = getattr(model, "primary", None)
    return bool(primary is not None and _tensornet_compile_enabled(primary))


def _tensornet_trainable_parameter_count(model: torch.nn.Module) -> int:
    for inner_model in _iter_pipeline_inner_models(model):
        wrapped = getattr(inner_model, "model", None)
        if wrapped is not None and hasattr(wrapped, "parameters"):
            return sum(1 for param in wrapped.parameters() if param.requires_grad)
    primary = getattr(model, "primary", None)
    if primary is not None:
        return _tensornet_trainable_parameter_count(primary)
    wrapped = getattr(model, "model", None)
    if wrapped is not None and hasattr(wrapped, "parameters"):
        return sum(1 for param in wrapped.parameters() if param.requires_grad)
    return 0


def _suggest_neighbor_list_method_for_batch(
    batch,
    *,
    cutoff: float,
) -> tuple[str, str]:
    try:
        from nvalchemiops.torch.neighbors import suggest_neighbor_list_method
    except ImportError:
        suggest_neighbor_list_method = None

    if suggest_neighbor_list_method is not None:
        return (
            suggest_neighbor_list_method(
                batch.batch_ptr,
                getattr(batch, "cell", None),
                getattr(batch, "pbc", None),
                cutoff=cutoff,
            ),
            "suggest_neighbor_list_method",
        )

    avg_atoms = batch.num_nodes // max(batch.num_graphs, 1)
    method = "batch_cell_list" if avg_atoms >= 2000 else "batch_naive"
    return method, "legacy_auto_dispatch"


def _clear_batch_neighbor_data(batch) -> None:
    atoms_group = getattr(batch, "_atoms_group", None)
    if atoms_group is not None:
        for key in ("neighbor_matrix", "num_neighbors", "neighbor_matrix_shifts"):
            if key in atoms_group:
                atoms_group.pop(key)
    storage = getattr(batch, "_storage", None)
    if storage is not None:
        storage.groups.pop("edges", None)
    batch.__dict__.pop("_neighbor_list_cutoff", None)


class _FixedMethodNeighborListHook:
    def __init__(self, config, *, method: str) -> None:
        from nvalchemi.hooks import NeighborListHook

        class FixedMethodHook(NeighborListHook):
            def __init__(self, *args, fixed_method: str, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.fixed_method = fixed_method

            def _alloc_nl_kwargs(self, *args, **kwargs) -> None:
                super()._alloc_nl_kwargs(*args, **kwargs)
                self._buf_nl_kwargs["method"] = self.fixed_method

        self.hook = FixedMethodHook(
            config,
            skin=config.skin,
            fixed_method=method,
        )

    def rebuild(self, batch) -> None:
        try:
            self.hook._rebuild(batch)
        except TypeError as exc:
            if "shift_range_per_dimension" in str(exc):
                self.hook._buf_nl_kwargs = {"method": self.hook.fixed_method}
            elif "max_atoms_per_system" in str(exc):
                self.hook._buf_nl_kwargs.pop("max_atoms_per_system", None)
            else:
                raise
            self.hook._rebuild(batch)


class _SeparateNeighborPrimaryD3Model(torch.nn.Module):
    def __init__(
        self,
        primary: torch.nn.Module,
        d3: torch.nn.Module,
        *,
        primary_name: str,
    ) -> None:
        super().__init__()
        from nvalchemi.models.base import ModelConfig

        self.primary = primary
        self.primary_name = primary_name
        self.d3 = d3
        self.benchmark_inner_models = [self.primary, self.d3]
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "stress"}),
            active_outputs={"energy", "forces"},
            supports_pbc=True,
            needs_pbc=False,
        )
        self.primary_neighbor_method = ""
        self.mace_neighbor_method = ""
        self.tnet_neighbor_method = ""
        self.d3_neighbor_method = ""
        self.neighbor_method_source = ""
        self._primary_hook: _FixedMethodNeighborListHook | None = None
        self._d3_hook: _FixedMethodNeighborListHook | None = None
        self._prepared_key: tuple[int, int, str, str] | None = None

    def make_neighbor_hooks(self) -> list[Any]:
        return []

    def prepare_benchmark_batch(self, batch) -> None:
        key = (
            int(batch.num_nodes),
            int(batch.num_graphs),
            str(batch.positions.device),
            str(batch.positions.dtype),
        )
        if key == self._prepared_key:
            return

        primary_nc = self.primary.model_config.neighbor_config
        d3_nc = self.d3.model_config.neighbor_config
        primary_method, primary_source = _suggest_neighbor_list_method_for_batch(
            batch,
            cutoff=float(primary_nc.cutoff),
        )
        d3_method, d3_source = _suggest_neighbor_list_method_for_batch(
            batch,
            cutoff=float(d3_nc.cutoff),
        )
        self.primary_neighbor_method = primary_method
        if self.primary_name == "mace":
            self.mace_neighbor_method = primary_method
        elif self.primary_name == "tnet":
            self.tnet_neighbor_method = primary_method
        self.d3_neighbor_method = d3_method
        self.neighbor_method_source = (
            primary_source
            if primary_source == d3_source
            else f"{self.primary_name}:{primary_source};d3:{d3_source}"
        )
        self._primary_hook = _FixedMethodNeighborListHook(
            primary_nc,
            method=primary_method,
        )
        self._d3_hook = _FixedMethodNeighborListHook(d3_nc, method=d3_method)
        self._prepared_key = key

    def forward(self, batch) -> dict[str, torch.Tensor]:
        self.prepare_benchmark_batch(batch)
        if self._primary_hook is None or self._d3_hook is None:
            raise RuntimeError("Separate primary+D3 neighbor hooks were not initialized.")

        _clear_batch_neighbor_data(batch)
        self._d3_hook.rebuild(batch)
        with torch.no_grad():
            d3_out = self.d3(batch)

        _clear_batch_neighbor_data(batch)
        self._primary_hook.rebuild(batch)
        primary_out = self.primary(batch)
        _clear_batch_neighbor_data(batch)

        out: dict[str, torch.Tensor] = {}
        for key in self.model_config.active_outputs:
            primary_value = primary_out.get(key)
            d3_value = d3_out.get(key)
            if primary_value is None and d3_value is None:
                continue
            if primary_value is None:
                out[key] = d3_value
            elif d3_value is None:
                out[key] = primary_value
            else:
                out[key] = primary_value + d3_value
        return out


class _FixedNeighborPrimaryModel(torch.nn.Module):
    def __init__(self, primary: torch.nn.Module, *, primary_name: str) -> None:
        super().__init__()
        self.primary = primary
        self.primary_name = primary_name
        self.benchmark_inner_models = [self.primary]
        self.model_config = primary.model_config
        self.primary_neighbor_method = ""
        self.mace_neighbor_method = ""
        self.tnet_neighbor_method = ""
        self.d3_neighbor_method = ""
        self.neighbor_method_source = ""
        self._primary_hook: _FixedMethodNeighborListHook | None = None
        self._prepared_key: tuple[int, int, str, str] | None = None

    def make_neighbor_hooks(self) -> list[Any]:
        return []

    def prepare_benchmark_batch(self, batch) -> None:
        key = (
            int(batch.num_nodes),
            int(batch.num_graphs),
            str(batch.positions.device),
            str(batch.positions.dtype),
        )
        if key == self._prepared_key:
            return

        primary_nc = self.primary.model_config.neighbor_config
        primary_method, primary_source = _suggest_neighbor_list_method_for_batch(
            batch,
            cutoff=float(primary_nc.cutoff),
        )
        self.primary_neighbor_method = primary_method
        if self.primary_name == "mace":
            self.mace_neighbor_method = primary_method
        elif self.primary_name == "tnet":
            self.tnet_neighbor_method = primary_method
        self.neighbor_method_source = primary_source
        self._primary_hook = _FixedMethodNeighborListHook(
            primary_nc,
            method=primary_method,
        )
        self._prepared_key = key

    def forward(self, batch):
        self.prepare_benchmark_batch(batch)
        if self._primary_hook is None:
            raise RuntimeError("Fixed primary neighbor hook was not initialized.")

        _clear_batch_neighbor_data(batch)
        self._primary_hook.rebuild(batch)
        out = self.primary(batch)
        _clear_batch_neighbor_data(batch)
        return out


def _build_separate_neighbor_mace_d3_model(
    config: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    stack = _stack_config(config)
    mace = _build_mace(stack, device, dtype)
    d3 = _build_d3(stack).to(device)
    model = _SeparateNeighborPrimaryD3Model(mace, d3, primary_name="mace")
    model.to(device)
    model.eval()
    return model


def _build_separate_neighbor_tnet_d3_model(
    config: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    stack = _stack_config(config)
    tnet = DirectSingleGradTensorNetModel(_build_tensornet(stack, device))
    d3 = _build_d3(stack).to(device)
    model = _SeparateNeighborPrimaryD3Model(tnet, d3, primary_name="tnet")
    model.to(device)
    model.eval()
    return model


def _make_neighbor_hooks(model: torch.nn.Module):
    make_hooks = getattr(model, "make_neighbor_hooks", None)
    return list(make_hooks()) if make_hooks is not None else []


def _rebuild_neighbors(model: torch.nn.Module, batch) -> None:
    for hook in _make_neighbor_hooks(model):
        rebuild = getattr(hook, "_rebuild", None)
        if rebuild is None:
            continue
        try:
            rebuild(batch)
        except TypeError as exc:
            if "shift_range_per_dimension" in str(exc):
                hook._buf_nl_kwargs = {}
            elif "max_atoms_per_system" in str(exc):
                hook._buf_nl_kwargs.pop("max_atoms_per_system", None)
            else:
                raise
            rebuild(batch)


def _call_model_with_neighbors(model: torch.nn.Module, batch) -> None:
    _rebuild_neighbors(model, batch)
    model(batch)


def _time_model(
    model: torch.nn.Module,
    batch,
    *,
    warmup_runs: int,
    timing_runs: int,
    device: torch.device,
) -> float:
    prepare_benchmark_batch = getattr(model, "prepare_benchmark_batch", None)
    if prepare_benchmark_batch is not None:
        prepare_benchmark_batch(batch)

    for _ in range(warmup_runs):
        _call_model_with_neighbors(model, batch)
    _sync_work(device)
    start = time.perf_counter()
    for _ in range(timing_runs):
        _call_model_with_neighbors(model, batch)
    _sync_work(device)
    return (time.perf_counter() - start) * 1000.0 / max(timing_runs, 1)


def _checkpoint_facts(config: dict[str, Any]) -> dict[str, str]:
    stack = config.get("model_stacks", {})
    mace_cfg = stack.get("mace", {}) if isinstance(stack, dict) else {}
    checkpoint = str(mace_cfg.get("checkpoint", "medium-mpa-0"))
    facts = {
        "model_checkpoint": checkpoint,
        "model_checkpoint_path": "",
        "model_checkpoint_sha256": "",
        "model_checkpoint_error": "",
    }
    checkpoint_path = Path(checkpoint).expanduser()
    try:
        if checkpoint_path.exists():
            facts["model_checkpoint_path"] = str(checkpoint_path.resolve())
            facts["model_checkpoint_sha256"] = _sha256(checkpoint_path)
        else:
            from mace.calculators.foundations_models import download_mace_mp_checkpoint

            resolved = Path(download_mace_mp_checkpoint(checkpoint)).expanduser()
            facts["model_checkpoint_path"] = str(resolved.resolve())
            facts["model_checkpoint_sha256"] = _sha256(resolved)
    except Exception as exc:
        facts["model_checkpoint_error"] = f"{type(exc).__name__}: {exc}"
    return facts


def _audit_facts(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    dtype: torch.dtype,
    methods: list[str],
    outputs: set[str],
) -> dict[str, Any]:
    stack = config.get("model_stacks", {})
    tnet_cfg = stack.get("tensornet", {}) if isinstance(stack, dict) else {}
    return {
        "dataset_path": str(args.dataset.resolve()),
        "dataset_sha256": _sha256(args.dataset),
        "nvalchemiops_path_override": os.environ.get("NVALCHEMIOPS_PATH", ""),
        "nvalchemiops_path_override_resolved": NVALCHEMIOPS_PATH_OVERRIDE,
        **_checkpoint_facts(config),
        **_module_facts("nvalchemi"),
        **_module_facts("nvalchemiops"),
        **_module_facts("matgl"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "mace_version": _version("mace-torch"),
        "cuequivariance_version": _version("cuequivariance"),
        "cuequivariance_ops_torch_version": (
            _version("cuequivariance-ops-torch-cu13")
            or _version("cuequivariance-ops-torch-cu12")
        ),
        "nvalchemi_toolkit_version": _version("nvalchemi-toolkit"),
        "nvalchemi_toolkit_ops_version": _version("nvalchemi-toolkit-ops"),
        "tensornet_model_name": str(tnet_cfg.get("model_name", "")),
        "tensornet_require_warp": bool(tnet_cfg.get("require_warp", True)),
        "tensornet_compile_requested": bool(tnet_cfg.get("compile_model", False)),
        "tensornet_compile_kwargs": dict(tnet_cfg.get("compile_kwargs", {}) or {}),
        "dtype": str(dtype).removeprefix("torch."),
        "mace_compile_requested": bool(
            config.get("model_stacks", {}).get("mace", {}).get("compile_model", False)
        ),
        "compile_enabled": bool(
            config.get("model_stacks", {}).get("mace", {}).get("compile_model", False)
            or tnet_cfg.get("compile_model", False)
        ),
        "stress_enabled": "stress" in outputs,
        "d3_enabled": any("d3" in method for method in methods),
        "methods": methods,
        "outputs": sorted(outputs),
        "sampling_mode": args.sampling_mode,
        "timing_scope": TIMING_SCOPE,
        "neighbor_list_included_in_timing": True,
        "warmup_runs": args.warmup_runs,
        "timing_runs": args.timing_runs,
    }


def _failure_row(
    *,
    method: str,
    resolved_method: str,
    target_atoms: int,
    error: Exception,
    outputs: str,
    inner_outputs: str,
    stress_enabled: bool,
    sampling_mode: str,
    tnet_compile_enabled: bool = False,
    tnet_trainable_parameters: int = 0,
) -> dict[str, Any]:
    return {
        "success": False,
        "method": method,
        "resolved_method": resolved_method,
        "sampling_mode": sampling_mode,
        "timing_scope": TIMING_SCOPE,
        "target_atoms": target_atoms,
        "total_atoms": math.nan,
        "num_graphs": math.nan,
        "avg_atoms_per_graph": math.nan,
        "avg_call_time_ms": math.nan,
        "time_us_per_atom_call": math.nan,
        "throughput_atom_calls_per_s": math.nan,
        "outputs": outputs,
        "inner_outputs": inner_outputs,
        "stress_enabled": stress_enabled,
        "tnet_compile_enabled": tnet_compile_enabled,
        "tnet_trainable_parameters": tnet_trainable_parameters,
        "primary_neighbor_method": "",
        "mace_neighbor_method": "",
        "tnet_neighbor_method": "",
        "d3_neighbor_method": "",
        "neighbor_method_source": "",
        "error_type": type(error).__name__,
        "error": str(error),
    }


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    _configure_torch_acceleration()
    config = _build_config(args.config)
    dtype = _dtype_from_config(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data, segments, starts = _load_archive(args.dataset)
    rows: list[dict[str, Any]] = []
    source_batch = None
    requested_outputs = _parse_outputs(args.outputs) if args.outputs is not None else None
    args.methods, requested_outputs, method_labels = _resolve_methods_and_outputs(
        args.methods,
        requested_outputs,
    )

    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print(f"dataset systems={int(segments.numel())} atoms={int(segments.sum())}")
    if args.target_atoms is None:
        print("targets=auto[1000, 10000, +10000 until OOM]")
    else:
        print(f"targets={args.target_atoms}")
    print(f"methods={args.methods}")
    print(f"sampling_mode={args.sampling_mode}")
    print(f"timing_scope={TIMING_SCOPE}")
    if args.sampling_mode == "prefix_rollover":
        source_batch = _make_source_batch(data, segments, starts, dtype=dtype)

    for method in args.methods:
        if method == "mace_d3":
            model = _build_separate_neighbor_mace_d3_model(config, device, dtype)
        elif method == "tnet":
            model = _FixedNeighborPrimaryModel(
                DirectSingleGradTensorNetModel(
                    _build_tensornet(_stack_config(config), device)
                ),
                primary_name="tnet",
            )
            model.to(device)
            model.eval()
        elif method == "tnet_d3":
            model = _build_separate_neighbor_tnet_d3_model(
                config,
                device,
            )
        else:
            model = _build_model_stack(method, config, device, dtype)
        outputs, inner_outputs, stress_enabled = _set_outputs(model, requested_outputs)
        for target_atoms in _target_atom_sequence(args.target_atoms):
            batch = None
            try:
                if args.sampling_mode == "prefix_rollover":
                    batch, indices, total_atoms = _make_prefix_rollover_batch(
                        source_batch,
                        segments,
                        int(target_atoms),
                        device=device,
                    )
                else:
                    batch, indices, total_atoms = _make_batch(
                        data,
                        segments,
                        starts,
                        int(target_atoms),
                        seed=args.seed,
                        sampling_mode=args.sampling_mode,
                        device=device,
                        dtype=dtype,
                    )
                avg_ms = _time_model(
                    model,
                    batch,
                    warmup_runs=args.warmup_runs,
                    timing_runs=args.timing_runs,
                    device=device,
                )
                row = {
                    "success": True,
                    "method": method_labels.get(method, method),
                    "resolved_method": method,
                    "sampling_mode": args.sampling_mode,
                    "timing_scope": TIMING_SCOPE,
                    "target_atoms": int(target_atoms),
                    "total_atoms": int(total_atoms),
                    "num_graphs": len(indices),
                    "avg_atoms_per_graph": total_atoms / max(len(indices), 1),
                    "avg_call_time_ms": avg_ms,
                    "time_us_per_atom_call": avg_ms * 1000.0 / max(total_atoms, 1),
                    "throughput_atom_calls_per_s": total_atoms / (avg_ms / 1000.0),
                    "outputs": outputs,
                    "inner_outputs": inner_outputs,
                    "stress_enabled": stress_enabled,
                    "tnet_compile_enabled": _tensornet_compile_enabled(model) if method.startswith("tnet") else False,
                    "tnet_trainable_parameters": _tensornet_trainable_parameter_count(model) if method.startswith("tnet") else 0,
                    "primary_neighbor_method": getattr(model, "primary_neighbor_method", ""),
                    "mace_neighbor_method": getattr(model, "mace_neighbor_method", ""),
                    "tnet_neighbor_method": getattr(model, "tnet_neighbor_method", ""),
                    "d3_neighbor_method": getattr(model, "d3_neighbor_method", ""),
                    "neighbor_method_source": getattr(model, "neighbor_method_source", ""),
                    "error_type": "",
                    "error": "",
                }
                print(
                    f"  {method} target={target_atoms} total={total_atoms} "
                    f"graphs={len(indices)}: {avg_ms:.4f} ms/call"
                )
            except torch.cuda.OutOfMemoryError as exc:
                row = _failure_row(
                    method=method_labels.get(method, method),
                    resolved_method=method,
                    target_atoms=int(target_atoms),
                    error=exc,
                    outputs=outputs,
                    inner_outputs=inner_outputs,
                    stress_enabled=stress_enabled,
                    sampling_mode=args.sampling_mode,
                    tnet_compile_enabled=_tensornet_compile_enabled(model) if method.startswith("tnet") else False,
                    tnet_trainable_parameters=_tensornet_trainable_parameter_count(model) if method.startswith("tnet") else 0,
                )
                print(f"  {method} target={target_atoms}: OOM")
                rows.append(row)
                break
            except Exception as exc:
                row = _failure_row(
                    method=method_labels.get(method, method),
                    resolved_method=method,
                    target_atoms=int(target_atoms),
                    error=exc,
                    outputs=outputs,
                    inner_outputs=inner_outputs,
                    stress_enabled=stress_enabled,
                    sampling_mode=args.sampling_mode,
                    tnet_compile_enabled=_tensornet_compile_enabled(model) if method.startswith("tnet") else False,
                    tnet_trainable_parameters=_tensornet_trainable_parameter_count(model) if method.startswith("tnet") else 0,
                )
                print(f"  {method} target={target_atoms}: FAILED - {exc}")
                rows.append(row)
                break
            finally:
                del batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            rows.append(row)
    if not args.no_audit_json:
        audit_path = args.audit_json or args.output_csv.with_suffix(".audit.json")
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("w") as handle:
            json.dump(
                _audit_facts(
                    args=args,
                    config=config,
                    dtype=dtype,
                    methods=args.methods,
                    outputs=requested_outputs,
                ),
                handle,
                indent=2,
                sort_keys=True,
            )
        print(f"Saved audit metadata to {audit_path}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--method", dest="methods", nargs="+", required=True)
    parser.add_argument("--target-atoms", type=int, nargs="+", default=None)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--timing-runs", type=int, default=6)
    parser.add_argument("--outputs")
    parser.add_argument(
        "--sampling-mode",
        choices=["random", "repeat_largest", "prefix_rollover"],
        default="prefix_rollover",
    )
    parser.add_argument("--audit-json", type=Path)
    parser.add_argument("--no-audit-json", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = run(args)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
