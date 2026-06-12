#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Toolkit model-stack dynamics benchmark helpers.

The unified dynamics runner is primarily a native Lennard-Jones integrator
benchmark. This module adds optional neural-potential stack methods without
moving or modifying any nvalchemi-toolkit APIs: MACE/TensorNet, DFT-D3, PME,
and Ewald are composed through Toolkit's public model wrappers and
``PipelineModelWrapper``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from benchmarks.dynamics.shared_utils import BenchmarkResult
from benchmarks.systems import (
    configs_for_mode,
    create_system,
    filter_configs_by_total_atoms,
    omat_dataset_system_count,
    omat_prefix_rollover_counts,
    planned_atom_counts,
    resolve_nh3_dir,
)

__all__ = [
    "MODEL_STACK_METHODS",
    "MODEL_STACK_METHOD_FAMILIES",
    "is_model_stack_method",
    "planned_model_stack_rows",
    "run_model_stack_case",
]

_MODEL_STACK_DEFINITIONS = {
    "tnet": {"stack": "tnet", "engine": "nvt", "family": "model_md"},
    "mace": {"stack": "mace", "engine": "nvt", "family": "model_md"},
    "tnet_d3": {"stack": "tnet_d3", "engine": "nvt", "family": "model_md"},
    "mace_d3": {"stack": "mace_d3", "engine": "nvt", "family": "model_md"},
    "mace_d3_pme": {
        "stack": "mace_d3_pme",
        "engine": "nvt",
        "family": "model_md",
    },
    "mace_d3_ewald": {
        "stack": "mace_d3_ewald",
        "engine": "nvt",
        "family": "model_md",
    },
    "tnet_fire": {
        "stack": "tnet",
        "engine": "fire",
        "family": "model_opt",
    },
    "tnet_fire2": {
        "stack": "tnet",
        "engine": "fire2",
        "family": "model_opt",
    },
    "mace_fire": {
        "stack": "mace",
        "engine": "fire",
        "family": "model_opt",
    },
    "mace_fire2": {
        "stack": "mace",
        "engine": "fire2",
        "family": "model_opt",
    },
    "tnet_d3_fire": {
        "stack": "tnet_d3",
        "engine": "fire",
        "family": "model_opt",
    },
    "tnet_d3_fire2": {
        "stack": "tnet_d3",
        "engine": "fire2",
        "family": "model_opt",
    },
    "mace_d3_fire": {
        "stack": "mace_d3",
        "engine": "fire",
        "family": "model_opt",
    },
    "mace_d3_fire2": {
        "stack": "mace_d3",
        "engine": "fire2",
        "family": "model_opt",
    },
    "mace_d3_pme_fire": {
        "stack": "mace_d3_pme",
        "engine": "fire",
        "family": "model_opt",
    },
    "mace_d3_pme_fire2": {
        "stack": "mace_d3_pme",
        "engine": "fire2",
        "family": "model_opt",
    },
    "mace_d3_ewald_fire": {
        "stack": "mace_d3_ewald",
        "engine": "fire",
        "family": "model_opt",
    },
    "mace_d3_ewald_fire2": {
        "stack": "mace_d3_ewald",
        "engine": "fire2",
        "family": "model_opt",
    },
}

MODEL_STACK_METHODS = frozenset(_MODEL_STACK_DEFINITIONS)
MODEL_STACK_METHOD_FAMILIES = {
    name: definition["family"] for name, definition in _MODEL_STACK_DEFINITIONS.items()
}

_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
}
_REPO_ROOT = Path(__file__).resolve().parents[2]
_OMAT_SOURCE_BATCH_CACHE: dict[tuple[str, str], Any] = {}


def is_model_stack_method(method: str) -> bool:
    """Return whether *method* is handled by this optional model-stack runner."""
    return method in MODEL_STACK_METHODS


def _method_definition(method: str) -> dict[str, str]:
    """Return the model-stack method definition for *method*."""
    try:
        return _MODEL_STACK_DEFINITIONS[method]
    except KeyError:
        raise ValueError(f"Unsupported model-stack dynamics method: {method}") from None


def _method_engine(method: str) -> str:
    """Return the benchmark engine for one model-stack method."""
    return _method_definition(method)["engine"]


def _method_stack_name(method: str) -> str:
    """Return the composed model stack name for one model-stack method."""
    return _method_definition(method)["stack"]


def _method_family(method: str) -> str:
    """Return the CSV method-family label for one model-stack method."""
    return _method_definition(method)["family"]


def _stack_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the optional ``model_stacks`` YAML subtree."""
    value = config.get("model_stacks", {})
    return value if isinstance(value, dict) else {}


def _method_config(config: dict[str, Any], method: str) -> dict[str, Any]:
    """Return the top-level YAML method entry for *method*."""
    for entry in config.get("methods", []):
        if entry.get("name") == method:
            return entry
    return {}


def _torch_dtype(value: str) -> torch.dtype:
    """Resolve a torch dtype token from YAML."""
    try:
        return _DTYPES[value]
    except KeyError:
        raise ValueError(
            f"Unsupported model-stack dtype {value!r}; expected one of {sorted(_DTYPES)}."
        ) from None


def _dtype_from_config(config: dict[str, Any]) -> torch.dtype:
    """Resolve the model-stack tensor dtype."""
    stack = _stack_config(config)
    value = str(
        stack.get(
            "dtype",
            config.get("parameters", {}).get("dtype", "float32"),
        )
    )
    return _torch_dtype(value)


def _method_steps(config: dict[str, Any], method: str) -> int:
    """Resolve timed NVT steps with the shared timing override semantics."""
    params = config.get("parameters", {})
    method_cfg = _method_config(config, method)
    if params.get("timing_runs") is not None:
        return int(params["timing_runs"])
    stack = _stack_config(config)
    if _method_engine(method) == "nvt":
        defaults = stack.get("integrator", {})
        return int(method_cfg.get("steps", defaults.get("steps", 20)))
    defaults = stack.get("optimizer", {})
    return int(
        method_cfg.get("max_steps", method_cfg.get("steps", defaults.get("steps", 20)))
    )


def _method_warmup(config: dict[str, Any], method: str) -> int:
    """Resolve warmup NVT steps with the shared warmup override semantics."""
    params = config.get("parameters", {})
    method_cfg = _method_config(config, method)
    if params.get("warmup_runs") is not None:
        return int(params["warmup_runs"])
    stack = _stack_config(config)
    if _method_engine(method) == "nvt":
        defaults = stack.get("integrator", {})
        return int(method_cfg.get("warmup_steps", defaults.get("warmup_steps", 10)))
    defaults = stack.get("optimizer", {})
    return int(method_cfg.get("warmup_steps", defaults.get("warmup_steps", 1)))


def _planned_row(
    *,
    backend: str,
    system_name: str,
    mode_name: str,
    method: str,
    case_config: dict[str, Any],
    atoms_per_system: int,
    batch_size: int,
    total_atoms: int,
    steps: int,
    warmup_steps: int,
    reason: str = "",
) -> dict[str, Any]:
    """Build one allocation-free model-stack plan row."""
    return {
        "benchmark": "dyn",
        "backend": backend,
        "system": system_name,
        "mode": mode_name,
        "scaling_mode": mode_name,
        "method": method,
        "method_family": _method_family(method),
        "engine": _method_engine(method),
        "target_atoms": int(
            case_config.get("target_atoms", case_config["num_atoms"])
        ),
        "atoms_per_system": atoms_per_system,
        "batch_size": batch_size,
        "total_atoms": total_atoms,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "pdb_path": str(case_config["pdb_path"]) if case_config.get("pdb_path") else "",
        "dataset_path": str(case_config["dataset_path"])
        if case_config.get("dataset_path")
        else "",
        "segment_index": case_config.get("segment_index", ""),
        "batch_construction": case_config.get("batch_construction", "repeat_segment"),
        "reason": reason,
    }


def _unavailable_system_rows(
    *,
    backend: str,
    system_name: str,
    mode_name: str,
    methods: list[str],
    config: dict[str, Any],
    reason: str,
) -> list[dict[str, Any]]:
    """Build explicit skip rows for unavailable optional model-stack systems."""
    return [
        {
            "benchmark": "dyn",
            "backend": backend,
            "system": system_name,
            "mode": mode_name,
            "scaling_mode": mode_name,
            "method": method,
            "method_family": _method_family(method),
            "engine": _method_engine(method),
            "target_atoms": 0,
            "atoms_per_system": 0,
            "batch_size": 0,
            "total_atoms": 0,
            "steps": _method_steps(config, method),
            "warmup_steps": _method_warmup(config, method),
            "pdb_path": "",
            "dataset_path": "",
            "segment_index": "",
            "batch_construction": "",
            "reason": reason,
        }
        for method in methods
    ]


def planned_model_stack_rows(
    config: dict[str, Any],
    backend: str,
    methods: list[str],
) -> list[dict[str, Any]]:
    """Build allocation-free rows for optional Toolkit model-stack methods.

    Parameters
    ----------
    config : dict
        Loaded dynamics YAML configuration.
    backend : str
        Runtime backend. Model-stack dynamics currently uses torch only.
    methods : list[str]
        Enabled model-stack method names, in suite order.

    Returns
    -------
    list[dict]
        Planned rows. Rows over ``max_total_atoms`` are included with a skip
        reason so dry-runs and CSV outputs remain explicit.
    """
    stack = _stack_config(config)
    if not stack.get("enabled", True):
        return []

    max_total_atoms = config.get("parameters", {}).get("max_total_atoms")
    rows: list[dict[str, Any]] = []
    for system_name, system_config in stack.get("systems", {}).items():
        if not system_config.get("enabled", True):
            continue
        nh3_dir = resolve_nh3_dir(system_config) if system_name == "nh3" else None
        for mode_name, mode_config in stack.get("scaling", {}).items():
            if not isinstance(mode_config, dict) or not mode_config.get(
                "enabled", True
            ):
                continue
            try:
                case_configs = configs_for_mode(
                    mode_name,
                    mode_config,
                    system_name,
                    system_config,
                    nh3_dir=nh3_dir,
                )
            except FileNotFoundError as exc:
                rows.extend(
                    _unavailable_system_rows(
                        backend=backend,
                        system_name=system_name,
                        mode_name=mode_name,
                        methods=methods,
                        config=config,
                        reason=str(exc),
                    )
                )
                continue
            kept, skipped = filter_configs_by_total_atoms(
                case_configs,
                system_name,
                int(max_total_atoms) if max_total_atoms is not None else None,
            )
            for case_config in kept:
                atoms_per_system, batch_size, total_atoms = planned_atom_counts(
                    system_name,
                    case_config,
                )
                rows.extend(
                    _planned_row(
                        backend=backend,
                        system_name=system_name,
                        mode_name=mode_name,
                        method=method,
                        case_config=case_config,
                        atoms_per_system=atoms_per_system,
                        batch_size=batch_size,
                        total_atoms=total_atoms,
                        steps=_method_steps(config, method),
                        warmup_steps=_method_warmup(config, method),
                    )
                    for method in methods
                )
            for case_config, total_atoms in skipped:
                atoms_per_system, batch_size, _ = planned_atom_counts(
                    system_name,
                    case_config,
                )
                rows.extend(
                    _planned_row(
                        backend=backend,
                        system_name=system_name,
                        mode_name=mode_name,
                        method=method,
                        case_config=case_config,
                        atoms_per_system=atoms_per_system,
                        batch_size=batch_size,
                        total_atoms=total_atoms,
                        steps=_method_steps(config, method),
                        warmup_steps=_method_warmup(config, method),
                        reason=f">{max_total_atoms} max_total_atoms",
                    )
                    for method in methods
                )
    return rows


def _expand_path(value: str | None) -> Path | None:
    """Expand a YAML path token while preserving ``None``."""
    if value in (None, ""):
        return None
    return Path(value).expanduser()


def _set_energy_forces(model: torch.nn.Module) -> torch.nn.Module:
    """Restrict a Toolkit model wrapper to energy and forces for NVT timing."""
    model.model_config.active_outputs = {"energy", "forces"}
    return model


def _freeze_model_parameters(model: torch.nn.Module) -> torch.nn.Module:
    """Disable parameter gradients for inference-only benchmark runs."""
    for param in model.parameters():
        param.requires_grad = False
    return model


def _compile_tensornet_model(model: torch.nn.Module, tnet_cfg: dict[str, Any]) -> torch.nn.Module:
    """Compile the inner MatGL TensorNet model when requested."""
    compile_model = bool(tnet_cfg.get("compile_model", False))
    setattr(model, "tnet_compile_enabled", False)
    if not compile_model:
        return model

    compile_kwargs = dict(tnet_cfg.get("compile_kwargs", {}) or {})
    uses_warp = bool(getattr(model.model, "_use_warp", False))
    model.model = torch.compile(model.model, **compile_kwargs)
    setattr(model.model, "_use_warp", uses_warp)
    setattr(model, "tnet_compile_enabled", True)
    return model


def _ensure_tensornet_warp_potential(potential):
    """Return a TensorNet Potential whose wrapped model uses Warp layers."""
    model = potential.model
    if bool(getattr(model, "_use_warp", False)):
        return potential

    from matgl.models import TensorNet

    if not isinstance(model, TensorNet):
        raise TypeError(
            f"Expected a MatGL TensorNet model, got {type(model).__name__}."
        )

    init_args = dict(getattr(model, "_init_args", {}))
    init_args["use_warp"] = True
    warp_model = TensorNet(**init_args)
    incompatible = warp_model.load_state_dict(model.state_dict(), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Could not rebuild the loaded TensorNet checkpoint with Warp layers: "
            f"missing_keys={incompatible.missing_keys}, "
            f"unexpected_keys={incompatible.unexpected_keys}."
        )
    potential.model = warp_model
    if hasattr(potential, "_init_args"):
        potential._init_args["model"] = warp_model
    return potential


def _build_tensornet(stack: dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Build a MatGL TensorNet wrapper using the packaged alchmtk integration."""
    tnet_cfg = stack.get("tensornet", {})
    try:
        import matgl
        from matgl.ext.alchmtk import TensorNetWrapper
    except ImportError as exc:
        raise ImportError(
            "TensorNet model-stack benchmarks require matgl with the alchmtk "
            "integration installed."
        ) from exc

    model_name = str(tnet_cfg.get("model_name", "TensorNet-PES-MatPES-PBE-2025.2"))
    potential = matgl.load_model(model_name)
    require_warp = bool(tnet_cfg.get("require_warp", True))
    if require_warp:
        potential = _ensure_tensornet_warp_potential(potential)
    model = TensorNetWrapper.from_potential(potential)
    model = _set_energy_forces(model)
    model.model = _freeze_model_parameters(model.model)
    model.to(device)
    model = _compile_tensornet_model(model, tnet_cfg)
    uses_warp = bool(getattr(model.model, "_use_warp", False))
    if require_warp and not uses_warp:
        raise RuntimeError(
            "Loaded TensorNet did not enable Warp kernels. Ensure matgl sees the "
            "local nvalchemi-toolkit-ops installation, or set "
            "model_stacks.tensornet.require_warp=false to benchmark the PyG path."
        )
    return model


def _build_mace(
    stack: dict[str, Any],
    device: torch.device,
    default_dtype: torch.dtype,
) -> torch.nn.Module:
    """Build the Toolkit MACE wrapper from a configured checkpoint."""
    mace_cfg = stack.get("mace", {})
    try:
        from nvalchemi.models.mace import MACEWrapper
    except ImportError as exc:
        raise ImportError(
            "MACE model-stack benchmarks require nvalchemi-toolkit with MACE extras."
        ) from exc
    dtype = _torch_dtype(
        str(mace_cfg.get("dtype", str(default_dtype).removeprefix("torch.")))
    )
    model = MACEWrapper.from_checkpoint(
        mace_cfg.get("checkpoint", "medium-mpa-0"),
        device=device,
        enable_cueq=bool(mace_cfg.get("enable_cueq", True)),
        dtype=dtype,
        compile_model=bool(mace_cfg.get("compile_model", False)),
        **mace_cfg.get("compile_kwargs", {}),
    )
    return _set_energy_forces(model)


def _build_d3(stack: dict[str, Any]) -> torch.nn.Module:
    """Build the Toolkit DFT-D3 wrapper."""
    d3_cfg = stack.get("d3", {})
    from nvalchemi.models.dftd3 import DFTD3ModelWrapper

    model = DFTD3ModelWrapper(
        a1=float(d3_cfg.get("a1", 0.4145)),
        a2=float(d3_cfg.get("a2", 4.8593)),
        s8=float(d3_cfg.get("s8", 1.2177)),
        cutoff=float(d3_cfg.get("cutoff", 15.0)),
        k1=float(d3_cfg.get("k1", 16.0)),
        k3=float(d3_cfg.get("k3", -4.0)),
        s6=float(d3_cfg.get("s6", 1.0)),
        smoothing_fraction=float(d3_cfg.get("smoothing_fraction", 0.2)),
        auto_download=bool(d3_cfg.get("auto_download", True)),
        param_file=_expand_path(d3_cfg.get("param_file")),
    )
    return _set_energy_forces(model)


def _build_pme(stack: dict[str, Any]) -> torch.nn.Module:
    """Build the Toolkit PME wrapper."""
    electro_cfg = stack.get("electrostatics", {})
    pme_cfg = electro_cfg.get("pme", {})
    from nvalchemi.models.pme import PMEModelWrapper

    model = PMEModelWrapper(
        cutoff=float(electro_cfg.get("cutoff", 10.0)),
        mesh_spacing=float(pme_cfg.get("mesh_spacing", 1.0)),
        mesh_dimensions=pme_cfg.get("mesh_dimensions"),
        spline_order=int(pme_cfg.get("spline_order", 4)),
        alpha=pme_cfg.get("alpha"),
        accuracy=float(electro_cfg.get("accuracy", 1.0e-4)),
        coulomb_constant=float(electro_cfg.get("coulomb_constant", 14.3996)),
        hybrid_forces=bool(electro_cfg.get("hybrid_forces", True)),
    )
    return _set_energy_forces(model)


def _build_ewald(stack: dict[str, Any]) -> torch.nn.Module:
    """Build the Toolkit Ewald wrapper."""
    electro_cfg = stack.get("electrostatics", {})
    ewald_cfg = electro_cfg.get("ewald", {})
    from nvalchemi.models.ewald import EwaldModelWrapper

    model = EwaldModelWrapper(
        cutoff=float(electro_cfg.get("cutoff", 10.0)),
        accuracy=float(ewald_cfg.get("accuracy", electro_cfg.get("accuracy", 1.0e-4))),
        coulomb_constant=float(electro_cfg.get("coulomb_constant", 14.3996)),
        hybrid_forces=bool(electro_cfg.get("hybrid_forces", True)),
    )
    return _set_energy_forces(model)


def _build_model_stack(
    method: str,
    config: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    """Compose a Toolkit pipeline for one model-stack method."""
    from nvalchemi.models.pipeline import PipelineGroup, PipelineModelWrapper

    stack = _stack_config(config)
    stack_name = _method_stack_name(method)
    if stack_name == "tnet":
        steps = [_build_tensornet(stack, device)]
    elif stack_name == "mace":
        steps = [_build_mace(stack, device, dtype)]
    elif stack_name == "tnet_d3":
        steps = [_build_tensornet(stack, device), _build_d3(stack)]
    elif stack_name == "mace_d3":
        steps = [_build_mace(stack, device, dtype), _build_d3(stack)]
    elif stack_name == "mace_d3_pme":
        steps = [_build_mace(stack, device, dtype), _build_d3(stack), _build_pme(stack)]
    elif stack_name == "mace_d3_ewald":
        steps = [
            _build_mace(stack, device, dtype),
            _build_d3(stack),
            _build_ewald(stack),
        ]
    else:
        raise ValueError(f"Unsupported model-stack dynamics method: {method}")

    model = PipelineModelWrapper(groups=[PipelineGroup(steps=steps)])
    model.model_config.active_outputs = {"energy", "forces"}
    model.to(device)
    model.eval()
    return model


def _cache_key(
    method: str,
    device: torch.device,
    dtype: torch.dtype,
    plan: dict[str, Any],
) -> tuple[str, ...]:
    """Return the model-cache key for one planned model-stack run."""
    stack_name = _method_stack_name(method)
    key = (stack_name, str(device), str(dtype))
    if stack_name in {"mace_d3_pme", "mace_d3_ewald"}:
        # PME/Ewald wrappers keep setup tensors shaped by the concrete
        # batch. Scope these caches to the plan geometry so system-size,
        # constant-workload, and batch-scaling cases cannot corrupt one
        # another through stale electrostatics state.
        key = (
            *key,
            f"atoms={int(plan['atoms_per_system'])}",
            f"batch={int(plan['batch_size'])}",
            f"total={int(plan['total_atoms'])}",
        )
    return key


def _get_or_build_model(
    method: str,
    config: dict[str, Any],
    model_cache: dict[tuple[str, ...], torch.nn.Module | Exception],
    device: torch.device,
    dtype: torch.dtype,
    plan: dict[str, Any],
) -> torch.nn.Module:
    """Return a cached model stack, caching construction failures as well."""
    key = _cache_key(method, device, dtype, plan)
    cached = model_cache.get(key)
    if isinstance(cached, Exception):
        raise RuntimeError(f"Cached model construction failed for {method}") from cached
    if cached is not None:
        return cached
    try:
        model = _build_model_stack(method, config, device, dtype)
    except Exception as exc:
        model_cache[key] = exc
        raise
    model_cache[key] = model
    return model


def _system_to_atomic_data(
    system: dict[str, Any],
    *,
    graph_index: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    perturbation: float,
):
    """Convert a benchmark system dict into one Toolkit ``AtomicData`` graph."""
    from nvalchemi.data import AtomicData

    positions = system["positions"].to(device=device, dtype=dtype).clone()
    if perturbation:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + graph_index * 1009 + positions.shape[0])
        positions = positions + perturbation * torch.randn(
            positions.shape,
            dtype=dtype,
            device=device,
            generator=generator,
        )
    atomic_numbers = system["atomic_numbers"].to(device=device, dtype=torch.long)
    charges = system.get("charges")
    charge = system.get("charge")
    if isinstance(charges, torch.Tensor):
        charges = charges.to(device=device, dtype=dtype).reshape(-1)
        if not isinstance(charge, torch.Tensor):
            charge = charges.sum().round().reshape(1, 1)
    else:
        charges = None
    if isinstance(charge, torch.Tensor):
        charge = charge.to(device=device, dtype=dtype).reshape(1, 1)
    else:
        charge = None
    cell = system.get("cell")
    if isinstance(cell, torch.Tensor):
        cell = cell.to(device=device, dtype=dtype).reshape(1, 3, 3)
    pbc = system.get("pbc")
    if isinstance(pbc, torch.Tensor):
        pbc = pbc.to(device=device, dtype=torch.bool).reshape(1, 3)
    n_atoms = positions.shape[0]
    return AtomicData(
        positions=positions,
        atomic_numbers=atomic_numbers,
        charges=charges,
        charge=charge,
        forces=torch.zeros(n_atoms, 3, dtype=dtype, device=device),
        energy=torch.zeros(1, 1, dtype=dtype, device=device),
        velocities=torch.zeros(n_atoms, 3, dtype=dtype, device=device),
        cell=cell,
        pbc=pbc,
    )


def _resolve_omat_dataset_path(dataset_path: str | None) -> Path:
    """Resolve an OMat dataset path from config semantics."""
    path = Path(dataset_path or "benchmarks/omat/omat24_sample.pt").expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


def _make_omat_atomic_data_from_archive(
    data: dict[str, torch.Tensor],
    starts: torch.Tensor,
    segment_index: int,
    *,
    dtype: torch.dtype,
):
    """Create one CPU ``AtomicData`` graph directly from a packed OMat archive."""
    from nvalchemi.data import AtomicData

    start = int(starts[segment_index])
    end = int(starts[segment_index + 1])
    n_atoms = end - start
    charge = data.get("charge")
    return AtomicData(
        positions=data["coord"][start:end].to(dtype=dtype).clone(),
        atomic_numbers=data["numbers"][start:end].to(dtype=torch.long),
        charge=(
            charge[segment_index].to(dtype=dtype).reshape(1, 1)
            if isinstance(charge, torch.Tensor)
            else None
        ),
        forces=torch.zeros(n_atoms, 3, dtype=dtype),
        energy=torch.zeros(1, 1, dtype=dtype),
        velocities=torch.zeros(n_atoms, 3, dtype=dtype),
        cell=data["cell"][segment_index].to(dtype=dtype).reshape(1, 3, 3),
        pbc=torch.ones(1, 3, dtype=torch.bool),
    )


def _get_omat_source_batch(dataset_path: str | None, dtype: torch.dtype):
    """Return a cached CPU ``Batch`` containing every OMat sample graph once."""
    from nvalchemi.data import Batch

    resolved = _resolve_omat_dataset_path(dataset_path)
    key = (str(resolved), str(dtype))
    cached = _OMAT_SOURCE_BATCH_CACHE.get(key)
    if cached is not None:
        return cached

    archive = torch.load(resolved, map_location="cpu", weights_only=False)
    data = archive["data"]
    segments = archive["segment_lengths"]["atoms"].to(dtype=torch.long)
    starts = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(segments, 0)])
    data_list = [
        _make_omat_atomic_data_from_archive(
            data,
            starts,
            segment_index,
            dtype=dtype,
        )
        for segment_index in range(int(segments.numel()))
    ]
    batch = Batch.from_data_list(data_list, device=torch.device("cpu"))
    _OMAT_SOURCE_BATCH_CACHE[key] = batch
    return batch


def _select_omat_prefix_rollover_batch(
    source_batch,
    *,
    dataset_path: str | None,
    target_atoms: int,
    device: torch.device,
):
    """Select a whole-system OMat prefix, repeating the source batch if needed."""
    required_methods = ("clone", "append", "index_select", "to")
    missing = [name for name in required_methods if not hasattr(source_batch, name)]
    if missing:
        raise AttributeError(f"Batch object is missing {', '.join(missing)}")

    systems_to_include, total_atoms = omat_prefix_rollover_counts(
        dataset_path,
        target_atoms,
    )
    dataset_systems = omat_dataset_system_count(dataset_path)
    repeats = max(1, (systems_to_include + dataset_systems - 1) // dataset_systems)

    batch = source_batch
    if repeats > 1:
        batch = source_batch.clone()
        for _ in range(repeats - 1):
            appended = batch.append(source_batch.clone())
            if appended is not None:
                batch = appended
    return batch.index_select(slice(0, systems_to_include)).to(device), total_atoms


def _initialize_md_velocities(
    batch,
    stack: dict[str, Any],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Initialize velocities on an already constructed Toolkit ``Batch``."""
    from nvalchemi.dynamics import initialize_velocities

    seed = int(stack.get("integrator", {}).get("random_seed", 42))
    temperature = torch.full(
        (batch.num_graphs,),
        float(stack.get("integrator", {}).get("temperature", 300.0)),
        dtype=dtype,
        device=device,
    )
    initialize_velocities(
        batch.velocities,
        batch.atomic_masses.to(dtype=dtype),
        temperature,
        batch.batch_idx.int(),
        random_seed=seed,
        remove_com=True,
        remove_rotations=False,
        rescale=True,
        positions=batch.positions,
    )


def _make_omat_prefix_rollover_batch(
    plan: dict[str, Any],
    config: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
):
    """Create a segmented OMat ``Batch`` by rolling over the packed dataset prefix."""
    from nvalchemi.data import Batch

    stack = _stack_config(config)
    seed = int(stack.get("integrator", {}).get("random_seed", 42))
    perturbation = float(stack.get("position_perturbation", 0.01))
    dataset_path = plan.get("dataset_path") or None
    systems_to_include, total_atoms = omat_prefix_rollover_counts(
        dataset_path,
        int(plan["target_atoms"]),
    )
    if int(plan["total_atoms"]) != total_atoms:
        raise RuntimeError(
            "OMat prefix-rollover plan drifted: "
            f"planned total_atoms={plan['total_atoms']} but built {total_atoms}."
        )

    base_systems = int(plan["batch_size"])
    if base_systems != systems_to_include:
        raise RuntimeError(
            "OMat prefix-rollover graph count drifted: "
            f"planned batch_size={base_systems} but built {systems_to_include}."
        )

    dataset_systems = omat_dataset_system_count(dataset_path)
    data_list = []
    for graph_index in range(systems_to_include):
        system = create_system(
            "omat",
            dataset_path=dataset_path,
            segment_index=graph_index % max(1, dataset_systems),
            batch_size=1,
            device="cpu",
            dtype=dtype,
            backend="torch",
        )
        data_list.append(
            _system_to_atomic_data(
                system,
                graph_index=graph_index,
                device=device,
                dtype=dtype,
                seed=seed,
                perturbation=perturbation,
            )
        )
    return Batch.from_data_list(data_list, device=device)


def _make_omat_prefix_rollover_batch_from_source(
    plan: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
):
    """Create an OMat prefix-rollover ``Batch`` through ``Batch.index_select``."""
    dataset_path = plan.get("dataset_path") or None
    source_batch = _get_omat_source_batch(dataset_path, dtype)
    batch, total_atoms = _select_omat_prefix_rollover_batch(
        source_batch,
        dataset_path=dataset_path,
        target_atoms=int(plan["target_atoms"]),
        device=device,
    )
    if int(plan["total_atoms"]) != total_atoms:
        raise RuntimeError(
            "OMat prefix-rollover plan drifted: "
            f"planned total_atoms={plan['total_atoms']} but built {total_atoms}."
        )
    return batch


def _make_batch(
    plan: dict[str, Any],
    config: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
    initialize_md_velocities: bool = True,
):
    """Create a Toolkit ``Batch`` for a planned model-stack case."""
    from nvalchemi.data import Batch

    stack = _stack_config(config)
    if (
        plan["system"] == "omat"
        and plan.get("batch_construction") == "prefix_rollover"
    ):
        try:
            batch = _make_omat_prefix_rollover_batch_from_source(
                plan,
                device=device,
                dtype=dtype,
            )
        except (AttributeError, TypeError):
            batch = _make_omat_prefix_rollover_batch(
                plan,
                config,
                device=device,
                dtype=dtype,
            )
        if initialize_md_velocities:
            _initialize_md_velocities(batch, stack, dtype=dtype, device=device)
        return batch

    seed = int(stack.get("integrator", {}).get("random_seed", 42))
    perturbation = float(stack.get("position_perturbation", 0.01))
    base_system = create_system(
        plan["system"],
        num_atoms=int(plan["target_atoms"]),
        pdb_path=plan.get("pdb_path") or None,
        dataset_path=plan.get("dataset_path") or None,
        segment_index=(
            int(plan["segment_index"]) if plan.get("segment_index") != "" else None
        ),
        batch_size=1,
        device=str(device),
        dtype=dtype,
        backend="torch",
    )
    data_list = [
        _system_to_atomic_data(
            base_system,
            graph_index=i,
            device=device,
            dtype=dtype,
            seed=seed,
            perturbation=perturbation,
        )
        for i in range(int(plan["batch_size"]))
    ]
    batch = Batch.from_data_list(data_list, device=device)
    if not initialize_md_velocities:
        return batch
    _initialize_md_velocities(batch, stack, dtype=dtype, device=device)
    return batch


def _sync_work(device: torch.device) -> None:
    """Synchronize Torch CUDA and Warp queues before reading wall time."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    try:
        import warp as wp
    except ImportError:
        return
    wp.synchronize()


def _run_nvt(
    model: torch.nn.Module,
    batch,
    *,
    config: dict[str, Any],
    steps: int,
    warmup_steps: int,
    device: torch.device,
) -> float:
    """Warm up, then return synchronized wall time for timed NVT steps."""
    from nvalchemi.dynamics import NVTLangevin

    if steps <= 0:
        raise ValueError(f"timed steps must be positive, got {steps}")
    integrator = _stack_config(config).get("integrator", {})

    def build_dynamics(n_steps: int) -> NVTLangevin:
        return NVTLangevin(
            model=model,
            dt=float(integrator.get("dt", 0.25)),
            temperature=float(integrator.get("temperature", 300.0)),
            friction=float(integrator.get("friction", 0.01)),
            random_seed=int(integrator.get("random_seed", 42)),
            n_steps=n_steps,
            hooks=model.make_neighbor_hooks(),
        )

    if warmup_steps > 0:
        build_dynamics(warmup_steps).run(batch)
    _sync_work(device)
    start = time.perf_counter()
    build_dynamics(steps).run(batch)
    _sync_work(device)
    return time.perf_counter() - start


def _build_optimizer(
    engine: str,
    model: torch.nn.Module,
    config: dict[str, Any],
    n_steps: int,
):
    """Build a Toolkit optimizer for one model-stack optimization run."""
    optimizer_cfg = _stack_config(config).get("optimizer", {})
    hooks = model.make_neighbor_hooks()
    if engine == "fire":
        from nvalchemi.dynamics import FIRE

        fire_cfg = optimizer_cfg.get("fire", {})
        dt = float(fire_cfg.get("dt_start", optimizer_cfg.get("dt_start", 0.05)))
        dt_max = fire_cfg.get("dt_max", fire_cfg.get("tmax"))
        dt_min = fire_cfg.get("dt_min", fire_cfg.get("tmin"))
        return FIRE(
            model=model,
            dt=dt,
            dt_max=float(dt_max) if dt_max is not None else None,
            dt_min=float(dt_min) if dt_min is not None else None,
            maxstep=float(fire_cfg.get("maxstep", optimizer_cfg.get("maxstep", 0.1))),
            n_min=int(fire_cfg.get("n_min", 5)),
            f_dec=float(fire_cfg.get("f_dec", 0.75)),
            f_inc=float(fire_cfg.get("f_inc", 1.05)),
            alpha_start=float(fire_cfg.get("alpha_start", 0.09)),
            f_alpha=float(fire_cfg.get("f_alpha", 0.985)),
            n_steps=n_steps,
            hooks=hooks,
            convergence_hook=None,
        )
    if engine == "fire2":
        from nvalchemi.dynamics import FIRE2

        fire2_cfg = optimizer_cfg.get("fire2", {})
        return FIRE2(
            model=model,
            dt=float(fire2_cfg.get("dt_start", optimizer_cfg.get("dt_start", 0.045))),
            delaystep=int(fire2_cfg.get("delaystep", 5)),
            dtgrow=float(fire2_cfg.get("dtgrow", 1.05)),
            dtshrink=float(fire2_cfg.get("dtshrink", 0.75)),
            alphashrink=float(fire2_cfg.get("alphashrink", 0.985)),
            alpha0=float(fire2_cfg.get("alpha0", 0.09)),
            tmax=float(fire2_cfg.get("tmax", 0.08)),
            tmin=float(fire2_cfg.get("tmin", 0.005)),
            maxstep=float(fire2_cfg.get("maxstep", optimizer_cfg.get("maxstep", 0.1))),
            n_steps=n_steps,
            hooks=hooks,
            convergence_hook=None,
        )
    raise ValueError(f"Unsupported model-stack optimization engine: {engine}")


def _run_optimizer_steps(
    model: torch.nn.Module,
    batch,
    *,
    config: dict[str, Any],
    engine: str,
    steps: int,
) -> None:
    """Run fixed-step Toolkit geometry optimization."""
    if steps <= 0:
        raise ValueError(f"optimizer steps must be positive, got {steps}")
    optimizer = _build_optimizer(engine, model, config, steps)
    optimizer.run(batch)


def _run_optimization(
    model: torch.nn.Module,
    timed_batch,
    *,
    config: dict[str, Any],
    engine: str,
    steps: int,
    device: torch.device,
    warmup_batch=None,
    warmup_steps: int = 0,
) -> float:
    """Warm up with a throwaway batch, then time fixed-step optimization."""
    if warmup_steps > 0:
        if warmup_batch is None:
            raise ValueError("warmup_batch is required when warmup_steps > 0")
        _run_optimizer_steps(
            model,
            warmup_batch,
            config=config,
            engine=engine,
            steps=warmup_steps,
        )
    _sync_work(device)
    start = time.perf_counter()
    _run_optimizer_steps(
        model,
        timed_batch,
        config=config,
        engine=engine,
        steps=steps,
    )
    _sync_work(device)
    return time.perf_counter() - start


def run_model_stack_case(
    plan: dict[str, Any],
    config: dict[str, Any],
    *,
    model_cache: dict[tuple[str, ...], torch.nn.Module | Exception],
    device: torch.device,
) -> BenchmarkResult:
    """Run one planned Toolkit model-stack NVT dynamics benchmark.

    Parameters
    ----------
    plan : dict
        One row produced by :func:`planned_model_stack_rows`.
    config : dict
        Loaded dynamics YAML configuration.
    model_cache : dict
        Mutable cache used to keep model construction out of timed loops.
    device : torch.device
        Runtime device.

    Returns
    -------
    BenchmarkResult
        Timed model-stack dynamics result using the same CSV schema as the
        native dynamics runner.
    """
    dtype = _dtype_from_config(config)
    model = _get_or_build_model(
        plan["method"],
        config,
        model_cache,
        device,
        dtype,
        plan,
    )
    engine = _method_engine(plan["method"])
    if engine == "nvt":
        batch = _make_batch(
            plan,
            config,
            device=device,
            dtype=dtype,
            initialize_md_velocities=True,
        )
        total_time = _run_nvt(
            model,
            batch,
            config=config,
            steps=int(plan["steps"]),
            warmup_steps=int(plan["warmup_steps"]),
            device=device,
        )
        ensemble = "NVT"
        dt = float(_stack_config(config).get("integrator", {}).get("dt", 0.25))
    else:
        warmup_batch = (
            _make_batch(
                plan,
                config,
                device=device,
                dtype=dtype,
                initialize_md_velocities=False,
            )
            if int(plan["warmup_steps"]) > 0
            else None
        )
        batch = _make_batch(
            plan,
            config,
            device=device,
            dtype=dtype,
            initialize_md_velocities=False,
        )
        total_time = _run_optimization(
            model,
            batch,
            config=config,
            engine=engine,
            steps=int(plan["steps"]),
            device=device,
            warmup_batch=warmup_batch,
            warmup_steps=int(plan["warmup_steps"]),
        )
        ensemble = "optimization"
        opt_cfg = _stack_config(config).get("optimizer", {})
        dt = float(
            opt_cfg.get(engine, {}).get("dt_start", opt_cfg.get("dt_start", 0.05))
        )
    step_time = total_time / max(int(plan["steps"]), 1)
    final_pe = None
    if isinstance(getattr(batch, "energy", None), torch.Tensor):
        final_pe = float(batch.energy.detach().sum().cpu())
    return BenchmarkResult(
        name=plan["method"],
        backend="torch",
        model_type=plan["method"],
        ensemble=ensemble,
        num_atoms=int(plan["atoms_per_system"]),
        num_steps=int(plan["steps"]),
        dt=dt,
        warmup_steps=int(plan["warmup_steps"]),
        total_time=total_time,
        step_times=[step_time] * int(plan["steps"]),
        batch_size=int(plan["batch_size"]),
        final_pe=final_pe,
    )
