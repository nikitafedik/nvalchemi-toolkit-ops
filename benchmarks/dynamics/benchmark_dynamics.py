#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unified batched dynamics benchmark runner.

This runner exposes the same suite contract as the NL/D3/EL benchmarks:
YAML-first configuration, shared CLI overrides, allocation-free dry-runs,
failure rows, and one CSV per ``system x scaling_mode``.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import torch

from benchmarks.config import (
    add_common_cli_args,
    enabled_method_names,
    load_yaml_config,
    merge_common_cli_overrides,
    normalize_method_name,
)
from benchmarks.dynamics.model_stacks import (
    MODEL_STACK_METHOD_FAMILIES,
    MODEL_STACK_METHODS,
    planned_model_stack_rows,
    run_model_stack_case,
)
from benchmarks.dynamics.shared_utils import (
    NvalchemiOpsBenchmark,
    create_fcc_argon,
    get_gpu_sku,
)
from benchmarks.utils import (
    create_run_directory,
    make_csv_name,
    save_results,
)

__all__ = [
    "dry_run_from_config",
    "main",
    "merge_cli_overrides",
    "parse_args",
    "run_from_config",
]

_SUPPORTED_BACKENDS = {"torch"}
_METHOD_FAMILY = {
    "velocity_verlet": "md",
    "langevin": "md",
    "npt": "md",
    "nph": "md",
    "fire": "opt",
    "fire2": "opt",
    **MODEL_STACK_METHOD_FAMILIES,
}
_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
}


def _merge_model_stack_cli_overrides(config: dict, args: argparse.Namespace) -> None:
    """Apply shared system/mode filters to the model-stack YAML subtree."""
    stack_config = config.get("model_stacks")
    if not isinstance(stack_config, dict):
        return
    if args.system is not None and "all" not in args.system:
        for sys_name, sys_config in stack_config.get("systems", {}).items():
            sys_config["enabled"] = sys_name in args.system
    if args.mode is not None and "all" not in args.mode:
        for mode_name, mode_config in stack_config.get("scaling", {}).items():
            if isinstance(mode_config, dict):
                mode_config["enabled"] = mode_name in args.mode


def merge_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply shared CLI overrides to the dynamics benchmark config."""
    config = merge_common_cli_overrides(config, args)
    _merge_model_stack_cli_overrides(config, args)
    return config


def _validate_backend(backend: str) -> None:
    """Raise if the requested backend is unsupported by dynamics benchmarks."""
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(
            "Dynamics benchmark supports the torch backend only; "
            f"got backend={backend!r}."
        )


def _enabled_dynamics_methods(config: dict) -> tuple[list[str], list[str]]:
    """Return supported and ignored method names after alias normalization."""
    supported: list[str] = []
    ignored: list[str] = []
    for method in enabled_method_names(config):
        name = normalize_method_name(method)
        if name in _METHOD_FAMILY:
            if name not in supported:
                supported.append(name)
        else:
            ignored.append(name)
    return supported, ignored


def _method_config(config: dict, method: str) -> dict:
    """Return the YAML entry for *method*, or an empty dict."""
    for entry in config.get("methods", []):
        if normalize_method_name(entry.get("name", "")) == method:
            return entry
    return {}


def _actual_fcc_atoms(target_atoms: int) -> tuple[int, int]:
    """Return ``(num_cells, actual_atoms)`` for an FCC target atom count."""
    num_cells = max(1, math.ceil((target_atoms / 4.0) ** (1.0 / 3.0)))
    return num_cells, 4 * num_cells**3


def _dtype_from_config(config: dict) -> torch.dtype:
    """Resolve torch dtype from YAML."""
    value = str(config.get("parameters", {}).get("dtype", "float64"))
    try:
        return _DTYPES[value]
    except KeyError:
        raise ValueError(
            f"Unsupported dynamics dtype {value!r}; expected one of {sorted(_DTYPES)}."
        ) from None


def _method_steps(config: dict, method: str) -> int:
    """Resolve timed steps for a method, with CLI timing override support."""
    params = config.get("parameters", {})
    method_cfg = _method_config(config, method)
    if params.get("timing_runs") is not None:
        return int(params["timing_runs"])
    default_steps = method_cfg.get("max_steps", params.get("steps", 1000))
    return int(method_cfg.get("steps", default_steps))


def _method_warmup(config: dict, method: str) -> int:
    """Resolve warmup steps for a method, with CLI warmup override support."""
    params = config.get("parameters", {})
    method_cfg = _method_config(config, method)
    if params.get("warmup_runs") is not None:
        return int(params["warmup_runs"])
    return int(method_cfg.get("warmup_steps", params.get("warmup_steps", 100)))


def _cases_for_mode(
    system_config: dict,
    mode_name: str,
    mode_config: dict,
) -> list[tuple[int, int, int]]:
    """Return ``(target_atoms, atoms_per_system, batch_size)`` cases."""
    cases: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()

    def append_case(target_atoms: int, atoms_per_system: int, batch_size: int) -> None:
        key = (atoms_per_system, batch_size)
        if key not in seen:
            seen.add(key)
            cases.append((target_atoms, atoms_per_system, batch_size))

    if mode_name == "system_size":
        batch_size = int(mode_config.get("batch_size", 1))
        for target_atoms in system_config.get("atom_counts", []):
            _, atoms_per_system = _actual_fcc_atoms(int(target_atoms))
            append_case(int(target_atoms), atoms_per_system, batch_size)
        return cases

    if mode_name == "constant_workload":
        target_total = int(mode_config.get("target_atoms", 0))
        atom_counts = mode_config.get(
            "atoms_per_system",
            system_config.get(
                "constant_atoms_sizes",
                system_config.get("atom_counts", []),
            ),
        )
        for target_atoms in atom_counts:
            _, atoms_per_system = _actual_fcc_atoms(int(target_atoms))
            batch_size = max(1, target_total // atoms_per_system)
            append_case(int(target_atoms), atoms_per_system, batch_size)
        return cases

    if mode_name == "batch_scaling":
        atom_counts = mode_config.get(
            "atoms_per_system",
            system_config.get(
                "batch_atom_counts", system_config.get("atom_counts", [])
            ),
        )
        for target_atoms in atom_counts:
            _, atoms_per_system = _actual_fcc_atoms(int(target_atoms))
            batch_sizes = mode_config.get(
                "batch_sizes",
                system_config.get("batch_sizes", [1]),
            )
            for batch_size in batch_sizes:
                append_case(int(target_atoms), atoms_per_system, int(batch_size))
        return cases

    return cases


def _planned_rows(config: dict, backend: str) -> list[dict[str, Any]]:
    """Build allocation-free dynamics plan rows."""
    methods, ignored = _enabled_dynamics_methods(config)
    if ignored:
        print(f"Dynamics dry-run ignoring non-dynamics methods: {', '.join(ignored)}")
    standard_methods = [
        method for method in methods if method not in MODEL_STACK_METHODS
    ]
    model_methods = [method for method in methods if method in MODEL_STACK_METHODS]

    params = config.get("parameters", {})
    max_total_atoms = params.get("max_total_atoms")
    rows: list[dict[str, Any]] = []
    for system_name, system_config in config.get("systems", {}).items():
        if not system_config.get("enabled", True):
            continue
        for mode_name, mode_config in config.get("scaling", {}).items():
            if not isinstance(mode_config, dict) or not mode_config.get(
                "enabled", True
            ):
                continue
            for target_atoms, atoms_per_system, batch_size in _cases_for_mode(
                system_config, mode_name, mode_config
            ):
                total_atoms = atoms_per_system * batch_size
                reason = ""
                if max_total_atoms is not None and total_atoms > int(max_total_atoms):
                    reason = f">{max_total_atoms} max_total_atoms"
                rows.extend(
                    {
                        "benchmark": "dyn",
                        "backend": backend,
                        "system": system_name,
                        "mode": mode_name,
                        "scaling_mode": mode_name,
                        "method": method,
                        "method_family": _METHOD_FAMILY[method],
                        "target_atoms": target_atoms,
                        "atoms_per_system": atoms_per_system,
                        "batch_size": batch_size,
                        "total_atoms": total_atoms,
                        "steps": _method_steps(config, method),
                        "warmup_steps": _method_warmup(config, method),
                        "reason": reason,
                    }
                    for method in standard_methods
                )
    if model_methods:
        rows.extend(planned_model_stack_rows(config, backend, model_methods))
    return rows


def dry_run_from_config(config: dict, backend: str | None = None) -> list[dict]:
    """Print and return the expanded dynamics benchmark plan."""
    if backend is None:
        backend = config.get("runtime", {}).get("backend", "torch")
    _validate_backend(backend)
    rows = _planned_rows(config, backend)
    print("Dynamics dry-run plan")
    for row in rows:
        suffix = f" SKIP {row['reason']}" if row["reason"] else ""
        print(
            "  {system}/{mode} backend={backend} method={method} "
            "N={atoms_per_system} batch={batch_size} total={total_atoms} "
            "steps={steps} warmup={warmup_steps}{suffix}".format(
                **row,
                suffix=suffix,
            )
        )
    print(f"Dynamics dry-run rows: {len(rows)}")
    return rows


def _make_batched_lj_system(
    *,
    target_atoms: int,
    batch_size: int,
    config: dict,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Create deterministic batched FCC argon systems."""
    system_cfg = config.get("system", {})
    lattice_constant = float(system_cfg.get("lattice_constant", 5.26))
    perturbation = float(
        config.get("parameters", {}).get("position_perturbation", 0.01)
    )
    seed = int(config.get("parameters", {}).get("seed", 42))
    num_cells, atoms_per_system = _actual_fcc_atoms(target_atoms)
    pos_np, cell_np = create_fcc_argon(num_unit_cells=num_cells, a=lattice_constant)

    base_positions = torch.as_tensor(pos_np, dtype=dtype, device=device)
    positions = base_positions.repeat(batch_size, 1)
    if perturbation:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + target_atoms * 1009 + batch_size)
        noise = torch.randn(
            positions.shape,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        positions = positions + noise * perturbation

    cell = torch.as_tensor(cell_np, dtype=dtype, device=device).repeat(batch_size, 1, 1)
    batch_idx = torch.repeat_interleave(
        torch.arange(batch_size, dtype=torch.int32, device=device),
        atoms_per_system,
    )
    pbc_values = system_cfg.get("pbc", [True, True, True])
    pbc = torch.tensor([pbc_values] * batch_size, dtype=torch.bool, device=device)
    return positions, cell, pbc, batch_idx, atoms_per_system


def _run_method(
    bench: NvalchemiOpsBenchmark,
    method: str,
    method_cfg: dict,
    steps: int,
    warmup_steps: int,
):
    """Run one dynamics method and return ``BenchmarkResult``."""
    if method == "velocity_verlet":
        return bench.run_velocity_verlet(
            dt=float(method_cfg.get("dt", 0.001)),
            num_steps=steps,
            warmup_steps=warmup_steps,
            log_interval=int(method_cfg.get("log_interval", 100)),
        )
    if method == "langevin":
        return bench.run_langevin(
            dt=float(method_cfg.get("dt", 0.001)),
            num_steps=steps,
            temperature=float(method_cfg.get("temperature", 94.4)),
            friction=float(method_cfg.get("friction", 0.01)),
            warmup_steps=warmup_steps,
            log_interval=int(method_cfg.get("log_interval", 100)),
        )
    if method == "npt":
        return bench.run_npt(
            dt=float(method_cfg.get("dt", 0.001)),
            num_steps=steps,
            temperature=float(method_cfg.get("temperature", 94.4)),
            pressure=float(method_cfg.get("pressure", 1.0)),
            tau_t=float(method_cfg.get("tau_t", 500.0)),
            tau_p=float(method_cfg.get("tau_p", 5000.0)),
            chain_length=int(method_cfg.get("chain_length", 3)),
            warmup_steps=warmup_steps,
            log_interval=int(method_cfg.get("log_interval", 100)),
        )
    if method == "nph":
        return bench.run_nph(
            dt=float(method_cfg.get("dt", 0.001)),
            num_steps=steps,
            temperature=float(method_cfg.get("temperature", 94.4)),
            pressure=float(method_cfg.get("pressure", 1.0)),
            tau_p=float(method_cfg.get("tau_p", 5000.0)),
            warmup_steps=warmup_steps,
            log_interval=int(method_cfg.get("log_interval", 100)),
        )
    if method == "fire":
        return bench.run_fire(
            max_steps=steps,
            force_tolerance=float(method_cfg.get("force_tolerance", 0.01)),
            dt_start=float(method_cfg.get("dt_start", 0.05)),
            dt_max=float(method_cfg.get("dt_max", 0.08)),
            dt_min=float(method_cfg.get("dt_min", 0.005)),
            alpha_start=float(method_cfg.get("alpha_start", 0.09)),
            n_min=int(method_cfg.get("n_min", 5)),
            f_inc=float(method_cfg.get("f_inc", 1.05)),
            f_dec=float(method_cfg.get("f_dec", 0.75)),
            f_alpha=float(method_cfg.get("f_alpha", 0.985)),
            maxstep=float(method_cfg.get("maxstep", 0.1)),
            warmup_steps=warmup_steps,
            log_interval=int(method_cfg.get("log_interval", 100)),
            check_interval=int(method_cfg.get("check_interval", 20)),
        )
    if method == "fire2":
        return bench.run_fire2(
            max_steps=steps,
            force_tolerance=float(method_cfg.get("force_tolerance", 0.01)),
            dt_start=float(method_cfg.get("dt_start", 0.045)),
            tmax=float(method_cfg.get("tmax", 0.08)),
            tmin=float(method_cfg.get("tmin", 0.005)),
            delaystep=int(method_cfg.get("delaystep", 5)),
            dtgrow=float(method_cfg.get("dtgrow", 1.05)),
            dtshrink=float(method_cfg.get("dtshrink", 0.75)),
            alpha0=float(method_cfg.get("alpha0", 0.09)),
            alphashrink=float(method_cfg.get("alphashrink", 0.985)),
            maxstep=float(method_cfg.get("maxstep", 0.1)),
            warmup_steps=warmup_steps,
            log_interval=int(method_cfg.get("log_interval", 100)),
            check_interval=int(method_cfg.get("check_interval", 20)),
        )
    raise ValueError(f"Unsupported dynamics method: {method}")


def _result_row(
    result, plan: dict[str, Any], *, success: bool = True
) -> dict[str, Any]:
    """Normalize a ``BenchmarkResult`` into the unified suite row schema."""
    row = {
        "success": success,
        "error": "" if success else "Dynamics benchmark returned no timed steps",
        "error_type": "" if success else "NoTimedSteps",
        "benchmark": "dyn",
        "system": plan["system"],
        "scaling_mode": plan["scaling_mode"],
        "method_family": plan["method_family"],
        "atoms_per_system": plan["atoms_per_system"],
        "target_atoms": plan["target_atoms"],
    }
    row.update(result.to_csv_row())
    row["backend"] = plan["backend"]
    row["method"] = plan["method"]
    row["engine"] = plan.get("engine", "")
    row["batch_construction"] = plan.get("batch_construction", "")
    row["batch_size"] = plan["batch_size"]
    row["total_atoms"] = plan["total_atoms"]
    if result.avg_step_time_ms:
        timed_atoms = max(plan["total_atoms"], 1)
        row["time_us_per_atom_step"] = result.avg_step_time_ms * 1000.0 / timed_atoms
    else:
        row["time_us_per_atom_step"] = math.nan
    return row


def _failure_row(
    plan: dict[str, Any],
    error: Exception | str,
    error_type: str,
) -> dict[str, Any]:
    """Build a failure row matching the successful dynamics schema."""
    return {
        "success": False,
        "error": str(error),
        "error_type": error_type,
        "benchmark": "dyn",
        "backend": plan["backend"],
        "system": plan["system"],
        "scaling_mode": plan["scaling_mode"],
        "method": plan["method"],
        "method_family": plan["method_family"],
        "engine": plan.get("engine", ""),
        "batch_construction": plan.get("batch_construction", ""),
        "target_atoms": plan["target_atoms"],
        "atoms_per_system": plan["atoms_per_system"],
        "batch_size": plan["batch_size"],
        "total_atoms": plan["total_atoms"],
        "steps": plan["steps"],
        "warmup_steps": plan["warmup_steps"],
        "avg_step_time_ms": math.nan,
        "total_time_s": math.nan,
        "throughput_steps_per_s": math.nan,
        "throughput_atom_steps_per_s": math.nan,
        "batch_throughput_system_steps_per_s": math.nan,
        "time_us_per_atom_step": math.nan,
    }


def run_from_config(
    config: dict,
    output_dir: Path | str | None = None,
    backend: str | None = None,
) -> list[dict]:
    """Run batched dynamics benchmarks from YAML config."""
    if backend is None:
        backend = config.get("runtime", {}).get("backend", "torch")
    _validate_backend(backend)
    if config.get("runtime", {}).get("dry_run", False):
        return dry_run_from_config(config, backend=backend)

    if output_dir is None:
        output_dir = create_run_directory(config["output"]["base_dir"], prefix="dyn")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = _dtype_from_config(config)
    potential = config.get("potential", {})
    gpu_sku = get_gpu_sku()
    print("Dynamics Benchmark Suite")
    print(f"GPU: {gpu_sku}")
    print(f"Backend: {backend}")
    print(f"Output: {output_dir}")

    all_results: list[dict] = []
    rows_by_file: dict[str, list[dict]] = {}
    model_cache: dict[tuple[str, ...], torch.nn.Module | Exception] = {}
    for plan in _planned_rows(config, backend):
        csv_name = make_csv_name("dyn", plan["system"], plan["scaling_mode"])
        if plan["reason"]:
            row = _failure_row(plan, plan["reason"], "SkippedByPolicy")
            rows_by_file.setdefault(csv_name, []).append(row)
            all_results.append(row)
            continue

        try:
            if plan["method"] in MODEL_STACK_METHODS:
                result = run_model_stack_case(
                    plan,
                    config,
                    model_cache=model_cache,
                    device=device,
                )
            else:
                method_cfg = _method_config(config, plan["method"])
                positions, cell, pbc, batch_idx, atoms_per_system = (
                    _make_batched_lj_system(
                        target_atoms=plan["target_atoms"],
                        batch_size=plan["batch_size"],
                        config=config,
                        device=device,
                        dtype=dtype,
                    )
                )
                plan = {**plan, "atoms_per_system": atoms_per_system}
                plan["total_atoms"] = atoms_per_system * plan["batch_size"]
                bench = NvalchemiOpsBenchmark(
                    positions=positions,
                    cell=cell,
                    pbc=pbc,
                    epsilon=float(potential.get("epsilon", 0.0104)),
                    sigma=float(potential.get("sigma", 3.40)),
                    cutoff=float(potential.get("cutoff", 8.5)),
                    skin=float(potential.get("skin", 1.0)),
                    neighbor_rebuild_interval=int(
                        potential.get("neighbor_rebuild_interval", 10)
                    ),
                    batch_idx=batch_idx,
                )
                result = _run_method(
                    bench,
                    plan["method"],
                    method_cfg,
                    plan["steps"],
                    plan["warmup_steps"],
                )
            success = bool(result.step_times) and result.total_time > 0.0
            row = _result_row(result, plan, success=success)
            print(
                "  {system}/{scaling_mode} {method} N={atoms_per_system} "
                "batch={batch_size}: {avg_step_time_ms:.4f} ms/step".format(**row)
            )
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            row = _failure_row(plan, e, type(e).__name__)
            print(
                "  {system}/{scaling_mode} {method} N={atoms_per_system} "
                "batch={batch_size}: OOM".format(**plan)
            )
        except Exception as e:
            row = _failure_row(plan, e, type(e).__name__)
            print(
                "  {system}/{scaling_mode} {method} N={atoms_per_system} "
                "batch={batch_size}: FAILED - {error}".format(**plan, error=e)
            )
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        rows_by_file.setdefault(csv_name, []).append(row)
        all_results.append(row)

    for csv_name, rows in rows_by_file.items():
        save_results(rows, output_dir / csv_name)
    return all_results


def _successful_rows(results: list[dict]) -> int:
    """Count successful runtime rows, treating dry-run rows as planned work."""
    return sum(1 for row in results if row.get("success", True) is not False)


def parse_args():
    """Parse command-line arguments for standalone dynamics benchmarks."""
    parser = argparse.ArgumentParser(
        description="Unified batched dynamics benchmark",
    )
    parser.add_argument("--config", type=Path, required=True)
    add_common_cli_args(parser)
    return parser.parse_args()


def main():
    """Standalone entry point."""
    args = parse_args()
    config = load_yaml_config(args.config)
    config = merge_cli_overrides(config, args)
    backend = args.backend or config.get("runtime", {}).get("backend", "torch")
    results = run_from_config(config, output_dir=args.output_dir, backend=backend)
    if not results:
        return 1
    if _successful_rows(results) <= 0:
        print(
            "ERROR: no successful dynamics benchmark rows were produced",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
