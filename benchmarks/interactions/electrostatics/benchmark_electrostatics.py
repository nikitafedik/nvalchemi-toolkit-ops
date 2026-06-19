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

"""Electrostatics Benchmark (Ewald + PME).

CRITICAL: Electrostatics requires float64 for positions and cells.
K-vectors are pre-computed ONCE outside the timing loop.

Usage (run from the repository root):
    python -m benchmarks.interactions.electrostatics.benchmark_electrostatics \
        --config benchmarks/interactions/electrostatics/benchmark_config.yaml
    python -m benchmarks.interactions.electrostatics.benchmark_electrostatics \
        --config benchmarks/interactions/electrostatics/benchmark_config.yaml \
        --output-dir docs/benchmarks/benchmark_results

    # JAX backend (the runner sets JAX env defaults before importing JAX)
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
        python -m benchmarks.interactions.electrostatics.benchmark_electrostatics \
        --config benchmarks/interactions/electrostatics/benchmark_config.yaml --backend jax

Backends
--------
``--backend torch`` (default) uses the warp-based torch kernels and CUDA
events for timing. ``--backend jax`` uses the JAX wrappers in
``nvalchemiops.jax.interactions.electrostatics`` and wall-clock timing.

JAX caveats:

- The runner enables ``jax_enable_x64`` before importing electrostatics;
  electrostatics requires float64 positions and cells.
- JAX EL rows benchmark the direct-output API path
  (``compute_forces=True`` and, for ``*_cg`` rows,
  ``compute_charge_gradients=True``). If a direct-output combination is not
  implemented by a backend, the runner records a ``success=False`` row and the
  plotter filters it from timing panels.

Environment variables for ``--backend jax``:

- ``XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`` — set by the runner before
  importing JAX unless the user already configured JAX memory behavior. JAX's
  normal preallocator avoids the fragmentation seen with on-demand allocation
  in large benchmark sweeps.
- ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` — optional user override. If unset,
  the runner prints a one-line note before JAX import and keeps the default
  preallocator capped by ``XLA_PYTHON_CLIENT_MEM_FRACTION``.
- ``JAX_ENABLE_X64=True`` — set by the suite/runner for electrostatics;
  exporting it explicitly is also safe.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import torch

__all__ = [
    "ElectrostaticsInputs",
    "benchmark_ewald",
    "benchmark_pme",
    "main",
    "merge_cli_overrides",
    "parse_args",
    "run_from_config",
]

from benchmarks.config import (
    add_common_cli_args,
    load_yaml_config,
    merge_common_cli_overrides,
)
from benchmarks.constants import DEFAULT_ATOMIC_DENSITY, DEFAULT_NL_SAFETY_FACTOR
from benchmarks.suite_systems import (
    configs_for_mode,
    create_system,
    filter_configs_by_total_atoms,
    planned_atom_counts,
    resolve_nh3_dir,
)
from benchmarks.suite_utils import (
    build_failure_result,
    build_result,
    build_skipped_result,
    clean_gpu,
    clean_jax,
    create_run_directory,
    cuda_timed_runs,
    current_alloc_gb,
    ensure_jax_available,
    failure_error_type,
    format_num,
    jax_timed_serial,
    lazy_import_jax,
    make_csv_name,
    make_row_meta,
    measure_memory_jax,
    measure_memory_torch,
    save_results,
)
from nvalchemiops.neighbors import estimate_max_neighbors
from nvalchemiops.torch.interactions.electrostatics import (
    estimate_ewald_parameters,
    estimate_pme_parameters,
    ewald_real_space,
    ewald_reciprocal_space,
    ewald_summation,
    generate_k_vectors_ewald_summation,
    generate_k_vectors_pme,
    particle_mesh_ewald,
    pme_reciprocal_space,
)
from nvalchemiops.torch.neighbors import batch_cell_list

# =============================================================================
# Config Loading
# =============================================================================


def merge_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply CLI overrides on top of YAML config.

    Adds EL-specific ``--accuracies`` on top of the shared flags.
    """
    config = merge_common_cli_overrides(config, args)
    if args.accuracies is not None:
        config["accuracies"] = args.accuracies
    return config


# =============================================================================
# Inputs Bundle
# =============================================================================


@dataclass(frozen=True)
class ElectrostaticsInputs:
    """Per-config tensor bundle passed to the benchmark kernels.

    Bundles the system state (positions/charges/cell/pbc/batch_idx) with the
    precomputed neighbor list (nl_data/nl_shifts/nl_ptr) so the kernel-level
    benchmark functions take one object instead of eight positional args.

    Frozen — the runner builds one per (system, accuracy) config and passes
    it through. Field types are ``Any`` because the same shape applies to
    both torch tensors and jax arrays; the ``backend`` field disambiguates.
    """

    positions: Any
    charges: Any
    cell: Any
    pbc: Any
    batch_idx: Any
    nl_data: Any
    nl_shifts: Any
    nl_ptr: Any
    backend: str
    # max_atoms_per_system is needed by the JAX ewald_summation API under
    # jax.jit (shape inference can't query batch_idx.max() inside a trace).
    # Our systems are uniform, so this equals the per-system atom count.
    max_atoms_per_system: int = 0


_JAX_EL_KERNEL_CACHE: dict[int, dict[str, Any]] = {}


def _get_jax_el_kernels(jax_api: dict) -> dict[str, Any]:
    """Return cached JAX EL kernels with arrays passed as runtime inputs."""
    jax = jax_api["jax"]
    cache_key = id(jax)
    cached = _JAX_EL_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    jax_pme = jax_api["particle_mesh_ewald"]
    jax_pme_reciprocal = jax_api["pme_reciprocal_space"]
    jax_ewald = jax_api["ewald_summation"]
    jax_real = jax_api["ewald_real_space"]
    jax_reciprocal = jax_api["ewald_reciprocal_space"]

    def pme_real_kernel(
        positions,
        charges,
        cell,
        alpha,
        batch_idx,
        nl_data,
        nl_ptr,
        nl_shifts,
        *,
        compute_cg,
    ):
        return jax_real(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            batch_idx=batch_idx,
            neighbor_list=nl_data,
            neighbor_ptr=nl_ptr,
            neighbor_shifts=nl_shifts,
            compute_forces=True,
            compute_charge_gradients=compute_cg,
        )

    def pme_reciprocal_kernel(
        positions,
        charges,
        cell,
        alpha,
        batch_idx,
        k_vectors,
        k_squared,
        volume,
        cell_inv_t,
        moduli_x,
        moduli_y,
        moduli_z,
        *,
        mesh_dims,
        spline_order,
        compute_cg,
    ):
        return jax_pme_reciprocal(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            spline_order=spline_order,
            batch_idx=batch_idx,
            k_vectors=k_vectors,
            k_squared=k_squared,
            volume=volume,
            cell_inv_t=cell_inv_t,
            moduli_x=moduli_x,
            moduli_y=moduli_y,
            moduli_z=moduli_z,
            compute_forces=True,
            compute_charge_gradients=compute_cg,
        )

    def pme_kernel(
        positions,
        charges,
        cell,
        alpha,
        batch_idx,
        k_vectors,
        k_squared,
        nl_data,
        nl_ptr,
        nl_shifts,
        volume,
        cell_inv_t,
        moduli_x,
        moduli_y,
        moduli_z,
        *,
        mesh_dims,
        spline_order,
        accuracy,
        compute_cg,
    ):
        return jax_pme(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            spline_order=spline_order,
            batch_idx=batch_idx,
            k_vectors=k_vectors,
            k_squared=k_squared,
            neighbor_list=nl_data,
            neighbor_ptr=nl_ptr,
            neighbor_shifts=nl_shifts,
            volume=volume,
            cell_inv_t=cell_inv_t,
            moduli_x=moduli_x,
            moduli_y=moduli_y,
            moduli_z=moduli_z,
            compute_forces=True,
            compute_charge_gradients=compute_cg,
            accuracy=accuracy,
        )

    def ewald_real_kernel(
        positions,
        charges,
        cell,
        alpha,
        batch_idx,
        nl_data,
        nl_ptr,
        nl_shifts,
        *,
        compute_cg,
    ):
        return jax_real(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            batch_idx=batch_idx,
            neighbor_list=nl_data,
            neighbor_ptr=nl_ptr,
            neighbor_shifts=nl_shifts,
            compute_forces=True,
            compute_charge_gradients=compute_cg,
        )

    def ewald_reciprocal_kernel(
        positions,
        charges,
        cell,
        alpha,
        batch_idx,
        k_vectors,
        *,
        max_atoms_per_system,
        compute_cg,
    ):
        return jax_reciprocal(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            batch_idx=batch_idx,
            max_atoms_per_system=max_atoms_per_system,
            k_vectors=k_vectors,
            compute_forces=True,
            compute_charge_gradients=compute_cg,
        )

    def ewald_kernel(
        positions,
        charges,
        cell,
        alpha,
        batch_idx,
        k_vectors,
        nl_data,
        nl_ptr,
        nl_shifts,
        *,
        k_cutoff,
        max_atoms_per_system,
        accuracy,
        compute_cg,
    ):
        return jax_ewald(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            batch_idx=batch_idx,
            max_atoms_per_system=max_atoms_per_system,
            neighbor_list=nl_data,
            neighbor_ptr=nl_ptr,
            neighbor_shifts=nl_shifts,
            k_vectors=k_vectors,
            compute_forces=True,
            compute_charge_gradients=compute_cg,
            accuracy=accuracy,
        )

    kernels = {
        "pme_real": jax.jit(pme_real_kernel, static_argnames=("compute_cg",)),
        "pme_reciprocal": jax.jit(
            pme_reciprocal_kernel,
            static_argnames=("mesh_dims", "spline_order", "compute_cg"),
        ),
        "pme": jax.jit(
            pme_kernel,
            static_argnames=("mesh_dims", "spline_order", "accuracy", "compute_cg"),
        ),
        "ewald_real": jax.jit(ewald_real_kernel, static_argnames=("compute_cg",)),
        "ewald_reciprocal": jax.jit(
            ewald_reciprocal_kernel,
            static_argnames=("max_atoms_per_system", "compute_cg"),
        ),
        "ewald": jax.jit(
            ewald_kernel,
            static_argnames=(
                "k_cutoff",
                "max_atoms_per_system",
                "accuracy",
                "compute_cg",
            ),
        ),
    }
    _JAX_EL_KERNEL_CACHE[cache_key] = kernels
    return kernels


# =============================================================================
# Core Benchmarks
# =============================================================================


def benchmark_pme(
    inputs: ElectrostaticsInputs,
    alpha: Any,
    mesh_dims: tuple[int, int, int],
    spline_order: int,
    accuracy: float,
    compute_cg: bool,
    num_runs: int,
    warmup_runs: int,
    jax_api: dict | None = None,
) -> dict:
    """Benchmark Particle Mesh Ewald for one config.

    Dispatches on ``inputs.backend``. Accepts pre-converted f64 tensors/arrays
    on ``inputs`` to avoid redundant GPU copies.
    """
    if inputs.backend == "jax":
        return _benchmark_pme_jax(
            inputs,
            alpha,
            mesh_dims,
            spline_order,
            accuracy,
            compute_cg,
            num_runs,
            warmup_runs,
            jax_api,
        )

    k_vectors, k_squared = generate_k_vectors_pme(inputs.cell, mesh_dims)

    def run_real():
        return ewald_real_space(
            positions=inputs.positions,
            charges=inputs.charges,
            cell=inputs.cell,
            alpha=alpha,
            batch_idx=inputs.batch_idx,
            neighbor_list=inputs.nl_data,
            neighbor_ptr=inputs.nl_ptr,
            neighbor_shifts=inputs.nl_shifts,
            compute_forces=True,
            compute_charge_gradients=compute_cg,
        )

    def run_reciprocal():
        return pme_reciprocal_space(
            positions=inputs.positions,
            charges=inputs.charges,
            cell=inputs.cell,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            spline_order=spline_order,
            batch_idx=inputs.batch_idx,
            k_vectors=k_vectors,
            k_squared=k_squared,
            compute_forces=True,
            compute_charge_gradients=compute_cg,
        )

    def run_pme():
        particle_mesh_ewald(
            positions=inputs.positions,
            charges=inputs.charges,
            cell=inputs.cell,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            spline_order=spline_order,
            batch_idx=inputs.batch_idx,
            k_vectors=k_vectors,
            k_squared=k_squared,
            neighbor_list=inputs.nl_data,
            neighbor_ptr=inputs.nl_ptr,
            neighbor_shifts=inputs.nl_shifts,
            compute_forces=True,
            compute_charge_gradients=compute_cg,
            accuracy=accuracy,
        )

    _, mem_info = measure_memory_torch(run_pme)
    time_real_sec = cuda_timed_runs(run_real, num_runs, warmup_runs=warmup_runs)
    time_reciprocal_sec = cuda_timed_runs(
        run_reciprocal, num_runs, warmup_runs=warmup_runs
    )
    time_sec = cuda_timed_runs(run_pme, num_runs, warmup_runs=warmup_runs)
    return {
        "time_seconds": time_sec,
        "time_real_seconds": time_real_sec,
        "time_reciprocal_seconds": time_reciprocal_sec,
        "mem_info": mem_info,
        "timing_method": "torch_cuda_events",
        "timing_method_real": "torch_cuda_events",
        "timing_method_reciprocal": "torch_cuda_events",
    }


def benchmark_ewald(
    inputs: ElectrostaticsInputs,
    alpha: Any,
    k_cutoff: float,
    accuracy: float,
    compute_cg: bool,
    num_runs: int,
    warmup_runs: int,
    jax_api: dict | None = None,
) -> dict:
    """Benchmark Ewald summation for one config via the unified API.

    Raises
    ------
    NotImplementedError
        When ``inputs.backend='jax'`` and ``compute_cg=True``. The JAX
        ``ewald_summation`` API does not yet support charge gradients.
    """
    if inputs.backend == "jax":
        return _benchmark_ewald_jax(
            inputs,
            alpha,
            k_cutoff,
            accuracy,
            compute_cg,
            num_runs,
            warmup_runs,
            jax_api,
        )

    k_vectors = generate_k_vectors_ewald_summation(inputs.cell, k_cutoff)
    if k_vectors.ndim == 2:
        k_vectors = k_vectors.unsqueeze(0)

    def run_real():
        return ewald_real_space(
            positions=inputs.positions,
            charges=inputs.charges,
            cell=inputs.cell,
            alpha=alpha,
            batch_idx=inputs.batch_idx,
            neighbor_list=inputs.nl_data,
            neighbor_ptr=inputs.nl_ptr,
            neighbor_shifts=inputs.nl_shifts,
            compute_forces=True,
            compute_charge_gradients=compute_cg,
        )

    def run_reciprocal():
        return ewald_reciprocal_space(
            positions=inputs.positions,
            charges=inputs.charges,
            cell=inputs.cell,
            alpha=alpha,
            batch_idx=inputs.batch_idx,
            k_vectors=k_vectors,
            compute_forces=True,
            compute_charge_gradients=compute_cg,
        )

    def run_ewald():
        ewald_summation(
            positions=inputs.positions,
            charges=inputs.charges,
            cell=inputs.cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            batch_idx=inputs.batch_idx,
            neighbor_list=inputs.nl_data,
            neighbor_ptr=inputs.nl_ptr,
            neighbor_shifts=inputs.nl_shifts,
            k_vectors=k_vectors,
            compute_forces=True,
            compute_charge_gradients=compute_cg,
            accuracy=accuracy,
        )

    _, mem_info = measure_memory_torch(run_ewald)
    time_real_sec = cuda_timed_runs(run_real, num_runs, warmup_runs=warmup_runs)
    time_reciprocal_sec = cuda_timed_runs(
        run_reciprocal, num_runs, warmup_runs=warmup_runs
    )
    time_sec = cuda_timed_runs(run_ewald, num_runs, warmup_runs=warmup_runs)
    return {
        "time_seconds": time_sec,
        "time_real_seconds": time_real_sec,
        "time_reciprocal_seconds": time_reciprocal_sec,
        "mem_info": mem_info,
        "timing_method": "torch_cuda_events",
        "timing_method_real": "torch_cuda_events",
        "timing_method_reciprocal": "torch_cuda_events",
    }


def _jax_timed_with_serial_fallback(
    fn,
    num_runs: int,
    warmup_runs: int,
) -> tuple[float, str]:
    """Use batched JAX timing, falling back to per-call blocking on OOM."""
    try:
        time_sec = cuda_timed_runs(fn, num_runs, warmup_runs=warmup_runs, backend="jax")
        return time_sec, "jax_wall_block_until_ready"
    except Exception as e:
        if failure_error_type(e) != "OutOfMemoryError":
            raise
        clean_gpu()
        clean_jax()
        print("      batched JAX timing OOM; retrying with per-call blocking")
        return (
            jax_timed_serial(fn, num_runs, warmup_runs=warmup_runs),
            "jax_wall_block_each",
        )


def _benchmark_pme_jax(
    inputs,
    alpha,
    mesh_dims,
    spline_order,
    accuracy,
    compute_cg,
    num_runs,
    warmup_runs,
    jax_api,
):
    """JAX backend implementation of :func:`benchmark_pme`."""
    jax = jax_api["jax"]
    jnp = jax_api["jnp"]
    k_pme = jax_api["generate_k_vectors_pme"]
    compute_bspline_moduli_1d = jax_api["compute_bspline_moduli_1d"]
    kernels = _get_jax_el_kernels(jax_api)

    k_vectors, k_squared = k_pme(inputs.cell, mesh_dims)
    mesh_nx, mesh_ny, mesh_nz = mesh_dims
    dtype = inputs.positions.dtype
    cell_3d = inputs.cell if inputs.cell.ndim == 3 else jnp.expand_dims(inputs.cell, 0)
    cell_inv_t = jnp.transpose(jnp.linalg.inv(cell_3d), (0, 2, 1)).astype(dtype)
    volume = jnp.abs(jnp.linalg.det(cell_3d)).astype(dtype)
    miller_x = jnp.fft.fftfreq(mesh_nx, d=1.0 / mesh_nx).astype(dtype)
    miller_y = jnp.fft.fftfreq(mesh_ny, d=1.0 / mesh_ny).astype(dtype)
    miller_z = jnp.fft.rfftfreq(mesh_nz, d=1.0 / mesh_nz).astype(dtype)
    moduli_x = compute_bspline_moduli_1d(miller_x, mesh_nx, spline_order)
    moduli_y = compute_bspline_moduli_1d(miller_y, mesh_ny, spline_order)
    moduli_z = compute_bspline_moduli_1d(miller_z, mesh_nz, spline_order)

    def run_real():
        return kernels["pme_real"](
            inputs.positions,
            inputs.charges,
            inputs.cell,
            alpha,
            inputs.batch_idx,
            inputs.nl_data,
            inputs.nl_ptr,
            inputs.nl_shifts,
            compute_cg=compute_cg,
        )

    def run_reciprocal():
        return kernels["pme_reciprocal"](
            inputs.positions,
            inputs.charges,
            inputs.cell,
            alpha,
            inputs.batch_idx,
            k_vectors,
            k_squared,
            volume,
            cell_inv_t,
            moduli_x,
            moduli_y,
            moduli_z,
            mesh_dims=mesh_dims,
            spline_order=spline_order,
            compute_cg=compute_cg,
        )

    def run_pme():
        return kernels["pme"](
            inputs.positions,
            inputs.charges,
            inputs.cell,
            alpha,
            inputs.batch_idx,
            k_vectors,
            k_squared,
            inputs.nl_data,
            inputs.nl_ptr,
            inputs.nl_shifts,
            volume,
            cell_inv_t,
            moduli_x,
            moduli_y,
            moduli_z,
            mesh_dims=mesh_dims,
            spline_order=spline_order,
            accuracy=accuracy,
            compute_cg=compute_cg,
        )

    # JAX memory is unavailable in this suite; avoid an extra allocation-heavy
    # execution that would only produce NaN memory fields.
    _, mem_info = measure_memory_jax(run_pme, jax)
    time_real_sec, timing_method_real = _jax_timed_with_serial_fallback(
        run_real, num_runs, warmup_runs
    )
    time_reciprocal_sec, timing_method_reciprocal = _jax_timed_with_serial_fallback(
        run_reciprocal, num_runs, warmup_runs
    )
    time_sec, timing_method = _jax_timed_with_serial_fallback(
        run_pme, num_runs, warmup_runs
    )
    return {
        "time_seconds": time_sec,
        "time_real_seconds": time_real_sec,
        "time_reciprocal_seconds": time_reciprocal_sec,
        "timing_method": timing_method,
        "timing_method_real": timing_method_real,
        "timing_method_reciprocal": timing_method_reciprocal,
        "mem_info": mem_info,
    }


def _benchmark_ewald_jax(
    inputs,
    alpha,
    k_cutoff,
    accuracy,
    compute_cg,
    num_runs,
    warmup_runs,
    jax_api,
):
    """JAX backend implementation of :func:`benchmark_ewald`."""

    jax = jax_api["jax"]
    k_ewald = jax_api["generate_k_vectors_ewald_summation"]
    kernels = _get_jax_el_kernels(jax_api)

    k_vectors = k_ewald(inputs.cell, k_cutoff)
    if k_vectors.ndim == 2:
        k_vectors = jax_api["jnp"].expand_dims(k_vectors, 0)

    def run_real():
        return kernels["ewald_real"](
            inputs.positions,
            inputs.charges,
            inputs.cell,
            alpha,
            inputs.batch_idx,
            inputs.nl_data,
            inputs.nl_ptr,
            inputs.nl_shifts,
            compute_cg=compute_cg,
        )

    def run_reciprocal():
        return kernels["ewald_reciprocal"](
            inputs.positions,
            inputs.charges,
            inputs.cell,
            alpha,
            inputs.batch_idx,
            k_vectors,
            max_atoms_per_system=inputs.max_atoms_per_system,
            compute_cg=compute_cg,
        )

    def run_ewald():
        return kernels["ewald"](
            inputs.positions,
            inputs.charges,
            inputs.cell,
            alpha,
            inputs.batch_idx,
            k_vectors,
            inputs.nl_data,
            inputs.nl_ptr,
            inputs.nl_shifts,
            k_cutoff=k_cutoff,
            max_atoms_per_system=inputs.max_atoms_per_system,
            accuracy=accuracy,
            compute_cg=compute_cg,
        )

    # JAX memory is unavailable in this suite; avoid an extra allocation-heavy
    # execution that would only produce NaN memory fields.
    _, mem_info = measure_memory_jax(run_ewald, jax)
    time_real_sec, timing_method_real = _jax_timed_with_serial_fallback(
        run_real, num_runs, warmup_runs
    )
    time_reciprocal_sec, timing_method_reciprocal = _jax_timed_with_serial_fallback(
        run_reciprocal, num_runs, warmup_runs
    )
    time_sec, timing_method = _jax_timed_with_serial_fallback(
        run_ewald, num_runs, warmup_runs
    )
    return {
        "time_seconds": time_sec,
        "time_real_seconds": time_real_sec,
        "time_reciprocal_seconds": time_reciprocal_sec,
        "timing_method": timing_method,
        "timing_method_real": timing_method_real,
        "timing_method_reciprocal": timing_method_reciprocal,
        "mem_info": mem_info,
    }


# =============================================================================
# run_from_config helpers
# =============================================================================


def _el_tensors_from_data(data, backend):
    """Convert a ``create_system`` dict to f64 tensors/arrays for electrostatics.

    Returns ``(positions, charges, cell, pbc, batch_idx)`` as a tuple. The
    caller drops its reference to ``data`` afterward so the f32 originals can
    be released.
    """
    pbc = data["pbc"]
    batch_idx = data["batch_idx"]
    if backend == "torch":
        positions = data["positions"].to(torch.float64)
        charges = data["charges"].to(torch.float64)
        cell = data["cell"].to(torch.float64)
    else:
        import jax.numpy as jnp

        positions = data["positions"].astype(jnp.float64)
        charges = data["charges"].astype(jnp.float64)
        cell = data["cell"].astype(jnp.float64)
    return positions, charges, cell, pbc, batch_idx


def _el_estimate_params(positions, cell, batch_idx, backend, accuracy, jax_api):
    """Estimate PME + Ewald parameters for one system at the given accuracy.

    Dispatches on backend; returns ``(pme_params, ewald_params)`` parameter
    dataclasses from the underlying library.
    """
    if backend == "torch":
        pme_params = estimate_pme_parameters(
            positions, cell, batch_idx=batch_idx, accuracy=accuracy
        )
        ewald_params = estimate_ewald_parameters(
            positions, cell, batch_idx=batch_idx, accuracy=accuracy
        )
    else:
        pme_params = jax_api["estimate_pme_parameters"](
            positions, cell, batch_idx=batch_idx, accuracy=accuracy
        )
        ewald_params = jax_api["estimate_ewald_parameters"](
            positions, cell, batch_idx=batch_idx, accuracy=accuracy
        )
    return pme_params, ewald_params


def _el_unpack_params(pme_params, ewald_params, backend):
    """Extract alpha / cutoffs / mesh_dims from the parameter dataclasses.

    Returns ``(alpha, real_cutoff, mesh_dims, k_cutoff)``. ``alpha`` keeps the
    per-system tensor/array shape the component kernels consume. For diagnostic
    printing, use ``float(alpha.mean())``.
    """
    if backend == "torch":
        alpha = pme_params.alpha.clone()
        real_cutoff = (
            pme_params.real_space_cutoff[0].item()
            if pme_params.real_space_cutoff.dim() > 0
            else pme_params.real_space_cutoff.item()
        )
        mesh_dims = tuple(pme_params.mesh_dimensions)
        k_cutoff = ewald_params.reciprocal_space_cutoff.max().item()
    else:
        alpha = pme_params.alpha
        real_cutoff = float(pme_params.real_space_cutoff[0])
        md = pme_params.mesh_dimensions
        mesh_dims = (int(md[0]), int(md[1]), int(md[2]))
        k_cutoff = float(ewald_params.reciprocal_space_cutoff.max())
    return alpha, real_cutoff, mesh_dims, k_cutoff


def _torch_neighbor_matrix_to_list_chunked(
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    neighbor_shift_matrix: torch.Tensor,
    *,
    fill_value: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a large Torch neighbor matrix to COO/CSR in row chunks.

    PyTorch's ``nonzero``/``where`` rejects tensors with more than INT_MAX
    elements. Large electrostatics reportable rows can exceed that matrix size
    even when the actual neighbor count fits in memory. Chunking avoids that
    conversion API ceiling without skipping atoms, reducing the benchmark grid,
    or changing the neighbor-list semantics.
    """
    if num_neighbors.shape[0] == 0:
        neighbor_list = torch.zeros(
            2,
            0,
            dtype=neighbor_matrix.dtype,
            device=neighbor_matrix.device,
        )
        neighbor_ptr = torch.zeros(1, dtype=torch.int32, device=neighbor_matrix.device)
        neighbor_list_shifts = torch.empty(
            0,
            3,
            dtype=neighbor_shift_matrix.dtype,
            device=neighbor_shift_matrix.device,
        )
        return neighbor_list, neighbor_ptr, neighbor_list_shifts

    max_found = int(num_neighbors.max().item())
    if max_found > neighbor_matrix.shape[1]:
        raise ValueError(
            f"neighbor matrix width {neighbor_matrix.shape[1]} is smaller than "
            f"observed max neighbors {max_found}"
        )

    neighbor_ptr = torch.zeros(
        num_neighbors.shape[0] + 1,
        dtype=torch.int32,
        device=neighbor_matrix.device,
    )
    torch.cumsum(num_neighbors, dim=0, out=neighbor_ptr[1:])
    total_pairs = int(neighbor_ptr[-1].item())
    neighbor_list = torch.empty(
        (2, total_pairs),
        dtype=neighbor_matrix.dtype,
        device=neighbor_matrix.device,
    )
    neighbor_list_shifts = torch.empty(
        (total_pairs, 3),
        dtype=neighbor_shift_matrix.dtype,
        device=neighbor_shift_matrix.device,
    )

    max_neighbors = max(int(neighbor_matrix.shape[1]), 1)
    max_chunk_elements = torch.iinfo(torch.int32).max // 8
    chunk_rows = max(1, max_chunk_elements // max_neighbors)

    for start in range(0, neighbor_matrix.shape[0], chunk_rows):
        end = min(start + chunk_rows, neighbor_matrix.shape[0])
        out_start = int(neighbor_ptr[start].item())
        out_end = int(neighbor_ptr[end].item())
        if out_start == out_end:
            continue

        matrix_chunk = neighbor_matrix[start:end]
        active = matrix_chunk != fill_value
        rows, _ = torch.where(active)
        neighbor_list[0, out_start:out_end] = rows.to(neighbor_matrix.dtype) + start
        neighbor_list[1, out_start:out_end] = matrix_chunk[active].to(
            neighbor_matrix.dtype
        )
        neighbor_list_shifts[out_start:out_end] = neighbor_shift_matrix[start:end][
            active
        ]

    return neighbor_list, neighbor_ptr, neighbor_list_shifts


def _el_build_nl(positions, cell, pbc, batch_idx, real_cutoff, backend, jax_api):
    """Build the real-space neighbor list in LIST (COO + ptr) format."""
    maxnb = estimate_max_neighbors(
        real_cutoff,
        atomic_density=DEFAULT_ATOMIC_DENSITY,
        safety_factor=DEFAULT_NL_SAFETY_FACTOR,
    )
    if backend == "torch":
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = batch_cell_list(
            positions=positions,
            cutoff=real_cutoff,
            batch_idx=batch_idx,
            pbc=pbc,
            cell=cell,
            max_neighbors=maxnb,
            return_neighbor_list=False,
        )
        return _torch_neighbor_matrix_to_list_chunked(
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
            fill_value=positions.shape[0],
        )
    jnp = jax_api["jnp"]
    num_systems = int(cell.shape[0]) if cell.ndim == 3 else 1
    atoms_per_system = positions.shape[0] // max(num_systems, 1)
    batch_ptr = jnp.arange(num_systems + 1, dtype=jnp.int32) * int(atoms_per_system)
    max_total_cells, _, neighbor_search_radius = jax_api[
        "estimate_batch_cell_list_sizes"
    ](
        positions=positions,
        batch_ptr=batch_ptr,
        cell=cell,
        cutoff=float(real_cutoff),
        pbc=pbc,
    )

    def build_nl_matrix(pos, cell_, pbc_, batch_idx_, batch_ptr_):
        return jax_api["neighbor_list"](
            positions=pos,
            cutoff=float(real_cutoff),
            cell=cell_,
            pbc=pbc_,
            batch_idx=batch_idx_,
            batch_ptr=batch_ptr_,
            method="batch_cell_list",
            return_neighbor_list=False,
            strategy="atom_centric",
            atom_centric_path="direct",
            max_neighbors=int(maxnb),
            max_total_cells=int(max_total_cells),
            neighbor_search_radius=neighbor_search_radius,
        )

    neighbor_matrix, num_neighbors, neighbor_matrix_shifts = jax_api["jax"].jit(
        build_nl_matrix
    )(
        positions,
        cell,
        pbc,
        batch_idx,
        batch_ptr,
    )
    jax_api["jax"].block_until_ready(neighbor_matrix)
    from nvalchemiops.jax.neighbors.neighbor_utils import (
        get_neighbor_list_from_neighbor_matrix,
    )

    nl = get_neighbor_list_from_neighbor_matrix(
        neighbor_matrix,
        num_neighbors,
        neighbor_shift_matrix=neighbor_matrix_shifts,
        fill_value=positions.shape[0],
    )
    jax_api["jax"].block_until_ready(nl[0])
    return nl


# =============================================================================
# Config-Driven Runner
# =============================================================================


class ElConfigSetup(NamedTuple):
    """Return shape of :func:`_el_setup_config`.

    Named over positional unpacking — the eight fields are a mix of
    backend-polymorphic tensors (``inputs``, ``alpha``) and plain scalars
    that the method loop consumes.
    """

    inputs: ElectrostaticsInputs
    alpha: Any  # torch 0-dim tensor (torch) or python float (jax)
    real_cutoff: float
    mesh_dims: tuple[int, int, int]
    k_cutoff: float
    atoms_per_system: int
    batch_size: int
    actual_total: int


class ElConfigFailure(NamedTuple):
    """Expected setup failure that still needs a CSV row."""

    error: str
    error_type: str
    failure_stage: str


def _ordered_configs_for_backend(configs: list[dict], sys_name: str, backend: str):
    """Order configs for stable memory behavior without changing planned rows."""
    if backend != "jax":
        return configs
    return sorted(
        configs,
        key=lambda cfg: planned_atom_counts(sys_name, cfg)[2],
        reverse=True,
    )


def _el_setup_config(
    cfg: dict, sys_name: str, accuracy: float, backend: str, jax_api: dict | None
) -> ElConfigSetup | ElConfigFailure:
    """Build the per-config :class:`ElectrostaticsInputs` + derived params.

    Returns an ``ElConfigFailure`` on expected failures (create_system error,
    params estimation failure, NL build failure) after printing a diagnostic
    line so callers can still emit ``success=False`` CSV rows.
    """
    n, bs = cfg["num_atoms"], cfg["batch_size"]
    try:
        data = create_system(
            sys_name,
            num_atoms=n,
            pdb_path=cfg.get("pdb_path"),
            batch_size=bs,
            backend=backend,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        error_type = failure_error_type(e)
        if error_type == "OutOfMemoryError":
            clean_gpu()
            if backend == "jax":
                clean_jax()
        print(f"    FAILED: {e}")
        return ElConfigFailure(str(e), error_type, "system_setup")

    atoms_per_system = data["atoms_per_system"]
    actual_total = data.get("total_atoms", atoms_per_system)
    batch_size = data.get("batch_size", 1)
    print(
        f"\n  {format_num(atoms_per_system)} atoms × {batch_size} batch = "
        f"{format_num(actual_total)} total  [GPU: {current_alloc_gb(backend):.1f} GB allocated]"
    )

    positions, charges, cell, pbc, batch_idx = _el_tensors_from_data(data, backend)
    del data

    try:
        pme_params, ewald_params = _el_estimate_params(
            positions, cell, batch_idx, backend, accuracy, jax_api
        )
    except (RuntimeError, ValueError) as e:
        error_type = failure_error_type(e)
        if error_type == "OutOfMemoryError":
            clean_gpu()
            if backend == "jax":
                clean_jax()
        print(f"    FAILED (params): {e}")
        return ElConfigFailure(str(e), error_type, "parameter_setup")

    alpha, real_cutoff, mesh_dims, k_cutoff = _el_unpack_params(
        pme_params, ewald_params, backend
    )
    del pme_params, ewald_params

    try:
        nl_data, nl_ptr, nl_shifts = _el_build_nl(
            positions, cell, pbc, batch_idx, real_cutoff, backend, jax_api
        )
    except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError) as e:
        error_type = failure_error_type(e)
        if error_type == "OutOfMemoryError":
            clean_gpu()
            if backend == "jax":
                clean_jax()
        print(f"    FAILED (NL): {e}")
        return ElConfigFailure(str(e), error_type, "neighbor_list_setup")

    inputs = ElectrostaticsInputs(
        positions=positions,
        charges=charges,
        cell=cell,
        pbc=pbc,
        batch_idx=batch_idx,
        nl_data=nl_data,
        nl_shifts=nl_shifts,
        nl_ptr=nl_ptr,
        backend=backend,
        max_atoms_per_system=int(atoms_per_system),
    )
    return ElConfigSetup(
        inputs=inputs,
        alpha=alpha,
        real_cutoff=real_cutoff,
        mesh_dims=mesh_dims,
        k_cutoff=k_cutoff,
        atoms_per_system=atoms_per_system,
        batch_size=batch_size,
        actual_total=actual_total,
    )


def _el_run_method(
    method,
    inputs,
    alpha,
    mesh_dims,
    k_cutoff,
    spline_order,
    accuracy,
    compute_cg,
    num_runs,
    warmup_runs,
    jax_api,
    row_meta,
):
    """Run one ``(method, compute_cg)`` combination and build a result row.

    Catches OOM, ``NotImplementedError``, and other exceptions as
    ``success=False`` rows for CSV visibility and plotter filtering.
    ``row_meta`` carries the identity fields for :func:`build_result`.
    """
    cg_label = "+cg" if compute_cg else ""
    label = f"{method.upper()}{cg_label}"
    method_col = f"{method}{'_cg' if compute_cg else ''}"
    try:
        if method == "pme":
            r = benchmark_pme(
                inputs,
                alpha,
                mesh_dims,
                spline_order,
                accuracy,
                compute_cg,
                num_runs,
                warmup_runs,
                jax_api=jax_api,
            )
        else:
            r = benchmark_ewald(
                inputs,
                alpha,
                k_cutoff,
                accuracy,
                compute_cg,
                num_runs,
                warmup_runs,
                jax_api=jax_api,
            )
        result = build_result(
            method=method_col,
            time_seconds=r["time_seconds"],
            mem_info=r["mem_info"],
            accuracy=accuracy,
            timing_runs=num_runs,
            warmup_runs=warmup_runs,
            timing_method=r.get("timing_method"),
            timing_method_real=r.get("timing_method_real"),
            timing_method_reciprocal=r.get("timing_method_reciprocal"),
            time_real_us_per_atom=(
                (r["time_real_seconds"] * 1e6) / row_meta["total_atoms"]
                if row_meta["total_atoms"] > 0
                else 0.0
            ),
            time_reciprocal_us_per_atom=(
                (r["time_reciprocal_seconds"] * 1e6) / row_meta["total_atoms"]
                if row_meta["total_atoms"] > 0
                else 0.0
            ),
            **row_meta,
        )
        mem_suffix = (
            f" | {result['mem_peak_gb']:.1f} GB" if inputs.backend == "torch" else ""
        )
        print(f"    {label:10s}: {result['time_us_per_atom']:.3f} μs/atom{mem_suffix}")
        return result
    except NotImplementedError as e:
        # Expected for JAX ewald+cg. Emit success=False so the plotter can filter.
        print(f"    {label:10s}: UNSUPPORTED - {e}")
        return build_failure_result(
            method=method_col,
            accuracy=accuracy,
            timing_runs=num_runs,
            warmup_runs=warmup_runs,
            error=str(e),
            error_type=type(e).__name__,
            failure_stage="method_unsupported",
            **row_meta,
        )
    except torch.cuda.OutOfMemoryError:
        print(f"    {label:10s}: OOM")
        clean_gpu()
        return build_failure_result(
            method=method_col,
            accuracy=accuracy,
            error="CUDA out of memory",
            error_type="OutOfMemoryError",
            failure_stage=f"{method_col}_timing",
            timing_runs=num_runs,
            warmup_runs=warmup_runs,
            **row_meta,
        )
    except Exception as e:
        error_type = failure_error_type(e)
        if error_type == "OutOfMemoryError":
            clean_gpu()
        print(f"    {label:10s}: FAILED - {e}")
        return build_failure_result(
            method=method_col,
            accuracy=accuracy,
            error=str(e),
            error_type=error_type,
            failure_stage=f"{method_col}_timing",
            timing_runs=num_runs,
            warmup_runs=warmup_runs,
            **row_meta,
        )


def dry_run_from_config(config: dict, backend: str | None = None) -> list[dict]:
    """Print and return the expanded electrostatics plan without allocation."""
    params = config["parameters"]
    max_total_atoms = params.get("max_total_atoms")
    accuracies = config["accuracies"]
    cg_options = config["compute_charge_gradients"]
    method_names = [m["name"] for m in config["methods"] if m.get("enabled", True)]
    plan_output = config.get("runtime", {}).get("plan_output", "dry_run")
    if backend is None:
        backend = config.get("runtime", {}).get("backend", "torch")
    rows = []
    for sys_name, sys_config in config["systems"].items():
        if not sys_config.get("enabled", True):
            continue
        nh3_dir = resolve_nh3_dir(sys_config)
        for mode_name, mode_config in config["scaling"].items():
            if not isinstance(mode_config, dict) or not mode_config.get(
                "enabled", True
            ):
                continue
            configs = configs_for_mode(
                mode_name,
                mode_config,
                sys_name,
                sys_config,
                nh3_dir,
                plan_only=True,
            )
            configs, skipped = filter_configs_by_total_atoms(
                configs, sys_name, max_total_atoms
            )
            for cfg, total_atoms in skipped:
                atoms_per_system, batch_size, _ = planned_atom_counts(sys_name, cfg)
                rows.extend(
                    {
                        "benchmark": "el",
                        "backend": backend,
                        "system": sys_name,
                        "mode": mode_name,
                        "atoms_per_system": atoms_per_system,
                        "batch_size": batch_size,
                        "total_atoms": total_atoms,
                        "method": method,
                        "accuracy": accuracy,
                        "compute_cg": compute_cg,
                        "reason": f">{max_total_atoms} max_total_atoms",
                    }
                    for accuracy in accuracies
                    for method in method_names
                    for compute_cg in cg_options
                )
            for cfg in configs:
                atoms_per_system, batch_size, total_atoms = planned_atom_counts(
                    sys_name, cfg
                )
                for accuracy in accuracies:
                    for method in method_names:
                        rows.extend(
                            [
                                {
                                    "benchmark": "el",
                                    "backend": backend,
                                    "system": sys_name,
                                    "mode": mode_name,
                                    "atoms_per_system": atoms_per_system,
                                    "batch_size": batch_size,
                                    "total_atoms": total_atoms,
                                    "method": method,
                                    "accuracy": accuracy,
                                    "compute_cg": compute_cg,
                                    "reason": "",
                                }
                                for compute_cg in cg_options
                            ]
                        )
    if plan_output != "count":
        print("EL dry-run plan")
        for row in rows:
            suffix = f" SKIP {row['reason']}" if row["reason"] else ""
            print(
                "  {system}/{mode} backend={backend} method={method} "
                "accuracy={accuracy} cg={compute_cg} N={atoms_per_system} "
                "batch={batch_size} total={total_atoms}{suffix}".format(
                    **row, suffix=suffix
                )
            )
    print(f"EL dry-run rows: {len(rows)}")
    return rows


def run_from_config(
    config: dict,
    output_dir: Path | str | None = None,
    backend: str | None = None,
) -> list[dict]:
    """Run electrostatics benchmarks driven by YAML config.

    Parameters
    ----------
    backend : str, optional
        ``'torch'`` or ``'jax'``. Pulled from ``config['runtime']['backend']``
        when None. Default is ``'torch'``.
    """
    params = config["parameters"]
    num_runs = params["timing_runs"]
    warmup_runs = params["warmup_runs"]
    accuracies = config["accuracies"]
    max_total_atoms = params.get("max_total_atoms")
    cg_options = config["compute_charge_gradients"]
    methods_config = config["methods"]
    method_names = [m["name"] for m in methods_config if m.get("enabled", True)]
    # YAML is authoritative for spline_order; None when PME isn't enabled.
    pme_spline_order = next(
        (m["spline_order"] for m in methods_config if m["name"] == "pme"),
        None,
    )

    if backend is None:
        backend = config.get("runtime", {}).get("backend", "torch")
    if backend == "warp":
        raise ValueError(
            "Electrostatics benchmark supports torch and jax backends, not warp."
        )
    if config.get("runtime", {}).get("dry_run", False):
        return dry_run_from_config(config, backend=backend)
    jax_api = lazy_import_jax(need_electrostatics=True) if backend == "jax" else None

    if output_dir is None:
        output_dir = create_run_directory(config["output"]["base_dir"], prefix="el")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        gpu_name = torch.cuda.get_device_name(0)
    except (AssertionError, RuntimeError):
        gpu_name = "N/A (no CUDA)"
    print(f"Electrostatics Benchmark | GPU: {gpu_name}")
    print(f"Backend: {backend}")
    print(f"Methods: {method_names} | Accuracies: {accuracies}")
    print(f"Timing: {num_runs} runs")
    print(f"Output: {output_dir}")

    all_results = []

    for sys_name, sys_config in config["systems"].items():
        if not sys_config.get("enabled", True):
            continue
        nh3_dir = resolve_nh3_dir(sys_config)

        for mode_name, mode_config in config["scaling"].items():
            if not isinstance(mode_config, dict) or not mode_config.get(
                "enabled", True
            ):
                continue
            print(f"\n{'=' * 70}")
            print(f"ELECTROSTATICS: {sys_name.upper()} / {mode_name}")
            print(f"{'=' * 70}")

            configs = configs_for_mode(
                mode_name, mode_config, sys_name, sys_config, nh3_dir
            )
            configs, skipped = filter_configs_by_total_atoms(
                configs, sys_name, max_total_atoms
            )
            configs = _ordered_configs_for_backend(configs, sys_name, backend)
            results = []
            for cfg, skipped_total in skipped:
                print(
                    f"  SKIP total atoms {format_num(skipped_total)} "
                    f"(>{format_num(max_total_atoms)})"
                )
                atoms_per_system, batch_size, total_atoms = planned_atom_counts(
                    sys_name, cfg
                )
                row_meta = make_row_meta(
                    sys_name,
                    mode_name,
                    backend,
                    atoms_per_system,
                    batch_size,
                    total_atoms,
                )
                reason = f">{max_total_atoms} max_total_atoms"
                results.extend(
                    build_skipped_result(
                        method=f"{method}{'_cg' if compute_cg else ''}",
                        accuracy=accuracy,
                        reason=reason,
                        timing_runs=num_runs,
                        warmup_runs=warmup_runs,
                        **row_meta,
                    )
                    for accuracy in accuracies
                    for method in method_names
                    for compute_cg in cg_options
                )
            if not configs:
                if results:
                    csv_name = make_csv_name("el", sys_name, mode_name)
                    save_results(
                        results, output_dir / csv_name, replace_backend=backend
                    )
                    all_results.extend(results)
                continue
            for accuracy in accuracies:
                print(f"\n  --- Accuracy: {accuracy:.0e} ---")

                for cfg in configs:
                    atoms_per_system, batch_size, total_atoms = planned_atom_counts(
                        sys_name, cfg
                    )
                    if max_total_atoms is not None and total_atoms > max_total_atoms:
                        print(
                            f"  {format_num(atoms_per_system)}×{batch_size}: "
                            f"SKIP (>{format_num(max_total_atoms)})"
                        )
                        row_meta = make_row_meta(
                            sys_name,
                            mode_name,
                            backend,
                            atoms_per_system,
                            batch_size,
                            total_atoms,
                        )
                        reason = f">{max_total_atoms} max_total_atoms"
                        results.extend(
                            build_skipped_result(
                                method=f"{method}{'_cg' if compute_cg else ''}",
                                accuracy=accuracy,
                                reason=reason,
                                timing_runs=num_runs,
                                warmup_runs=warmup_runs,
                                **row_meta,
                            )
                            for method in method_names
                            for compute_cg in cg_options
                        )
                        continue

                    if backend == "jax":
                        clean_jax()
                    clean_gpu()
                    setup = _el_setup_config(cfg, sys_name, accuracy, backend, jax_api)
                    if isinstance(setup, ElConfigFailure):
                        row_meta = make_row_meta(
                            sys_name,
                            mode_name,
                            backend,
                            atoms_per_system,
                            batch_size,
                            total_atoms,
                        )
                        results.extend(
                            build_failure_result(
                                method=f"{method}{'_cg' if compute_cg else ''}",
                                accuracy=accuracy,
                                error=setup.error,
                                error_type=setup.error_type,
                                failure_stage=setup.failure_stage,
                                timing_runs=num_runs,
                                warmup_runs=warmup_runs,
                                **row_meta,
                            )
                            for method in method_names
                            for compute_cg in cg_options
                        )
                        if backend == "jax":
                            clean_jax()
                        continue

                    print(
                        f"    alpha={float(setup.alpha.mean()):.4f}, "
                        f"r_cut={setup.real_cutoff:.2f}Å, "
                        f"NL pairs={setup.inputs.nl_data.shape[1]:,}"
                    )

                    row_meta = make_row_meta(
                        sys_name,
                        mode_name,
                        backend,
                        setup.atoms_per_system,
                        setup.batch_size,
                        setup.actual_total,
                    )
                    method_variants = [
                        (method, compute_cg)
                        for method in method_names
                        for compute_cg in cg_options
                    ]
                    for idx, (method, compute_cg) in enumerate(method_variants):
                        result = _el_run_method(
                            method,
                            setup.inputs,
                            setup.alpha,
                            setup.mesh_dims,
                            setup.k_cutoff,
                            pme_spline_order,
                            accuracy,
                            compute_cg,
                            num_runs,
                            warmup_runs,
                            jax_api,
                            row_meta,
                        )
                        if result is not None:
                            results.append(result)
                        if (
                            backend == "jax"
                            and result is not None
                            and result.get("success") is False
                            and result.get("error_type") == "OutOfMemoryError"
                        ):
                            clean_jax()
                            for next_method, next_compute_cg in method_variants[
                                idx + 1 :
                            ]:
                                next_method_col = (
                                    f"{next_method}_cg"
                                    if next_compute_cg
                                    else next_method
                                )
                                results.append(
                                    build_failure_result(
                                        method=next_method_col,
                                        accuracy=accuracy,
                                        error=(
                                            "Previous JAX method OOM for this "
                                            "configuration; remaining variants "
                                            "were not executed in the same process."
                                        ),
                                        error_type="SkippedAfterOOM",
                                        failure_stage="post_oom_containment",
                                        timing_runs=num_runs,
                                        warmup_runs=warmup_runs,
                                        **row_meta,
                                    )
                                )
                            break

                    del setup
                    if backend == "jax":
                        clean_jax()

            if results:
                csv_name = make_csv_name("el", sys_name, mode_name)
                save_results(results, output_dir / csv_name, replace_backend=backend)
                all_results.extend(results)

    print(f"\nCOMPLETE: {len(all_results)} results in {output_dir}")
    return all_results


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    """Parse command-line arguments for electrostatics benchmarks."""
    parser = argparse.ArgumentParser(
        description="Electrostatics Benchmark (2 systems × 3 modes)"
    )
    parser.add_argument("--config", type=Path, required=True)
    add_common_cli_args(parser)
    parser.add_argument(
        "--accuracies",
        "-a",
        type=float,
        nargs="+",
        default=None,
        help="Override target accuracies (Hartree/atom)",
    )
    return parser.parse_args()


def main():
    """Run electrostatics benchmarks."""
    args = parse_args()
    config = load_yaml_config(args.config)
    config = merge_cli_overrides(config, args)

    backend = args.backend or config.get("runtime", {}).get("backend", "torch")
    plan_only = (
        getattr(args, "dry_run", False)
        or getattr(args, "list_plan", False)
        or getattr(args, "count_plan", False)
    )
    if backend == "jax" and not plan_only:
        ensure_jax_available(need_electrostatics=True)

    results = run_from_config(config, output_dir=args.output_dir, backend=backend)
    if not results:
        return 1
    if not any(row.get("success", True) is not False for row in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
