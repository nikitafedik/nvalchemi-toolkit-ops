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

"""Neighbor List Benchmark.

Benchmarks naive O(N²) and cell-list O(N) neighbor list construction
across two chemical systems (CsCl, NH3) and three scaling modes.
Configuration is loaded from a per-module YAML file.

Usage (run from the repository root):
    python -m benchmarks.neighborlist.benchmark_neighborlist \
        --config benchmarks/neighborlist/benchmark_config.yaml
    python -m benchmarks.neighborlist.benchmark_neighborlist \
        --config benchmarks/neighborlist/benchmark_config.yaml \
        --system cscl --mode system_size
    python -m benchmarks.neighborlist.benchmark_neighborlist \
        --config benchmarks/neighborlist/benchmark_config.yaml \
        --output-dir docs/benchmarks/benchmark_results

    # JAX backend (the runner also sets this defensively before importing JAX)
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python -m benchmarks.neighborlist.benchmark_neighborlist \
        --config benchmarks/neighborlist/benchmark_config.yaml --backend jax

Backends
--------
``--backend torch`` (default) uses the warp-based torch kernels and CUDA
events for timing. ``--backend jax`` uses the JAX wrappers in
``nvalchemiops.jax.neighbors`` and wall-clock timing with
``jax.block_until_ready``.

Environment variables for ``--backend jax``:

- ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` — set by the runner before importing
  JAX; exporting it explicitly is also safe.
- ``JAX_ENABLE_X64=True`` — optional (electrostatics is the only benchmark
  that hard-requires this).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

__all__ = [
    "benchmark_nl",
    "main",
    "merge_cli_overrides",
    "parse_args",
    "run_from_config",
]

from benchmarks.config import (
    add_common_cli_args,
    enabled_method_names,
    load_yaml_config,
    merge_common_cli_overrides,
    normalize_method_name,
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
    create_run_directory,
    cuda_timed_runs,
    current_alloc_gb,
    ensure_jax_available,
    format_num,
    lazy_import_jax,
    make_csv_name,
    make_row_meta,
    measure_memory_jax,
    measure_memory_torch,
    save_results,
    sync_gpu,
)

# Official nvalchemiops public APIs used by the neighbor-list runner.
from nvalchemiops.neighbors import estimate_max_neighbors
from nvalchemiops.torch.neighbors import (
    batch_cell_list,
    batch_naive_neighbor_list,
    cell_list,
    naive_neighbor_list,
)
from nvalchemiops.torch.neighbors import (
    neighbor_list as torch_neighbor_list,
)

_SUPPORTED_NL_METHODS = {
    "cell_list",
    "batch_cell_list",
    "cluster_tile",
    "batch_cluster_tile",
    "naive_neighbor_list",
    "batch_naive_neighbor_list",
}

_CLUSTER_TILE_METHODS = {"cluster_tile", "batch_cluster_tile"}
_REQUIRED_WARP_TILE_BUILTINS = (
    "tile_from_thread_dispatch_func",
    "tile_arange_dispatch_func",
    "tile_load_tuple_dispatch_func",
    "tile_store_dispatch_func",
    "untile_value_func",
)


def merge_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply CLI overrides on top of YAML config.

    Common flags are merged by :func:`merge_common_cli_overrides`; this
    wrapper adds the NL-specific ``--cutoffs`` handling.
    """
    config = merge_common_cli_overrides(config, args)
    if args.cutoffs is not None:
        config["parameters"]["cutoffs"] = args.cutoffs
    return config


def _nl_supported_methods(methods: list[str]) -> tuple[list[str], list[str]]:
    """Split requested method names into NL-supported and ignored tokens."""
    supported = []
    ignored = []
    for method in methods:
        method = normalize_method_name(method)
        if method in _SUPPORTED_NL_METHODS:
            supported.append(method)
        else:
            ignored.append(method)
    return supported, ignored


def _nl_method_for_case(method: str, batch_size: int, explicit: bool) -> str | None:
    """Resolve a configured NL method to the concrete API for this batch shape."""
    method = normalize_method_name(method)
    if batch_size == 1:
        return method.removeprefix("batch_")
    if method.startswith("batch_"):
        return method
    if method == "cell_list":
        return "batch_cell_list"
    if method == "cluster_tile":
        return "batch_cluster_tile"
    if method == "naive_neighbor_list":
        return "batch_naive_neighbor_list"
    return method


def _nl_backend_skip_reason(backend: str, method: str) -> str:
    """Return a policy-skip reason for backend/method pairs not runnable here."""
    if backend == "warp" and method in _CLUSTER_TILE_METHODS:
        return "warp backend does not support cluster_tile"
    return ""


def _cutoff_limit(cutoff_limits: dict, cutoff: float) -> int | None:
    """Return the configured total-atom cap for ``cutoff`` if one exists."""
    return cutoff_limits.get(cutoff) or cutoff_limits.get(str(cutoff))


def _all_cutoffs_limited(
    cutoffs: list[float],
    cutoff_limits: dict,
    total_atoms: int,
) -> bool:
    """Return True when every cutoff is policy-skipped for ``total_atoms``."""
    return bool(cutoffs) and all(
        (limit := _cutoff_limit(cutoff_limits, cutoff)) is not None
        and total_atoms > limit
        for cutoff in cutoffs
    )


def _resolved_methods_for_case(
    methods: list[str], batch_size: int, explicit: bool
) -> list[str]:
    """Resolve and de-duplicate method names for one concrete NL case."""
    resolved = []
    for method in methods:
        concrete = _nl_method_for_case(method, batch_size, explicit)
        if concrete is not None and concrete not in resolved:
            resolved.append(concrete)
    return resolved


def _require_warp_tile_api(method: str) -> None:
    """Raise a concise dependency error when cluster-tile kernels cannot compile."""
    if normalize_method_name(method) not in _CLUSTER_TILE_METHODS:
        return

    try:
        import warp._src.builtins as wp_builtins
    except Exception as exc:
        raise RuntimeError(
            f"{method} requires Warp tile primitives, but Warp builtins could not be inspected"
        ) from exc

    missing = [
        name for name in _REQUIRED_WARP_TILE_BUILTINS if not hasattr(wp_builtins, name)
    ]
    if missing:
        raise RuntimeError(
            f"{method} requires Warp tile primitives missing from this environment: "
            f"{', '.join(missing)}"
        )


# =============================================================================
# Core Benchmark Function
# =============================================================================


def benchmark_nl(
    data: dict,
    cutoff: float,
    method: str,
    num_runs: int,
    warmup_runs: int = 3,
    backend: str = "torch",
) -> dict:
    """Benchmark a single NL configuration.

    Optimized: no redundant clean_gpu or extra kernel calls.
    Memory is measured from a single warmup run. Neighbor count
    captured from warmup result. clean_gpu() is the caller's
    responsibility (once per atom-size group, not per config).

    Parameters
    ----------
    data : dict
        System data from create_system() (backend-specific arrays).
    cutoff : float
        Cutoff distance in Angstroms.
    method : str
        Public neighborlist API name.
    num_runs : int
        Number of timing iterations.
    warmup_runs : int
        Number of warmup iterations.
    backend : str, default='torch'
        ``'torch'``, ``'jax'``, or ``'warp'``.

    Returns
    -------
    dict
        Timing and memory results with NL-specific extras.
    """
    method = normalize_method_name(method)
    _require_warp_tile_api(method)
    if backend == "jax":
        return _benchmark_nl_jax(data, cutoff, method, num_runs, warmup_runs)
    if backend == "warp":
        return _benchmark_nl_warp(data, cutoff, method, num_runs, warmup_runs)

    positions = data["positions"]
    cell = data["cell"]
    pbc = data["pbc"]
    batch_idx = data["batch_idx"]
    batch_size = int(data.get("batch_size", 1))
    total_atoms = data.get("total_atoms", data["atoms_per_system"])

    maxnb = estimate_max_neighbors(
        cutoff,
        atomic_density=DEFAULT_ATOMIC_DENSITY,
        safety_factor=DEFAULT_NL_SAFETY_FACTOR,
    )

    def run_nl():
        if method == "cell_list":
            return cell_list(
                positions=positions,
                cell=cell,
                pbc=pbc,
                cutoff=cutoff,
                max_neighbors=maxnb,
            )
        if method == "naive_neighbor_list":
            return naive_neighbor_list(
                positions=positions,
                cell=cell,
                pbc=pbc,
                cutoff=cutoff,
                max_neighbors=maxnb,
            )
        if method == "batch_cell_list":
            return batch_cell_list(
                positions=positions,
                cell=cell,
                pbc=pbc,
                cutoff=cutoff,
                batch_idx=batch_idx,
                max_neighbors=maxnb,
            )
        if method == "batch_naive_neighbor_list":
            return batch_naive_neighbor_list(
                positions=positions,
                cell=cell,
                pbc=pbc,
                cutoff=cutoff,
                batch_idx=batch_idx,
                max_neighbors=maxnb,
                max_atoms_per_system=int(total_atoms // batch_size),
            )
        if method == "cluster_tile":
            return torch_neighbor_list(
                positions=positions,
                cell=cell,
                pbc=pbc,
                cutoff=cutoff,
                method="cluster_tile",
                max_neighbors=maxnb,
                return_neighbor_list=False,
            )
        if method == "batch_cluster_tile":
            return torch_neighbor_list(
                positions=positions,
                cell=cell,
                pbc=pbc,
                cutoff=cutoff,
                batch_idx=batch_idx,
                method="batch_cluster_tile",
                max_neighbors=maxnb,
                return_neighbor_list=False,
            )
        raise ValueError(f"Unsupported NL method for torch backend: {method}")

    # Single warmup run: captures neighbor count + peak memory
    result, mem_info = measure_memory_torch(run_nl)
    n_neighbors = int(result[0].shape[1]) if hasattr(result[0], "shape") else 0

    # Timing (warmup inside cuda_timed_runs handles GPU pipeline warmup)
    time_sec = cuda_timed_runs(run_nl, num_runs, warmup_runs=warmup_runs)

    return {
        "time_seconds": time_sec,
        "mem_info": mem_info,
        "max_neighbors": n_neighbors,
        "total_neighbor_pairs": total_atoms * n_neighbors,
    }


def _benchmark_nl_jax(data, cutoff, method, num_runs, warmup_runs):
    """JAX backend implementation of :func:`benchmark_nl`.

    Uses wall-clock timing (CUDA events cannot observe JAX work).
    JAX memory fields are reported as NaN because the XLA pool makes
    per-call memory attribution misleading.
    """
    jax_api = lazy_import_jax()
    jax = jax_api["jax"]
    jnp = jax_api["jnp"]
    jax_nl = jax_api["neighbor_list"]
    estimate_bcl_sizes = jax_api["estimate_batch_cell_list_sizes"]
    compute_naive_shifts = jax_api["compute_naive_num_shifts"]

    positions = data["positions"]
    cell = data["cell"]
    pbc = data["pbc"]
    batch_idx = data["batch_idx"]
    atoms_per_system = int(data["atoms_per_system"])
    batch_size = int(data.get("batch_size", 1))
    total_atoms = data.get("total_atoms", atoms_per_system)

    # Under jax.jit the batched NL wrappers need static batch pointers
    # and precomputed sizing metadata.
    batch_ptr = jnp.arange(batch_size + 1, dtype=jnp.int32) * atoms_per_system

    jax_method = {
        "cell_list": "cell_list",
        "cluster_tile": "cluster_tile",
        "naive_neighbor_list": "naive",
        "batch_cell_list": "batch_cell_list",
        "batch_cluster_tile": "batch_cluster_tile",
        "batch_naive_neighbor_list": "batch_naive",
    }.get(method)
    if jax_method is None:
        raise ValueError(f"Unsupported NL method for jax backend: {method}")
    is_batch_method = method.startswith("batch_")

    maxnb = estimate_max_neighbors(
        cutoff,
        atomic_density=DEFAULT_ATOMIC_DENSITY,
        safety_factor=DEFAULT_NL_SAFETY_FACTOR,
    )

    # Pre-compute sizing outside jit so the closure can be traced.
    if jax_method == "cell_list":
        max_total_cells, _, _ = estimate_bcl_sizes(
            positions=positions,
            batch_ptr=jnp.asarray([0, atoms_per_system], dtype=jnp.int32),
            cell=cell,
            cutoff=float(cutoff),
            pbc=pbc,
        )
        nl_kwargs = dict(max_total_cells=int(max_total_cells))
    elif jax_method == "batch_cell_list":
        max_total_cells, _, _ = estimate_bcl_sizes(
            positions=positions,
            batch_ptr=batch_ptr,
            cell=cell,
            cutoff=float(cutoff),
            pbc=pbc,
        )
        nl_kwargs = dict(max_total_cells=int(max_total_cells))
    elif jax_method in {"naive", "batch_naive"}:
        shift_range, num_shifts, max_shifts = compute_naive_shifts(
            cell=cell, pbc=pbc, cutoff=float(cutoff)
        )
        nl_kwargs = dict(
            shift_range_per_dimension=shift_range,
            num_shifts_per_system=num_shifts,
            max_shifts_per_system=int(max_shifts),
        )
        if jax_method == "batch_naive":
            nl_kwargs["max_atoms_per_system"] = atoms_per_system
    else:
        nl_kwargs = {}

    def run_nl():
        kwargs = dict(
            positions=positions,
            cutoff=float(cutoff),
            cell=cell,
            pbc=pbc,
            method=jax_method,
            return_neighbor_list=False,
            max_neighbors=int(maxnb),
            **nl_kwargs,
        )
        if is_batch_method:
            kwargs.update(batch_idx=batch_idx, batch_ptr=batch_ptr)
        return jax_nl(
            **kwargs,
        )

    # jit so the timed loop reflects steady-state per-call cost (the
    # number an MD/MC user would observe), not Python-side tracing.
    run_nl_jit = jax.jit(run_nl)

    # Memory via NVML (XLA pool makes torch-side measurement useless)
    result, mem_info = measure_memory_jax(run_nl_jit, jax)
    n_neighbors = int(result[0].shape[1]) if hasattr(result[0], "shape") else 0

    time_sec = cuda_timed_runs(
        run_nl_jit, num_runs, warmup_runs=warmup_runs, backend="jax"
    )

    return {
        "time_seconds": time_sec,
        "mem_info": mem_info,
        "max_neighbors": n_neighbors,
        "total_neighbor_pairs": total_atoms * n_neighbors,
    }


def _benchmark_nl_warp(data, cutoff, method, num_runs, warmup_runs):
    """Benchmark root Warp neighbor-list launchers through preallocated buffers."""
    import warp as wp

    from nvalchemiops.neighbors import (
        batch_build_cell_list as wp_batch_build_cell_list,
    )
    from nvalchemiops.neighbors import (
        batch_naive_neighbor_matrix_pbc,
        naive_neighbor_matrix_pbc,
    )
    from nvalchemiops.neighbors import (
        batch_query_cell_list as wp_batch_query_cell_list,
    )
    from nvalchemiops.neighbors import (
        build_cell_list as wp_build_cell_list,
    )
    from nvalchemiops.neighbors import (
        query_cell_list as wp_query_cell_list,
    )
    from nvalchemiops.torch.neighbors.batch_cell_list import (
        estimate_batch_cell_list_sizes,
    )
    from nvalchemiops.torch.neighbors.cell_list import estimate_cell_list_sizes
    from nvalchemiops.torch.neighbors.neighbor_utils import (
        allocate_cell_list,
        compute_naive_num_shifts,
        prepare_batch_idx_ptr,
    )
    from nvalchemiops.torch.types import (
        get_wp_dtype,
        get_wp_mat_dtype,
        get_wp_vec_dtype,
    )

    positions = data["positions"]
    cell = data["cell"]
    pbc = data["pbc"]
    batch_idx = data["batch_idx"]
    batch_size = int(data.get("batch_size", 1))
    total_atoms = int(data.get("total_atoms", data["atoms_per_system"]))
    device = positions.device
    wp_device = str(device)
    wp_dtype = get_wp_dtype(positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(positions.dtype)
    wp_mat_dtype = get_wp_mat_dtype(positions.dtype)
    maxnb = estimate_max_neighbors(
        cutoff,
        atomic_density=DEFAULT_ATOMIC_DENSITY,
        safety_factor=DEFAULT_NL_SAFETY_FACTOR,
    )

    neighbor_matrix = torch.empty(
        (total_atoms, maxnb), dtype=torch.int32, device=device
    )
    neighbor_matrix_shifts = torch.empty(
        (total_atoms, maxnb, 3), dtype=torch.int32, device=device
    )
    num_neighbors = torch.empty((total_atoms,), dtype=torch.int32, device=device)
    wp_positions = wp.from_torch(positions, dtype=wp_vec_dtype, return_ctype=True)
    wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
    wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)
    wp_neighbor_matrix = wp.from_torch(
        neighbor_matrix, dtype=wp.int32, return_ctype=True
    )
    wp_neighbor_matrix_shifts = wp.from_torch(
        neighbor_matrix_shifts, dtype=wp.vec3i, return_ctype=True
    )
    wp_num_neighbors = wp.from_torch(num_neighbors, dtype=wp.int32, return_ctype=True)

    def zero_outputs() -> None:
        neighbor_matrix.fill_(total_atoms)
        neighbor_matrix_shifts.zero_()
        num_neighbors.zero_()

    if method == "naive_neighbor_list":
        shift_range, num_shifts, _ = compute_naive_num_shifts(cell, cutoff, pbc)
        wp_shift_range = wp.from_torch(shift_range, dtype=wp.vec3i, return_ctype=True)

        def run_nl():
            zero_outputs()
            naive_neighbor_matrix_pbc(
                wp_positions,
                cutoff,
                wp_cell,
                wp_shift_range,
                int(num_shifts[0].item()),
                wp_neighbor_matrix,
                wp_neighbor_matrix_shifts,
                wp_num_neighbors,
                wp_dtype,
                wp_device,
            )

    elif method == "batch_naive_neighbor_list":
        batch_idx, batch_ptr = prepare_batch_idx_ptr(
            batch_idx, None, total_atoms, device
        )
        shift_range, num_shifts, max_shifts = compute_naive_num_shifts(
            cell, cutoff, pbc
        )
        wp_batch_idx = wp.from_torch(batch_idx, dtype=wp.int32, return_ctype=True)
        wp_batch_ptr = wp.from_torch(batch_ptr, dtype=wp.int32, return_ctype=True)
        wp_shift_range = wp.from_torch(shift_range, dtype=wp.vec3i, return_ctype=True)
        wp_num_shifts = wp.from_torch(num_shifts, dtype=wp.int32, return_ctype=True)

        def run_nl():
            zero_outputs()
            batch_naive_neighbor_matrix_pbc(
                wp_positions,
                wp_cell,
                cutoff,
                wp_batch_ptr,
                wp_batch_idx,
                wp_shift_range,
                wp_num_shifts,
                max_shifts,
                wp_neighbor_matrix,
                wp_neighbor_matrix_shifts,
                wp_num_neighbors,
                wp_dtype,
                wp_device,
                int(data["atoms_per_system"]),
            )

    elif method == "cell_list":
        wp_pbc_single = wp.from_torch(pbc.squeeze(0), dtype=wp.bool, return_ctype=True)
        max_total_cells, neighbor_search_radius = estimate_cell_list_sizes(
            cell, pbc, cutoff
        )
        cell_cache = allocate_cell_list(
            total_atoms, max_total_cells, neighbor_search_radius, device
        )
        (
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        ) = cell_cache
        wp_cells_per_dimension = wp.from_torch(
            cells_per_dimension, dtype=wp.int32, return_ctype=True
        )
        wp_neighbor_search_radius = wp.from_torch(
            neighbor_search_radius, dtype=wp.int32, return_ctype=True
        )
        wp_atom_periodic_shifts = wp.from_torch(
            atom_periodic_shifts, dtype=wp.vec3i, return_ctype=True
        )
        wp_atom_to_cell_mapping = wp.from_torch(
            atom_to_cell_mapping, dtype=wp.vec3i, return_ctype=True
        )
        wp_atoms_per_cell_count = wp.from_torch(atoms_per_cell_count, dtype=wp.int32)
        wp_cell_atom_start_indices = wp.from_torch(
            cell_atom_start_indices, dtype=wp.int32
        )
        wp_cell_atom_list = wp.from_torch(
            cell_atom_list, dtype=wp.int32, return_ctype=True
        )

        def run_nl():
            zero_outputs()
            for tensor in cell_cache:
                tensor.zero_()
            wp_build_cell_list(
                wp_positions,
                wp_cell,
                wp_pbc_single,
                cutoff,
                wp_cells_per_dimension,
                wp_atom_periodic_shifts,
                wp_atom_to_cell_mapping,
                wp_atoms_per_cell_count,
                wp_cell_atom_start_indices,
                wp_cell_atom_list,
                wp_dtype,
                wp_device,
            )
            wp_query_cell_list(
                wp_positions,
                wp_cell,
                wp_pbc_single,
                cutoff,
                wp_cells_per_dimension,
                wp_neighbor_search_radius,
                wp_atom_periodic_shifts,
                wp_atom_to_cell_mapping,
                wp.from_torch(atoms_per_cell_count, dtype=wp.int32, return_ctype=True),
                wp.from_torch(
                    cell_atom_start_indices, dtype=wp.int32, return_ctype=True
                ),
                wp_cell_atom_list,
                wp_neighbor_matrix,
                wp_neighbor_matrix_shifts,
                wp_num_neighbors,
                wp_dtype,
                wp_device,
            )

    elif method == "batch_cell_list":
        max_total_cells, neighbor_search_radius = estimate_batch_cell_list_sizes(
            cell, pbc, cutoff
        )
        cell_cache = allocate_cell_list(
            total_atoms, max_total_cells, neighbor_search_radius, device
        )
        (
            cells_per_dimension,
            neighbor_search_radius,
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
        ) = cell_cache
        cell_offsets = torch.zeros((batch_size,), dtype=torch.int32, device=device)
        cells_per_system = torch.zeros((batch_size,), dtype=torch.int32, device=device)
        wp_batch_idx = wp.from_torch(batch_idx, dtype=wp.int32, return_ctype=True)
        wp_cells_per_dimension = wp.from_torch(
            cells_per_dimension, dtype=wp.vec3i, return_ctype=True
        )
        wp_neighbor_search_radius = wp.from_torch(
            neighbor_search_radius, dtype=wp.vec3i, return_ctype=True
        )
        wp_cell_offsets = wp.from_torch(cell_offsets, dtype=wp.int32)
        wp_cells_per_system = wp.from_torch(cells_per_system, dtype=wp.int32)
        wp_atom_periodic_shifts = wp.from_torch(
            atom_periodic_shifts, dtype=wp.vec3i, return_ctype=True
        )
        wp_atom_to_cell_mapping = wp.from_torch(
            atom_to_cell_mapping, dtype=wp.vec3i, return_ctype=True
        )
        wp_atoms_per_cell_count = wp.from_torch(atoms_per_cell_count, dtype=wp.int32)
        wp_cell_atom_start_indices = wp.from_torch(
            cell_atom_start_indices, dtype=wp.int32
        )
        wp_cell_atom_list = wp.from_torch(
            cell_atom_list, dtype=wp.int32, return_ctype=True
        )

        def run_nl():
            zero_outputs()
            for tensor in (*cell_cache, cell_offsets, cells_per_system):
                tensor.zero_()
            wp_batch_build_cell_list(
                wp_positions,
                wp_cell,
                wp_pbc,
                cutoff,
                wp_batch_idx,
                wp_cells_per_dimension,
                wp_cell_offsets,
                wp_cells_per_system,
                wp_atom_periodic_shifts,
                wp_atom_to_cell_mapping,
                wp_atoms_per_cell_count,
                wp_cell_atom_start_indices,
                wp_cell_atom_list,
                wp_dtype,
                wp_device,
            )
            cells_per_system_counts = cells_per_dimension.prod(dim=1)
            cell_offsets.zero_()
            if batch_size > 1:
                torch.cumsum(cells_per_system_counts[:-1], dim=0, out=cell_offsets[1:])
            wp_batch_query_cell_list(
                wp_positions,
                wp_cell,
                wp_pbc,
                cutoff,
                wp_batch_idx,
                wp_cells_per_dimension,
                wp_neighbor_search_radius,
                wp.from_torch(cell_offsets, dtype=wp.int32, return_ctype=True),
                wp_atom_periodic_shifts,
                wp_atom_to_cell_mapping,
                wp.from_torch(atoms_per_cell_count, dtype=wp.int32, return_ctype=True),
                wp.from_torch(
                    cell_atom_start_indices, dtype=wp.int32, return_ctype=True
                ),
                wp_cell_atom_list,
                wp_neighbor_matrix,
                wp_neighbor_matrix_shifts,
                wp_num_neighbors,
                wp_dtype,
                wp_device,
            )

    else:
        raise ValueError(f"Unsupported NL method for warp backend: {method}")

    _, mem_info = measure_memory_torch(run_nl)
    n_neighbors = int(num_neighbors.max().item()) if total_atoms else 0
    time_sec = cuda_timed_runs(run_nl, num_runs, warmup_runs=warmup_runs)
    sync_gpu()
    return {
        "time_seconds": time_sec,
        "mem_info": mem_info,
        "max_neighbors": n_neighbors,
        "total_neighbor_pairs": int(num_neighbors.sum().item()) if total_atoms else 0,
    }


# =============================================================================
# Config-Driven Runner
# =============================================================================


def _nl_run_one_method(data, cutoff, method, num_runs, warmup_runs, backend, row_meta):
    """Run :func:`benchmark_nl` for one ``(cutoff, method)`` and build a result row.

    Catches OOM and other exceptions and emits ``success=False`` rows.
    Keeps the inner loop in :func:`run_from_config` free of try/except nesting.
    """
    try:
        r = benchmark_nl(data, cutoff, method, num_runs, warmup_runs, backend=backend)
        result = build_result(
            method=method,
            time_seconds=r["time_seconds"],
            mem_info=r["mem_info"],
            cutoff=cutoff,
            **row_meta,
        )
        throughput_matoms = result["throughput_atoms_per_sec"] / 1e6
        mem_suffix = (
            f" | {result['mem_delta_mb']:.1f} MB"
            if backend in {"torch", "warp"}
            else ""
        )
        print(
            f"    {cutoff}Å {method}: "
            f"{result['time_us_per_atom']:.3f} μs/atom | "
            f"{throughput_matoms:.1f} Matom/s{mem_suffix}"
        )
        return result
    except torch.cuda.OutOfMemoryError as e:
        print(f"    {cutoff}Å {method}: OOM")
        clean_gpu()
        return build_failure_result(
            method=method,
            cutoff=cutoff,
            error=str(e),
            error_type=type(e).__name__,
            **row_meta,
        )
    except Exception as e:
        print(f"    {cutoff}Å {method}: FAILED - {e}")
        return build_failure_result(
            method=method,
            cutoff=cutoff,
            error=str(e),
            error_type=type(e).__name__,
            **row_meta,
        )


def dry_run_from_config(config: dict, backend: str | None = None) -> list[dict]:
    """Print and return the expanded NL benchmark plan without allocation."""
    params = config["parameters"]
    cutoffs = params["cutoffs"]
    cutoff_limits = params.get("cutoff_limits", {})
    max_total_atoms = params.get("max_total_atoms")
    if backend is None:
        backend = config.get("runtime", {}).get("backend", "torch")
    methods, ignored_methods = _nl_supported_methods(enabled_method_names(config))
    if ignored_methods:
        print(f"NL dry-run ignoring non-NL methods: {', '.join(ignored_methods)}")
    explicit = bool(config.get("runtime", {}).get("explicit_methods", False))
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
                resolved_methods = _resolved_methods_for_case(
                    methods, batch_size, explicit
                )
                rows.extend(
                    {
                        "benchmark": "nl",
                        "backend": backend,
                        "system": sys_name,
                        "mode": mode_name,
                        "atoms_per_system": atoms_per_system,
                        "batch_size": batch_size,
                        "total_atoms": total_atoms,
                        "method": method,
                        "cutoff": cutoff,
                        "reason": f">{max_total_atoms} max_total_atoms",
                    }
                    for cutoff in cutoffs
                    for method in resolved_methods
                )
            for cfg in configs:
                atoms_per_system, batch_size, total_atoms = planned_atom_counts(
                    sys_name, cfg
                )
                resolved_methods = _resolved_methods_for_case(
                    methods, batch_size, explicit
                )
                for cutoff in cutoffs:
                    limit = _cutoff_limit(cutoff_limits, cutoff)
                    for method in resolved_methods:
                        reason = ""
                        if limit and total_atoms > limit:
                            reason = f">{limit} cutoff_limit"
                        elif _nl_backend_skip_reason(backend, method):
                            reason = _nl_backend_skip_reason(backend, method)
                        rows.append(
                            {
                                "benchmark": "nl",
                                "backend": backend,
                                "system": sys_name,
                                "mode": mode_name,
                                "atoms_per_system": atoms_per_system,
                                "batch_size": batch_size,
                                "total_atoms": total_atoms,
                                "method": method,
                                "cutoff": cutoff,
                                "reason": reason,
                            }
                        )
    print("NL dry-run plan")
    for row in rows:
        suffix = f" SKIP {row['reason']}" if row["reason"] else ""
        print(
            "  {system}/{mode} backend={backend} method={method} "
            "N={atoms_per_system} batch={batch_size} total={total_atoms} "
            "cutoff={cutoff}{suffix}".format(**row, suffix=suffix)
        )
    print(f"NL dry-run rows: {len(rows)}")
    return rows


def run_from_config(
    config: dict,
    output_dir: Path | str | None = None,
    backend: str | None = None,
) -> list[dict]:
    """Run NL benchmarks driven entirely by YAML config.

    This is the main entry point, used both standalone and from benchmark_suite.py.

    Parameters
    ----------
    config : dict
        Merged config (YAML + CLI overrides).
    output_dir : Path, optional
        Override output directory. If None, uses config['output']['base_dir'].
    backend : str, optional
        ``'torch'`` or ``'jax'``. If None, pulled from
        ``config['runtime']['backend']`` (merged in by ``merge_cli_overrides``),
        defaulting to ``'torch'``.

    Returns
    -------
    list[dict]
        All benchmark results.
    """
    params = config["parameters"]
    num_runs = params["timing_runs"]
    warmup_runs = params["warmup_runs"]
    cutoffs = params["cutoffs"]
    cutoff_limits = params.get("cutoff_limits", {})
    max_total_atoms = params.get("max_total_atoms")

    if backend is None:
        backend = config.get("runtime", {}).get("backend", "torch")

    if config.get("runtime", {}).get("dry_run", False):
        return dry_run_from_config(config, backend=backend)

    # Resolve output directory
    if output_dir is None:
        output_dir = create_run_directory(config["output"]["base_dir"], prefix="nl")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect enabled methods
    methods, ignored_methods = _nl_supported_methods(enabled_method_names(config))
    if ignored_methods:
        print(f"NL ignoring non-NL methods: {', '.join(ignored_methods)}")
    explicit_methods = bool(config.get("runtime", {}).get("explicit_methods", False))

    # Eagerly validate JAX availability so the error surfaces now, not
    # partway through the benchmark loop below.
    if backend == "jax":
        lazy_import_jax()

    try:
        gpu_name = torch.cuda.get_device_name(0)
    except (AssertionError, RuntimeError):
        gpu_name = "N/A (no CUDA)"
    print("NL Benchmark Suite")
    print(f"GPU: {gpu_name}")
    print(f"Backend: {backend}")
    print(f"Cutoffs: {cutoffs} Å | Methods: {methods}")
    print(f"Timing: {num_runs} runs")
    print(f"Output: {output_dir}")

    all_results = []

    # Iterate: systems × scaling modes
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
            print(f"NL: {sys_name.upper()} / {mode_name}")
            print(f"{'=' * 70}")

            configs = configs_for_mode(
                mode_name, mode_config, sys_name, sys_config, nh3_dir
            )
            configs, skipped = filter_configs_by_total_atoms(
                configs, sys_name, max_total_atoms
            )
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
                resolved_methods = _resolved_methods_for_case(
                    methods, batch_size, explicit_methods
                )
                reason = f">{max_total_atoms} max_total_atoms"
                results.extend(
                    build_skipped_result(
                        method=method,
                        cutoff=cutoff,
                        reason=reason,
                        **row_meta,
                    )
                    for cutoff in cutoffs
                    for method in resolved_methods
                )
            if not configs:
                if results:
                    csv_name = make_csv_name("nl", sys_name, mode_name)
                    save_results(results, output_dir / csv_name)
                    all_results.extend(results)
                continue

            for cfg in configs:
                n, bs = cfg["num_atoms"], cfg["batch_size"]
                planned_n, planned_bs, planned_total = planned_atom_counts(
                    sys_name, cfg
                )
                planned_row_meta = make_row_meta(
                    sys_name,
                    mode_name,
                    backend,
                    planned_n,
                    planned_bs,
                    planned_total,
                )
                planned_methods = _resolved_methods_for_case(
                    methods,
                    planned_bs,
                    explicit_methods,
                )
                if planned_methods and all(
                    _nl_backend_skip_reason(backend, method)
                    for method in planned_methods
                ):
                    for cutoff in cutoffs:
                        results.extend(
                            build_skipped_result(
                                method=method,
                                cutoff=cutoff,
                                reason=_nl_backend_skip_reason(backend, method),
                                **planned_row_meta,
                            )
                            for method in planned_methods
                        )
                    continue
                if _all_cutoffs_limited(cutoffs, cutoff_limits, planned_total):
                    for cutoff in cutoffs:
                        limit = _cutoff_limit(cutoff_limits, cutoff)
                        print(f"    {cutoff}Å: SKIP (>{format_num(limit)} limit)")
                        reason = f">{limit} cutoff_limit"
                        results.extend(
                            build_skipped_result(
                                method=method,
                                cutoff=cutoff,
                                reason=reason,
                                **planned_row_meta,
                            )
                            for method in planned_methods
                        )
                    continue

                clean_gpu()

                try:
                    data = create_system(
                        sys_name,
                        num_atoms=n,
                        pdb_path=cfg.get("pdb_path"),
                        batch_size=bs,
                        backend="torch" if backend == "warp" else backend,
                    )
                except (FileNotFoundError, RuntimeError, ValueError) as e:
                    print(f"    SKIP: {e}")
                    results.extend(
                        build_failure_result(
                            method=method,
                            cutoff=cutoff,
                            error=str(e),
                            error_type=type(e).__name__,
                            **planned_row_meta,
                        )
                        for cutoff in cutoffs
                        for method in planned_methods
                    )
                    continue

                actual_total = data.get("total_atoms", data["atoms_per_system"])
                actual_n = data["atoms_per_system"]
                print(
                    f"\n  {format_num(actual_n)} atoms × {bs} batch = {format_num(actual_total)} total"
                )
                print(f"  [GPU: {current_alloc_gb(backend):.1f} GB allocated]")
                row_meta = make_row_meta(
                    sys_name,
                    mode_name,
                    backend,
                    actual_n,
                    data.get("batch_size", 1),
                    actual_total,
                )
                resolved_methods = _resolved_methods_for_case(
                    methods,
                    int(data.get("batch_size", 1)),
                    explicit_methods,
                )

                for cutoff in cutoffs:
                    limit = _cutoff_limit(cutoff_limits, cutoff)
                    if limit and actual_total > limit:
                        print(f"    {cutoff}Å: SKIP (>{format_num(limit)} limit)")
                        reason = f">{limit} cutoff_limit"
                        results.extend(
                            build_skipped_result(
                                method=method,
                                cutoff=cutoff,
                                reason=reason,
                                **row_meta,
                            )
                            for method in resolved_methods
                        )
                        continue

                    # Note: cell_size < 2*cutoff violates minimum image convention
                    # but we still benchmark for completeness. Document in sphinx docs.
                    if data["cell_size"] < 2 * cutoff:
                        print(
                            f"    {cutoff}Å: WARNING cell {data['cell_size']:.1f}Å < 2×cutoff (benchmarking anyway)"
                        )

                    for method in resolved_methods:
                        reason = _nl_backend_skip_reason(backend, method)
                        if reason:
                            print(f"    {cutoff}Å {method}: SKIP - {reason}")
                            results.append(
                                build_skipped_result(
                                    method=method,
                                    cutoff=cutoff,
                                    reason=reason,
                                    **row_meta,
                                )
                            )
                            continue
                        result = _nl_run_one_method(
                            data,
                            cutoff,
                            method,
                            num_runs,
                            warmup_runs,
                            backend,
                            row_meta,
                        )
                        if result is not None:
                            results.append(result)

                del data

            # Save per-(system, mode) CSV with standardized name
            if results:
                csv_name = make_csv_name("nl", sys_name, mode_name)
                save_results(results, output_dir / csv_name)
                all_results.extend(results)

    print(f"\n{'=' * 70}")
    print(f"COMPLETE: {len(all_results)} results saved to {output_dir}")
    print(f"{'=' * 70}")

    return all_results


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    """Parse command-line arguments for neighbor list benchmarks."""
    parser = argparse.ArgumentParser(
        description="Neighbor List Benchmark (2 systems × 3 modes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (run from the repository root):
    python -m benchmarks.neighborlist.benchmark_neighborlist \\
        --config benchmarks/neighborlist/benchmark_config.yaml
    python -m benchmarks.neighborlist.benchmark_neighborlist \\
        --config benchmarks/neighborlist/benchmark_config.yaml \\
        --system cscl --mode system_size
    python -m benchmarks.neighborlist.benchmark_neighborlist \\
        --config benchmarks/neighborlist/benchmark_config.yaml \\
        --cutoffs 6 15 --method cell_list
    python -m benchmarks.neighborlist.benchmark_neighborlist \\
        --config benchmarks/neighborlist/benchmark_config.yaml \\
        --output-dir docs/benchmarks/benchmark_results
        """,
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to benchmark_config.yaml"
    )
    add_common_cli_args(parser)
    parser.add_argument(
        "--cutoffs",
        "-c",
        type=float,
        nargs="+",
        default=None,
        help="Override cutoff radii in Angstroms",
    )
    return parser.parse_args()


def main():
    """Run neighbor list benchmarks."""
    args = parse_args()

    # Load YAML config, merge CLI overrides
    config = load_yaml_config(args.config)
    config = merge_cli_overrides(config, args)

    # Resolve backend: explicit CLI arg wins; otherwise honor config; else torch.
    backend = args.backend or config.get("runtime", {}).get("backend", "torch")

    if backend == "jax":
        ensure_jax_available()

    results = run_from_config(config, output_dir=args.output_dir, backend=backend)
    if not results:
        return 1
    if not any(row.get("success", True) is not False for row in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
