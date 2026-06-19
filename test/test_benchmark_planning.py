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

"""Tests for benchmark planning and result-schema helpers."""

import csv
import importlib
import inspect
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from benchmarks import benchmark_suite
from benchmarks.benchmark_suite import (
    validate_backend_selection,
    validate_method_selection,
)
from benchmarks.config import (
    enabled_method_names,
    load_yaml_config,
    merge_common_cli_overrides,
    normalize_method_name,
)
from benchmarks.interactions.dispersion import benchmark_dftd3
from benchmarks.interactions.dispersion.benchmark_dftd3 import (
    dry_run_from_config as dry_run_d3,
)
from benchmarks.interactions.electrostatics import benchmark_electrostatics
from benchmarks.interactions.electrostatics.benchmark_electrostatics import (
    _el_unpack_params,
    _torch_neighbor_matrix_to_list_chunked,
)
from benchmarks.neighborlist import benchmark_neighborlist
from benchmarks.neighborlist.benchmark_neighborlist import (
    _nl_method_for_case,
)
from benchmarks.neighborlist.benchmark_neighborlist import (
    dry_run_from_config as dry_run_nl,
)
from benchmarks.plotting import plot_benchmarks
from benchmarks.plotting.plot_benchmarks import load_csv
from benchmarks.suite_systems import (
    configs_for_mode,
    filter_configs_by_total_atoms,
    planned_atom_counts,
)
from benchmarks.suite_utils import (
    build_failure_result,
    build_result,
    build_skipped_result,
    clean_jax,
    configure_jax_environment,
    jax_timed_batch,
    jax_timed_stateful,
    measure_memory_jax,
    save_results,
    write_run_log,
)
from docs.benchmarks import generate_plots as docs_generate_plots
from nvalchemiops.jax.neighbors.batch_cell_list import (
    batch_cell_list as jax_batch_cell_list,
)
from nvalchemiops.jax.neighbors.batch_cell_list import (
    estimate_batch_cell_list_sizes as jax_estimate_batch_cell_list_sizes,
)
from nvalchemiops.jax.neighbors.cell_list import (
    estimate_cell_list_sizes as jax_estimate_cell_list_sizes,
)

jax_batch_cell_list_module = importlib.import_module(
    "nvalchemiops.jax.neighbors.batch_cell_list",
)


class TestBenchmarkMethodSelection:
    """Test benchmark method normalization and CLI selection."""

    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            ("cell", "cell_list"),
            ("naive", "naive_neighbor_list"),
            ("naive-scalar", "naive_scalar"),
            ("naive-tile", "naive_tile"),
            ("cell-list-atom-centric", "cell_list_atom_centric"),
            ("cell-list-pair-centric", "cell_list_pair_centric"),
            ("batch-cell-list", "batch_cell_list"),
            ("batch-naive-scalar", "batch_naive_scalar"),
            ("batch-naive-tile", "batch_naive_tile"),
            (
                "batch-cell-list-atom-centric",
                "batch_cell_list_atom_centric",
            ),
            (
                "batch-cell-list-pair-centric",
                "batch_cell_list_pair_centric",
            ),
            ("batch_naive", "batch_naive_neighbor_list"),
            ("cluster-tile", "cluster_tile"),
            ("tile", "cluster_tile"),
            ("batch-cluster-tile", "batch_cluster_tile"),
            ("d3", "dftd3"),
            ("pme", "pme"),
        ],
    )
    def test_normalize_method_aliases(self, token, expected):
        """Legacy aliases normalize to the public API method names."""
        assert normalize_method_name(token) == expected

    def test_selected_methods_preserve_explicit_batch_api(self):
        """Explicit CLI methods are returned even when absent from YAML."""
        config = {
            "parameters": {},
            "runtime": {},
            "systems": {"cscl": {"enabled": True}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [{"name": "cell_list", "enabled": True}],
        }
        args = type(
            "Args",
            (),
            {
                "timing_runs": None,
                "warmup_runs": None,
                "system": None,
                "mode": None,
                "output_dir": None,
                "backend": None,
                "methods": ["cell_list", "batch_cell_list"],
                "dry_run": False,
                "max_total_atoms": None,
            },
        )()

        merged = merge_common_cli_overrides(config, args)

        assert enabled_method_names(merged) == ["cell_list", "batch_cell_list"]
        assert merged["runtime"]["explicit_methods"] is True

    def test_jax_benchmarks_do_not_jit_zero_arg_array_closures(self):
        """JAX benchmark wrappers pass large arrays as jit arguments."""
        functions = (
            benchmark_electrostatics._benchmark_pme_jax,
            benchmark_electrostatics._benchmark_ewald_jax,
            benchmark_neighborlist._benchmark_nl_jax,
            benchmark_dftd3._benchmark_d3_jax,
        )

        for function in functions:
            assert "jax.jit(run_" not in inspect.getsource(function)

    def test_jax_benchmark_kernel_caches_are_defined(self):
        """EL and NL keep reusable jitted entrypoints at module scope."""
        assert isinstance(benchmark_electrostatics._JAX_EL_KERNEL_CACHE, dict)
        assert isinstance(benchmark_neighborlist._JAX_NL_KERNEL_CACHE, dict)

    def test_jax_cleanup_preserves_executable_caches_by_default(self):
        """Successful JAX rows do not clear compiled executables per cutoff."""
        signature = inspect.signature(clean_jax)
        source = inspect.getsource(clean_jax)

        assert signature.parameters["clear_executables"].default is False
        assert "if clear_executables:" in source
        assert "jax.clear_caches()" in source

    def test_jax_nl_timing_falls_back_to_serial_after_oom(self):
        """Large JAX NL rows can time serially when batched dispatch OOMs."""
        fallback_source = inspect.getsource(
            benchmark_neighborlist._jax_nl_timed_with_serial_fallback
        )
        benchmark_source = inspect.getsource(benchmark_neighborlist._benchmark_nl_jax)

        assert 'failure_error_type(e) != "OutOfMemoryError"' in fallback_source
        assert "if prefer_serial:" in fallback_source
        assert "clean_jax(clear_executables=True)" in fallback_source
        assert "jax_timed_serial" in fallback_source
        assert '"jax_wall_block_each"' in fallback_source
        assert "_jax_nl_timed_with_serial_fallback(" in benchmark_source
        assert "prefer_serial=num_runs > 1" not in benchmark_source
        assert '"timing_method": timing_method' in benchmark_source

    def test_warp_nl_timing_declares_warp_backend(self):
        """Direct Warp API timings use the shared CUDA-event path explicitly."""
        source = inspect.getsource(benchmark_neighborlist._benchmark_nl_warp)

        assert 'backend="warp"' in source

    def test_suite_sets_jax_allocator_env_before_runner_imports(self):
        """Suite-level JAX env includes the throughput allocator policy."""
        source = inspect.getsource(benchmark_suite.main)
        env_block = source.split("if args.plot_only:", 1)[0]

        assert "configure_jax_environment(need_x64=True" in env_block
        helper_source = inspect.getsource(configure_jax_environment)
        assert 'os.environ.setdefault("JAX_ENABLE_X64", "1")' in helper_source
        assert (
            'os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")'
            in helper_source
        )
        assert "XLA_PYTHON_CLIENT_PREALLOCATE" in helper_source
        assert "on-demand allocation" in helper_source
        assert "TF_GPU_ALLOCATOR" not in helper_source

    def test_suite_rejects_profile_selector(self, monkeypatch, capsys):
        """The reportable suite has no reduced profile selector."""
        monkeypatch.setattr(
            sys,
            "argv",
            ["benchmark_suite.py", "--benchmark", "all", "--profile"],
        )

        with pytest.raises(SystemExit) as exc_info:
            benchmark_suite.parse_args()

        captured = capsys.readouterr()
        assert exc_info.value.code == 2
        assert "unrecognized arguments: --profile" in captured.err

    def test_jax_env_warns_when_preallocation_policy_is_unset(
        self, monkeypatch, capsys
    ):
        """DallasF allocator concern is visible without disabling preallocation."""
        monkeypatch.delenv("JAX_ENABLE_X64", raising=False)
        monkeypatch.delenv("XLA_PYTHON_CLIENT_MEM_FRACTION", raising=False)
        monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
        monkeypatch.delenv("XLA_PYTHON_CLIENT_ALLOCATOR", raising=False)
        monkeypatch.setattr(
            "benchmarks.suite_utils._JAX_ALLOCATOR_WARNING_EMITTED", False
        )

        configure_jax_environment(need_x64=True, context="test context")

        captured = capsys.readouterr()
        assert "XLA_PYTHON_CLIENT_PREALLOCATE is unset" in captured.out
        assert "XLA_PYTHON_CLIENT_MEM_FRACTION=0.95" in captured.out
        assert "on-demand allocation" in captured.out
        assert "XLA_PYTHON_CLIENT_PREALLOCATE" not in captured.err

    def test_jax_naive_benchmark_uses_direct_batched_timing(self):
        """JAX naive NL avoids donated-buffer state leaking into later rows."""
        kernel_source = inspect.getsource(benchmark_neighborlist._get_jax_nl_kernels)
        benchmark_source = inspect.getsource(benchmark_neighborlist._benchmark_nl_jax)

        assert "neighbor_matrix=neighbor_matrix" in kernel_source
        assert "neighbor_matrix_shifts=neighbor_matrix_shifts" in kernel_source
        assert "num_neighbors=num_neighbors" in kernel_source
        assert "donate_argnums=(8, 9, 10)" in kernel_source
        assert "donate_argnums=(10, 11, 12)" in kernel_source
        assert "use_direct_jax_nl = jax_family in" in benchmark_source
        assert "return jax_neighbor_list(" in benchmark_source
        assert (
            'shift_range_per_dimension=nl_kwargs["shift_range_per_dimension"]'
            in benchmark_source
        )
        assert 'num_shifts_per_system=nl_kwargs["num_shifts_per_system"]' in (
            benchmark_source
        )
        assert 'max_shifts_per_system=nl_kwargs["max_shifts_per_system"]' in (
            benchmark_source
        )
        assert "compiled_nl = jax.jit(direct_nl_kernel)" in benchmark_source
        assert (
            "time_sec, timing_method = _jax_nl_timed_with_serial_fallback("
            in benchmark_source
        )
        assert "jax_timed_stateful(" not in benchmark_source

    def test_jax_cell_list_benchmark_uses_direct_batched_timing(self):
        """JAX cell-list NL avoids the slow stateful compile path."""
        kernel_source = inspect.getsource(benchmark_neighborlist._get_jax_nl_kernels)
        benchmark_source = inspect.getsource(benchmark_neighborlist._benchmark_nl_jax)

        assert "build_cell_list(" in kernel_source
        assert "query_cell_list(" in kernel_source
        assert "batch_build_cell_list(" in kernel_source
        assert "batch_query_cell_list(" in kernel_source
        assert "cells_per_dimension=cells_per_dimension" in kernel_source
        assert "atom_periodic_shifts=atom_periodic_shifts" in kernel_source
        assert "cell_atom_list=cell_atom_list" in kernel_source
        assert "donate_argnums=(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)" in kernel_source
        assert "donate_argnums=(6, 7, 8, 9, 10, 11, 12, 13, 14)" in kernel_source
        assert "use_direct_jax_nl = jax_family in" in benchmark_source
        assert "return jax_neighbor_list(" in benchmark_source
        assert 'max_total_cells=nl_kwargs["max_total_cells"]' in benchmark_source
        assert 'neighbor_search_radius=nl_kwargs["neighbor_search_radius"]' in (
            benchmark_source
        )
        assert "compiled_nl = jax.jit(direct_nl_kernel)" in benchmark_source
        assert (
            "time_sec, timing_method = _jax_nl_timed_with_serial_fallback("
            in benchmark_source
        )

    def test_jax_pme_benchmark_precomputes_reusable_metadata(self):
        """JAX PME timings pass cacheable mesh/cell metadata explicitly."""
        source = inspect.getsource(benchmark_electrostatics._benchmark_pme_jax)
        kernel_source = inspect.getsource(benchmark_electrostatics._get_jax_el_kernels)

        assert (
            'compute_bspline_moduli_1d = jax_api["compute_bspline_moduli_1d"]' in source
        )
        assert "cell_inv_t = jnp.transpose(jnp.linalg.inv(cell_3d)" in source
        assert "volume = jnp.abs(jnp.linalg.det(cell_3d))" in source
        assert "moduli_x = compute_bspline_moduli_1d" in source
        assert "volume=volume" in kernel_source
        assert "cell_inv_t=cell_inv_t" in kernel_source
        assert "moduli_x=moduli_x" in kernel_source

    def test_torch_el_benchmarks_report_cuda_event_timing_metadata(self):
        """Torch EL total/real/reciprocal timing labels are explicit in CSV rows."""
        pme_source = inspect.getsource(benchmark_electrostatics.benchmark_pme)
        ewald_source = inspect.getsource(benchmark_electrostatics.benchmark_ewald)

        for source in (pme_source, ewald_source):
            assert '"timing_method": "torch_cuda_events"' in source
            assert '"timing_method_real": "torch_cuda_events"' in source
            assert '"timing_method_reciprocal": "torch_cuda_events"' in source

    def test_results_readme_documents_el_component_timing_metadata(self):
        """CSV schema docs include EL component timing labels."""
        readme = Path("docs/benchmarks/benchmark_results/README.md").read_text(
            encoding="utf-8"
        )

        assert "`timing_method_real`" in readme
        assert "`timing_method_reciprocal`" in readme

    def test_d3_timing_excludes_neighbor_list_setup(self):
        """D3 benchmark rows time D3 separately from neighbor-list setup."""
        torch_source = inspect.getsource(benchmark_dftd3.benchmark_d3)
        jax_source = inspect.getsource(benchmark_dftd3._benchmark_d3_jax)
        row_source = inspect.getsource(benchmark_dftd3._d3_run_one_cutoff)

        assert "Neighbor-list setup is performed outside" in torch_source
        assert "time_d3 = cuda_timed_runs(run_d3, num_runs" in torch_source
        assert "def _run_d3_kernel(" in jax_source
        assert "jax.jit(_run_d3_kernel)" in jax_source
        assert "return _run_d3_kernel_jit(" in jax_source
        assert "time_d3 = cuda_timed_runs(run_d3" in jax_source
        assert '"time_d3_seconds": time_d3' in torch_source
        assert '"time_d3_seconds": time_d3' in jax_source
        assert '"neighbor_setup_method": neighbor_setup_method' in torch_source
        assert '"neighbor_setup_method": neighbor_setup_method' in jax_source
        assert "time_d3_us_per_atom=time_d3_us_per_atom" in row_source
        assert 'neighbor_setup_method=r["neighbor_setup_method"]' in row_source

    def test_el_setup_uses_cell_list_not_all_pairs_neighbor_setup(self):
        """EL setup avoids all-pairs neighbor-list construction for large grids."""
        source = inspect.getsource(benchmark_electrostatics._el_build_nl)

        assert "batch_cell_list(" in source
        assert "return_neighbor_list=False" in source
        assert "_torch_neighbor_matrix_to_list_chunked" in source
        assert 'method="batch_cell_list"' in source
        assert "batch_naive_neighbor_list" not in source
        assert 'method="batch_naive"' not in source

    def test_el_torch_chunked_matrix_to_list_preserves_coo_order(self):
        """Chunked Torch COO conversion matches row-major neighbor ordering."""
        neighbor_matrix = torch.tensor(
            [
                [1, 3, 4],
                [0, 4, 4],
                [0, 1, 4],
            ],
            dtype=torch.int32,
        )
        num_neighbors = torch.tensor([2, 1, 2], dtype=torch.int32)
        shifts = torch.arange(27, dtype=torch.int32).reshape(3, 3, 3)

        nl, ptr, nl_shifts = _torch_neighbor_matrix_to_list_chunked(
            neighbor_matrix,
            num_neighbors,
            shifts,
            fill_value=4,
        )

        assert nl.tolist() == [[0, 0, 1, 2, 2], [1, 3, 0, 0, 1]]
        assert ptr.tolist() == [0, 2, 3, 5]
        assert nl_shifts.tolist() == [
            shifts[0, 0].tolist(),
            shifts[0, 1].tolist(),
            shifts[1, 0].tolist(),
            shifts[2, 0].tolist(),
            shifts[2, 1].tolist(),
        ]

    def test_jax_el_runs_large_configs_first_and_cleans_jax(self):
        """JAX EL avoids shape-sweep fragmentation where possible."""
        order_source = inspect.getsource(
            benchmark_electrostatics._ordered_configs_for_backend
        )
        run_source = inspect.getsource(benchmark_electrostatics.run_from_config)
        d3_run_source = inspect.getsource(benchmark_dftd3.run_from_config)

        assert 'backend != "jax"' in order_source
        assert "planned_atom_counts(sys_name, cfg)[2]" in order_source
        assert "reverse=True" in order_source
        assert "configs = _ordered_configs_for_backend" in run_source
        assert "clean_jax()" in run_source
        assert "clean_jax()" in d3_run_source

    def test_jax_el_records_failure_stage_and_serial_timing_fallback(self):
        """JAX EL distinguishes setup OOMs from batched timing OOMs."""
        setup_source = inspect.getsource(benchmark_electrostatics._el_setup_config)
        method_source = inspect.getsource(benchmark_electrostatics._el_run_method)
        fallback_source = inspect.getsource(
            benchmark_electrostatics._jax_timed_with_serial_fallback
        )

        assert "neighbor_list_setup" in setup_source
        assert "failure_stage=setup.failure_stage" in inspect.getsource(
            benchmark_electrostatics.run_from_config
        )
        assert 'failure_stage=f"{method_col}_timing"' in method_source
        assert "jax_timed_serial" in fallback_source
        assert "jax_wall_block_each" in fallback_source

    def test_jax_el_uses_explicit_atom_centric_nl_setup(self):
        """EL setup avoids auto pair-centric JAX FFI during reportable runs."""
        source = inspect.getsource(benchmark_electrostatics._el_build_nl)

        assert 'method="batch_cell_list"' in source
        assert 'strategy="atom_centric"' in source
        assert 'atom_centric_path="direct"' in source

    def test_d3_el_generic_oom_paths_clean_gpu(self):
        """Generic JAX-style OOM exceptions clean state before continuing."""
        for function in (
            benchmark_dftd3._d3_run_one_cutoff,
            benchmark_electrostatics._el_setup_config,
            benchmark_electrostatics._el_run_method,
        ):
            source = inspect.getsource(function)
            assert "error_type = failure_error_type(e)" in source
            assert 'if error_type == "OutOfMemoryError":' in source
            assert "clean_gpu()" in source

    def test_d3_missing_parameter_file_is_generated(self, monkeypatch, tmp_path):
        """D3 benchmark can populate the configured parameter cache path."""
        utils_module = ModuleType("examples.dispersion.utils")

        def fake_extract_dftd3_parameters():
            return {
                "rcov": torch.ones(1),
                "r4r2": torch.ones(1),
                "c6ab": torch.ones(1),
                "cn_ref": torch.ones(1),
            }

        utils_module.extract_dftd3_parameters = fake_extract_dftd3_parameters
        monkeypatch.setitem(sys.modules, "examples", ModuleType("examples"))
        monkeypatch.setitem(
            sys.modules, "examples.dispersion", ModuleType("examples.dispersion")
        )
        monkeypatch.setitem(sys.modules, "examples.dispersion.utils", utils_module)
        params_path = tmp_path / "cache" / "dftd3_parameters.pt"

        benchmark_dftd3._ensure_d3_parameter_file(params_path)

        assert params_path.exists()
        loaded = torch.load(params_path, map_location="cpu", weights_only=True)
        assert sorted(loaded) == ["c6ab", "cn_ref", "r4r2", "rcov"]

    def test_d3_default_parameter_path_honors_xdg_cache_home(
        self, monkeypatch, tmp_path
    ):
        """Default D3 cache path can be redirected to scratch with XDG_CACHE_HOME."""
        scratch_cache = tmp_path / "scratch-cache"
        monkeypatch.setenv("XDG_CACHE_HOME", str(scratch_cache))

        resolved = benchmark_dftd3._resolve_d3_params_path(
            "~/.cache/nvalchemiops/dftd3_parameters.pt"
        )
        explicit = benchmark_dftd3._resolve_d3_params_path(
            tmp_path / "explicit" / "dftd3_parameters.pt"
        )

        assert resolved == scratch_cache / "nvalchemiops" / "dftd3_parameters.pt"
        assert explicit == tmp_path / "explicit" / "dftd3_parameters.pt"

    def test_d3_params_path_override_only_updates_d3_config(self, tmp_path):
        """Unified suite can point D3 at a pre-seeded scratch parameter file."""
        params_path = tmp_path / "scratch" / "dftd3_parameters.pt"
        args = SimpleNamespace(
            timing_runs=None,
            warmup_runs=None,
            system=None,
            mode=None,
            output_dir=None,
            backend=None,
            methods=None,
            dry_run=False,
            max_total_atoms=None,
            d3_params_path=params_path,
        )

        d3_merged = merge_common_cli_overrides(
            load_yaml_config(benchmark_suite.RUNNERS["d3"]["config"]),
            args,
        )
        nl_merged = merge_common_cli_overrides(
            load_yaml_config(benchmark_suite.RUNNERS["nl"]["config"]),
            args,
        )

        assert d3_merged["params_path"] == str(params_path)
        assert "params_path" not in nl_merged

    def test_jax_batch_cluster_tile_uses_static_batch_ptr_metadata(self):
        """The benchmark wrapper keeps batch_ptr concrete for cluster-tile."""
        source = inspect.getsource(benchmark_neighborlist._benchmark_nl_jax)

        assert "batch_ptr_static = tuple" in source
        assert "batch_ptr=batch_ptr" in source

    def test_jax_cell_list_uses_precomputed_radius_metadata(self):
        """JAX cell-list benchmarks keep sizing metadata consistent."""
        source = inspect.getsource(benchmark_neighborlist._benchmark_nl_jax)
        kernel_source = inspect.getsource(benchmark_neighborlist._get_jax_nl_kernels)
        batch_source = source.rsplit('elif jax_family == "batch_cell_list":', 1)[
            1
        ].split('elif jax_family == "naive":', 1)[0]
        batch_signature = inspect.signature(jax_batch_cell_list)

        assert "use_direct_jax_nl = jax_family in" in source
        assert 'neighbor_search_radius=nl_kwargs["neighbor_search_radius"]' in source
        assert (
            'neighbor_search_radius=nl_kwargs["neighbor_search_radius"]' in batch_source
        )
        assert kernel_source.count('atom_centric_path="direct"') >= 2
        assert "neighbor_search_radius" in batch_signature.parameters
        assert "cell_list_min_cells = 1 if" in source
        assert "min_cells_per_dimension=cell_list_min_cells" in source

    def test_jax_cell_list_estimator_honors_min_cells_policy(self):
        """JAX sizing mirrors the Warp min-cells rule used by Torch."""
        import jax.numpy as jnp

        positions = jnp.zeros((2, 3), dtype=jnp.float32)
        cell = jnp.eye(3, dtype=jnp.float32) * 10.0
        pbc = jnp.ones((3,), dtype=jnp.bool_)

        max_cells_min1, cells_min1, radius_min1 = jax_estimate_cell_list_sizes(
            positions,
            cell,
            cutoff=6.0,
            pbc=pbc,
            min_cells_per_dimension=1,
        )
        max_cells_min4, cells_min4, radius_min4 = jax_estimate_cell_list_sizes(
            positions,
            cell,
            cutoff=6.0,
            pbc=pbc,
            min_cells_per_dimension=4,
        )
        batch_ptr = jnp.array([0, 2, 4], dtype=jnp.int32)
        batch_cell = jnp.broadcast_to(cell, (2, 3, 3))
        batch_pbc = jnp.broadcast_to(pbc, (2, 3))
        batch_positions = jnp.zeros((4, 3), dtype=jnp.float32)
        max_batch_min1, batch_cells_min1, batch_radius_min1 = (
            jax_estimate_batch_cell_list_sizes(
                batch_positions,
                batch_ptr=batch_ptr,
                cell=batch_cell,
                cutoff=6.0,
                pbc=batch_pbc,
                min_cells_per_dimension=1,
            )
        )

        assert max_cells_min1 == 1
        assert max_cells_min4 == 64
        assert cells_min1.tolist() == [1, 1, 1]
        assert cells_min4.tolist() == [4, 4, 4]
        assert radius_min1.tolist() == [1, 1, 1]
        assert radius_min4.tolist() == [3, 3, 3]
        assert max_batch_min1 == 2
        assert batch_cells_min1.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert batch_radius_min1.tolist() == [[1, 1, 1], [1, 1, 1]]

    def test_jax_batch_cell_list_has_direct_atom_centric_path(self):
        """JAX batch cell-list can skip gather for direct atom-centric queries."""
        module_source = inspect.getsource(jax_batch_cell_list_module)
        query_source = inspect.getsource(
            jax_batch_cell_list_module.batch_query_cell_list
        )

        assert (
            "_jax_batch_build_neighbor_matrix_local_count_direct_f32" in module_source
        )
        assert (
            "_jax_batch_build_neighbor_matrix_local_count_direct_f64" in module_source
        )
        assert 'atom_centric_path="direct"' in module_source
        assert (
            'use_direct = atom_centric_path == "direct" and not half_fill'
            in query_source
        )
        assert "if use_direct:" in query_source
        assert "sorted_positions = jnp.zeros((0, 3)" in query_source

    def test_jax_tail_fill_callback_avoids_graph_capture(self):
        """Tail fill must not load Warp modules while a JAX callback captures."""
        source = inspect.getsource(
            importlib.import_module("nvalchemiops.jax.neighbors.cell_list")
        )
        tail_section = source.split("_jax_fill_neighbor_matrix_tail = jax_callable", 1)[
            1
        ].split("def _resolve_cell_strategy", 1)[0]

        assert "graph_mode=GraphMode.NONE" in tail_section
        assert "graph_mode=GraphMode.WARP" not in tail_section

    def test_torch_cell_list_uses_preallocated_sizing_metadata(self):
        """Torch cell-list timings exclude sizing and scratch allocation."""
        source = inspect.getsource(benchmark_neighborlist.benchmark_nl)
        torch_cell_source = inspect.getsource(
            importlib.import_module("nvalchemiops.torch.neighbors.cell_list").cell_list
        )
        torch_batch_source = inspect.getsource(
            importlib.import_module(
                "nvalchemiops.torch.neighbors.batch_cell_list"
            ).batch_cell_list
        )

        assert "estimate_cell_list_sizes" in source
        assert "estimate_batch_cell_list_sizes" in source
        assert "allocate_cell_list" in source
        assert '"cells_per_dimension": cells_per_dimension' in source
        assert '"neighbor_search_radius": neighbor_search_radius' in source
        assert "cell_list_min_cells = 4" not in torch_cell_source
        assert "cell_list_min_cells = 4" not in torch_batch_source

    def test_torch_naive_uses_preallocated_output_and_shift_metadata(self):
        """Torch naive timings match JAX by reusing outputs and PBC shift metadata."""
        source = inspect.getsource(benchmark_neighborlist.benchmark_nl)

        assert "compute_naive_num_shifts" in source
        assert '"neighbor_matrix": torch.empty' in source
        assert '"neighbor_matrix_shifts": torch.empty' in source
        assert '"num_neighbors": torch.empty' in source
        assert '"shift_range_per_dimension": shift_range' in source
        assert '"num_shifts_per_system": num_shifts' in source
        assert '"max_shifts_per_system": int(max_shifts)' in source

    def test_explicit_pair_centric_uses_chunking_launcher(self):
        """Explicit pair-centric paths are not capped by the old launch guard."""
        torch_cell_source = Path(
            "nvalchemiops/torch/neighbors/cell_list.py"
        ).read_text()
        torch_batch_source = Path(
            "nvalchemiops/torch/neighbors/batch_cell_list.py"
        ).read_text()
        jax_cell_source = Path("nvalchemiops/jax/neighbors/cell_list.py").read_text()
        jax_batch_source = Path(
            "nvalchemiops/jax/neighbors/batch_cell_list.py"
        ).read_text()

        for source in (
            torch_cell_source,
            torch_batch_source,
            jax_cell_source,
            jax_batch_source,
        ):
            assert "_raise_unsafe_pair_centric_launch" not in source
            assert "is_pair_centric_launch_safe" not in source
            assert "The Warp launcher chunks oversized logical pair grids" in source
        assert 'use_pair = strategy == "pair_centric"' not in torch_cell_source
        assert 'use_pair_centric = strategy == "pair_centric"' not in torch_batch_source

    def test_max_total_atoms_override_updates_batch_scaling_grid(self):
        """CLI atom caps also resize batch-scaling config generation."""
        config = {
            "parameters": {},
            "runtime": {},
            "systems": {"cscl": {"enabled": True}},
            "scaling": {
                "system_size": {"enabled": True},
                "batch_scaling": {"enabled": True, "max_total_atoms": 128},
            },
            "methods": [{"name": "cell_list", "enabled": True}],
            "output": {"base_dir": "unused"},
        }
        args = SimpleNamespace(
            timing_runs=None,
            warmup_runs=None,
            system=None,
            mode=None,
            output_dir=None,
            backend=None,
            methods=None,
            dry_run=False,
            max_total_atoms=4096,
        )

        merged = merge_common_cli_overrides(config, args)

        assert merged["parameters"]["max_total_atoms"] == 4096
        assert merged["scaling"]["batch_scaling"]["max_total_atoms"] == 4096
        assert "max_total_atoms" not in merged["scaling"]["system_size"]

    def test_shipped_config_keeps_reportable_grid_without_hidden_overrides(self):
        """Shipped configs stay full/reportable unless CLI filters are explicit."""
        args = SimpleNamespace(
            timing_runs=None,
            warmup_runs=None,
            system=None,
            mode=None,
            output_dir=None,
            backend=None,
            methods=None,
            dry_run=False,
            max_total_atoms=None,
        )

        for runner in benchmark_suite.RUNNERS.values():
            config = load_yaml_config(runner["config"])
            merged = merge_common_cli_overrides(config, args)

            assert "profiles" not in merged
            assert merged.get("active_profile") is None
            assert merged["parameters"]["warmup_runs"] == 3
            assert merged["parameters"]["timing_runs"] == 10
            assert merged["scaling"]["constant_workload"]["target_atoms"] == 131072
            assert merged["scaling"]["batch_scaling"]["max_total_atoms"] == 131072
            assert "max_total_atoms" not in merged["parameters"]

            for system_config in merged["systems"].values():
                if "atom_counts" in system_config:
                    assert system_config["atom_counts"][-1] == 131072
                    assert 131072 in system_config["atom_counts"]

    def test_shipped_configs_use_canonical_methods(self):
        """Shipped YAMLs avoid stale review-era method names."""
        nl_config = load_yaml_config(benchmark_suite.RUNNERS["nl"]["config"])
        d3_config = load_yaml_config(benchmark_suite.RUNNERS["d3"]["config"])
        el_config = load_yaml_config(benchmark_suite.RUNNERS["el"]["config"])

        assert [m["name"] for m in nl_config["methods"]] == [
            "naive_scalar",
            "naive_tile",
            "cell_list_atom_centric",
            "cell_list_pair_centric",
            "cluster_tile",
        ]
        assert [m["name"] for m in d3_config["methods"]] == ["dftd3"]
        assert [m["name"] for m in el_config["methods"]] == ["pme", "ewald"]

        shipped_methods = {
            method["name"]
            for config in (nl_config, d3_config, el_config)
            for method in config["methods"]
        }
        assert "cell" not in shipped_methods
        assert not Path(
            "benchmarks/interactions/dispersion/validate_d3_energies.py"
        ).exists()

    def test_reportable_helper_uses_full_protocol_without_atom_cap(self):
        """The reportable helper does not hide a reduced workload."""
        source = Path("benchmarks/run_reportable_suite.sh").read_text()
        run_suite_body = source.split("run_suite() {", 1)[1].split("for backend in", 1)[
            0
        ]

        assert "--timing-runs 10" in run_suite_body
        assert "--warmup-runs 3" in run_suite_body
        assert "--max-total-atoms" not in run_suite_body

    def test_reportable_helper_exposes_hardware_neutral_shard_filters(self):
        """Reportable runs can be sharded without changing benchmark grids."""
        source = Path("benchmarks/run_reportable_suite.sh").read_text()

        assert 'BENCHMARK="all"' in source
        assert 'SYSTEM_FILTER=""' in source
        assert 'MODE_FILTER=""' in source
        assert "--benchmark all|nl|d3|el" in source
        assert "--system SYSTEM" in source
        assert "--mode MODE" in source
        assert '--benchmark "$BENCHMARK"' in source
        assert 'common_args+=(--system "$SYSTEM_FILTER")' in source
        assert 'common_args+=(--mode "$MODE_FILTER")' in source
        assert "full_suite_selection()" in source
        assert (
            "Skipping full-suite CSV completeness check for selected shard." in source
        )

    def test_reportable_helper_keeps_outputs_and_caches_off_home(self):
        """Reportable cluster runs route writable state to scratch."""
        source = Path("benchmarks/run_reportable_suite.sh").read_text()

        assert 'reject_home_path "BENCHMARK_SCRATCH" "$SCRATCH"' in source
        assert 'reject_home_path "output directory" "$RESULT_DIR"' in source
        assert 'reject_home_path "D3 parameter path" "$D3_PARAMS_PATH"' in source
        assert 'export HOME="$SCRATCH/home"' in source
        assert "BENCHMARK_D3_PARAMS_PATH" in source
        assert "--d3-params-path" in source
        for cache_var in (
            "XDG_CACHE_HOME",
            "UV_CACHE_DIR",
            "PRE_COMMIT_HOME",
            "UV_PROJECT_ENVIRONMENT",
            "WARP_CACHE_PATH",
            "TORCH_EXTENSIONS_DIR",
            "PYTORCH_KERNEL_CACHE_PATH",
            "JAX_COMPILATION_CACHE_DIR",
            "MPLCONFIGDIR",
            "CUDA_CACHE_PATH",
        ):
            assert f"export {cache_var}=" in source

    def test_reportable_helper_uses_compatible_sync_defaults(self):
        """Reportable cluster sync avoids mutually exclusive CUDA extras."""
        source = Path("benchmarks/run_reportable_suite.sh").read_text()

        assert "sync --all-extras" not in source
        assert (
            'UV_SYNC_ARGS="${UV_SYNC_ARGS:---extra torch --extra jax --group docs}"'
            in source
        )
        assert 'read -r -a uv_sync_args <<< "$UV_SYNC_ARGS"' in source
        assert '"$UV_BIN" sync "${uv_sync_args[@]}"' in source
        assert '--python "${UV_PROJECT_ENVIRONMENT}/bin/python"' in source
        assert "BENCHMARK_PIP_PACKAGES" in source
        assert "pyyaml>=6.0.3" in source
        assert "nvidia-ml-py==13.590.48" in source
        assert 'echo "uv_sync_args=$UV_SYNC_ARGS"' in source

    def test_reportable_helper_preflights_nh3_inputs(self):
        """Reportable runs fail early when generated NH3 PDBs are missing."""
        source = Path("benchmarks/run_reportable_suite.sh").read_text()

        assert "CONFIG_PATHS" in source
        assert "ammonia_pbc_{atom_count}.pdb" in source
        assert "Missing NH3 PBC benchmark inputs" in source
        assert "generate_pbc_pdbs.sh" in source

    def test_cluster_tile_resolves_to_batch_api_for_batched_cases(self):
        """Method expansion maps cluster-tile to the batch-shaped API."""
        assert _nl_method_for_case("cluster_tile", batch_size=1, explicit=False) == (
            "cluster_tile"
        )
        assert _nl_method_for_case("cluster_tile", batch_size=4, explicit=False) == (
            "batch_cluster_tile"
        )
        assert _nl_method_for_case("cluster_tile", batch_size=4, explicit=True) == (
            "batch_cluster_tile"
        )

    def test_explicit_nl_methods_still_follow_batch_shape(self):
        """CLI method filters select a family, not a mismatched batch API."""
        assert _nl_method_for_case("batch_cell_list", batch_size=1, explicit=True) == (
            "cell_list"
        )
        assert _nl_method_for_case("cell_list", batch_size=4, explicit=True) == (
            "batch_cell_list"
        )
        assert _nl_method_for_case("naive_scalar", batch_size=4, explicit=True) == (
            "batch_naive_scalar"
        )
        assert _nl_method_for_case(
            "batch_cell_list_pair_centric", batch_size=1, explicit=True
        ) == ("cell_list_pair_centric")


class TestSuiteBackendSelection:
    """Test suite-level backend compatibility guards."""

    def test_warp_backend_accepts_neighbor_list_only(self):
        """The Warp backend is valid for the NL benchmark."""
        validate_backend_selection("warp", {"nl"})

    def test_warp_backend_rejects_non_neighbor_list_benchmarks(self):
        """The suite rejects Warp for D3/EL before dispatch."""
        with pytest.raises(ValueError, match="not supported"):
            validate_backend_selection("warp", {"nl", "d3", "el"})

    def test_electrostatics_rejects_multipole_methods(self):
        """The unified suite keeps multipole EL outside this benchmark path."""
        with pytest.raises(ValueError, match="Multipole"):
            validate_method_selection(["multipole_ewald"], {"el"})


class TestBenchmarkSuitePlotting:
    """Test suite-level plotting control flow."""

    def test_nl_plot_labels_cluster_tile_family(self):
        """NL plot labels preserve user-facing strategy distinctions."""
        assert plot_benchmarks._nl_method_family("cluster_tile") == "cluster_tile"
        assert plot_benchmarks._nl_method_family("batch_cluster_tile") == "cluster_tile"
        assert plot_benchmarks._nl_method_label("batch_cluster_tile") == "Cluster tile"
        assert plot_benchmarks._nl_method_family("batch_naive_scalar") == (
            "naive_scalar"
        )
        assert plot_benchmarks._nl_method_label("batch_naive_tile") == "Naive tile"
        assert plot_benchmarks._nl_method_family("cell_list_pair_centric") == (
            "cell_list_pair_centric"
        )
        assert plot_benchmarks._nl_method_label("batch_cell_list_atom_centric") == (
            "Cell atom"
        )

    def test_plotting_does_not_assume_h100_vram_reference(self):
        """Memory plot hardware references must come from explicit metadata."""
        assert plot_benchmarks.GPU_VRAM_REFS == {}

    def test_comparison_plotter_filters_noncomparable_and_serial_rows(self):
        """Backend overlays keep CSV-visible but non-comparable rows out."""
        assert plot_benchmarks._is_backend_comparison_row(
            {
                "success": True,
                "backend_comparable": True,
                "timing_scope": "backend_comparison",
                "timing_method": "jax_wall_block_until_ready",
            }
        )
        assert not plot_benchmarks._is_backend_comparison_row(
            {
                "success": True,
                "backend_comparable": False,
                "timing_scope": "coverage_only_pair_centric",
                "timing_method": "jax_wall_block_until_ready",
            }
        )
        assert not plot_benchmarks._is_backend_comparison_row(
            {
                "success": True,
                "backend_comparable": True,
                "timing_scope": "backend_comparison",
                "timing_method": "jax_wall_block_each",
            }
        )

    def test_gitignore_allows_unified_benchmark_csvs(self):
        """Current unified NL/D3/EL docs CSVs are not ignored."""
        gitignore = Path(".gitignore").read_text(encoding="utf-8")

        assert "!docs/benchmarks/benchmark_results/nl-*.csv" in gitignore
        assert "!docs/benchmarks/benchmark_results/d3-*.csv" in gitignore
        assert "!docs/benchmarks/benchmark_results/el-*.csv" in gitignore

    def test_plot_data_line_skips_empty_memory_series(self):
        """All-missing memory series do not create ghost legend entries."""
        fig, ax = plot_benchmarks.plt.subplots()
        try:
            plotted = plot_benchmarks._plot_data_line(
                ax,
                [128, 256],
                [None, math.nan],
                color="black",
                linestyle="-",
                marker="o",
                label="D3 jax",
            )
            assert plotted is False
            assert list(ax.lines) == []
        finally:
            plot_benchmarks.plt.close(fig)

    def test_plot_only_short_circuits_inside_main(self, monkeypatch, tmp_path):
        """``main`` handles plot-only mode before importing any runners."""

        def fake_parse_args():
            return SimpleNamespace(
                benchmark=["all"],
                backend=None,
                plot_only=tmp_path,
                plots=["time"],
            )

        called = {}

        def fake_generate_plots(results_dir, plots=None):
            called["results_dir"] = Path(results_dir)
            called["plots"] = plots
            return True

        def fail_import(_module_name):
            pytest.fail("plot-only mode should not import benchmark runners")

        monkeypatch.setattr(benchmark_suite, "parse_args", fake_parse_args)
        monkeypatch.setattr(benchmark_suite, "_generate_plots", fake_generate_plots)
        monkeypatch.setattr(benchmark_suite.importlib, "import_module", fail_import)

        assert benchmark_suite.main() == 0
        assert called == {"results_dir": tmp_path, "plots": ["time"]}

    def test_run_dir_writes_results_without_nested_timestamp(
        self, monkeypatch, tmp_path
    ):
        """``--run-dir`` writes directly into a caller-owned result directory."""
        run_dir = tmp_path / "merged-results"
        calls = {}

        def fake_parse_args():
            return SimpleNamespace(
                benchmark=["nl"],
                backend="jax",
                plot_only=None,
                plots=["time"],
                system=None,
                mode=None,
                output_dir=tmp_path / "unused-base",
                run_dir=run_dir,
                timing_runs=None,
                warmup_runs=None,
                methods=None,
                dry_run=False,
                max_total_atoms=None,
                no_plot=True,
                cutoffs=None,
                accuracies=None,
            )

        fake_runner = SimpleNamespace(
            run_from_config=lambda _config, output_dir=None: (
                calls.setdefault("output_dir", Path(output_dir)) and [{"success": True}]
            )
        )

        monkeypatch.setattr(benchmark_suite, "parse_args", fake_parse_args)
        monkeypatch.setattr(
            benchmark_suite,
            "load_yaml_config",
            lambda _path: {
                "parameters": {},
                "runtime": {},
                "systems": {},
                "scaling": {},
                "methods": [],
                "output": {"base_dir": str(tmp_path)},
            },
        )
        monkeypatch.setattr(
            benchmark_suite.importlib,
            "import_module",
            lambda _name: fake_runner,
        )

        assert benchmark_suite.main() == 0
        assert calls["output_dir"] == run_dir
        assert run_dir.is_dir()
        assert not list((tmp_path / "unused-base").glob("run_*"))
        assert "- **Backend**: jax" in (run_dir / "RUN_LOG.md").read_text(
            encoding="utf-8"
        )

    def test_count_mode_uses_dry_plan_without_row_listing(self, monkeypatch, capsys):
        """``--count`` prints row counts through the no-allocation planning path."""

        def fake_parse_args():
            return SimpleNamespace(
                benchmark=["nl"],
                backend="torch",
                plot_only=None,
                plots=["time"],
                system=None,
                mode=None,
                output_dir=None,
                run_dir=None,
                timing_runs=None,
                warmup_runs=None,
                methods=None,
                dry_run=False,
                list_plan=False,
                count_plan=True,
                max_total_atoms=None,
                no_plot=True,
                cutoffs=None,
                accuracies=None,
                d3_params_path=None,
            )

        calls = {}

        def fake_dry_run(config):
            calls["plan_output"] = config["runtime"]["plan_output"]
            return [{"benchmark": "nl"}, {"benchmark": "nl"}]

        fake_runner = SimpleNamespace(
            dry_run_from_config=fake_dry_run,
            run_from_config=lambda *_args, **_kwargs: pytest.fail(
                "count mode must not run benchmarks"
            ),
        )

        monkeypatch.setattr(benchmark_suite, "parse_args", fake_parse_args)
        monkeypatch.setattr(
            benchmark_suite.importlib,
            "import_module",
            lambda _module_name: fake_runner,
        )

        assert benchmark_suite.main() == 0
        out = capsys.readouterr().out
        assert calls == {"plan_output": "count"}
        assert "COUNT COMPLETE: 2 planned row(s)" in out

    def test_generate_plots_fails_when_all_attempts_fail(self, monkeypatch, tmp_path):
        """A plot pass with CSV input returns False when every plot fails."""
        csv_path = tmp_path / "nl-cscl-system-size-scaling.csv"
        csv_path.write_text(
            "success,backend,method,total_atoms\nTrue,torch,cell_list,2\n"
        )

        def fail_plot(*_args, **_kwargs):
            raise RuntimeError("plot boom")

        monkeypatch.setattr(plot_benchmarks, "detect_and_plot", fail_plot)
        monkeypatch.setattr(plot_benchmarks, "plot_single_panel", fail_plot)

        assert benchmark_suite._generate_plots(tmp_path) is False

    def test_generate_plots_honors_panel_filter(self, monkeypatch, tmp_path):
        """``plots=['time']`` skips 3-panel plots and renders only time panels."""
        csv_path = tmp_path / "nl-cscl-system-size-scaling.csv"
        csv_path.write_text(
            "success,backend,method,total_atoms\nTrue,torch,cell_list,2\n"
        )
        panels = []

        def fail_three_panel(*_args, **_kwargs):
            pytest.fail(
                "filtered single-panel plotting should not render 3-panel plots"
            )

        def record_single_panel(_csv_path, panel, output_path, **kwargs):
            Path(output_path).write_text("png", encoding="utf-8")
            panels.append((panel, kwargs.get("filters")))
            return True

        monkeypatch.setattr(plot_benchmarks, "detect_and_plot", fail_three_panel)
        monkeypatch.setattr(plot_benchmarks, "plot_single_panel", record_single_panel)

        assert benchmark_suite._generate_plots(tmp_path, plots=["time"]) is True
        assert panels == [
            ("time", None),
            ("time", {"cutoff": 6.0}),
            ("time", {"cutoff": 15.0}),
            ("time", {"cutoff": 25.0}),
        ]

    def test_generate_plots_ignores_legacy_non_suite_csvs(self, monkeypatch, tmp_path):
        """Suite plot-only skips legacy or unrelated CSVs in docs result dirs."""
        suite_csv = tmp_path / "nl-cscl-system-size-scaling.csv"
        suite_csv.write_text(
            "success,backend,method,total_atoms,cutoff\nTrue,torch,cell_list,2,6.0\n",
            encoding="utf-8",
        )
        legacy_csv = tmp_path / "dftd3_benchmark_torch_h100-80gb-hbm3.csv"
        legacy_csv.write_text(
            "success,backend,method,total_atoms\nTrue,torch,dftd3,2\n",
            encoding="utf-8",
        )
        seen = []

        def record_single_panel(csv_path, panel, output_path, **kwargs):
            seen.append(Path(csv_path).name)
            Path(output_path).write_text("png", encoding="utf-8")
            return True

        monkeypatch.setattr(plot_benchmarks, "plot_single_panel", record_single_panel)

        assert benchmark_suite._generate_plots(tmp_path, plots=["time"]) is True
        assert seen == [
            "nl-cscl-system-size-scaling.csv",
            "nl-cscl-system-size-scaling.csv",
            "nl-cscl-system-size-scaling.csv",
            "nl-cscl-system-size-scaling.csv",
        ]

    def test_docs_backend_comparison_omits_noncomparable_pair_centric(
        self, monkeypatch, tmp_path
    ):
        """Torch-vs-JAX docs panels include direct cell-list and omit pair-centric."""
        csv_path = tmp_path / "nl-cscl-system-size-scaling.csv"
        rows = [
            "success,system,backend,method,cutoff,total_atoms,batch_size,time_us_per_atom,throughput_atoms_per_sec",
        ]
        for method in (
            "naive_scalar",
            "naive_tile",
            "cell_list_atom_centric",
            "cell_list_pair_centric",
        ):
            rows.extend(
                f"True,cscl,{backend},{method},15,128,1,1.0,128000000.0"
                for backend in ("torch", "jax")
            )
        csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

        batch_csv_path = tmp_path / "nl-cscl-batch-scaling.csv"
        batch_rows = [
            "success,system,backend,method,cutoff,total_atoms,batch_size,time_us_per_atom,throughput_atoms_per_sec",
        ]
        for method in (
            "batch_naive_scalar",
            "batch_naive_tile",
            "batch_cell_list_atom_centric",
            "batch_cell_list_pair_centric",
        ):
            batch_rows.extend(
                f"True,cscl,{backend},{method},15,128,4,1.0,128000000.0"
                for backend in ("torch", "jax")
            )
        batch_csv_path.write_text("\n".join(batch_rows) + "\n", encoding="utf-8")

        labels = []

        def record_series(series, output_path, **_kwargs):
            Path(output_path).write_text("png", encoding="utf-8")
            labels.extend(series)

        monkeypatch.setattr(docs_generate_plots, "plot_series", record_series)

        docs_generate_plots.generate_nl_backend_comparison_plots(tmp_path, tmp_path)

        assert any("naive scalar" in label for label in labels)
        assert any("cell list atom centric" in label for label in labels)
        assert any("batch cell list atom centric" in label for label in labels)
        assert not any("pair centric" in label for label in labels)

    def test_docs_backend_comparison_requires_matched_x_values(
        self, monkeypatch, tmp_path
    ):
        """Torch-vs-JAX docs panels only plot x-points where both backends succeeded."""
        csv_path = tmp_path / "nl-cscl-system-size-scaling.csv"
        csv_path.write_text(
            "\n".join(
                [
                    "success,system,backend,method,cutoff,total_atoms,batch_size,time_us_per_atom,throughput_atoms_per_sec,backend_comparable,timing_scope",
                    "True,cscl,torch,naive_scalar,15,128,1,1.0,128000000.0,True,backend_comparison",
                    "True,cscl,torch,naive_scalar,15,256,1,1.0,256000000.0,True,backend_comparison",
                    "True,cscl,jax,naive_scalar,15,128,1,2.0,64000000.0,True,backend_comparison",
                    "True,cscl,jax,naive_scalar,15,512,1,3.0,42666666.0,True,backend_comparison",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        plotted = {}

        def record_series(series, output_path, **_kwargs):
            Path(output_path).write_text("png", encoding="utf-8")
            plotted.update(series)

        monkeypatch.setattr(docs_generate_plots, "plot_series", record_series)

        docs_generate_plots.generate_nl_backend_comparison_plots(tmp_path, tmp_path)

        assert set(plotted) == {"naive scalar (torch)", "naive scalar (jax)"}
        assert plotted["naive scalar (torch)"][0].tolist() == [128.0]
        assert plotted["naive scalar (jax)"][0].tolist() == [128.0]

    def test_nl_method_metadata_marks_backend_comparable_methods(self):
        """NL rows label comparison scope explicitly in the CSV schema."""
        comparable = benchmark_neighborlist._nl_method_metadata("naive_tile")
        cell = benchmark_neighborlist._nl_method_metadata("cell_list_atom_centric")
        batch_cell = benchmark_neighborlist._nl_method_metadata(
            "batch_cell_list_atom_centric"
        )
        pair = benchmark_neighborlist._nl_method_metadata("cell_list_pair_centric")
        cluster = benchmark_neighborlist._nl_method_metadata("batch_cluster_tile")

        assert comparable == {
            "backend_comparable": True,
            "timing_scope": "backend_comparison",
        }
        assert cell == {
            "backend_comparable": True,
            "timing_scope": "backend_comparison",
        }
        assert batch_cell == {
            "backend_comparable": True,
            "timing_scope": "backend_comparison",
        }
        assert pair == {
            "backend_comparable": False,
            "timing_scope": "coverage_only_pair_centric",
        }
        assert cluster == {
            "backend_comparable": False,
            "timing_scope": "torch_cluster_tile_only",
        }

    def test_docs_backend_comparison_ignores_legacy_when_unified_csv_is_torch_only(
        self, monkeypatch, tmp_path
    ):
        """Legacy backend CSVs must not mask missing current JAX rows."""
        unified = tmp_path / "nl-cscl-system-size-scaling.csv"
        unified.write_text(
            "\n".join(
                [
                    "success,system,backend,method,cutoff,total_atoms,batch_size,time_us_per_atom,throughput_atoms_per_sec",
                    "True,cscl,torch,naive_scalar,15,128,1,1.0,128000000.0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        legacy = tmp_path / "nl-backend-cscl-system-size-scaling.csv"
        legacy.write_text(
            "\n".join(
                [
                    "success,system,backend,method,cutoff,total_atoms,batch_size,time_us_per_atom,throughput_atoms_per_sec",
                    "True,cscl,torch,naive,15,128,1,1.0,128000000.0",
                    "True,cscl,jax,naive,15,128,1,1.2,106000000.0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        outputs = []

        def record_series(_series, output_path, **_kwargs):
            Path(output_path).write_text("png", encoding="utf-8")
            outputs.append(Path(output_path).name)

        monkeypatch.setattr(docs_generate_plots, "plot_series", record_series)

        docs_generate_plots.generate_nl_backend_comparison_plots(tmp_path, tmp_path)

        assert outputs == []

    def test_docs_cutoff_selector_writes_placeholder_for_empty_filtered_view(
        self, monkeypatch, tmp_path
    ):
        """Cutoff selector targets should not silently fall back to all-cutoff plots."""
        csv_path = tmp_path / "nl-cscl-constant-workload-scaling.csv"
        csv_path.write_text(
            "\n".join(
                [
                    "success,system,backend,method,cutoff,total_atoms,batch_size,time_us_per_atom,throughput_atoms_per_sec,error_type",
                    "True,cscl,torch,batch_cell_list_atom_centric,25,131072,1,1.0,131072000.0,",
                    "False,cscl,jax,batch_cell_list_atom_centric,25,131072,1,nan,nan,OutOfMemoryError",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        def record_single_panel(_csv, _panel, output_path, **kwargs):
            filters = kwargs.get("filters") or {}
            if filters.get("backend") == "jax" and filters.get("cutoff") == 25.0:
                return False
            Path(output_path).write_text("png", encoding="utf-8")
            return True

        placeholders = []

        def record_placeholder(output_path, title, details):
            placeholders.append((Path(output_path).name, title, details))
            Path(output_path).write_text("placeholder", encoding="utf-8")

        monkeypatch.setattr(plot_benchmarks, "plot_single_panel", record_single_panel)
        monkeypatch.setattr(
            plot_benchmarks,
            "plot_comparison_panel",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            docs_generate_plots,
            "_write_no_data_placeholder",
            record_placeholder,
        )

        docs_generate_plots.generate_suite_csv_plots(tmp_path, tmp_path)

        assert (
            "nl-cscl-constant-workload-scaling-cutoff-25A-jax-time.png",
            "No successful benchmark rows",
            (
                "nl-cscl-constant-workload-scaling, time, JAX, 25A cutoff. "
                "See the CSV error_type column for failed rows."
            ),
        ) in placeholders
        assert (
            "nl-cscl-constant-workload-scaling-cutoff-25A-jax-throughput.png",
            "No successful benchmark rows",
            (
                "nl-cscl-constant-workload-scaling, throughput, JAX, 25A cutoff. "
                "See the CSV error_type column for failed rows."
            ),
        ) in placeholders

    def test_docs_d3_el_comparison_generates_memory_panels(self, monkeypatch, tmp_path):
        """D3/EL comparison plots refresh memory panels as well as timing panels."""
        for name in (
            "d3-cscl-system-size-scaling.csv",
            "el-cscl-system-size-scaling.csv",
        ):
            (tmp_path / name).write_text(
                "\n".join(
                    [
                        "success,system,scaling_mode,backend,method,total_atoms,batch_size,cutoff,accuracy,time_us_per_atom,throughput_atoms_per_sec,mem_peak_mb,backend_comparable,timing_scope",
                        "True,cscl,system_size,torch,pme,128,1,15,1e-6,1.0,128000000.0,10.0,True,backend_comparison",
                        "True,cscl,system_size,jax,pme,128,1,15,1e-6,2.0,64000000.0,nan,True,backend_comparison",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

        comparisons = []

        def record_comparison(csv_path, panel, output_path, module):
            comparisons.append(
                (Path(csv_path).name, panel, Path(output_path).name, module)
            )

        monkeypatch.setattr(
            plot_benchmarks,
            "plot_single_panel",
            lambda _csv, _panel, output_path, **_kwargs: (
                Path(output_path).write_text("png", encoding="utf-8") or True
            ),
        )
        monkeypatch.setattr(plot_benchmarks, "plot_comparison_panel", record_comparison)

        docs_generate_plots.generate_suite_csv_plots(tmp_path, tmp_path)

        assert (
            "d3-cscl-system-size-scaling.csv",
            "memory",
            "d3-cscl-system-size-comparison-memory.png",
            "d3",
        ) in comparisons
        assert (
            "el-cscl-system-size-scaling.csv",
            "memory",
            "el-cscl-system-size-comparison-memory.png",
            "el",
        ) in comparisons

    def test_comparison_plotter_requires_matched_x_values(self):
        """Shared comparison panels drop rows missing on either backend."""
        grouped = {
            ("naive_scalar", "torch", None): [
                {"total_atoms": 128, "backend": "torch"},
                {"total_atoms": 256, "backend": "torch"},
            ],
            ("naive_scalar", "jax", None): [
                {"total_atoms": 128, "backend": "jax"},
                {"total_atoms": 512, "backend": "jax"},
            ],
            ("cell_list_atom_centric", "torch", None): [
                {"total_atoms": 128, "backend": "torch"},
            ],
        }

        filtered = plot_benchmarks._filter_grouped_to_matched_backend_x(
            grouped,
            "total_atoms",
        )

        assert set(filtered) == {
            ("naive_scalar", "torch", None),
            ("naive_scalar", "jax", None),
        }
        assert [
            row["total_atoms"] for row in filtered[("naive_scalar", "torch", None)]
        ] == [128]
        assert [
            row["total_atoms"] for row in filtered[("naive_scalar", "jax", None)]
        ] == [128]

    def test_generate_plots_fails_when_single_panel_has_no_data(self, tmp_path):
        """All-failed CSVs are not counted as successfully rendered plots."""
        csv_path = tmp_path / "nl-cscl-system-size-scaling.csv"
        csv_path.write_text(
            "success,backend,method,total_atoms\nFalse,torch,cell_list,2\n"
        )

        assert benchmark_suite._generate_plots(tmp_path, plots=["time"]) is False

    def test_three_panel_detect_returns_false_when_csv_has_no_data(self, tmp_path):
        """The 3-panel plotter reports empty/all-failed CSVs to orchestration."""
        csv_path = tmp_path / "nl-cscl-system-size-scaling.csv"
        csv_path.write_text(
            "success,backend,method,total_atoms\nFalse,torch,cell_list,2\n",
            encoding="utf-8",
        )

        assert plot_benchmarks.detect_and_plot(csv_path, tmp_path) is False

    def test_suite_detects_yaml_selected_jax_backend(self, monkeypatch, tmp_path):
        """Suite-level JAX env setup honors YAML runtime backend, not just CLI."""
        config_path = tmp_path / "benchmark_config.yaml"
        config_path.write_text("runtime:\n  backend: jax\n", encoding="utf-8")
        monkeypatch.setitem(
            benchmark_suite.RUNNERS,
            "nl",
            {
                "label": "NL",
                "config": config_path,
                "module": "benchmarks.neighborlist.benchmark_neighborlist",
            },
        )
        args = SimpleNamespace(backend=None)

        assert benchmark_suite._suite_needs_jax_env(args, {"nl"}) is True

    def test_dry_run_with_no_planned_rows_fails(self, monkeypatch):
        """Dry-run exits nonzero when CLI filters produce an empty plan."""

        def fake_parse_args():
            return SimpleNamespace(
                benchmark=["el"],
                backend="torch",
                plot_only=None,
                plots=None,
                system=["cscl"],
                mode=["system_size"],
                output_dir=None,
                timing_runs=None,
                warmup_runs=None,
                methods=["cell_list"],
                dry_run=True,
                max_total_atoms=512,
                no_plot=True,
                accuracies=None,
            )

        monkeypatch.setattr(benchmark_suite, "parse_args", fake_parse_args)

        assert benchmark_suite.main() == 1

    def test_non_dry_run_with_no_successful_rows_fails(self, monkeypatch, tmp_path):
        """Runtime exits nonzero when every emitted row is failed or skipped."""

        def fake_parse_args():
            return SimpleNamespace(
                benchmark=["nl"],
                backend=None,
                plot_only=None,
                plots=["time"],
                system=None,
                mode=None,
                output_dir=tmp_path,
                timing_runs=None,
                warmup_runs=None,
                methods=None,
                dry_run=False,
                max_total_atoms=None,
                no_plot=True,
                cutoffs=None,
                accuracies=None,
            )

        fake_runner = SimpleNamespace(
            run_from_config=lambda _config, output_dir=None: [
                {"success": False, "error_type": "RuntimeError"}
            ]
        )

        monkeypatch.setattr(benchmark_suite, "parse_args", fake_parse_args)
        monkeypatch.setattr(
            benchmark_suite,
            "load_yaml_config",
            lambda _path: {
                "parameters": {},
                "runtime": {},
                "systems": {},
                "scaling": {},
                "methods": [],
                "output": {"base_dir": str(tmp_path)},
            },
        )
        monkeypatch.setattr(
            benchmark_suite.importlib,
            "import_module",
            lambda _name: fake_runner,
        )

        assert benchmark_suite.main() == 1

    def test_non_dry_run_fails_when_one_requested_module_has_no_success(
        self, monkeypatch, tmp_path
    ):
        """A failing requested module is not masked by another module's success."""

        def fake_parse_args():
            return SimpleNamespace(
                benchmark=["nl", "el"],
                backend=None,
                plot_only=None,
                plots=["time"],
                system=None,
                mode=None,
                output_dir=tmp_path,
                timing_runs=None,
                warmup_runs=None,
                methods=None,
                dry_run=False,
                max_total_atoms=None,
                no_plot=True,
                cutoffs=None,
                accuracies=None,
            )

        def fake_import(module_name):
            if module_name.endswith("benchmark_neighborlist"):
                return SimpleNamespace(
                    run_from_config=lambda _config, output_dir=None: [{"success": True}]
                )
            return SimpleNamespace(
                run_from_config=lambda _config, output_dir=None: [
                    {"success": False, "error_type": "RuntimeError"}
                ]
            )

        monkeypatch.setattr(benchmark_suite, "parse_args", fake_parse_args)
        monkeypatch.setattr(
            benchmark_suite,
            "load_yaml_config",
            lambda _path: {
                "parameters": {},
                "runtime": {},
                "systems": {},
                "scaling": {},
                "methods": [],
                "output": {"base_dir": str(tmp_path)},
            },
        )
        monkeypatch.setattr(benchmark_suite.importlib, "import_module", fake_import)

        assert benchmark_suite.main() == 1


class TestBenchmarkAtomPlanning:
    """Test allocation-free atom count planning."""

    def test_planned_atom_counts_for_cscl_supercell(self):
        """CsCl planning uses the rounded valid supercell atom count."""
        atoms_per_system, batch_size, total_atoms = planned_atom_counts(
            "cscl", {"num_atoms": 100, "batch_size": 4}
        )

        assert atoms_per_system == 128
        assert batch_size == 4
        assert total_atoms == 512

    def test_filter_configs_by_total_atoms_splits_before_allocation(self):
        """Configs over the atom cap are reported as skipped rows."""
        configs = [
            {"num_atoms": 100, "batch_size": 1},
            {"num_atoms": 100, "batch_size": 4},
        ]

        kept, skipped = filter_configs_by_total_atoms(configs, "cscl", 256)

        assert kept == [{"num_atoms": 100, "batch_size": 1}]
        assert skipped == [({"num_atoms": 100, "batch_size": 4}, 512)]

    def test_plan_only_nh3_configs_do_not_require_pdb_files(self, tmp_path):
        """Dry-run NH3 planning uses YAML sizes without generated PDB files."""
        configs = configs_for_mode(
            "system_size",
            {"enabled": True},
            "nh3",
            {"enabled": True, "atom_counts": [128, 256]},
            tmp_path / "missing-nh3",
            plan_only=True,
        )

        assert configs == [
            {"num_atoms": 128, "pdb_path": None, "batch_size": 1},
            {"num_atoms": 256, "pdb_path": None, "batch_size": 1},
        ]

    def test_actual_nh3_missing_pdbs_fall_back_to_planned_configs(self, tmp_path):
        """Actual runs keep row accounting when generated NH3 PDBs are absent."""
        configs = configs_for_mode(
            "system_size",
            {"enabled": True},
            "nh3",
            {"enabled": True, "atom_counts": [128]},
            tmp_path / "missing-nh3",
            plan_only=False,
        )

        assert configs == [{"num_atoms": 128, "pdb_path": None, "batch_size": 1}]

    def test_cscl_constant_workload_uses_yaml_atom_counts(self):
        """Constant-workload CsCl rows are driven by config, not baked-in grids."""
        configs = configs_for_mode(
            "constant_workload",
            {"enabled": True, "target_atoms": 1024},
            "cscl",
            {"enabled": True, "atom_counts": [100, 500]},
        )

        assert configs == [
            {"num_atoms": 100, "pdb_path": None, "batch_size": 8},
            {"num_atoms": 500, "pdb_path": None, "batch_size": 1},
        ]


class TestFailureRows:
    """Test failure row schema used by benchmark CSV output."""

    def test_build_failure_result_sets_success_and_error_fields(self):
        """Failures are written into main result rows with explicit metadata."""
        row = build_failure_result(
            error="boom",
            error_type="RuntimeError",
            benchmark="nl",
            backend="torch",
            system="cscl",
            scaling_mode="system_size",
            method="cell_list",
            atoms_per_system=128,
            batch_size=1,
            total_atoms=128,
            timing_runs=10,
            warmup_runs=3,
        )

        assert row["success"] is False
        assert row["error"] == "boom"
        assert row["error_type"] == "RuntimeError"
        assert row["method"] == "cell_list"
        assert row["timing_runs"] == 10
        assert row["warmup_runs"] == 3
        assert math.isnan(row["time_us_per_atom"])
        assert math.isnan(row["throughput_atoms_per_sec"])

    def test_success_result_uses_stable_error_columns(self):
        """Successful rows still carry empty error columns for append stability."""
        row = build_result(
            benchmark="nl",
            backend="torch",
            system="cscl",
            scaling_mode="system_size",
            method="cell_list",
            atoms_per_system=128,
            batch_size=1,
            total_atoms=128,
            time_seconds=1.0,
            timing_runs=10,
            warmup_runs=3,
            mem_info={"mem_delta_mb": 0.0, "mem_peak_gb": 0.0},
        )

        assert row["success"] is True
        assert row["error"] == ""
        assert row["error_type"] == ""
        assert row["timing_runs"] == 10
        assert row["warmup_runs"] == 3
        assert row["timing_method"] == "torch_cuda_events"
        assert row["compile_policy"] == "warmup_excluded"

    def test_jax_success_result_records_timing_contract(self):
        """JAX rows explicitly record wall-clock block-until-ready timing."""
        row = build_result(
            benchmark="nl",
            backend="jax",
            system="cscl",
            scaling_mode="system_size",
            method="cell_list",
            atoms_per_system=128,
            batch_size=1,
            total_atoms=128,
            time_seconds=1.0,
            timing_runs=10,
            warmup_runs=3,
            mem_info={"mem_delta_mb": math.nan, "mem_peak_gb": math.nan},
        )

        assert row["timing_method"] == "jax_wall_block_until_ready"
        assert row["compile_policy"] == "warmup_excluded"

    def test_result_row_treats_none_timing_metadata_as_backend_default(self):
        """Accidental None values do not erase timing metadata in CSV rows."""
        row = build_result(
            benchmark="el",
            backend="torch",
            system="cscl",
            scaling_mode="system_size",
            method="pme",
            atoms_per_system=128,
            batch_size=1,
            total_atoms=128,
            time_seconds=1.0,
            timing_runs=10,
            warmup_runs=3,
            mem_info={"mem_delta_mb": 0.0, "mem_peak_gb": 0.0},
            timing_method=None,
            compile_policy=None,
        )

        assert row["timing_method"] == "torch_cuda_events"
        assert row["compile_policy"] == "warmup_excluded"

    def test_oom_failure_result_uses_concise_error_message(self):
        """OOM rows keep a stable error type without embedding backend tracebacks."""
        row = build_failure_result(
            error="RESOURCE_EXHAUSTED: Failed to allocate request for 19.20GiB",
            error_type="OutOfMemoryError",
            benchmark="nl",
            backend="jax",
            system="cscl",
            scaling_mode="system_size",
            method="cell_list",
            atoms_per_system=128,
            batch_size=1024,
            total_atoms=131072,
            timing_runs=10,
            warmup_runs=3,
        )

        assert row["error_type"] == "OutOfMemoryError"
        assert row["error"] == (
            "Out of memory during benchmark execution; see run logs for backend details."
        )

    def test_save_results_preserves_existing_rows_when_schema_expands(self, tmp_path):
        """Appending failure rows to legacy CSV headers does not clobber data."""
        csv_path = tmp_path / "results.csv"
        csv_path.write_text("system,success\ncscl,True\n", encoding="utf-8")
        row = build_failure_result(
            error="boom",
            error_type="RuntimeError",
            benchmark="nl",
            backend="torch",
            system="nh3",
            scaling_mode="system_size",
            method="cell_list",
            atoms_per_system=128,
            batch_size=1,
            total_atoms=128,
            timing_runs=10,
            warmup_runs=3,
        )

        save_results([row], csv_path, append=True)

        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert [row["system"] for row in rows] == ["cscl", "nh3"]
        assert rows[1]["error"] == "boom"

    def test_save_results_replaces_existing_rows_by_default(self, tmp_path):
        """Default writes avoid silently mixing old benchmark generations."""
        csv_path = tmp_path / "results.csv"
        csv_path.write_text("system,success\nold,True\n", encoding="utf-8")
        row = build_result(
            benchmark="nl",
            backend="torch",
            system="new",
            scaling_mode="system_size",
            method="cell_list",
            atoms_per_system=128,
            batch_size=1,
            total_atoms=128,
            time_seconds=1.0,
            timing_runs=10,
            warmup_runs=3,
            mem_info={"mem_delta_mb": 0.0, "mem_peak_gb": 0.0},
        )

        save_results([row], csv_path)

        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert [row["system"] for row in rows] == ["new"]

    def test_save_results_replaces_only_requested_backend(self, tmp_path):
        """Shared docs CSVs keep other backends while refreshing one backend."""
        csv_path = tmp_path / "results.csv"
        torch_old = build_result(
            benchmark="nl",
            backend="torch",
            system="old_torch",
            scaling_mode="system_size",
            method="cell_list",
            atoms_per_system=128,
            batch_size=1,
            total_atoms=128,
            time_seconds=1.0,
            timing_runs=10,
            warmup_runs=3,
            mem_info={"mem_delta_mb": 0.0, "mem_peak_gb": 0.0},
        )
        jax_row = build_result(
            benchmark="nl",
            backend="jax",
            system="jax",
            scaling_mode="system_size",
            method="cell_list",
            atoms_per_system=128,
            batch_size=1,
            total_atoms=128,
            time_seconds=1.0,
            timing_runs=10,
            warmup_runs=3,
            mem_info={"mem_delta_mb": 0.0, "mem_peak_gb": 0.0},
            timing_method="jax_wall_block_last",
        )
        torch_new = build_result(
            benchmark="nl",
            backend="torch",
            system="new_torch",
            scaling_mode="system_size",
            method="cell_list",
            atoms_per_system=256,
            batch_size=1,
            total_atoms=256,
            time_seconds=1.0,
            timing_runs=10,
            warmup_runs=3,
            mem_info={"mem_delta_mb": 0.0, "mem_peak_gb": 0.0},
        )

        save_results([torch_old], csv_path, replace_backend="torch")
        save_results([jax_row], csv_path, replace_backend="jax")
        save_results([torch_new], csv_path, replace_backend="torch")

        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert [(row["backend"], row["system"]) for row in rows] == [
            ("jax", "jax"),
            ("torch", "new_torch"),
        ]

    def test_run_log_describes_jax_timing_contract(self, tmp_path):
        """RUN_LOG.md does not describe JAX as CUDA-event timed."""
        start = benchmark_suite.datetime(2026, 1, 1, 0, 0, 0)

        write_run_log(tmp_path, start)

        text = (tmp_path / "RUN_LOG.md").read_text(encoding="utf-8")
        assert "Torch/Warp CUDA timing pattern" in text
        assert "JAX timing pattern" in text
        assert "block_until_ready(last)" in text
        assert "JAX/XLA Environment" in text
        assert "XLA_PYTHON_CLIENT_MEM_FRACTION" in text
        assert "Runtime Cache Environment" in text
        assert "XDG_CACHE_HOME" in text
        assert "PYTORCH_KERNEL_CACHE_PATH" in text
        assert "Reported timings exclude warm-up/compile/load iterations." in text

    def test_build_skipped_result_sets_policy_error_type(self):
        """Policy skips use the same failure-row schema as runtime failures."""
        row = build_skipped_result(
            reason=">64 max_total_atoms",
            benchmark="nl",
            backend="jax",
            system="cscl",
            scaling_mode="system_size",
            method="cell_list",
            atoms_per_system=128,
            batch_size=1,
            total_atoms=128,
            cutoff=25.0,
            timing_runs=10,
            warmup_runs=3,
        )

        assert row["success"] is False
        assert row["error"] == ">64 max_total_atoms"
        assert row["error_type"] == "SkippedByPolicy"
        assert row["method"] == "cell_list"
        assert row["timing_runs"] == 10
        assert row["warmup_runs"] == 3

    def test_result_rows_require_timing_metadata(self):
        """Every benchmark row must carry timing run-count metadata."""
        with pytest.raises(ValueError, match="timing_runs is required"):
            build_result(
                benchmark="nl",
                backend="torch",
                system="cscl",
                scaling_mode="system_size",
                method="cell_list",
                atoms_per_system=128,
                batch_size=1,
                total_atoms=128,
                time_seconds=1.0,
                timing_runs=None,
                warmup_runs=3,
                mem_info={"mem_delta_mb": 0.0, "mem_peak_gb": 0.0},
            )


class TestDryRunSkipPlanning:
    """Test allocation-free planning of policy-skipped benchmark rows."""

    def test_nl_dry_run_expands_max_atom_skips_per_method_and_cutoff(self):
        """NL max-atom caps are visible for every planned method/cutoff row."""
        config = {
            "parameters": {
                "cutoffs": [6.0, 25.0],
                "max_total_atoms": 64,
            },
            "runtime": {},
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [
                {"name": "naive_neighbor_list", "enabled": True},
                {"name": "cell_list", "enabled": True},
            ],
        }

        rows = dry_run_nl(config, backend="jax")

        assert len(rows) == 4
        assert {row["method"] for row in rows} == {
            "naive_neighbor_list",
            "cell_list",
        }
        assert {row["cutoff"] for row in rows} == {6.0, 25.0}
        assert {row["reason"] for row in rows} == {">64 max_total_atoms"}

    def test_nl_dry_run_includes_cluster_tile_family(self):
        """Cluster-tile is a first-class planned NL method."""
        config = {
            "parameters": {
                "cutoffs": [15.0],
                "max_total_atoms": 1024,
            },
            "runtime": {},
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [
                {"name": "cell_list", "enabled": True},
                {"name": "cluster_tile", "enabled": True},
            ],
        }

        rows = dry_run_nl(config, backend="torch")

        assert [row["method"] for row in rows] == ["cell_list", "cluster_tile"]

    def test_nl_warp_default_omits_unsupported_cluster_tile_family(self):
        """Default Warp dry-runs include only runnable Warp NL methods."""
        config = {
            "parameters": {
                "cutoffs": [15.0],
                "max_total_atoms": 1024,
            },
            "runtime": {},
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [
                {"name": "cell_list", "enabled": True},
                {"name": "cluster_tile", "enabled": True},
            ],
        }

        rows = dry_run_nl(config, backend="warp")

        assert [row["method"] for row in rows] == ["cell_list"]
        assert {row["reason"] for row in rows} == {""}

    def test_nl_jax_default_omits_coverage_and_unsupported_methods(self):
        """Default JAX dry-runs keep production scope to comparable methods."""
        config = {
            "parameters": {
                "cutoffs": [15.0],
                "max_total_atoms": 1024,
            },
            "runtime": {},
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [
                {"name": "cell_list", "enabled": True},
                {"name": "cell_list_pair_centric", "enabled": True},
                {"name": "cluster_tile", "enabled": True},
            ],
        }

        rows = dry_run_nl(config, backend="jax")

        assert [row["method"] for row in rows] == ["cell_list"]
        assert {row["reason"] for row in rows} == {""}

    def test_nl_jax_explicit_pair_centric_is_planned(self):
        """Coverage-only JAX pair-centric remains available when requested."""
        config = {
            "parameters": {
                "cutoffs": [15.0],
                "max_total_atoms": 1024,
            },
            "runtime": {
                "explicit_methods": True,
                "selected_methods": ["cell_list_pair_centric"],
            },
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [{"name": "cell_list_pair_centric", "enabled": True}],
        }

        rows = dry_run_nl(config, backend="jax")

        assert [row["method"] for row in rows] == ["cell_list_pair_centric"]
        assert {row["reason"] for row in rows} == {""}

    def test_nl_jax_explicit_cluster_tile_is_policy_skipped(self):
        """Explicit JAX cluster-tile rows are visible policy skips."""
        config = {
            "parameters": {
                "cutoffs": [15.0],
                "max_total_atoms": 1024,
            },
            "runtime": {
                "explicit_methods": True,
                "selected_methods": ["cluster_tile"],
            },
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [{"name": "cluster_tile", "enabled": True}],
        }

        rows = dry_run_nl(config, backend="jax")

        assert [row["method"] for row in rows] == ["cluster_tile"]
        assert {row["reason"] for row in rows} == {
            "jax backend does not support cluster_tile in this benchmark suite"
        }

    def test_nl_dry_run_batches_cluster_tile_for_default_methods(self):
        """Default method expansion uses batch_cluster_tile for batched inputs."""
        config = {
            "parameters": {
                "cutoffs": [15.0],
                "max_total_atoms": 1024,
            },
            "runtime": {},
            "systems": {"cscl": {"enabled": True, "atom_counts": [128, 500]}},
            "scaling": {"constant_workload": {"enabled": True, "target_atoms": 1024}},
            "methods": [{"name": "cluster_tile", "enabled": True}],
        }

        rows = dry_run_nl(config, backend="torch")

        assert rows
        assert {row["method"] for row in rows} == {
            "cluster_tile",
            "batch_cluster_tile",
        }

    def test_d3_dry_run_respects_method_filter(self):
        """D3 does not plan rows when CLI-selected methods exclude dftd3."""
        config = {
            "parameters": {
                "cutoffs": [15.0],
                "max_total_atoms": 256,
            },
            "runtime": {"explicit_methods": True, "selected_methods": ["pme"]},
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [{"name": "dftd3", "enabled": False}],
        }

        rows = dry_run_d3(config, backend="jax")

        assert rows == []

    def test_d3_dry_run_respects_disabled_yaml_method(self):
        """D3 YAML method switches are authoritative."""
        config = {
            "parameters": {
                "cutoffs": [15.0],
                "max_total_atoms": 256,
            },
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [{"name": "dftd3", "enabled": False}],
        }

        rows = dry_run_d3(config, backend="jax")

        assert rows == []

    def test_el_setup_failure_is_written_as_failure_row(self, monkeypatch, tmp_path):
        """EL setup failures emit explicit failure rows instead of disappearing."""
        config = {
            "parameters": {"timing_runs": 1, "warmup_runs": 1, "max_total_atoms": 1024},
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [{"name": "pme", "enabled": True, "spline_order": 5}],
            "accuracies": [1.0e-4],
            "compute_charge_gradients": [False],
            "output": {"base_dir": str(tmp_path)},
        }

        def fail_create_system(*_args, **_kwargs):
            raise RuntimeError("setup boom")

        monkeypatch.setattr(benchmark_electrostatics, "clean_gpu", lambda: None)
        monkeypatch.setattr(
            benchmark_electrostatics,
            "create_system",
            fail_create_system,
        )

        rows = benchmark_electrostatics.run_from_config(
            config,
            output_dir=tmp_path,
            backend="torch",
        )

        assert len(rows) == 1
        assert rows[0]["success"] is False
        assert rows[0]["error"] == "setup boom"
        assert rows[0]["error_type"] == "RuntimeError"
        assert rows[0]["method"] == "pme"
        assert rows[0]["timing_runs"] == 1
        assert rows[0]["warmup_runs"] == 1

    def test_nl_setup_failure_is_written_as_failure_rows(self, monkeypatch, tmp_path):
        """NL setup failures emit one row per planned method/cutoff."""
        config = {
            "parameters": {"timing_runs": 1, "warmup_runs": 1, "cutoffs": [6.0]},
            "runtime": {},
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [{"name": "cell_list", "enabled": True}],
            "output": {"base_dir": str(tmp_path)},
        }

        monkeypatch.setattr(benchmark_neighborlist, "clean_gpu", lambda: None)
        monkeypatch.setattr(
            benchmark_neighborlist,
            "create_system",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("setup boom")),
        )

        rows = benchmark_neighborlist.run_from_config(
            config,
            output_dir=tmp_path,
            backend="torch",
        )

        assert len(rows) == 1
        assert rows[0]["success"] is False
        assert rows[0]["error"] == "setup boom"
        assert rows[0]["error_type"] == "RuntimeError"

    def test_nl_warp_cluster_tile_is_policy_skipped_before_allocation(
        self, monkeypatch, tmp_path
    ):
        """Warp cluster-tile rows are explicit skips, not unsupported failures."""
        config = {
            "parameters": {"timing_runs": 1, "warmup_runs": 1, "cutoffs": [6.0]},
            "runtime": {},
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [{"name": "cluster_tile", "enabled": True}],
            "output": {"base_dir": str(tmp_path)},
        }

        def fail_create_system(*_args, **_kwargs):
            pytest.fail("policy-skipped cluster_tile should not allocate")

        monkeypatch.setattr(benchmark_neighborlist, "create_system", fail_create_system)

        rows = benchmark_neighborlist.run_from_config(
            config,
            output_dir=tmp_path,
            backend="warp",
        )

        assert len(rows) == 1
        assert rows[0]["success"] is False
        assert rows[0]["error_type"] == "SkippedByPolicy"
        assert rows[0]["error"] == "warp backend does not support cluster_tile"

    def test_d3_setup_failure_is_written_as_failure_rows(self, monkeypatch, tmp_path):
        """D3 setup failures emit one row per planned cutoff."""
        params_path = tmp_path / "d3_params.pt"
        torch.save({"rcov": torch.tensor([1.0])}, params_path)
        config = {
            "params_path": str(params_path),
            "parameters": {
                "timing_runs": 1,
                "warmup_runs": 1,
                "cutoffs": [6.0, 15.0],
            },
            "runtime": {},
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [{"name": "dftd3", "enabled": True}],
            "dftd3_parameters": {"a1": 0.4289, "a2": 4.4407, "s8": 0.7875},
            "output": {"base_dir": str(tmp_path)},
        }

        monkeypatch.setattr(
            benchmark_dftd3,
            "_torch_d3_params_to_device",
            lambda params, _device: params,
        )
        monkeypatch.setattr(
            benchmark_dftd3,
            "create_system",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("setup boom")),
        )

        rows = benchmark_dftd3.run_from_config(
            config,
            output_dir=tmp_path,
            backend="torch",
        )

        assert len(rows) == 2
        assert {row["cutoff"] for row in rows} == {6.0, 15.0}
        assert {row["error"] for row in rows} == {"setup boom"}

    def test_d3_missing_parameters_are_written_as_failure_rows(self, tmp_path):
        """Missing D3 parameter files emit planned CSV failure rows."""
        config = {
            "params_path": str(tmp_path / "missing_d3_params.pt"),
            "parameters": {
                "timing_runs": 1,
                "warmup_runs": 1,
                "cutoffs": [6.0, 15.0],
            },
            "runtime": {},
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [{"name": "dftd3", "enabled": True}],
            "dftd3_parameters": {"a1": 0.4289, "a2": 4.4407, "s8": 0.7875},
            "output": {"base_dir": str(tmp_path)},
        }

        rows = benchmark_dftd3.run_from_config(
            config,
            output_dir=tmp_path,
            backend="torch",
        )

        assert len(rows) == 2
        assert {row["cutoff"] for row in rows} == {6.0, 15.0}
        assert {row["success"] for row in rows} == {False}
        assert {row["error_type"] for row in rows} == {"FileNotFoundError"}
        assert (tmp_path / "d3-cscl-system-size-scaling.csv").exists()


class TestJaxMemoryContract:
    """Test JAX memory metadata behavior."""

    def test_jax_timed_batch_compiles_before_timing_when_zero_warmups(
        self, monkeypatch
    ):
        """JAX batch timing excludes compilation even when zero warmups are requested."""
        calls = []
        blocked = []

        fake_jax = SimpleNamespace(
            block_until_ready=lambda state: blocked.append(state),
        )
        monkeypatch.setitem(sys.modules, "jax", fake_jax)

        def step():
            calls.append(len(calls) + 1)
            return calls[-1]

        elapsed = jax_timed_batch(step, num_runs=2, warmup_runs=0)

        assert elapsed >= 0.0
        assert calls == [1, 2, 3]
        assert blocked == [1, 3]

    def test_jax_timed_stateful_threads_state(self, monkeypatch):
        """Stateful JAX timing carries donated buffers through every call."""
        seen_states = []
        blocked_states = []

        fake_jax = SimpleNamespace(
            block_until_ready=lambda state: blocked_states.append(state),
        )
        monkeypatch.setitem(sys.modules, "jax", fake_jax)

        def step(state):
            seen_states.append(state)
            return state + 1

        elapsed, final_state = jax_timed_stateful(
            step,
            state=0,
            num_runs=3,
            warmup_runs=2,
        )

        assert elapsed >= 0.0
        assert final_state == 5
        assert seen_states == [0, 1, 2, 3, 4]
        assert blocked_states == [1, 2, 5]

    def test_jax_timed_stateful_compiles_before_timing_when_zero_warmups(
        self, monkeypatch
    ):
        """Stateful JAX timing also excludes compile when zero warmups are requested."""
        seen_states = []
        blocked_states = []

        fake_jax = SimpleNamespace(
            block_until_ready=lambda state: blocked_states.append(state),
        )
        monkeypatch.setitem(sys.modules, "jax", fake_jax)

        def step(state):
            seen_states.append(state)
            return state + 1

        elapsed, final_state = jax_timed_stateful(
            step,
            state=0,
            num_runs=2,
            warmup_runs=0,
        )

        assert elapsed >= 0.0
        assert final_state == 3
        assert seen_states == [0, 1, 2]
        assert blocked_states == [1, 3]

    def test_measure_memory_jax_reports_nan_without_running_function(self):
        """JAX memory is unavailable and must not trigger an extra benchmark run."""

        class FakeJax:
            @staticmethod
            def block_until_ready(result):
                """Return the already-computed fake result."""
                return result

        def fail_if_called():
            raise AssertionError("JAX memory probe should not execute benchmarks")

        result, mem_info = measure_memory_jax(fail_if_called, FakeJax)

        assert result is None
        assert math.isnan(mem_info["mem_delta_mb"])
        assert math.isnan(mem_info["mem_peak_gb"])

    def test_plot_memory_suppresses_jax_rows(self):
        """JAX memory rows are not plotted even if stale CSVs contain values."""
        values = plot_benchmarks._get_memory_y(
            [
                {
                    "backend": "jax",
                    "mem_delta_mb": 123.0,
                    "mem_peak_gb": 80.0,
                },
            ]
        )

        assert values == [None]


class TestElectrostaticsParameterUnpack:
    """Test electrostatics benchmark parameter shape handling."""

    def test_torch_alpha_keeps_per_system_shape(self):
        """Torch component timing receives vector alpha, not a collapsed scalar."""
        pme_params = SimpleNamespace(
            alpha=torch.tensor([0.1, 0.2], dtype=torch.float64),
            real_space_cutoff=torch.tensor([8.0, 8.0], dtype=torch.float64),
            mesh_dimensions=(16, 16, 16),
        )
        ewald_params = SimpleNamespace(
            reciprocal_space_cutoff=torch.tensor([4.0, 4.0], dtype=torch.float64)
        )

        alpha, real_cutoff, mesh_dims, k_cutoff = _el_unpack_params(
            pme_params, ewald_params, "torch"
        )

        assert alpha.shape == (2,)
        assert torch.allclose(alpha, pme_params.alpha)
        assert real_cutoff == 8.0
        assert mesh_dims == (16, 16, 16)
        assert k_cutoff == 4.0

    def test_jax_alpha_object_is_preserved(self):
        """JAX component timing receives the original array-like alpha."""
        alpha_value = object()
        pme_params = SimpleNamespace(
            alpha=alpha_value,
            real_space_cutoff=[8.0],
            mesh_dimensions=(16, 16, 16),
        )
        ewald_params = SimpleNamespace(
            reciprocal_space_cutoff=SimpleNamespace(max=lambda: 4.0)
        )

        alpha, real_cutoff, mesh_dims, k_cutoff = _el_unpack_params(
            pme_params, ewald_params, "jax"
        )

        assert alpha is alpha_value
        assert real_cutoff == 8.0
        assert mesh_dims == (16, 16, 16)
        assert k_cutoff == 4.0


class TestBenchmarkCsvLoading:
    """Test benchmark CSV type conversion used by plotting."""

    def test_load_csv_parses_nan_tokens(self, tmp_path):
        """JAX memory NaN fields are parsed as floats, not strings."""
        csv_path = tmp_path / "results.csv"
        csv_path.write_text(
            "backend,success,mem_delta_mb,mem_peak_gb,time_us_per_atom\n"
            "jax,True,nan,nan,1.25\n",
            encoding="utf-8",
        )

        rows = load_csv(csv_path)

        assert len(rows) == 1
        assert isinstance(rows[0]["mem_delta_mb"], float)
        assert math.isnan(rows[0]["mem_delta_mb"])
        assert isinstance(rows[0]["mem_peak_gb"], float)
        assert math.isnan(rows[0]["mem_peak_gb"])
