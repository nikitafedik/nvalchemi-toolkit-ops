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
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmarks import benchmark_suite
from benchmarks.benchmark_suite import (
    validate_backend_selection,
    validate_method_selection,
)
from benchmarks.config import (
    enabled_method_names,
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
)
from benchmarks.interactions.electrostatics.benchmark_electrostatics import (
    dry_run_from_config as dry_run_el,
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
    measure_memory_jax,
    save_results,
)


class TestBenchmarkMethodSelection:
    """Test benchmark method normalization and CLI selection."""

    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            ("cell", "cell_list"),
            ("naive", "naive_neighbor_list"),
            ("batch-cell-list", "batch_cell_list"),
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
        """NL plot labels collapse batch/unbatch cluster methods cleanly."""
        assert plot_benchmarks._nl_method_family("cluster_tile") == "cluster_tile"
        assert plot_benchmarks._nl_method_family("batch_cluster_tile") == "cluster_tile"
        assert plot_benchmarks._nl_method_label("batch_cluster_tile") == "Cluster tile"

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

        def record_single_panel(_csv_path, panel, output_path):
            Path(output_path).write_text("png", encoding="utf-8")
            panels.append(panel)
            return True

        monkeypatch.setattr(plot_benchmarks, "detect_and_plot", fail_three_panel)
        monkeypatch.setattr(plot_benchmarks, "plot_single_panel", record_single_panel)

        assert benchmark_suite._generate_plots(tmp_path, plots=["time"]) is True
        assert panels == ["time"]

    def test_generate_plots_fails_when_single_panel_has_no_data(self, tmp_path):
        """All-failed CSVs are not counted as successfully rendered plots."""
        csv_path = tmp_path / "nl-cscl-system-size-scaling.csv"
        csv_path.write_text(
            "success,backend,method,total_atoms\nFalse,torch,cell_list,2\n"
        )

        assert benchmark_suite._generate_plots(tmp_path, plots=["time"]) is False

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
            time_us_per_atom=math.nan,
        )

        assert row["success"] is False
        assert row["error"] == "boom"
        assert row["error_type"] == "RuntimeError"
        assert row["method"] == "cell_list"

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
            mem_info={"mem_delta_mb": 0.0, "mem_peak_gb": 0.0},
        )

        assert row["success"] is True
        assert row["error"] == ""
        assert row["error_type"] == ""

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
        )

        save_results([row], csv_path)

        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert [row["system"] for row in rows] == ["cscl", "nh3"]
        assert rows[1]["error"] == "boom"

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
        )

        assert row["success"] is False
        assert row["error"] == ">64 max_total_atoms"
        assert row["error_type"] == "SkippedByPolicy"
        assert row["method"] == "cell_list"


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

    def test_nl_dry_run_batches_cluster_tile_for_default_methods(self):
        """Default method expansion uses batch_cluster_tile for batched inputs."""
        config = {
            "parameters": {
                "cutoffs": [15.0],
                "max_total_atoms": 1024,
            },
            "runtime": {},
            "systems": {"cscl": {"enabled": True}},
            "scaling": {"constant_workload": {"enabled": True, "target_atoms": 1024}},
            "methods": [{"name": "cluster_tile", "enabled": True}],
        }

        rows = dry_run_nl(config, backend="torch")

        assert rows
        assert {row["method"] for row in rows} == {
            "cluster_tile",
            "batch_cluster_tile",
        }

    def test_d3_dry_run_marks_cutoff_limit_skips(self):
        """D3 shares the cutoff-limit planning contract with NL."""
        config = {
            "parameters": {
                "cutoffs": [15.0, 25.0],
                "cutoff_limits": {25.0: 64},
                "max_total_atoms": 256,
            },
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
        }

        rows = dry_run_d3(config, backend="jax")

        assert len(rows) == 2
        skipped = [row for row in rows if row["reason"]]
        assert len(skipped) == 1
        assert skipped[0]["method"] == "dftd3"
        assert skipped[0]["cutoff"] == 25.0
        assert skipped[0]["reason"] == ">64 cutoff_limit"

    def test_d3_dry_run_respects_method_filter(self):
        """D3 does not plan rows when CLI-selected methods exclude dftd3."""
        config = {
            "parameters": {
                "cutoffs": [15.0],
                "cutoff_limits": {},
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
                "cutoff_limits": {},
                "max_total_atoms": 256,
            },
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [{"name": "dftd3", "enabled": False}],
        }

        rows = dry_run_d3(config, backend="jax")

        assert rows == []

    def test_el_dry_run_marks_high_accuracy_policy_skips(self):
        """EL high-accuracy OOM policy is visible before allocation."""
        config = {
            "parameters": {"max_total_atoms": 1024},
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [
                {"name": "pme", "enabled": True},
                {"name": "ewald", "enabled": True},
            ],
            "accuracies": [1.0e-4, 1.0e-6],
            "compute_charge_gradients": [False, True],
            "skip_accuracy_for_large": {1.0e-6: 128},
            "max_atoms": 1024,
        }

        rows = dry_run_el(config, backend="jax")

        skipped = [row for row in rows if row["reason"]]
        assert len(rows) == 8
        assert len(skipped) == 4
        assert {row["accuracy"] for row in skipped} == {1.0e-6}
        assert {row["reason"] for row in skipped} == {">=128 skip_accuracy_for_large"}

    def test_el_setup_failure_is_written_as_failure_row(self, monkeypatch, tmp_path):
        """EL setup failures emit explicit failure rows instead of disappearing."""
        config = {
            "parameters": {"timing_runs": 1, "warmup_runs": 1, "max_total_atoms": 1024},
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [{"name": "pme", "enabled": True, "spline_order": 5}],
            "accuracies": [1.0e-4],
            "compute_charge_gradients": [False],
            "skip_accuracy_for_large": {},
            "max_atoms": 1024,
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


class TestJaxMemoryContract:
    """Test JAX memory metadata behavior."""

    def test_measure_memory_jax_reports_nan_memory(self):
        """JAX memory is intentionally unavailable rather than reported as zero."""

        class FakeJax:
            @staticmethod
            def block_until_ready(result):
                """Return the already-computed fake result."""
                return result

        result, mem_info = measure_memory_jax(lambda: 7, FakeJax)

        assert result == 7
        assert math.isnan(mem_info["mem_delta_mb"])
        assert math.isnan(mem_info["mem_peak_gb"])


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
