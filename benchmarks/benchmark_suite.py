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

"""Unified Benchmark Suite for NVIDIA ALCHEMI Toolkit-Ops.

Loads per-module YAML configs and dispatches benchmarks in-process.
CLI flags override YAML values. Each sub-benchmark can also run standalone.
Plots are generated automatically unless --no-plot is specified.

Usage:
    python benchmark_suite.py --benchmark all
    python benchmark_suite.py --benchmark nl --system cscl --mode system_size
    python benchmark_suite.py --benchmark d3 el --system nh3
    python benchmark_suite.py --no-plot --benchmark nl
    python benchmark_suite.py --plot-only benchmarks/benchmark-results/run_2026-02-17/
"""

import argparse
import importlib
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

from benchmarks.config import (
    add_common_cli_args,
    load_yaml_config,
    merge_common_cli_overrides,
)
from benchmarks.utils import (
    create_run_directory,
    write_run_log,
)

SCRIPT_DIR = Path(__file__).parent

# Per-module config + runner module mapping. ``label`` is the pretty
# suffix used in the results summary; ``module`` is the import path to
# the runner's ``run_from_config`` entry point.
RUNNERS = {
    "nl": {
        "label": "NL",
        "config": SCRIPT_DIR / "neighborlist" / "benchmark_config.yaml",
        "module": "benchmarks.neighborlist.benchmark_neighborlist",
    },
    "d3": {
        "label": "D3",
        "config": SCRIPT_DIR / "interactions" / "dispersion" / "benchmark_config.yaml",
        "module": "benchmarks.interactions.dispersion.benchmark_dftd3",
    },
    "el": {
        "label": "EL",
        "config": SCRIPT_DIR
        / "interactions"
        / "electrostatics"
        / "benchmark_config.yaml",
        "module": "benchmarks.interactions.electrostatics.benchmark_electrostatics",
    },
    "dyn": {
        "label": "DYN",
        "config": SCRIPT_DIR / "dynamics" / "benchmark_config.yaml",
        "module": "benchmarks.dynamics.benchmark_dynamics",
    },
}

SUPPORTED_BACKENDS = {
    "nl": {"torch", "jax", "warp"},
    "d3": {"torch", "jax"},
    "el": {"torch", "jax"},
    "dyn": {"torch"},
}


def validate_backend_selection(backend: str | None, benchmarks: set[str]) -> None:
    """Validate suite-level backend compatibility for the selected benchmarks."""
    if backend is None:
        return
    unsupported = sorted(
        key for key in benchmarks if backend not in SUPPORTED_BACKENDS.get(key, set())
    )
    if unsupported:
        raise ValueError(
            f"Backend {backend!r} is not supported for requested benchmark(s): "
            f"{', '.join(unsupported)}. Supported backends: "
            + ", ".join(
                f"{key}={sorted(SUPPORTED_BACKENDS[key])}" for key in sorted(unsupported)
            )
        )


def _count_successful_rows(results: list[dict]) -> int:
    """Count rows that represent successful benchmark measurements."""
    return sum(1 for row in results if row.get("success", True) is not False)


def _labels_with_no_rows(summary: dict[str, int]) -> list[str]:
    """Return benchmark labels that produced no rows."""
    return sorted(label for label, count in summary.items() if count <= 0)


def parse_args():
    """Parse command-line arguments for the benchmark suite."""
    parser = argparse.ArgumentParser(
        description="Unified Benchmark Suite (NL + D3 + Electrostatics + Dynamics)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmark_suite.py --benchmark all
    python benchmark_suite.py --benchmark d3 --system cscl --mode system_size
    python benchmark_suite.py --benchmark all --timing-runs 50
    python benchmark_suite.py --benchmark nl --no-plot
    python benchmark_suite.py --plot-only benchmark-results/run_2026-02-17/

Benchmark aliases:
    nl      Neighbor List
    d3      DFT-D3 Dispersion
    el      Electrostatics (Ewald + PME)
    dyn     Dynamics and optimization
    all     All benchmarks

Each module reads its own benchmark_config.yaml. Global CLI flags override
YAML values across all modules. Run individual benchmarks standalone for
module-specific CLI options.
        """,
    )
    parser.add_argument(
        "--benchmark",
        "-b",
        nargs="+",
        default=["all"],
        choices=["nl", "d3", "el", "dyn", "all"],
        help="Benchmarks to run",
    )
    add_common_cli_args(parser)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting after benchmarks",
    )
    parser.add_argument(
        "--plot-only",
        type=Path,
        default=None,
        metavar="RESULTS_DIR",
        help="Skip benchmarks, only generate plots from existing results directory",
    )
    parser.add_argument(
        "--cutoffs",
        "-c",
        type=float,
        nargs="+",
        default=None,
        help="Override cutoff radii for NL/D3 benchmarks",
    )
    parser.add_argument(
        "--accuracies",
        "-a",
        type=float,
        nargs="+",
        default=None,
        help="Override electrostatics target accuracies",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        default=["all"],
        choices=["all", "time", "throughput", "memory"],
        help="Plot panels to generate after benchmarks",
    )
    return parser.parse_args()


def main():
    """Run the unified benchmark suite."""
    args = parse_args()

    benchmarks = set(RUNNERS) if "all" in args.benchmark else set(args.benchmark)

    # JAX_ENABLE_X64 must be set BEFORE the first `import jax`. EL needs
    # f64 (PME/Ewald accuracy); NL/D3 are f32-safe but share the same
    # Python process in the suite, so JAX commits to whatever x64 was
    # when NL ran first. Set it unconditionally when any JAX benchmark
    # is queued so the env is consistent regardless of module order.
    if args.backend == "jax":
        os.environ.setdefault("JAX_ENABLE_X64", "1")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    if args.plot_only:
        return 0 if _generate_plots(args.plot_only, plots=args.plots) else 1

    try:
        validate_backend_selection(args.backend, benchmarks)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print("=" * 70)
    print("NVIDIA ALCHEMI Toolkit-Ops Benchmark Suite")
    print("=" * 70)
    try:
        gpu_name = torch.cuda.get_device_name(0)
    except (AssertionError, RuntimeError):
        gpu_name = "N/A (no CUDA)"
    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Benchmarks: {', '.join(sorted(benchmarks))}")
    print(f"Systems: {args.system or 'all (from YAML)'}")
    print(f"Modes: {args.mode or 'all (from YAML)'}")
    print("=" * 70)

    start_time = datetime.now()

    run_dir = None
    if not args.dry_run:
        base_dir = args.output_dir or (SCRIPT_DIR / "benchmark-results")
        run_dir = create_run_directory(base_dir, prefix="run")
        print(f"\nOutput: {run_dir}")

    results_summary = {}
    success_summary = {}
    for key in ("nl", "d3", "el", "dyn"):
        if key not in benchmarks:
            continue
        info = RUNNERS[key]
        config_path = info["config"]
        if not config_path.exists():
            print(
                f"\nWARNING: {info['label']} config not found at {config_path}, skipping"
            )
            results_summary[info["label"]] = 0
            if not args.dry_run:
                success_summary[info["label"]] = 0
            continue
        runner = importlib.import_module(info["module"])
        config = load_yaml_config(config_path)
        if hasattr(runner, "merge_cli_overrides"):
            config = runner.merge_cli_overrides(config, args)
        else:
            config = merge_common_cli_overrides(config, args)
        if args.dry_run:
            results = runner.dry_run_from_config(config)
        else:
            results = runner.run_from_config(config, output_dir=run_dir)
        results_summary[info["label"]] = len(results)
        if not args.dry_run:
            success_summary[info["label"]] = _count_successful_rows(results)

    if args.dry_run:
        total = sum(results_summary.values())
        print(f"\nDRY RUN COMPLETE: {total} planned row(s)")
        empty = _labels_with_no_rows(results_summary)
        if empty:
            print(
                "ERROR: no planned rows for requested benchmark(s): "
                + ", ".join(empty),
                file=sys.stderr,
            )
            return 1
        return 0 if total > 0 else 1

    end_time = datetime.now()
    extra = {
        "Benchmarks run": ", ".join(sorted(benchmarks)),
        "Systems": str(args.system or "all"),
        "Modes": str(args.mode or "all"),
    }
    for name, count in results_summary.items():
        extra[f"{name} results"] = count
    for name, count in success_summary.items():
        extra[f"{name} successful results"] = count
    write_run_log(run_dir, start_time, end_time, extra_info=extra)

    # --- Plotting ---
    if not args.no_plot:
        plot_ok = _generate_plots(run_dir, plots=args.plots)
    else:
        plot_ok = True

    # Summary
    print(f"\n{'=' * 70}")
    print("BENCHMARK SUITE COMPLETE")
    for name, count in results_summary.items():
        successes = success_summary.get(name)
        if successes is None:
            print(f"  {name}: {count} results")
        else:
            print(f"  {name}: {successes}/{count} successful results")
    total = sum(results_summary.values())
    successful_total = sum(success_summary.values())
    print(f"  Total: {total} results")
    print(f"  Successful: {successful_total} results")
    print(f"  Output: {run_dir}")
    print(f"  Run log: {run_dir / 'RUN_LOG.md'}")
    print("=" * 70)

    if total <= 0:
        return 1
    empty = _labels_with_no_rows(results_summary)
    if empty:
        print(
            "ERROR: no result rows for requested benchmark(s): " + ", ".join(empty),
            file=sys.stderr,
        )
        return 1
    failed = _labels_with_no_rows(success_summary)
    if failed:
        print(
            "ERROR: no successful rows for requested benchmark(s): "
            + ", ".join(failed),
            file=sys.stderr,
        )
        return 1
    if successful_total <= 0:
        print("ERROR: no successful benchmark rows were produced", file=sys.stderr)
        return 1
    return 0 if plot_ok else 1


def _generate_plots(results_dir, plots=None):
    """Generate 3-panel review plots and single-panel docs plots from CSVs.

    Returns ``False`` only when CSVs existed and every attempted plot failed.
    """
    try:
        from benchmarks.plotting.plot_benchmarks import (
            detect_and_plot,
            plot_single_panel,
        )
    except ImportError as e:
        print(
            f"\n  WARNING: plotting unavailable ({e}). Results saved to {results_dir}"
        )
        return True

    results_dir = Path(results_dir)
    csvs = sorted(results_dir.glob("*.csv"))
    if not csvs:
        print("\n  No CSVs found, skipping plots")
        return True

    requested = set(plots or ["all"])
    all_panels = "all" in requested
    panels = ("time", "throughput", "memory") if all_panels else tuple(plots)
    plot_errors = []
    attempted = 0
    succeeded = 0

    print(f"\n{'=' * 70}")
    print(f"GENERATING PLOTS ({len(csvs)} CSVs)")
    print("=" * 70)

    # 3-panel review plots
    three_panel_dir = results_dir
    if all_panels:
        for csv in csvs:
            attempted += 1
            try:
                detect_and_plot(csv, three_panel_dir)
                succeeded += 1
            except Exception as e:
                msg = f"3-panel {csv.name}: {e}"
                plot_errors.append(msg)
                print(f"  ERROR ({msg})")

    # Single-panel plots for docs
    single_dir = results_dir / "single-panels"
    single_dir.mkdir(exist_ok=True)
    for csv in csvs:
        for panel in panels:
            attempted += 1
            try:
                out = single_dir / f"{csv.stem}-{panel}.png"
                plot_single_panel(csv, panel, out)
                succeeded += 1
            except Exception as e:
                msg = f"single {csv.stem}-{panel}: {e}"
                plot_errors.append(msg)
                print(f"  ERROR ({msg})")

    print(f"  3-panel plots: {three_panel_dir}")
    print(f"  Single-panel plots: {single_dir}")
    if plot_errors:
        print(f"\n{len(plot_errors)} plot(s) failed:")
        for err in plot_errors:
            print(f"  - {err}")
    return not (attempted > 0 and succeeded == 0)


if __name__ == "__main__":
    sys.exit(main())
