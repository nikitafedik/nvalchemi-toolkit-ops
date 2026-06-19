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

"""
Generate plots from benchmark CSV files.

This script is run during the Sphinx documentation build to create
visualization plots from benchmark results.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_REPO_BENCHMARKS = str(REPO_ROOT / "benchmarks")
_BENCHMARKS_PACKAGE = sys.modules.get("benchmarks")
if _BENCHMARKS_PACKAGE is not None and hasattr(_BENCHMARKS_PACKAGE, "__path__"):
    package_path = _BENCHMARKS_PACKAGE.__path__
    if _REPO_BENCHMARKS not in package_path:
        package_path.append(_REPO_BENCHMARKS)

NL_DOC_CUTOFFS = (6.0, 15.0, 25.0)
NL_DOC_CUTOFF_COMBINATIONS = (
    (6.0, 15.0),
    (6.0, 25.0),
    (15.0, 25.0),
)
SUITE_RESULTS_ENV = "BENCHMARK_SUITE_RESULTS_DIR"


def _filter_successful_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return only successful benchmark rows when the CSV has status data."""
    if "success" not in df.columns:
        return df
    return df[df["success"].astype(str).str.lower() == "true"]


def plot_series(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    title: str | None = None,
    x_label: str = "Number of atoms",
    y_label: str = "Value",
    caption: str | None = None,
) -> None:
    """
    Plot multiple data series on a log-log scale.

    Parameters
    ----------
    series
        Dictionary mapping series labels to (x, y) tuples.
    output_path
        Path to save the plot.
    title
        Plot title.
    x_label
        X-axis label.
    y_label
        Y-axis label.
    caption
        Caption text below the plot.
    """
    num_series = len(series)

    # Determine figure size based on number of series (accommodate legend)
    fig_width = 10 if num_series > 3 else 8
    fig, ax = plt.subplots(figsize=(fig_width, 5.5), constrained_layout=True)

    # Use YlGn sequential colormap
    if num_series == 1:
        colors = ["#2E7D32"]  # Single dark green
    else:
        # Use YlGn colormap, avoiding very light colors
        cmap = plt.cm.YlGn
        colors = [cmap(0.3 + 0.7 * i / (num_series - 1)) for i in range(num_series)]

    for idx, (label, (xs, ys)) in enumerate(series.items()):
        if xs is None or ys is None:
            continue

        color = colors[idx]

        # matplotlib automatically skips nan values, creating gaps in lines
        ax.plot(
            xs,
            ys,
            marker="o",
            linestyle="-",
            linewidth=2.5,
            markersize=6.0,
            label=label,
            color=color,
            markeredgewidth=0.5,
            markeredgecolor="black",
            alpha=0.9,
        )

    # Axis labels and scales
    ax.set_xlabel(x_label, fontsize=14, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Ensure sufficient tick marks on both axes
    # Use LogLocator with numticks parameter for better control
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))

    # Add minor ticks for additional reference points
    ax.xaxis.set_minor_locator(
        ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20)
    )
    ax.yaxis.set_minor_locator(
        ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20)
    )

    # Enhance tick labels
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=10)

    # Title with proper spacing
    if title is not None:
        ax.set_title(title, fontsize=16, fontweight="bold", pad=15)

    # Refined grid
    ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.3, color="gray")
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.2, color="gray")

    # Legend placement: outside plot area to avoid overlap
    if num_series <= 4:
        # Few series: place inside upper left
        ax.legend(
            frameon=False,
            fontsize=12,
            loc="upper left",
            framealpha=0.95,
            edgecolor="gray",
            fancybox=False,
        )
    else:
        # Many series: place outside to the right
        ax.legend(
            frameon=False,
            fontsize=11,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            framealpha=0.95,
            edgecolor="gray",
            fancybox=False,
        )

    # Caption if provided
    if caption is not None:
        fig.text(
            0.5,
            0.02,
            caption,
            wrap=True,
            horizontalalignment="center",
            fontsize=11,
            style="italic",
        )

    plt.savefig(output_path.as_posix(), dpi=300, bbox_inches="tight")
    plt.close()


def plot_throughput(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    title: str | None = None,
    caption: str | None = None,
) -> None:
    """
    Plot throughput (atoms/s) vs system size.

    Parameters
    ----------
    series
        Dictionary mapping series labels to (total_atoms, median_time_ms) tuples.
    output_path
        Path to save the plot.
    title
        Plot title.
    caption
        Caption text below the plot.
    """
    # Convert time series to throughput
    throughput_series = {}
    for label, (atoms, times_ms) in series.items():
        if atoms is None or times_ms is None:
            continue
        # Division with nan propagates nan, which matplotlib will skip.
        throughput = atoms / times_ms * 1000.0
        throughput_series[label] = (atoms, throughput)

    plot_series(
        throughput_series,
        output_path,
        title=title,
        x_label="Number of atoms",
        y_label="Throughput (atoms/s)",
        caption=caption,
    )


def plot_memory(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    title: str | None = None,
    caption: str | None = None,
) -> None:
    """
    Plot memory utilization vs system size.

    Parameters
    ----------
    series
        Dictionary mapping series labels to (total_atoms, peak_memory_mb) tuples.
    output_path
        Path to save the plot.
    title
        Plot title.
    caption
        Caption text below the plot.
    """
    plot_series(
        series,
        output_path,
        title=title,
        x_label="Number of atoms",
        y_label="Peak memory (MB)",
        caption=caption,
    )


def load_dynamics_csv(filepath: Path) -> pd.DataFrame:
    """
    Load dynamics benchmark results from CSV file.

    Parameters
    ----------
    filepath
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with dynamics benchmark data.
        Detects single-system vs batched based on presence of batch_size column.
    """
    df = pd.read_csv(filepath)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return _filter_successful_rows(df)


def _parse_dynamics_filename(filename: str) -> dict[str, str]:
    """
    Parse dynamics benchmark filename.

    Expected format: dynamics_{md|opt}_{single|batch}_{backend}_{gpu_sku}.csv

    Parameters
    ----------
    filename
        CSV filename.

    Returns
    -------
    dict
        Dictionary with keys: benchmark_type, system_type, backend, gpu_sku
    """
    parts = filename.replace(".csv", "").split("_")
    if len(parts) < 5 or parts[0] != "dynamics":
        return {}

    return {
        "benchmark_type": parts[1],  # md or opt
        "system_type": parts[2],  # single or batch
        "backend": parts[3],  # nvalchemiops, ase, torchsim
        "gpu_sku": "_".join(parts[4:]),  # rest is GPU SKU
    }


def generate_dynamics_plots(results_dir: Path, output_dir: Path) -> None:
    """
    Generate plots for dynamics benchmarks.

    Creates plots for:
    - Single-system MD benchmarks
    - Single-system optimization benchmarks
    - Batched MD benchmarks
    - Batched optimization benchmarks

    Parameters
    ----------
    results_dir
        Directory containing benchmark CSV files.
    output_dir
        Directory to save plots.
    """
    print("\nGenerating dynamics benchmark plots...")

    dynamics_files = list(results_dir.glob("dynamics_*.csv"))
    if not dynamics_files:
        print("  No dynamics benchmark results found")
        return

    files_by_category = {}
    for filepath in dynamics_files:
        info = _parse_dynamics_filename(filepath.name)
        if not info:
            continue

        category = f"{info['benchmark_type']}_{info['system_type']}"
        files_by_category.setdefault(category, {})

        backend = info["backend"]
        files_by_category[category].setdefault(backend, {})
        files_by_category[category][backend] = {
            "path": filepath,
            "gpu_sku": info["gpu_sku"],
        }

    for category, backends in files_by_category.items():
        benchmark_type, system_type = category.split("_")
        print(f"\n  Processing {benchmark_type.upper()} {system_type} benchmarks...")

        is_batched = system_type == "batch"
        all_data = {}
        gpu_sku = "unknown"
        for backend, file_info in backends.items():
            df = load_dynamics_csv(file_info["path"])
            all_data[backend] = df
            gpu_sku = file_info["gpu_sku"]

        if len(all_data) > 1:
            print("    Creating comparison plots...")
            _generate_dynamics_comparison_plots(
                all_data, benchmark_type, system_type, is_batched, gpu_sku, output_dir
            )

        for backend, df in all_data.items():
            print(f"    Creating {backend} detail plots...")
            _generate_dynamics_backend_plots(
                df,
                backend,
                benchmark_type,
                system_type,
                is_batched,
                gpu_sku,
                output_dir,
            )


def _generate_dynamics_comparison_plots(
    data_by_backend: dict[str, pd.DataFrame],
    benchmark_type: str,
    system_type: str,
    is_batched: bool,
    gpu_sku: str,
    output_dir: Path,
) -> None:
    """Generate comparison plots across backends."""
    # Scaling plot: num_atoms vs avg_step_time_ms
    series = {}
    for backend, df in data_by_backend.items():
        if is_batched:
            # For batched, average across batch sizes for each num_atoms
            grouped = df.groupby("num_atoms")["avg_step_time_ms"].mean()
            series[backend] = (grouped.index.values, grouped.values)
        else:
            # For single-system, average across methods for each num_atoms
            grouped = df.groupby("num_atoms")["avg_step_time_ms"].mean()
            series[backend] = (grouped.index.values, grouped.values)

    output_path = (
        output_dir
        / f"dynamics_{benchmark_type}_{system_type}_scaling_comparison_{gpu_sku}.png"
    )
    plot_series(
        series,
        output_path,
        title=f"{benchmark_type.upper()} {system_type.title()} Scaling Comparison",
        x_label="Number of atoms",
        y_label="Avg step time (ms)",
    )
    print(f"      Generated: {output_path.name}")

    # Throughput plot: num_atoms vs throughput_atom_steps_per_s
    series = {}
    for backend, df in data_by_backend.items():
        if is_batched:
            grouped = df.groupby("num_atoms")["throughput_atom_steps_per_s"].mean()
            series[backend] = (grouped.index.values, grouped.values)
        else:
            grouped = df.groupby("num_atoms")["throughput_atom_steps_per_s"].mean()
            series[backend] = (grouped.index.values, grouped.values)

    output_path = (
        output_dir
        / f"dynamics_{benchmark_type}_{system_type}_throughput_comparison_{gpu_sku}.png"
    )
    plot_series(
        series,
        output_path,
        title=f"{benchmark_type.upper()} {system_type.title()} Throughput Comparison",
        x_label="Number of atoms",
        y_label="Atom-steps/s",
    )
    print(f"      Generated: {output_path.name}")

    # For batched: batch scaling plot
    if is_batched and "batch_throughput_system_steps_per_s" in df.columns:
        series = {}
        for backend, df in data_by_backend.items():
            # Average across num_atoms for each batch_size
            grouped = df.groupby("batch_size")[
                "batch_throughput_system_steps_per_s"
            ].mean()
            series[backend] = (grouped.index.values, grouped.values)

        output_path = (
            output_dir
            / f"dynamics_{benchmark_type}_{system_type}_batch_scaling_comparison_{gpu_sku}.png"
        )
        plot_series(
            series,
            output_path,
            title=f"{benchmark_type.upper()} Batch Scaling Comparison",
            x_label="Batch size",
            y_label="System-steps/s",
        )
        print(f"      Generated: {output_path.name}")


def _generate_dynamics_backend_plots(
    df: pd.DataFrame,
    backend: str,
    benchmark_type: str,
    system_type: str,
    is_batched: bool,
    gpu_sku: str,
    output_dir: Path,
) -> None:
    """Generate per-backend detail plots."""
    if is_batched:
        # For batched data, use total_atoms as x-axis.
        # Choose series grouping: model_type if multiple, otherwise method.
        model_types = df["model_type"].dropna().replace("", pd.NA).dropna().unique()
        if len(model_types) > 1:
            group_col = "model_type"
        else:
            group_col = "method"
        group_vals = df[group_col].unique()
        x_col = "total_atoms"
        x_label = "Total atoms (num_atoms × batch_size)"
    else:
        group_col = "method"
        group_vals = df[group_col].unique()
        x_col = "num_atoms"
        x_label = "Number of atoms"

    # Scaling plot
    series = {}
    for val in group_vals:
        df_sub = df[df[group_col] == val]
        grouped = df_sub.groupby(x_col)["avg_step_time_ms"].mean()
        series[val] = (grouped.index.values, grouped.values)

    if series:
        output_path = (
            output_dir
            / f"dynamics_{benchmark_type}_{system_type}_{backend}_scaling_{gpu_sku}.png"
        )
        plot_series(
            series,
            output_path,
            title=f"{benchmark_type.upper()} {system_type.title()} Scaling ({backend})",
            x_label=x_label,
            y_label="Avg step time (ms)",
        )
        print(f"      Generated: {output_path.name}")

    # Throughput plot
    series = {}
    for val in group_vals:
        df_sub = df[df[group_col] == val]
        grouped = df_sub.groupby(x_col)["throughput_atom_steps_per_s"].mean()
        series[val] = (grouped.index.values, grouped.values)

    if series:
        output_path = (
            output_dir
            / f"dynamics_{benchmark_type}_{system_type}_{backend}_throughput_{gpu_sku}.png"
        )
        plot_series(
            series,
            output_path,
            title=f"{benchmark_type.upper()} {system_type.title()} Throughput ({backend})",
            x_label=x_label,
            y_label="Atom-steps/s",
        )
        print(f"      Generated: {output_path.name}")


def _suite_csv_dirs(results_dir: Path) -> list[Path]:
    """Return suite CSV search directories, honoring the docs-build override."""
    dirs = [results_dir]
    override = os.getenv(SUITE_RESULTS_ENV)
    if override:
        override_dir = Path(override).expanduser().resolve()
        dirs.insert(0, override_dir)
    return dirs


def _write_no_data_placeholder(output_path: Path, title: str, details: str) -> None:
    """Write an explicit placeholder for selector views with no successful rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    ax.axis("off")
    ax.text(
        0.5,
        0.58,
        title,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.42,
        details,
        ha="center",
        va="center",
        fontsize=11,
        wrap=True,
    )
    fig.savefig(output_path.as_posix(), dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_suite_csv_plots(results_dir: Path, output_dir: Path) -> None:
    """Generate docs panels from the unified suite's standardized CSV names."""
    try:
        from benchmarks.plotting.plot_benchmarks import (
            plot_comparison_panel,
            plot_single_panel,
        )
    except ImportError as exc:
        print(f"Skipping suite CSV plots: {exc}")
        return

    seen: set[str] = set()
    csv_files = []
    for csv_dir in _suite_csv_dirs(results_dir):
        for pattern in ("nl-*.csv", "d3-*.csv", "el-*.csv"):
            for path in sorted(csv_dir.glob(pattern)):
                if path.name in seen:
                    continue
                seen.add(path.name)
                csv_files.append(path)

    if not csv_files:
        print("No unified suite CSV files found")
        return

    nl_cutoff_views = [(cutoff,) for cutoff in NL_DOC_CUTOFFS]
    nl_cutoff_views.extend(NL_DOC_CUTOFF_COMBINATIONS)
    print(f"\nGenerating unified suite plots ({len(csv_files)} CSVs)...")
    for csv_file in csv_files:
        # Backend-comparison files (merged torch+jax) are rendered by
        # generate_nl_backend_comparison_plots instead. plot_single_panel
        # filters to torch when multiple backends are present, which would
        # silently drop their JAX rows here.
        if csv_file.name.startswith("nl-backend-"):
            continue
        for panel in ("time", "throughput", "memory"):
            output_path = output_dir / f"{csv_file.stem}-{panel}.png"
            if plot_single_panel(csv_file, panel, output_path):
                print(f"      Generated: {output_path.name}")
            else:
                print(f"      Skipped: {csv_file.name} ({panel}, no data)")
            if csv_file.name.startswith("nl-"):
                if panel in {"time", "throughput"}:
                    jax_output = output_dir / f"{csv_file.stem}-jax-{panel}.png"
                    if plot_single_panel(
                        csv_file,
                        panel,
                        jax_output,
                        filters={"backend": "jax"},
                        title_suffix="JAX",
                    ):
                        print(f"      Generated: {jax_output.name}")
                    else:
                        print(f"      Skipped: {csv_file.name} ({panel}, jax, no data)")
                for cutoffs in nl_cutoff_views:
                    cutoff_label = "-".join(f"{cutoff:g}A" for cutoff in cutoffs)
                    cutoff_output = (
                        output_dir
                        / f"{csv_file.stem}-cutoff-{cutoff_label}-{panel}.png"
                    )
                    filters = {"cutoff": cutoffs[0] if len(cutoffs) == 1 else cutoffs}
                    title_suffix = (
                        f"{cutoffs[0]:g}A cutoff"
                        if len(cutoffs) == 1
                        else f"{', '.join(f'{cutoff:g}A' for cutoff in cutoffs)} cutoffs"
                    )
                    if plot_single_panel(
                        csv_file,
                        panel,
                        cutoff_output,
                        filters=filters,
                        title_suffix=title_suffix,
                    ):
                        print(f"      Generated: {cutoff_output.name}")
                    else:
                        _write_no_data_placeholder(
                            cutoff_output,
                            "No successful benchmark rows",
                            (
                                f"{csv_file.stem}, {panel}, {title_suffix}. "
                                "See the CSV error_type column for failed rows."
                            ),
                        )
                        print(
                            f"      Placeholder: {csv_file.name} "
                            f"({panel}, cutoff={cutoff_label}, no successful data)"
                        )
                    if panel not in {"time", "throughput"}:
                        continue
                    jax_cutoff_output = (
                        output_dir
                        / f"{csv_file.stem}-cutoff-{cutoff_label}-jax-{panel}.png"
                    )
                    jax_filters = {
                        "backend": "jax",
                        "cutoff": cutoffs[0] if len(cutoffs) == 1 else cutoffs,
                    }
                    if plot_single_panel(
                        csv_file,
                        panel,
                        jax_cutoff_output,
                        filters=jax_filters,
                        title_suffix=f"JAX, {title_suffix}",
                    ):
                        print(f"      Generated: {jax_cutoff_output.name}")
                    else:
                        _write_no_data_placeholder(
                            jax_cutoff_output,
                            "No successful benchmark rows",
                            (
                                f"{csv_file.stem}, {panel}, JAX, {title_suffix}. "
                                "See the CSV error_type column for failed rows."
                            ),
                        )
                        print(
                            f"      Placeholder: {csv_file.name} "
                            f"({panel}, jax, cutoff={cutoff_label}, no successful data)"
                        )
        if csv_file.name.startswith(("d3-", "el-")):
            module = csv_file.name.split("-", 1)[0]
            for panel in ("time", "throughput", "memory"):
                output_path = (
                    output_dir
                    / f"{csv_file.stem.replace('-scaling', f'-comparison-{panel}')}.png"
                )
                plot_comparison_panel(csv_file, panel, output_path, module)


NL_BACKEND_COMPARISON_CUTOFF = 15.0
NL_BACKEND_COMPARISON_METHODS = {
    "system_size": (
        "naive_scalar",
        "naive_tile",
        "cell_list_atom_centric",
    ),
    "batch": (
        "batch_naive_scalar",
        "batch_naive_tile",
        "batch_cell_list_atom_centric",
    ),
}


def generate_nl_backend_comparison_plots(results_dir: Path, output_dir: Path) -> None:
    """Generate Torch-vs-JAX comparison panels for the public NL APIs.

    Prefer the unified suite ``nl-*.csv`` files so the comparison stays tied to
    the same granular method rows as the selector plots. Separate
    ``nl-backend-*.csv`` files are intentionally ignored so earlier comparison
    rows cannot masquerade as current backend-comparison evidence.
    Only time and throughput are emitted: JAX's XLA pre-allocation pins peak
    memory near the device capacity, so memory is not comparable across
    backends. Single-system atom-centric cell-list rows are included because
    the benchmark forces JAX's direct atom-centric path to match Torch/Warp's
    direct query boundary. Batched atom-centric cell-list rows are included for
    the same reason: the benchmark forces JAX's direct batched atom-centric
    path to match Torch/Warp's direct query boundary. Pair-centric JAX is eager
    / host-sized, so pair-centric rows remain coverage-only in the full suite
    CSVs and selector plots. A single representative cutoff keeps each panel
    readable.
    """
    seen: set[str] = set()
    csv_files: list[Path] = []
    for csv_dir in _suite_csv_dirs(results_dir):
        for path in sorted(csv_dir.glob("nl-*.csv")):
            if path.name.startswith("nl-backend-"):
                continue
            if path.name not in seen:
                seen.add(path.name)
                csv_files.append(path)

    if not csv_files:
        print("No NL backend-comparison CSV files found")
        return

    print(f"\nGenerating NL backend-comparison plots ({len(csv_files)} CSVs)...")

    def render_backend_csv(csv_file: Path) -> bool:
        df = pd.read_csv(csv_file)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = _filter_successful_rows(df)
        if "cutoff" in df.columns:
            cutoff_col = df["cutoff"].astype(float)
            df = df[np.isclose(cutoff_col, NL_BACKEND_COMPARISON_CUTOFF)]
        if df.empty:
            print(f"      Skipped: {csv_file.name} (no rows at representative cutoff)")
            return False

        system = str(df["system"].iloc[0]) if "system" in df.columns else "unknown"
        stem = csv_file.stem
        if "constant-workload" in stem:
            mode = "constant_workload"
        elif "batch-scaling" in stem:
            mode = "batch_scaling"
        else:
            mode = "system_size"
        x_field = (
            "batch_size"
            if mode in {"batch_scaling", "constant_workload"}
            else "total_atoms"
        )
        x_label = (
            "Batch size"
            if mode in {"batch_scaling", "constant_workload"}
            else "Number of atoms"
        )

        time_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        throughput_series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if mode == "system_size":
            method_order = NL_BACKEND_COMPARISON_METHODS["system_size"]
        else:
            method_order = NL_BACKEND_COMPARISON_METHODS["batch"]
        if {"backend_comparable", "timing_scope"}.issubset(df.columns):
            comparable = df["backend_comparable"]
            if comparable.dtype != bool:
                comparable = comparable.astype(str).str.lower().isin({"true", "1"})
            df = df[
                comparable & (df["timing_scope"].astype(str) == "backend_comparison")
            ]
        if "timing_method" in df.columns:
            df = df[df["timing_method"].astype(str) != "jax_wall_block_each"]
        for method in method_order:
            method_df = df[df["method"] == method]
            method_backends = {
                str(backend) for backend in method_df["backend"].dropna().unique()
            }
            if not {"torch", "jax"}.issubset(method_backends):
                continue
            torch_x = set(method_df.loc[method_df["backend"] == "torch", x_field])
            jax_x = set(method_df.loc[method_df["backend"] == "jax", x_field])
            shared_x = torch_x & jax_x
            if not shared_x:
                continue
            for backend in ("torch", "jax"):
                sub = method_df[
                    (method_df["backend"] == backend)
                    & (method_df[x_field].isin(shared_x))
                ]
                if sub.empty:
                    continue
                sub = sub.sort_values(x_field)
                xs = sub[x_field].to_numpy(dtype=float)
                label = f"{method.replace('_', ' ')} ({backend})"
                time_series[label] = (xs, sub["time_us_per_atom"].to_numpy(dtype=float))
                throughput_series[label] = (
                    xs,
                    sub["throughput_atoms_per_sec"].to_numpy(dtype=float),
                )

        if not time_series:
            print(f"      Skipped: {csv_file.name} (no torch/jax series)")
            return False

        mode_label = (mode or "system_size").replace("_", " ")
        title = (
            f"NL {system.upper()} {mode_label} - Torch vs JAX "
            f"({NL_BACKEND_COMPARISON_CUTOFF:g} A cutoff)"
        )
        out_stem = stem if stem.startswith("nl-backend-") else f"nl-backend-{stem[3:]}"
        time_out = output_dir / f"{out_stem}-time.png"
        plot_series(
            time_series,
            time_out,
            title=title,
            x_label=x_label,
            y_label="Time per atom (us)",
        )
        print(f"      Generated: {time_out.name}")

        throughput_out = output_dir / f"{out_stem}-throughput.png"
        plot_series(
            throughput_series,
            throughput_out,
            title=title,
            x_label=x_label,
            y_label="Throughput (atoms/s)",
        )
        print(f"      Generated: {throughput_out.name}")
        return True

    for csv_file in csv_files:
        render_backend_csv(csv_file)


def main() -> None:
    """Generate all plots from benchmark results."""
    print("Generating benchmark plots...")

    # Determine paths relative to this script
    results_dir = Path(__file__).parent / "benchmark_results"
    output_dir = Path(__file__).parent / "_static"

    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Generate plots for each benchmark type
    generate_suite_csv_plots(results_dir, output_dir)
    generate_nl_backend_comparison_plots(results_dir, output_dir)
    generate_dynamics_plots(results_dir, output_dir)

    print("\nPlot generation complete!")


if __name__ == "__main__":
    main()
