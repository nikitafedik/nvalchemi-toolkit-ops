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

"""Generate styled benchmark comparison figures for two suite result directories.

The expected inputs are CSV files produced by :mod:`benchmarks.benchmark_suite`.
Rows are joined by workload identity and then plotted as raw timing overlays and
``baseline / candidate`` speedup curves.

Neighbor-list comparisons intentionally pair the 0.3.1 cell-list path with the
0.4 cluster-tile path. D3 and electrostatics compare the same public method name
across versions. Multipole electrostatics rows are ignored by construction.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.plotting.styles import (
    ACCURACY_COLORS,
    CUTOFF_COLORS,
    D3_CUTOFF_COLORS,
    GRAY,
    NVIDIA_GREEN,
    SINGLE_PANEL_SIZE,
    format_accuracy,
    setup_log2_xaxis,
    setup_plot_style,
)

__all__ = [
    "build_comparisons",
    "main",
    "parse_args",
    "write_comparison_figures",
]

MODULES = ("nl", "d3", "el")
VERSION_BASELINE = "0.3.1"
VERSION_CANDIDATE = "0.4"
SUMMARY_FILENAME = "comparison_summary.csv"


def _csv_module(path: Path) -> str | None:
    """Infer the benchmark module from a suite CSV filename."""
    stem = path.stem
    for module in MODULES:
        if stem == module or stem.startswith(f"{module}-"):
            return module
    return None


def _convert_cell(value: str) -> Any:
    """Convert one CSV value to bool/int/float when possible."""
    if value in {"True", "False"}:
        return value == "True"
    if value == "":
        return value
    try:
        lowered = value.lower()
    except AttributeError:
        return value
    if lowered in {"nan", "inf", "+inf", "-inf"}:
        return float(value)
    try:
        if "." in value or "e" in lowered:
            return float(value)
        if value.lstrip("-").isdigit():
            return int(value)
    except ValueError:
        return value
    return value


def _load_rows(results_dir: Path) -> list[dict[str, Any]]:
    """Load successful suite CSV rows under ``results_dir``."""
    rows: list[dict[str, Any]] = []
    for csv_path in sorted(results_dir.rglob("*.csv")):
        module = _csv_module(csv_path)
        if module is None:
            continue
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                converted = {key: _convert_cell(value) for key, value in row.items()}
                if converted.get("success") is False:
                    continue
                converted["module"] = module
                converted["_source_csv"] = str(csv_path)
                rows.append(converted)
    return rows


def _identity(row: dict[str, Any], method: str) -> tuple[Any, ...]:
    """Build the comparison identity for one row and comparison method."""
    common = (
        row.get("system"),
        row.get("scaling_mode"),
        row.get("backend"),
        int(row.get("atoms_per_system", 0)),
        int(row.get("batch_size", 0)),
        int(row.get("total_atoms", 0)),
    )
    module = row["module"]
    if module in {"nl", "d3"}:
        return common + (method, float(row.get("cutoff", math.nan)))
    if module == "el":
        return common + (method, float(row.get("accuracy", math.nan)))
    raise ValueError(f"Unsupported module: {module}")


def _aggregate(rows: list[dict[str, Any]], method_for_row) -> dict[tuple[Any, ...], dict]:
    """Aggregate duplicate rows by identity using median timing."""
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        method = method_for_row(row)
        if method is None:
            continue
        buckets[_identity(row, method)].append(row)

    out = {}
    for key, grouped in buckets.items():
        representative = dict(grouped[0])
        for field in ("time_us_per_atom", "throughput_atoms_per_sec", "mem_delta_mb"):
            values = [
                float(row[field])
                for row in grouped
                if field in row and _is_finite_number(row[field])
            ]
            if values:
                representative[field] = statistics.median(values)
        out[key] = representative
    return out


def _is_finite_number(value: Any) -> bool:
    """Return whether ``value`` can be treated as a finite float."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _nl_comparison_method(row: dict[str, Any], version: str) -> str | None:
    """Map NL rows to the comparison method bucket."""
    method = str(row.get("method", ""))
    batch_size = int(row.get("batch_size", 1))
    if version == "baseline":
        # 0.3.1 module-level CSVs use the legacy label ``cell`` for both
        # single-system and batched cell-list rows. Keep the newer labels here
        # too so suite-produced baseline CSVs remain comparable.
        expected = {"cell", "cell_list"}
        if batch_size > 1:
            expected.add("batch_cell_list")
        return "cell_to_cluster_tile" if method in expected else None
    expected = "cluster_tile" if batch_size == 1 else "batch_cluster_tile"
    return "cell_to_cluster_tile" if method == expected else None


def _same_method(row: dict[str, Any]) -> str | None:
    """Return the row method for same-method comparisons."""
    method = str(row.get("method", ""))
    if method.startswith("multipole"):
        return None
    return method


def build_comparisons(
    baseline_dir: Path,
    candidate_dir: Path,
    modules: set[str],
) -> list[dict[str, Any]]:
    """Build joined comparison records for selected modules."""
    baseline_rows = [row for row in _load_rows(baseline_dir) if row["module"] in modules]
    candidate_rows = [row for row in _load_rows(candidate_dir) if row["module"] in modules]

    comparisons: list[dict[str, Any]] = []
    for module in MODULES:
        if module not in modules:
            continue
        base_module_rows = [row for row in baseline_rows if row["module"] == module]
        cand_module_rows = [row for row in candidate_rows if row["module"] == module]
        if module == "nl":
            baseline = _aggregate(
                base_module_rows, lambda row: _nl_comparison_method(row, "baseline")
            )
            candidate = _aggregate(
                cand_module_rows, lambda row: _nl_comparison_method(row, "candidate")
            )
        else:
            baseline = _aggregate(base_module_rows, _same_method)
            candidate = _aggregate(cand_module_rows, _same_method)

        for key in sorted(set(baseline) & set(candidate)):
            old = baseline[key]
            new = candidate[key]
            old_time = float(old["time_us_per_atom"])
            new_time = float(new["time_us_per_atom"])
            if old_time <= 0 or new_time <= 0:
                continue
            comparisons.append(
                {
                    "module": module,
                    "system": old["system"],
                    "scaling_mode": old["scaling_mode"],
                    "backend": old["backend"],
                    "atoms_per_system": int(old["atoms_per_system"]),
                    "batch_size": int(old["batch_size"]),
                    "total_atoms": int(old["total_atoms"]),
                    "method": key[6],
                    "cutoff": key[7] if module in {"nl", "d3"} else "",
                    "accuracy": key[7] if module == "el" else "",
                    "baseline_time_us_per_atom": old_time,
                    "candidate_time_us_per_atom": new_time,
                    "speedup": old_time / new_time,
                    "baseline_source_csv": old["_source_csv"],
                    "candidate_source_csv": new["_source_csv"],
                }
            )
    return comparisons


def _label_for(record: dict[str, Any]) -> str:
    """Build a compact line label for one comparison group."""
    mode = str(record["scaling_mode"]).replace("_", " ")
    backend = str(record["backend"])
    module = record["module"]
    if module == "el":
        method = str(record["method"]).replace("_cg", "+cg")
        return (
            f"{backend} {mode} {method} "
            f"{format_accuracy(float(record['accuracy']))}"
        )
    cutoff = int(float(record["cutoff"]))
    return f"{backend} {mode} {cutoff}A"


def _style_for(record: dict[str, Any]) -> dict[str, Any]:
    """Return matplotlib style kwargs for one comparison line."""
    module = record["module"]
    mode = str(record["scaling_mode"])
    linestyle = "-" if mode == "system_size" else (0, (4, 2))
    marker = "o" if mode == "system_size" else "s"
    if module == "d3":
        color = D3_CUTOFF_COLORS.get(float(record["cutoff"]), NVIDIA_GREEN)
    elif module == "nl":
        color = CUTOFF_COLORS.get(float(record["cutoff"]), NVIDIA_GREEN)
    else:
        color = ACCURACY_COLORS.get(float(record["accuracy"]), NVIDIA_GREEN)
        if str(record["method"]).startswith("ewald"):
            marker = "^"
    return {
        "color": color,
        "linestyle": linestyle,
        "marker": marker,
        "linewidth": 2,
        "markersize": 5,
        "alpha": 0.8,
    }


def _group_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group records into plot lines."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_label_for(record)].append(record)
    return grouped


def _plot_module_raw(module: str, records: list[dict[str, Any]], output_dir: Path) -> Path:
    """Plot raw baseline/candidate time overlays for one module."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=SINGLE_PANEL_SIZE)
    for label, grouped in sorted(_group_records(records).items()):
        grouped = sorted(grouped, key=lambda row: row["total_atoms"])
        x = [row["total_atoms"] for row in grouped]
        y_old = [row["baseline_time_us_per_atom"] for row in grouped]
        y_new = [row["candidate_time_us_per_atom"] for row in grouped]
        style = _style_for(grouped[0])
        ax.plot(x, y_old, label=f"{VERSION_BASELINE} {label}", color=GRAY, **{
            key: value for key, value in style.items() if key != "color"
        })
        ax.plot(x, y_new, label=f"{VERSION_CANDIDATE} {label}", **style)

    _finish_axes(ax, module, "Time per atom (us)", log_y=True)
    ax.legend(loc="best", frameon=True)
    path = output_dir / f"{module}_031_vs_04_raw_time.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_module_speedup(
    module: str, records: list[dict[str, Any]], output_dir: Path
) -> Path:
    """Plot baseline/candidate speedup curves for one module."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=SINGLE_PANEL_SIZE)
    for label, grouped in sorted(_group_records(records).items()):
        grouped = sorted(grouped, key=lambda row: row["total_atoms"])
        ax.plot(
            [row["total_atoms"] for row in grouped],
            [row["speedup"] for row in grouped],
            label=label,
            **_style_for(grouped[0]),
        )

    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2, alpha=0.7)
    _finish_axes(ax, module, f"{VERSION_BASELINE} / {VERSION_CANDIDATE} speedup")
    ax.legend(loc="best", frameon=True)
    if module == "nl":
        filename = "nl_031_vs_04_cluster_tile_speedup.png"
    else:
        filename = f"{module}_031_vs_04_speedup.png"
    path = output_dir / filename
    fig.savefig(path)
    plt.close(fig)
    return path


def _finish_axes(ax, module: str, ylabel: str, *, log_y: bool = False) -> None:
    """Apply common axis styling."""
    module_label = {
        "nl": "Neighbor list",
        "d3": "DFT-D3",
        "el": "Electrostatics",
    }[module]
    ax.set_title(f"{module_label}: {VERSION_BASELINE} vs {VERSION_CANDIDATE}")
    ax.set_ylabel(ylabel)
    setup_log2_xaxis(ax, label="Total atoms")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.3)


def _plot_summary(comparisons: list[dict[str, Any]], output_dir: Path) -> Path:
    """Plot median speedup summary bars by module."""
    setup_plot_style()
    modules = [module for module in MODULES if any(r["module"] == module for r in comparisons)]
    medians = []
    labels = []
    for module in modules:
        values = [float(row["speedup"]) for row in comparisons if row["module"] == module]
        medians.append(statistics.median(values))
        labels.append(module.upper())

    fig, ax = plt.subplots(figsize=SINGLE_PANEL_SIZE)
    colors = [NVIDIA_GREEN, "#31688E", "#E67E22"][: len(labels)]
    ax.bar(labels, medians, color=colors, edgecolor="black", linewidth=0.8)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.set_title(f"Median speedup: {VERSION_BASELINE} vs {VERSION_CANDIDATE}")
    ax.set_ylabel(f"{VERSION_BASELINE} / {VERSION_CANDIDATE} speedup")
    for idx, value in enumerate(medians):
        ax.text(idx, value, f"{value:.2f}x", ha="center", va="bottom")
    path = output_dir / "summary_speedup_panel.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _write_summary(comparisons: list[dict[str, Any]], output_dir: Path) -> Path:
    """Write the joined comparison data used for plotting."""
    path = output_dir / SUMMARY_FILENAME
    if not comparisons:
        path.write_text("")
        return path
    fieldnames = list(comparisons[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparisons)
    return path


def write_comparison_figures(
    comparisons: list[dict[str, Any]],
    output_dir: Path,
    modules: set[str],
) -> list[Path]:
    """Write all requested comparison figures and return generated paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = [_write_summary(comparisons, output_dir)]
    for module in MODULES:
        if module not in modules:
            continue
        records = [row for row in comparisons if row["module"] == module]
        if not records:
            continue
        written.append(_plot_module_raw(module, records, output_dir))
        written.append(_plot_module_speedup(module, records, output_dir))
    if comparisons:
        written.append(_plot_summary(comparisons, output_dir))
    return written


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate 0.3.1 vs 0.4 benchmark comparison figures."
    )
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--modules",
        nargs="+",
        default=["all"],
        choices=[*MODULES, "all"],
        help="Benchmark modules to compare.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the comparison plotter."""
    args = parse_args()
    modules = set(MODULES) if "all" in args.modules else set(args.modules)
    comparisons = build_comparisons(args.baseline_dir, args.candidate_dir, modules)
    written = write_comparison_figures(comparisons, args.output_dir, modules)

    print(
        f"Compared {len(comparisons)} joined row(s) from "
        f"{args.baseline_dir} and {args.candidate_dir}"
    )
    if not comparisons:
        print("No comparable successful rows found.")
        return 1
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
