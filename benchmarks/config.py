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

"""Shared CLI argument definitions, YAML loader, and override merging.

Each per-module benchmark runner reuses these helpers so the common flags
(``--system``, ``--mode``, ``--timing-runs``, ``--warmup-runs``,
``--output-dir``, ``--backend``, ``--dry-run``, ``--max-total-atoms``)
stay in sync. Module-specific flags (``--cutoffs``, ``--accuracies``) are added
by each runner's own ``parse_args`` on top of :func:`add_common_cli_args`.
The shared method selector accepts both ``--method`` and the legacy
``--methods`` spelling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

__all__ = [
    "add_common_cli_args",
    "enabled_method_names",
    "load_yaml_config",
    "merge_common_cli_overrides",
    "normalize_method_name",
]

_METHOD_ALIASES = {
    "naive": "naive_neighbor_list",
    "batch_naive": "batch_naive_neighbor_list",
    "cell": "cell_list",
    "batch-cell-list": "batch_cell_list",
    "batch-naive": "batch_naive_neighbor_list",
    "cluster": "cluster_tile",
    "tile": "cluster_tile",
    "cluster-tile": "cluster_tile",
    "batch_cluster": "batch_cluster_tile",
    "batch-cluster": "batch_cluster_tile",
    "batch-cluster-tile": "batch_cluster_tile",
    "d3": "dftd3",
}


def normalize_method_name(method: str) -> str:
    """Return the canonical benchmark method name for a CLI/config token."""
    return _METHOD_ALIASES.get(method, method)


def enabled_method_names(config: dict) -> list[str]:
    """Return canonical names for enabled methods in a benchmark config."""
    selected = config.get("runtime", {}).get("selected_methods")
    if selected:
        return [normalize_method_name(m) for m in selected]
    return [
        normalize_method_name(m["name"])
        for m in config.get("methods", [])
        if m.get("enabled", True)
    ]


def load_yaml_config(config_path: str | Path) -> dict:
    """Load benchmark configuration from a YAML file.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def add_common_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register the flags shared across every benchmark runner and the suite.

    Does NOT add ``--config`` — runners declare that themselves
    (``required=True``) while the suite resolves per-module configs from its
    own ``RUNNERS`` map.
    """
    parser.add_argument(
        "--system",
        "-s",
        nargs="+",
        default=None,
        help="Filter systems (subset of config['systems'] keys, or 'all')",
    )
    parser.add_argument(
        "--mode",
        "-m",
        nargs="+",
        default=None,
        help="Filter scaling modes (system_size, constant_workload, batch_scaling, or all)",
    )
    parser.add_argument(
        "--timing-runs",
        "-n",
        type=int,
        default=None,
        help="Override timing iterations",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=None,
        help="Override warmup iterations",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=["torch", "jax", "warp"],
        help="Framework backend (default: torch)",
    )
    parser.add_argument(
        "--method",
        "--methods",
        dest="methods",
        nargs="+",
        default=None,
        help="Restrict benchmark methods/APIs. Accepted values are module-specific.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the expanded benchmark plan and exit without GPU allocation.",
    )
    parser.add_argument(
        "--max-total-atoms",
        type=int,
        default=None,
        help="Skip concrete cases above this total atom count before allocation.",
    )


def merge_common_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply the shared CLI flags on top of a YAML config.

    Only non-None CLI values override. Mutates and returns ``config``.
    Module-specific flags (cutoffs, methods, accuracies) are handled by
    each runner's own ``merge_cli_overrides`` on top of this.
    """
    if args.timing_runs is not None:
        config["parameters"]["timing_runs"] = args.timing_runs
    if args.warmup_runs is not None:
        config["parameters"]["warmup_runs"] = args.warmup_runs

    if args.system is not None and "all" not in args.system:
        for sys_name in list(config["systems"].keys()):
            config["systems"][sys_name]["enabled"] = sys_name in args.system

    if args.mode is not None and "all" not in args.mode:
        for mode_name in list(config["scaling"].keys()):
            if isinstance(config["scaling"][mode_name], dict):
                config["scaling"][mode_name]["enabled"] = mode_name in args.mode

    if args.output_dir is not None:
        config["output"]["base_dir"] = str(args.output_dir)
    if getattr(args, "backend", None) is not None:
        config.setdefault("runtime", {})["backend"] = args.backend
    if getattr(args, "dry_run", False):
        config.setdefault("runtime", {})["dry_run"] = True
    if getattr(args, "max_total_atoms", None) is not None:
        config.setdefault("parameters", {})["max_total_atoms"] = args.max_total_atoms
    if getattr(args, "methods", None) is not None and "methods" in config:
        selected_list = [normalize_method_name(m) for m in args.methods]
        selected = set(selected_list)
        config.setdefault("runtime", {})["explicit_methods"] = True
        config.setdefault("runtime", {})["selected_methods"] = selected_list
        for method in config["methods"]:
            method["enabled"] = normalize_method_name(method["name"]) in selected

    return config
