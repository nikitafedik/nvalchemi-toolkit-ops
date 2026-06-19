#!/usr/bin/env bash
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

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
ORIGINAL_HOME="${HOME:-}"

BACKEND="both"
BENCHMARK="all"
SYSTEM_FILTER=""
MODE_FILTER=""
DRY_RUN=0
RUN_PLOTS=1
RESULT_DIR=""
SKIP_SYNC=0
USE_CURRENT_ENV=0
UV_BIN="${UV_BIN:-uv}"
UV_SYNC_ARGS="${UV_SYNC_ARGS:---extra torch --extra jax --group docs}"
BENCHMARK_PIP_PACKAGES="${BENCHMARK_PIP_PACKAGES:-pyyaml>=6.0.3 nvidia-ml-py==13.590.48}"
D3_PARAMS_PATH="${BENCHMARK_D3_PARAMS_PATH:-}"

usage() {
    cat <<'USAGE'
Usage:
  benchmarks/run_reportable_suite.sh [options]

Options:
  --backend torch|jax|both   Backend pass(es) to run. Default: both.
  --benchmark all|nl|d3|el   Benchmark module to run. Default: all.
  --system SYSTEM            Run one system shard (for example: cscl or nh3).
  --mode MODE                Run one scaling-mode shard (for example:
                             system_size, constant_workload, or batch_scaling).
  --dry-run                  Expand plans only; do not allocate GPU memory.
  --output-dir DIR           Exact result directory. Must be outside /home.
  --no-plot                  Skip plot generation after benchmark passes.
  --skip-sync                Do not run uv sync before benchmark passes.
  --use-current-env          Do not force UV_PROJECT_ENVIRONMENT into scratch.
  --d3-params-path PATH      Pre-seeded DFT-D3 parameter .pt file or cache path.
                             Must be outside /home.
  -h, --help                 Show this help.

Environment:
  BENCHMARK_SCRATCH          Required unless /scratch/$USER exists. All caches,
                             venvs, logs, and default results go under this tree.
  BENCHMARK_D3_PARAMS_PATH   Same as --d3-params-path.
  UV_BIN                     uv executable to use. Default: uv.
  UV_SYNC_ARGS               Arguments for uv sync. Default selects compatible
                             CUDA 13 Torch/JAX extras plus docs plotting deps.
  BENCHMARK_PIP_PACKAGES     Extra runtime packages installed into the uv env.
                             Default: pyyaml and nvidia-ml-py.

This helper runs the reportable NL/D3/EL grid with 3 warmups and 10 timed runs.
It does not pass --max-total-atoms or any hardware-specific skip limits; OOMs
are recorded by the benchmark suite as success=False CSV rows and omitted from
plots.

Use --benchmark/--system/--mode with a shared --output-dir to shard reportable
runs across multiple processes or scheduler jobs. Generate plots once after all
shards have written their CSV rows.
USAGE
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            [[ $# -ge 2 ]] || die "missing value for --backend"
            BACKEND="$2"
            shift 2
            ;;
        --benchmark)
            [[ $# -ge 2 ]] || die "missing value for --benchmark"
            BENCHMARK="$2"
            shift 2
            ;;
        --system)
            [[ $# -ge 2 ]] || die "missing value for --system"
            SYSTEM_FILTER="$2"
            shift 2
            ;;
        --mode)
            [[ $# -ge 2 ]] || die "missing value for --mode"
            MODE_FILTER="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --output-dir)
            [[ $# -ge 2 ]] || die "missing value for --output-dir"
            RESULT_DIR="$2"
            shift 2
            ;;
        --no-plot)
            RUN_PLOTS=0
            shift
            ;;
        --skip-sync)
            SKIP_SYNC=1
            shift
            ;;
        --use-current-env)
            USE_CURRENT_ENV=1
            shift
            ;;
        --d3-params-path)
            [[ $# -ge 2 ]] || die "missing value for --d3-params-path"
            D3_PARAMS_PATH="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unexpected argument: $1"
            ;;
    esac
done

case "$BACKEND" in
    torch) BACKENDS=(torch) ;;
    jax) BACKENDS=(jax) ;;
    both) BACKENDS=(torch jax) ;;
    *) die "unsupported backend: $BACKEND" ;;
esac

case "$BENCHMARK" in
    all|nl|d3|el) ;;
    *) die "unsupported benchmark: $BENCHMARK" ;;
esac

if [[ -n "${BENCHMARK_SCRATCH:-}" ]]; then
    SCRATCH="$BENCHMARK_SCRATCH"
elif [[ -d "/scratch/${USER:-}" ]]; then
    SCRATCH="/scratch/${USER}/nvalchemiops-benchmarks"
else
    die "set BENCHMARK_SCRATCH to a scratch filesystem path"
fi

mkdir -p "$SCRATCH"
SCRATCH="$(cd "$SCRATCH" && pwd -P)"

reject_home_path() {
    local label="$1"
    local path="$2"
    case "$path" in
        /home|/home/*)
            die "$label must not be under /home: $path"
            ;;
    esac
    if [[ -n "$ORIGINAL_HOME" ]]; then
        case "$path" in
            "$ORIGINAL_HOME"|"$ORIGINAL_HOME"/*)
                die "$label must not be under HOME: $path"
                ;;
        esac
    fi
}

reject_home_path "BENCHMARK_SCRATCH" "$SCRATCH"

if [[ -n "$D3_PARAMS_PATH" ]]; then
    D3_PARAMS_DIR="$(dirname "$D3_PARAMS_PATH")"
    mkdir -p "$D3_PARAMS_DIR"
    D3_PARAMS_DIR="$(cd "$D3_PARAMS_DIR" && pwd -P)"
    D3_PARAMS_PATH="${D3_PARAMS_DIR}/$(basename "$D3_PARAMS_PATH")"
    reject_home_path "D3 parameter path" "$D3_PARAMS_PATH"
fi

if [[ -z "$RESULT_DIR" ]]; then
    STAMP="$(date +%Y%m%d-%H%M%S)"
    RESULT_DIR="${SCRATCH}/results/reportable-suite-${STAMP}"
fi
mkdir -p "$RESULT_DIR"
RESULT_DIR="$(cd "$RESULT_DIR" && pwd -P)"
reject_home_path "output directory" "$RESULT_DIR"

mkdir -p \
    "$RESULT_DIR/logs" \
    "$SCRATCH/cache/xdg" \
    "$SCRATCH/cache/uv" \
    "$SCRATCH/cache/pre-commit" \
    "$SCRATCH/cache/warp" \
    "$SCRATCH/cache/torch-extensions" \
    "$SCRATCH/cache/pytorch-kernels" \
    "$SCRATCH/cache/jax" \
    "$SCRATCH/cache/matplotlib" \
    "$SCRATCH/cache/cuda" \
    "$SCRATCH/home"

export HOME="$SCRATCH/home"
export XDG_CACHE_HOME="$SCRATCH/cache/xdg"
export UV_CACHE_DIR="$SCRATCH/cache/uv"
export PRE_COMMIT_HOME="$SCRATCH/cache/pre-commit"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
if [[ "$USE_CURRENT_ENV" -eq 0 ]]; then
    export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$SCRATCH/venv}"
fi
export WARP_CACHE_PATH="$SCRATCH/cache/warp"
export TORCH_EXTENSIONS_DIR="$SCRATCH/cache/torch-extensions"
export PYTORCH_KERNEL_CACHE_PATH="$SCRATCH/cache/pytorch-kernels"
export JAX_COMPILATION_CACHE_DIR="$SCRATCH/cache/jax"
export MPLCONFIGDIR="$SCRATCH/cache/matplotlib"
export CUDA_CACHE_PATH="$SCRATCH/cache/cuda"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"

cd "$ROOT"

LOG_PATH="$RESULT_DIR/logs/reportable-suite.log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "reportable_suite_started_at=$(date -Is)"
echo "root=$ROOT"
echo "scratch=$SCRATCH"
echo "result_dir=$RESULT_DIR"
echo "backend=$BACKEND"
echo "benchmark=$BENCHMARK"
echo "system_filter=${SYSTEM_FILTER:-<all>}"
echo "mode_filter=${MODE_FILTER:-<all>}"
echo "dry_run=$DRY_RUN"
echo "run_plots=$RUN_PLOTS"
echo "skip_sync=$SKIP_SYNC"
echo "use_current_env=$USE_CURRENT_ENV"
echo "uv_bin=$UV_BIN"
echo "uv_sync_args=$UV_SYNC_ARGS"
echo "benchmark_pip_packages=${BENCHMARK_PIP_PACKAGES:-<none>}"
echo "uv_project_environment=${UV_PROJECT_ENVIRONMENT:-<current>}"
echo "d3_params_path=${D3_PARAMS_PATH:-<xdg-cache-default>}"
echo

git rev-parse --abbrev-ref HEAD | tee "$RESULT_DIR/logs/git-branch.txt"
git rev-parse HEAD | tee "$RESULT_DIR/logs/git-head.txt"
git status --short --branch | tee "$RESULT_DIR/logs/git-status.txt"
git diff --stat | tee "$RESULT_DIR/logs/git-diff-stat.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi | tee "$RESULT_DIR/logs/nvidia-smi.txt"
fi

"$UV_BIN" --version
if [[ "$SKIP_SYNC" -eq 0 ]]; then
    read -r -a uv_sync_args <<< "$UV_SYNC_ARGS"
    "$UV_BIN" sync "${uv_sync_args[@]}"
    if [[ -n "$BENCHMARK_PIP_PACKAGES" ]]; then
        read -r -a benchmark_pip_packages <<< "$BENCHMARK_PIP_PACKAGES"
        if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" && -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]]; then
            "$UV_BIN" pip install --python "${UV_PROJECT_ENVIRONMENT}/bin/python" "${benchmark_pip_packages[@]}"
        else
            "$UV_BIN" pip install "${benchmark_pip_packages[@]}"
        fi
    fi
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
    "$UV_BIN" run python - <<'PY'
from __future__ import annotations

from pathlib import Path

from benchmarks.config import load_yaml_config

CONFIG_PATHS = (
    Path("benchmarks/neighborlist/benchmark_config.yaml"),
    Path("benchmarks/interactions/dispersion/benchmark_config.yaml"),
    Path("benchmarks/interactions/electrostatics/benchmark_config.yaml"),
)

missing: set[Path] = set()
for config_path in CONFIG_PATHS:
    config = load_yaml_config(config_path)
    nh3_config = config.get("systems", {}).get("nh3", {})
    if not nh3_config.get("enabled", True):
        continue
    pdb_dir = Path(nh3_config.get("pdb_dir", "benchmarks/nh3"))
    if not pdb_dir.is_absolute():
        pdb_dir = Path("benchmarks") / pdb_dir.name
    for atom_count in nh3_config.get("atom_counts", []):
        path = pdb_dir / f"ammonia_pbc_{atom_count}.pdb"
        if not path.exists():
            missing.add(path)

if missing:
    files = "\n".join(f"  - {path}" for path in sorted(missing))
    raise SystemExit(
        "Missing NH3 PBC benchmark inputs for reportable run:\n"
        f"{files}\n"
        "Generate them with: cd benchmarks/nh3 && printf '1-11\\n' | "
        "bash generate_pbc_pdbs.sh"
    )
PY
fi

run_suite() {
    local backend="$1"
    local log_suffix="$backend"
    [[ "$BENCHMARK" == "all" ]] || log_suffix="${log_suffix}-${BENCHMARK}"
    [[ -z "$SYSTEM_FILTER" ]] || log_suffix="${log_suffix}-${SYSTEM_FILTER}"
    [[ -z "$MODE_FILTER" ]] || log_suffix="${log_suffix}-${MODE_FILTER}"
    local log_file="$RESULT_DIR/logs/${log_suffix}.log"
    local common_args=(
        --benchmark "$BENCHMARK"
        --backend "$backend"
        --timing-runs 10
        --warmup-runs 3
    )
    if [[ -n "$SYSTEM_FILTER" ]]; then
        common_args+=(--system "$SYSTEM_FILTER")
    fi
    if [[ -n "$MODE_FILTER" ]]; then
        common_args+=(--mode "$MODE_FILTER")
    fi
    if [[ -n "$D3_PARAMS_PATH" ]]; then
        common_args+=(--d3-params-path "$D3_PARAMS_PATH")
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        "$UV_BIN" run python -m benchmarks.benchmark_suite \
            "${common_args[@]}" \
            --dry-run | tee "$log_file"
        return "${PIPESTATUS[0]}"
    fi

    "$UV_BIN" run python -m benchmarks.benchmark_suite \
        "${common_args[@]}" \
        --run-dir "$RESULT_DIR" \
        --no-plot | tee "$log_file"
    local status="${PIPESTATUS[0]}"
    if [[ "$status" -ne 0 ]]; then
        return "$status"
    fi
    cp "$RESULT_DIR/RUN_LOG.md" "$RESULT_DIR/RUN_LOG-${log_suffix}.md"
}

for backend in "${BACKENDS[@]}"; do
    echo
    echo "=== backend=$backend ==="
    run_suite "$backend"
done

full_suite_selection() {
    [[ "$BENCHMARK" == "all" && -z "$SYSTEM_FILTER" && -z "$MODE_FILTER" ]]
}

if [[ "$DRY_RUN" -eq 0 ]]; then
    if ! full_suite_selection; then
        echo
        echo "Skipping full-suite CSV completeness check for selected shard."
    else
    "$UV_BIN" run python - "$RESULT_DIR" "${BACKENDS[@]}" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
expected_backends = set(sys.argv[2:])
expected_prefixes = ("nl-", "d3-", "el-")
csv_paths = sorted(result_dir.glob("*.csv"))
if len(csv_paths) != 18:
    raise SystemExit(f"expected 18 suite CSVs, found {len(csv_paths)} in {result_dir}")

summary: dict[tuple[str, str], list[bool]] = {}
for path in csv_paths:
    prefix = path.name.split("-", 1)[0]
    if f"{prefix}-" not in expected_prefixes:
        raise SystemExit(f"unexpected CSV name: {path.name}")
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"empty CSV: {path.name}")
    for row in rows:
        backend = row.get("backend", "")
        success = row.get("success", "True") == "True"
        summary.setdefault((prefix, backend), []).append(success)

for prefix in ("nl", "d3", "el"):
    for backend in expected_backends:
        rows = summary.get((prefix, backend), [])
        if not rows:
            raise SystemExit(f"missing rows for {prefix}/{backend}")
        if not any(rows):
            raise SystemExit(f"no successful rows for {prefix}/{backend}")
        print(
            f"{prefix}/{backend}: rows={len(rows)} "
            f"successes={sum(rows)} failures={len(rows) - sum(rows)}"
        )
PY
    fi

    if [[ "$RUN_PLOTS" -eq 1 ]]; then
        "$UV_BIN" run python -m benchmarks.benchmark_suite \
            --plot-only "$RESULT_DIR" \
            --plots all
    fi
fi

echo
echo "reportable_suite_finished_at=$(date -Is)"
echo "result_dir=$RESULT_DIR"
