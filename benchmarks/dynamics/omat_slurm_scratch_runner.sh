#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  benchmarks/dynamics/omat_slurm_scratch_runner.sh <h100|b200> <md-dry|md|efs>

Runs OMat benchmark probes without writing to remote /home. The script is
intended for the Slurm frontend: it tars the current checkout into /tmp,
allocates one GPU, broadcasts the tarball to the allocated node with sbcast,
runs uv with all caches in node-local /tmp, and prints generated CSVs to stdout.

Environment:
  OMAT_SOURCE_ROOT       Checkout to stage (default: current working directory)
  OMAT_SCRATCH_BASE      Frontend scratch dir (default: /tmp/$USER)
  OMAT_TARGET_ATOMS      Space-separated total-atom targets
                         (default: "1024 5000" for md-dry, "1024" otherwise)
  OMAT_METHODS           Space-separated methods
                         (default: "mace tnet" for efs, "mace" otherwise)
  OMAT_GPU_PARTITION     Override Slurm partition
  OMAT_H100_PARTITION    H100 partition override
  OMAT_B200_PARTITION    B200 partition override
  OMAT_TIME              Allocation time (default: 00:45:00)
  OMAT_CPUS              CPUs per task (default: 16)
  OMAT_MEM               Slurm memory (default: 128G)
  OMAT_WARMUP_RUNS       Warmup steps/calls override
  OMAT_TIMING_RUNS       Timed steps/calls override
  OMAT_SAMPLING_MODE     E/F/S sampling mode (default: prefix_rollover)
  OMAT_OUTPUTS           E/F/S outputs (default: energy,forces,stress)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

gpu_label="${1:?missing GPU label: h100 or b200}"
bench_mode="${2:?missing benchmark mode: md-dry, md, or efs}"

case "${gpu_label}" in
  h100)
    default_partition="h100-80gb-hbm3@ts6/mg62g4100/1gpu-32cpu-256gb"
    partition="${OMAT_GPU_PARTITION:-${OMAT_H100_PARTITION:-${default_partition}}}"
    ;;
  b200)
    default_partition="b200@ts5/genoad12m3nl/1gpu-32cpu-512gb"
    partition="${OMAT_GPU_PARTITION:-${OMAT_B200_PARTITION:-${default_partition}}}"
    ;;
  *)
    echo "Unsupported GPU label: ${gpu_label}" >&2
    usage >&2
    exit 2
    ;;
esac

case "${bench_mode}" in
  md-dry|md|efs) ;;
  *)
    echo "Unsupported benchmark mode: ${bench_mode}" >&2
    usage >&2
    exit 2
    ;;
esac

source_root="${OMAT_SOURCE_ROOT:-$(pwd)}"
source_root="$(cd "${source_root}" && pwd)"
source_name="$(basename "${source_root}")"
scratch_base="${OMAT_SCRATCH_BASE:-/tmp/${USER}}"
mkdir -p "${scratch_base}"

required_paths=(
  "benchmark-env"
  "benchmarks/dynamics/benchmark_config_omat_200k.yaml"
  "benchmarks/dynamics/benchmark_omat_efs.py"
  "benchmarks/dynamics/benchmark_dynamics.py"
  "benchmarks/dynamics/model_stacks.py"
  "benchmarks/omat/omat24_sample.pt"
)
for required_path in "${required_paths[@]}"; do
  if [[ ! -e "${source_root}/${required_path}" ]]; then
    echo "Missing required benchmark source path: ${source_root}/${required_path}" >&2
    exit 2
  fi
done
if [[ -f "${source_root}/benchmark-env/pyproject.toml" ]] \
  && grep -q "../vendor/nvalchemi-toolkit" "${source_root}/benchmark-env/pyproject.toml" \
  && [[ ! -d "${source_root}/vendor/nvalchemi-toolkit" ]]; then
  echo "benchmark-env references ../vendor/nvalchemi-toolkit, but it is missing under ${source_root}/vendor" >&2
  exit 2
fi

tarball="${scratch_base}/omat-${source_name}-${gpu_label}-${bench_mode}.tar"
node_runner="${scratch_base}/omat-node-runner-${gpu_label}-${bench_mode}.sh"
uv_staged="${scratch_base}/uv"
uv_source="${OMAT_UV_BIN:-$(command -v uv)}"
cp "${uv_source}" "${uv_staged}"
chmod +x "${uv_staged}"

tar \
  --exclude ".git" \
  --exclude ".venv" \
  --exclude "benchmark-results" \
  --exclude "cache" \
  --exclude "logs" \
  -C "$(dirname "${source_root}")" \
  -cf "${tarball}" \
  "${source_name}"

case "${bench_mode}" in
  md-dry)
    default_targets="1024 5000"
    default_methods="mace"
    ;;
  md)
    default_targets="1024"
    default_methods="mace"
    ;;
  efs)
    default_targets="1024"
    default_methods="mace tnet"
    ;;
esac

export OMAT_BENCH_MODE="${bench_mode}"
export OMAT_GPU_LABEL="${gpu_label}"
export OMAT_METHODS="${OMAT_METHODS:-${default_methods}}"
export OMAT_TARGET_ATOMS="${OMAT_TARGET_ATOMS:-${default_targets}}"
export OMAT_WARMUP_RUNS="${OMAT_WARMUP_RUNS:-}"
export OMAT_TIMING_RUNS="${OMAT_TIMING_RUNS:-}"
export OMAT_SAMPLING_MODE="${OMAT_SAMPLING_MODE:-prefix_rollover}"
export OMAT_OUTPUTS="${OMAT_OUTPUTS:-energy,forces,stress}"

cat > "${node_runner}" <<'NODE_RUNNER'
#!/usr/bin/env bash
set -euo pipefail

job_tmp="${1:?missing job tmp}"
source_name="${2:?missing source name}"
bench_mode="${OMAT_BENCH_MODE:?missing OMAT_BENCH_MODE}"

work_dir="${job_tmp}/work"
root="${work_dir}/${source_name}"
venv_dir="${job_tmp}/venv"
out_dir="${job_tmp}/out"
cache_dir="${job_tmp}/cache"
config_src="${root}/benchmarks/dynamics/benchmark_config_omat_200k.yaml"
config_run="${job_tmp}/benchmark_config_omat_targets.yaml"
dataset="${root}/benchmarks/omat/omat24_sample.pt"

mkdir -p \
  "${work_dir}" "${out_dir}" "${cache_dir}" \
  "${job_tmp}/uv-cache" "${job_tmp}/warp-cache" "${job_tmp}/torch-ext" \
  "${job_tmp}/mpl" "${job_tmp}/pycache" \
  "${cache_dir}/xdg" "${cache_dir}/torch" "${cache_dir}/matgl" "${cache_dir}/mace"

tar -C "${work_dir}" -xf "${job_tmp}/src.tar"

export PATH="${job_tmp}:${PATH}"
export UV_CACHE_DIR="${job_tmp}/uv-cache"
export UV_PROJECT_ENVIRONMENT="${venv_dir}"
export WARP_CACHE_PATH="${job_tmp}/warp-cache"
export TORCH_EXTENSIONS_DIR="${job_tmp}/torch-ext"
export MPLCONFIGDIR="${job_tmp}/mpl"
export PYTHONPYCACHEPREFIX="${job_tmp}/pycache"
export XDG_CACHE_HOME="${cache_dir}/xdg"
export TORCH_HOME="${cache_dir}/torch"
export MATGL_CACHE="${cache_dir}/matgl"
export MACE_CACHE_DIR="${cache_dir}/mace"
export PYTHONPATH="${root}"
export PYTHONUNBUFFERED=1
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== node ==="
date
hostname
nvidia-smi -L || true
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
echo "ROOT=${root}"
echo "JOB_TMP=${job_tmp}"
echo "MODE=${bench_mode}"
echo "METHODS=${OMAT_METHODS}"
echo "TARGETS=${OMAT_TARGET_ATOMS}"
echo "OMAT_MACE_COMPILE_MODEL=${OMAT_MACE_COMPILE_MODEL:-}"
echo "TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-}"
echo "TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-}"

echo "=== uv sync ==="
cd "${root}/benchmark-env"
uv sync --no-progress

"${venv_dir}/bin/python" - "${config_src}" "${config_run}" <<'PY'
import os
import sys
import yaml

src, dst = sys.argv[1], sys.argv[2]
with open(src) as handle:
    config = yaml.safe_load(handle)
targets = [int(item) for item in os.environ["OMAT_TARGET_ATOMS"].split()]
config["model_stacks"]["systems"]["omat"]["atom_count_targets"] = targets
config["model_stacks"]["position_perturbation"] = 0.0
if "OMAT_MACE_COMPILE_MODEL" in os.environ:
    config["model_stacks"].setdefault("mace", {})["compile_model"] = (
        os.environ["OMAT_MACE_COMPILE_MODEL"].lower()
        in {"1", "true", "yes", "on"}
    )
with open(dst, "w") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY

cd "${root}"
if [[ "${bench_mode}" == "efs" ]]; then
  read -r -a methods <<< "${OMAT_METHODS}"
  read -r -a targets <<< "${OMAT_TARGET_ATOMS}"
  warmup="${OMAT_WARMUP_RUNS:-3}"
  timing="${OMAT_TIMING_RUNS:-10}"
  "${venv_dir}/bin/python" "${root}/benchmarks/dynamics/benchmark_omat_efs.py" \
    --config "${config_run}" \
    --dataset "${dataset}" \
    --output-csv "${out_dir}/omat-efs.csv" \
    --method "${methods[@]}" \
    --target-atoms "${targets[@]}" \
    --warmup-runs "${warmup}" \
    --timing-runs "${timing}" \
    --outputs "${OMAT_OUTPUTS}" \
    --sampling-mode "${OMAT_SAMPLING_MODE}"
else
  read -r -a methods <<< "${OMAT_METHODS}"
  warmup="${OMAT_WARMUP_RUNS:-}"
  timing="${OMAT_TIMING_RUNS:-}"
  args=(
    -m benchmarks.dynamics.benchmark_dynamics
    --config "${config_run}"
    --backend torch
    --system omat
    --mode batch_scaling
    --method "${methods[@]}"
    --output-dir "${out_dir}"
  )
  if [[ -n "${warmup}" ]]; then
    args+=(--warmup-runs "${warmup}")
  fi
  if [[ -n "${timing}" ]]; then
    args+=(--timing-runs "${timing}")
  fi
  if [[ "${bench_mode}" == "md-dry" ]]; then
    args+=(--dry-run)
  fi
  "${venv_dir}/bin/python" "${args[@]}"
fi

echo "=== csv ==="
find "${out_dir}" -type f -name "*.csv" -print -exec sed -n '1,220p' {} \;
echo "=== done ==="
date
NODE_RUNNER
chmod +x "${node_runner}"

allocation_time="${OMAT_TIME:-00:45:00}"
cpus="${OMAT_CPUS:-16}"
mem="${OMAT_MEM:-128G}"

echo "=== frontend ==="
date
hostname
echo "SOURCE_ROOT=${source_root}"
echo "SCRATCH_BASE=${scratch_base}"
echo "TARBALL=${tarball}"
du -sh "${tarball}"
echo "UV_STAGED=${uv_staged}"
echo "PARTITION=${partition}"
echo "MODE=${bench_mode}"
echo "METHODS=${OMAT_METHODS}"
echo "TARGETS=${OMAT_TARGET_ATOMS}"

salloc \
  --partition="${partition}" \
  --gres=gpu:1 \
  --ntasks=1 \
  --cpus-per-task="${cpus}" \
  --mem="${mem}" \
  --chdir=/tmp \
  --time="${allocation_time}" \
  bash -lc '
    set -euo pipefail
    job_tmp="/tmp/${USER}/omat-${SLURM_JOB_ID}-${OMAT_GPU_LABEL}-${OMAT_BENCH_MODE}"
    srun --ntasks=1 --chdir=/tmp mkdir -p "${job_tmp}"
    sbcast -f "'"${tarball}"'" "${job_tmp}/src.tar"
    sbcast -f "'"${node_runner}"'" "${job_tmp}/node_runner.sh"
    sbcast -f "'"${uv_staged}"'" "${job_tmp}/uv"
    srun --ntasks=1 --chdir=/tmp chmod +x "${job_tmp}/uv"
    srun --ntasks=1 --chdir=/tmp bash "${job_tmp}/node_runner.sh" "${job_tmp}" "'"${source_name}"'"
  '
