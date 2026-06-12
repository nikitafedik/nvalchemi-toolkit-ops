# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# OMat Model-Stack Benchmark Repro

This benchmark exercises OMat model-stack batch scaling for direct
energy/force/stress calls and NVT MD steps.

## Scope

The current branch exercises two paths:

- E/F/S inference calls with `benchmarks/dynamics/benchmark_omat_efs.py`
- NVT MD steps with `benchmarks/dynamics/benchmark_dynamics.py`

Both paths use OMat prefix-rollover batching from
`benchmarks/omat/omat24_sample.pt`.

TensorNet is MatGL `TensorNet-PES-MatPES-PBE-2025.2` with Warp required
(`model_stacks.tensornet.require_warp: true`). This branch is not using an
R2SCAN TensorNet checkpoint.

## Required Local Layout

The runner expects this layout in the staged source root:

```text
benchmark-env/pyproject.toml
benchmarks/dynamics/benchmark_config_omat_200k.yaml
benchmarks/dynamics/benchmark_omat_efs.py
benchmarks/dynamics/benchmark_dynamics.py
benchmarks/dynamics/direct_tensornet.py
benchmarks/dynamics/model_stacks.py
benchmarks/dynamics/omat_slurm_scratch_runner.sh
benchmarks/omat/omat24_sample.pt
```

`benchmark-env/` is the uv environment for the benchmark stack. Use
`uv.lock` or `benchmark-env/pyproject.toml` as the source of truth for pinned
package versions.

To benchmark with an alternate local `nvalchemiops` checkout, set:

```bash
export NVALCHEMIOPS_PATH=/path/to/nvalchemi-toolkit-ops-checkout
```

The direct E/F/S benchmark applies this path before importing `nvalchemiops`
and records the imported module path in the audit JSON.

## Scratch Staging

`/tmp` on the Slurm frontends is frontend-local. If `cl` load-balances to a
different frontend, a `/tmp/$USER/...` checkout created on another frontend is
not visible. Either run staging and benchmark launch in one SSH session, or
clone/extract the branch directly into `/tmp/$USER` on the frontend where the
commands are run.

Example staging target:

```bash
scratch=/tmp/$USER/toolkit-omat-bench
mkdir -p /tmp/$USER
cd "$scratch"
```

If starting from an existing checkout, copy it into scratch before running:

```bash
scratch=/tmp/$USER/toolkit-omat-bench
mkdir -p "$scratch"
rsync -a --delete \
  --exclude .git \
  --exclude .venv \
  --exclude benchmark-results \
  --exclude cache \
  --exclude logs \
  ./ "$scratch"/
cd "$scratch"
```

## Smoke Commands

These smokes keep writable state under `/tmp/$USER`.

Syntax checks:

```bash
mkdir -p /tmp/$USER/pycache
PYTHONPYCACHEPREFIX=/tmp/$USER/pycache \
python3 -m py_compile \
  benchmarks/dynamics/benchmark_omat_efs.py \
  benchmarks/dynamics/direct_tensornet.py \
  benchmarks/dynamics/model_stacks.py \
  benchmarks/dynamics/benchmark_dynamics.py \
  benchmarks/systems.py

bash -n benchmarks/dynamics/omat_slurm_scratch_runner.sh
```

Direct EF-only MACE smoke up to about 30k atoms with prefix subbatches:

```bash
python benchmarks/dynamics/benchmark_omat_efs.py \
  --config benchmarks/dynamics/benchmark_config_omat_200k.yaml \
  --dataset benchmarks/omat/omat24_sample.pt \
  --output-csv /tmp/$USER/omat-mace-prefix-ef-smoke.csv \
  --method mace_ef \
  --target-atoms 1024 8232 20000 32796 \
  --warmup-runs 1 \
  --timing-runs 1 \
  --sampling-mode prefix_rollover
```

Direct EF-only MACE smoke for the random-structure workload:

```bash
python benchmarks/dynamics/benchmark_omat_efs.py \
  --config benchmarks/dynamics/benchmark_config_omat_200k.yaml \
  --dataset benchmarks/omat/omat24_sample.pt \
  --output-csv /tmp/$USER/omat-mace-random-ef-smoke.csv \
  --method mace_ef \
  --target-atoms 1024 8232 20000 32796 \
  --warmup-runs 1 \
  --timing-runs 1 \
  --sampling-mode random
```

H100 E/F/S smoke through Slurm:

```bash
OMAT_SOURCE_ROOT=/tmp/$USER/toolkit-omat-bench \
OMAT_TARGET_ATOMS="1024" \
OMAT_METHODS="mace tnet" \
OMAT_WARMUP_RUNS=1 \
OMAT_TIMING_RUNS=1 \
OMAT_TIME=00:15:00 \
benchmarks/dynamics/omat_slurm_scratch_runner.sh h100 efs \
  | tee /tmp/$USER/omat-h100-efs-smoke.log
```

H100 MD smoke:

```bash
OMAT_SOURCE_ROOT=/tmp/$USER/toolkit-omat-bench \
OMAT_TARGET_ATOMS="1024" \
OMAT_METHODS="mace tnet" \
OMAT_WARMUP_RUNS=1 \
OMAT_TIMING_RUNS=1 \
OMAT_TIME=00:15:00 \
benchmarks/dynamics/omat_slurm_scratch_runner.sh h100 md \
  | tee /tmp/$USER/omat-h100-md-smoke.log
```

Override the B200 partition when needed:

```bash
export OMAT_B200_PARTITION="b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb"
```

## Boundary Probes

MACE compile probe at a fixed target:

```bash
OMAT_SOURCE_ROOT=/tmp/$USER/toolkit-omat-bench \
OMAT_B200_PARTITION="b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb" \
OMAT_TARGET_ATOMS="100000" \
OMAT_METHODS="mace" \
OMAT_OUTPUTS="energy,forces" \
OMAT_WARMUP_RUNS=1 \
OMAT_TIMING_RUNS=2 \
OMAT_MACE_COMPILE_MODEL=1 \
TORCH_COMPILE_DISABLE=0 \
TORCHDYNAMO_DISABLE=0 \
OMAT_TIME=00:45:00 \
benchmarks/dynamics/omat_slurm_scratch_runner.sh b200 efs \
  | tee /tmp/$USER/omat-b200-mace-100k-compile.log
```

## Extract CSV From Runner Logs

The runner prints CSV content between `=== csv ===` and `=== done ===`.

```bash
awk 'BEGIN{flag=0} /^success,/{flag=1} /^=== done ===/{flag=0} flag{print}' \
  /tmp/$USER/omat-b200-mace-100k-compile.log \
  > /tmp/$USER/omat-b200-mace-100k-compile.csv
```
