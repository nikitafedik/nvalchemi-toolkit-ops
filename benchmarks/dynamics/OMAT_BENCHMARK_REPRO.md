# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# OMat Benchmark Forensics Repro

This is a forensic/WIP benchmark branch for reproducing OMat batch-scaling
behavior on H100/B200. It is not a polished benchmark product branch.

## Scope

The current branch exercises two paths:

- E/F/S inference calls with `benchmarks/dynamics/benchmark_omat_efs.py`
- NVT MD steps with `benchmarks/dynamics/benchmark_dynamics.py`

Both paths use OMat prefix-rollover batching from
`benchmarks/omat/omat24_sample.pt`.

TensorNet is MatGL `TensorNet-PES-MatPES-PBE-2025.2` with Warp required
(`model_stacks.tensornet.require_warp: true`). This branch is not using an
R2SCAN TensorNet checkpoint.

## Known Finding

TensorNet with Warp reaches about 200k atoms on B200 in the current runs.
MACE through the current `nvalchemi` wrapper does not: a fresh B200 MACE
100k run OOMs even with D3 disabled, stress disabled, and MACE compile enabled.

The leading mismatch to investigate is stack identity:

- Roman reference path: `alchemistudio.data.Batch` + `alchemistudio.models.mace.MACE`
- Current branch path: `nvalchemi.data.Batch` + `nvalchemi.models.mace.MACEWrapper`

Do not treat the MACE results in this branch as Roman-comparable until that
stack-path mismatch is resolved.

## Required Local Layout

Run from a scratch checkout, not from `/home/nfedik`.

The runner expects this layout in the staged source root:

```text
benchmark-env/pyproject.toml
benchmarks/dynamics/benchmark_config_omat_200k.yaml
benchmarks/dynamics/benchmark_omat_efs.py
benchmarks/dynamics/benchmark_dynamics.py
benchmarks/dynamics/model_stacks.py
benchmarks/dynamics/omat_slurm_scratch_runner.sh
benchmarks/omat/omat24_sample.pt
```

`benchmark-env/` is the uv environment for the benchmark stack. The current
branch pins `nvalchemi-toolkit==0.1.0`, `torch==2.12.0+cu130`,
`matgl==4.0.2`, and `cuequivariance-ops-torch-cu13==0.10.0`; it uses the
staged checkout as editable `nvalchemi-toolkit-ops`.

If testing an unpublished Toolkit build, point `benchmark-env/pyproject.toml`
at a staged `vendor/nvalchemi-toolkit/` checkout and include that directory in
the scratch source root.

## Scratch Staging

`/tmp` on the Slurm frontends is frontend-local. If `cl` load-balances to a
different frontend, a `/tmp/$USER/...` checkout created on another frontend is
not visible. Either run staging and benchmark launch in one SSH session, or
clone/extract the branch directly into `/tmp/$USER` on the frontend where the
commands are run.

Example staging target:

```bash
scratch=/tmp/$USER/toolkit-omat-forensics
mkdir -p /tmp/$USER
cd "$scratch"
```

If starting from an existing checkout, copy it into scratch before running:

```bash
scratch=/tmp/$USER/toolkit-omat-forensics
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
  benchmarks/dynamics/model_stacks.py \
  benchmarks/dynamics/benchmark_dynamics.py \
  benchmarks/systems.py

bash -n benchmarks/dynamics/omat_slurm_scratch_runner.sh
```

H100 E/F/S smoke:

```bash
OMAT_SOURCE_ROOT=/tmp/$USER/toolkit-omat-forensics \
OMAT_TARGET_ATOMS="1024" \
OMAT_METHODS="mace tnet" \
OMAT_WARMUP_RUNS=1 \
OMAT_TIMING_RUNS=1 \
OMAT_TIME=00:15:00 \
benchmarks/dynamics/omat_slurm_scratch_runner.sh h100 efs \
  | tee /tmp/$USER/omat-h100-efs-smoke.log
```

CL smoke observed on June 12, 2026:

```text
GPU: NVIDIA H100 80GB HBM3
methods=['mace', 'tnet']
target=1024 actual_total_atoms=1032 graphs=8
mace: 234.3301 ms/call, 227.0640 us/atom/call
tnet: 19.1768 ms/call, 18.5822 us/atom/call
```

H100 MD smoke:

```bash
OMAT_SOURCE_ROOT=/tmp/$USER/toolkit-omat-forensics \
OMAT_TARGET_ATOMS="1024" \
OMAT_METHODS="mace tnet" \
OMAT_WARMUP_RUNS=1 \
OMAT_TIMING_RUNS=1 \
OMAT_TIME=00:15:00 \
benchmarks/dynamics/omat_slurm_scratch_runner.sh h100 md \
  | tee /tmp/$USER/omat-h100-md-smoke.log
```

B200 runs use the larger B200 partition seen during forensics:

```bash
export OMAT_B200_PARTITION="b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb"
```

## Boundary Probes

MACE compile probe at the observed failing boundary:

```bash
OMAT_SOURCE_ROOT=/tmp/$USER/toolkit-omat-forensics \
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

Expected current result: OOM around 100k atoms with resident memory near the
B200 limit.

## Extract CSV From Runner Logs

The runner prints CSV content between `=== csv ===` and `=== done ===`.

```bash
awk 'BEGIN{flag=0} /^success,/{flag=1} /^=== done ===/{flag=0} flag{print}' \
  /tmp/$USER/omat-b200-mace-100k-compile.log \
  > /tmp/$USER/omat-b200-mace-100k-compile.csv
```

## Current Evidence Artifacts

Local forensic logs produced during this investigation:

```text
benchmarks/benchmark-results/omat-scratch-runs/
/tmp/nfedik/omat-scratch-runs/probes/
/tmp/nfedik/omat-scratch-runs/plots/
```

The `/tmp/nfedik` paths are node-local working evidence, not source artifacts.
Copy them into an issue or shared artifact store if the team needs them.
