# MD/OPT Model-Stack Benchmark Handoff

This is the handoff for the current chat. It is intentionally stored inside the
separated MD checkout:

```bash
cd /home/nfedik/projects/toolkit-ops/0.4-pre-md
```

Do not continue this work from `/home/nfedik/projects/toolkit-ops/0.3-tme`.
That checkout is historical context for the 0.3 benchmark PR and should not be
polluted with the MD/OPT model-stack work.

## Current State

- Checkout: `/home/nfedik/projects/toolkit-ops/0.4-pre-md`
- Branch: `benchmarks-dynamics-md-opt`
- Remote branch: `origin/benchmarks-dynamics-md-opt`
- No commit has been made for the current model-stack changes.
- GPU work is paused. Last verified `ws-loc` GPU state was idle: `0%`, `151 MiB`.
- Results and plots are present locally and synced back to `ws-loc`.

Current dirty tree at handoff time:

```text
 M benchmarks/dynamics/benchmark_config.yaml
 M benchmarks/dynamics/benchmark_dynamics.py
 M benchmarks/plotting/plot_benchmarks.py
 M docs/benchmarks/dynamics.md
?? MD_BENCHMARK_HANDOFF.md
?? benchmark-results-md-opt-apples-wsloc-full/
?? benchmarks/dynamics/model_stacks.py
```

## User Goal

Apples-to-apples benchmark coverage for:

- TensorNet + D3
- MACE + D3 + electrostatics
- MD and optimization
- All implemented engines for this model-stack path
- System-size variation and constant-workload modes from the benchmark suite

The implemented MACE + D3 + electrostatics coverage is:

- `mace_d3_pme`
- `mace_d3_ewald`
- FIRE and FIRE2 optimization variants for both

## Implemented Files

Primary new model-stack implementation:

- `benchmarks/dynamics/model_stacks.py`

Touched integration/config/docs:

- `benchmarks/dynamics/benchmark_config.yaml`
- `benchmarks/dynamics/benchmark_dynamics.py`
- `benchmarks/plotting/plot_benchmarks.py`
- `docs/benchmarks/dynamics.md`

The implementation uses Toolkit public APIs only. It does not modify Toolkit
source. The model stack composes public wrappers through Toolkit's pipeline
path:

- `nvalchemi.data.AtomicData`, `Batch`
- `nvalchemi.dynamics.NVTLangevin`, `FIRE`, `FIRE2`, `initialize_velocities`
- `nvalchemi.models.pipeline.PipelineModelWrapper`, `PipelineGroup`
- `DFTD3ModelWrapper`, `PMEModelWrapper`, `EwaldModelWrapper`, `MACEWrapper`
- TensorNet through `matgl.ext.alchmtk.TensorNetWrapper`

## Benchmark Methods

MD:

```text
tnet_d3
mace_d3
mace_d3_pme
mace_d3_ewald
```

OPT:

```text
tnet_d3_fire
tnet_d3_fire2
mace_d3_fire
mace_d3_fire2
mace_d3_pme_fire
mace_d3_pme_fire2
mace_d3_ewald_fire
mace_d3_ewald_fire2
```

The full apples-to-apples run used this subset:

```text
tnet_d3
mace_d3_pme
mace_d3_ewald
tnet_d3_fire
tnet_d3_fire2
mace_d3_pme_fire
mace_d3_pme_fire2
mace_d3_ewald_fire
mace_d3_ewald_fire2
```

## Important Technical Details

TensorNet:

- MatGL packaged model name used successfully:
  `TensorNet-PES-MatPES-PBE-2025.2`
- The MatGL loader does not forward `use_warp=True` into the saved model
  constructor.
- `model_stacks.py` rebuilds the loaded TensorNet potential with Warp layers,
  loads the same state dict, replaces `potential.model`, and enforces
  `require_warp: true`.

MACE:

- Uses `MACEWrapper.from_checkpoint(... enable_cueq=True, dtype=float32,
  compile_model=False)`.

PME/Ewald:

- Uses fixed charges carried by the benchmark CsCl/NH3 systems.
- These rows measure an additive fixed-charge long-range term, not a
  learned-charge model.

Timing:

- Warmups are excluded.
- Timed loops synchronize Torch CUDA and Warp before and after the measured
  loop.
- MD warmup/timing run on the same batch.
- OPT warmup uses a throwaway batch, then timed FIRE/FIRE2 starts from the
  original geometry with zero velocities.
- OPT is fixed-step timing, not convergence timing, to keep work comparable.

Metric correction:

- `time_us_per_atom_step` must be computed using `total_atoms`, not
  `atoms_per_system`.
- This matters for batched constant-workload rows.
- The code, copied CSVs, and regenerated plots have been corrected and synced.

## Successful Full Run

Full result directory, local and synced to `ws-loc`:

```text
/home/nfedik/projects/toolkit-ops/0.4-pre-md/benchmark-results-md-opt-apples-wsloc-full/run_2026-06-10_23-35-47
```

Run summary:

```text
DYN: 99/99 successful results
system_size: 36 rows, 36 success
constant_workload: 63 rows, 63 success
backend: torch
system: cscl
method families: model_md, model_opt
engines: nvt, fire, fire2
```

CSV files:

```text
dyn-cscl-system-size-scaling.csv
dyn-cscl-constant-workload-scaling.csv
```

Main PNG plots:

```text
dyn-cscl-system-size-scaling.png
dyn-cscl-constant-workload-scaling.png
```

Single-panel PNG plots:

```text
single-panels/dyn-cscl-system-size-scaling-time.png
single-panels/dyn-cscl-system-size-scaling-throughput.png
single-panels/dyn-cscl-system-size-scaling-memory.png
single-panels/dyn-cscl-constant-workload-scaling-time.png
single-panels/dyn-cscl-constant-workload-scaling-throughput.png
single-panels/dyn-cscl-constant-workload-scaling-memory.png
```

The combined dynamics plots were visually inspected after regeneration. The
third panel is currently used as clean legend space because dynamics memory is
not captured.

## Representative Corrected Speeds

The speed metric is microseconds per total atom-step:

```text
time_us_per_atom_step = avg_step_time_ms * 1000 / total_atoms
```

System-size, largest single system tested: `2662` atoms, `batch=1`.

```text
tnet_d3              NVT    5.530 us/atom-step
tnet_d3_fire         FIRE   5.538
tnet_d3_fire2        FIRE2  5.532

mace_d3_pme          NVT   12.182
mace_d3_pme_fire     FIRE  12.158
mace_d3_pme_fire2    FIRE2 12.156

mace_d3_ewald        NVT   12.444
mace_d3_ewald_fire   FIRE  12.510
mace_d3_ewald_fire2  FIRE2 12.519
```

Constant workload, about `8192` total atoms:

```text
128 atoms x batch 64:
tnet_d3              3.440 us/atom-step
mace_d3_pme          8.970
mace_d3_ewald        8.958

8192 atoms x batch 1:
tnet_d3              3.719
mace_d3_pme          9.535
mace_d3_ewald        9.848
```

## Remote Environment Used

The successful run used `ws-loc` and this Python environment:

```text
/home/nfedik/projects/tutorials/.venv-toolkit
```

Important environment variables used for remote runs:

```bash
export PYTHONPATH=/home/nfedik/projects/toolkit-ops/0.4-pre-md:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/home/nfedik/projects/tutorials/.venv-toolkit/lib/python3.12/site-packages/nvidia/cu13/lib:/home/nfedik/projects/tutorials/.venv-toolkit/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}
export WARP_CACHE_PATH=/tmp/warp-cache-toolkit-ops
export TORCH_EXTENSIONS_DIR=/tmp/torch-ext-toolkit-ops
export MPLCONFIGDIR=/tmp/matplotlib-toolkit-ops
export PYTHONPYCACHEPREFIX=/tmp/toolkit-ops-pycache
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
```

The full-run command was:

```bash
cd /home/nfedik/projects/toolkit-ops/0.4-pre-md
/home/nfedik/projects/tutorials/.venv-toolkit/bin/python -m benchmarks.benchmark_suite \
  --benchmark dyn \
  --backend torch \
  --system cscl \
  --mode system_size constant_workload \
  --method \
    tnet_d3 \
    mace_d3_pme \
    mace_d3_ewald \
    tnet_d3_fire \
    tnet_d3_fire2 \
    mace_d3_pme_fire \
    mace_d3_pme_fire2 \
    mace_d3_ewald_fire \
    mace_d3_ewald_fire2 \
  --output-dir /home/nfedik/projects/toolkit-ops/0.4-pre-md/benchmark-results-md-opt-apples-wsloc-full \
  --plots all
```

## Validation Performed

Local, CPU-only checks after the final plot cleanup:

```text
ruff check benchmarks/plotting/plot_benchmarks.py benchmarks/dynamics/benchmark_dynamics.py docs/benchmarks/dynamics.md
python -m compileall -q benchmarks/plotting/plot_benchmarks.py benchmarks/dynamics/benchmark_dynamics.py
git diff --check
```

Earlier checks also passed:

```text
compileall on benchmark_dynamics.py, model_stacks.py, plot_benchmarks.py
ruff check on benchmark_dynamics.py, model_stacks.py, plot_benchmarks.py, docs/benchmarks/dynamics.md
all-method tiny system-size smoke on ws-loc
constant-workload OPT smoke on ws-loc
full apples-to-apples ws-loc run
CSV audit for expected methods, engines, modes, and success rows
PNG file validation and visual inspection
```

Current remote sync was verified with:

```text
ws-loc GPU: 0%, 151 MiB
remote main PNGs present:
  dyn-cscl-constant-workload-scaling.png
  dyn-cscl-system-size-scaling.png
```

## CPU-Only Plot Regeneration

If plots need to be regenerated from the existing CSVs without rerunning the GPU
benchmark:

```bash
cd /home/nfedik/projects/toolkit-ops/0.4-pre-md

PYTHONPATH=/home/nfedik/projects/toolkit-ops/0.4-pre-md \
MPLCONFIGDIR=/tmp/matplotlib-toolkit-ops \
PYTHONPYCACHEPREFIX=/tmp/toolkit-ops-pycache \
UV_CACHE_DIR=/tmp/uv-cache-toolkit-ops \
uv run --with matplotlib python - <<'PY'
from pathlib import Path
from benchmarks.plotting.plot_benchmarks import detect_and_plot, plot_single_panel

run_dir = Path(
    "/home/nfedik/projects/toolkit-ops/0.4-pre-md/"
    "benchmark-results-md-opt-apples-wsloc-full/run_2026-06-10_23-35-47"
)
single_dir = run_dir / "single-panels"
single_dir.mkdir(exist_ok=True)

for csv_path in sorted(run_dir.glob("*.csv")):
    if csv_path.stem.endswith("-failures"):
        continue
    detect_and_plot(csv_path, run_dir)
    for panel in ("time", "throughput", "memory"):
        plot_single_panel(csv_path, panel, single_dir / f"{csv_path.stem}-{panel}.png")
PY
```

## Caveats And Decisions For PR Handoff

- The full result artifacts are currently untracked. Decide whether the PR should
  commit them, move them elsewhere, or keep them as handoff evidence only.
- The branch is dirty and intentionally uncommitted. Do not commit unless the
  user explicitly approves in the current conversation.
- If committing, use `git commit -s`.
- The dynamics memory panel is a placeholder because these model-stack runs do
  not capture dynamics VRAM. The combined plot now uses the third panel as clean
  legend space.
- The `mace_d3` and `mace_d3_fire/fire2` non-electrostatic methods are implemented
  but were not part of the final apples-to-apples run requested after the user
  emphasized MACE + D3 + electrostatics.
- Do not re-run GPU work until the user explicitly asks. The last user direction
  was to pause GPU use.

## Clean Transfer Checklist

For a new agent:

1. Start in `/home/nfedik/projects/toolkit-ops/0.4-pre-md`.
2. Run `git status --short --branch`; confirm branch is
   `benchmarks-dynamics-md-opt`.
3. Read this file before editing.
4. Treat `benchmark-results-md-opt-apples-wsloc-full/run_2026-06-10_23-35-47`
   as the authoritative completed run artifact.
5. Do not touch `/home/nfedik/projects/toolkit-ops/0.3-tme` unless the user
   explicitly asks.
6. Do not launch GPU benchmarks unless the user explicitly resumes GPU work.
7. Before PR handoff, decide artifact policy and run final lint/compile/diff
   checks again.

## Suggested Skills

- `handoff`: refresh this document before another context transfer.
- `diagnose`: use if benchmark results drift, a mode fails, or GPU/cache
  behavior differs from the validated run.
- `cc:review` or `cc:adversarial-review`: use for a second-pass branch-quality
  review before PR handoff.
- `tdd`: use if changing suite planning, backend guards, result schemas, or
  failure-row behavior.
