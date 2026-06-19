# Toolkit-Ops Benchmark Branch Manual

This is an agent/user manual for `origin/benchmarks-0.4-rc-clean`.

## Fetch The Code Branch

```bash
git fetch origin benchmarks-0.4-rc-clean
git switch --track origin/benchmarks-0.4-rc-clean
```

If the branch already exists locally:

```bash
git switch benchmarks-0.4-rc-clean
git pull --ff-only
```

Inspect against the 0.4 RC base:

```bash
git fetch upstream 0.4.0-rc
git log --oneline upstream/0.4.0-rc..origin/benchmarks-0.4-rc-clean
git diff --stat upstream/0.4.0-rc..origin/benchmarks-0.4-rc-clean
git show --stat 4755bbc525baf9fb50a8a4201608a4964cb7cd64
```

## Fetch Agent-Only Handoff Material

This branch is pullable for future agents but should not go upstream:

```bash
git fetch origin benchmarks-0.4-rc-clean-agent-handoff
git worktree add /tmp/toolkit-ops-agent-handoff origin/benchmarks-0.4-rc-clean-agent-handoff
```

Read:

- `/tmp/toolkit-ops-agent-handoff/AGENT_HANDOFF.md`
- `/tmp/toolkit-ops-agent-handoff/AGENT_MANUAL.md`

## Build Docs For Inspection

Use scratch/tmp caches so the docs build does not write under home:

```bash
env \
  UV_CACHE_DIR=/tmp/codex-uv-cache \
  XDG_CACHE_HOME=/tmp/codex-xdg-cache \
  MPLCONFIGDIR=/tmp/codex-mpl-cache \
  WARP_CACHE_PATH=/tmp/codex-warp-cache \
  TORCH_EXTENSIONS_DIR=/tmp/codex-torch-extensions \
  PLOT_GALLERY=False \
  RUN_STALE_EXAMPLES=False \
  uv run sphinx-build -q -b html docs docs/_build-codex-docs-inspect \
    -w /tmp/codex-sphinx-warnings-inspect.log
```

Expected local/offline warning residue:

- Intersphinx inventory DNS warnings if the environment cannot resolve docs
  hosts.
- `config.cache` warning for `sphinx_gallery_conf`.

Unexpected and worth fixing:

- Any source-level docutils errors.
- Any title underline warnings from generated examples.
- Any missing toctree/doc reference warnings.

## Regenerate Benchmark Plots

```bash
env MPLCONFIGDIR=/tmp/codex-mpl-cache \
  uv run python docs/benchmarks/generate_plots.py
```

Generated PNGs are ignored by git. Review them locally against the raw CSVs in
`docs/benchmarks/benchmark_results/`.

## Run Focused Local Checks

```bash
env UV_CACHE_DIR=/tmp/codex-uv-cache \
  uv run pytest test/test_benchmark_planning.py -q

env UV_TOOL_DIR=/tmp/codex-uv-tools UV_CACHE_DIR=/tmp/codex-uv-cache \
  uvx --from ruff==0.11.13 ruff check \
    benchmarks docs nvalchemiops test

env UV_TOOL_DIR=/tmp/codex-uv-tools UV_CACHE_DIR=/tmp/codex-uv-cache \
  uvx --from ruff==0.11.13 ruff format --check \
    benchmarks docs nvalchemiops test
```

The broad Ruff commands may include unrelated repo areas; narrow them if an
unrelated pre-existing issue appears.

## Dry-Run The Reportable Plan

The reportable helper requires scratch outside `/home`:

```bash
export BENCHMARK_SCRATCH=/scratch/$USER/nvalchemiops-benchmarks

benchmarks/run_reportable_suite.sh \
  --backend both \
  --benchmark all \
  --dry-run \
  --output-dir /scratch/$USER/nvalchemiops-benchmarks/results/dry-run \
  --no-plot
```

The helper sets caches under scratch and rejects `/home` paths for scratch,
outputs, and D3 parameter cache.

## Run Reportable Shards

Use one shared output directory when sharding. Example small shard:

```bash
export BENCHMARK_SCRATCH=/scratch/$USER/nvalchemiops-benchmarks
export OUT=/scratch/$USER/nvalchemiops-benchmarks/results/reportable-manual

benchmarks/run_reportable_suite.sh \
  --backend torch \
  --benchmark nl \
  --system cscl \
  --mode system_size \
  --output-dir "$OUT" \
  --no-plot
```

Full suite:

```bash
export BENCHMARK_SCRATCH=/scratch/$USER/nvalchemiops-benchmarks
export OUT=/scratch/$USER/nvalchemiops-benchmarks/results/reportable-full

benchmarks/run_reportable_suite.sh \
  --backend both \
  --benchmark all \
  --output-dir "$OUT"
```

Do not pass `--max-total-atoms` for default reportable docs data. That flag is
an opt-in policy filter and should write `SkippedByPolicy` rows only when a
human explicitly wants such a filtered run.

## Benchmark Policy Reminders

- Reportable profile: 3 warmups + 10 timed runs.
- JAX timing must use wall-clock measurement with `block_until_ready()`.
- Torch/Warp timing must synchronize appropriately.
- OOMs/timeouts/failures should become explicit CSV rows with `success=False`,
  `error`, and `error_type`.
- Plots should omit unsuccessful rows but raw CSVs should keep them.
- Inspect suspicious plots against raw data before reporting conclusions.

## Before Upstream Merge

- Rebuild docs from a clean checkout.
- Inspect raw CSVs and regenerated PNGs together.
- Run the planning tests.
- If possible, run at least a small GPU shard on the target cluster.
- Confirm no agent-only handoff/manual files are included in the upstream PR.
