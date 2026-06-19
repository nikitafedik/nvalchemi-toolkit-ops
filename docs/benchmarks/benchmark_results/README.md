---
orphan: true
---

# Benchmark Results

Pre-computed CSVs consumed by the Sphinx docs build. The shipped numbers
under this directory were produced on an **NVIDIA H100 80 GB HBM3
(Hopper)** and cover three modules (neighbor list, DFT-D3 dispersion,
electrostatics) across two chemical systems (CsCl, NH₃) and three
scaling modes.

See the per-module doc pages for how to read the plots and how to
reproduce:

- `../neighborlist.md`
- `../dftd3.md`
- `../electrostatics.md`

## File naming

Names follow the scheme emitted by
`benchmarks.suite_utils.make_csv_name(module, system, mode)`:

```text
{module}-{system}-{mode-slug}.csv
```

Where `module` ∈ `{nl, d3, el}`, `system` ∈ `{cscl, nh3}`, and
`mode-slug` ∈ `{system-size-scaling, constant-workload-scaling,
batch-scaling}`. Example: `nl-cscl-system-size-scaling.csv`.

The current NL/D3/EL suite uses only root-level `nl-*.csv`, `d3-*.csv`,
and `el-*.csv` files. Legacy dynamics and segment-operation CSVs may also
remain at the root for their own docs pages. Archived pre-suite NL/D3/EL
comparison CSVs remain under `archive/` and are not read by the current suite
plot generation path.

## CSV schema

Emitted by `benchmarks.suite_utils.build_result`:

| Column | Type | Description |
|---|---|---|
| `system` | str | `cscl` or `nh3` |
| `scaling_mode` | str | `system_size`, `constant_workload`, or `batch_scaling` |
| `method` | str | NL strategy (`naive_scalar`, `naive_tile`, `cell_list_atom_centric`, `cell_list_pair_centric`, `cluster_tile`, plus batch-prefixed concrete APIs where applicable), `dftd3` (D3), or `pme` / `pme_cg` / `ewald` / `ewald_cg` (EL) |
| `backend` | str | `torch`, `jax`, or `warp` where supported |
| `atoms_per_system` | int | Atoms in one system |
| `batch_size` | int | Number of systems in the batch |
| `total_atoms` | int | `atoms_per_system` × `batch_size` |
| `time_us_per_atom` | float | Mean μs per atom across the batch timing |
| `throughput_atoms_per_sec` | float | Derived throughput |
| `mem_delta_mb` | float | Memory delta from the pre-timing measurement call (MB); NaN for JAX |
| `mem_peak_gb` | float | Peak GPU memory (GB); NaN for JAX |
| `timing_runs` | int | Number of timed calls represented by the row |
| `warmup_runs` | int | Number of untimed warmup calls before measurement |
| `timing_method` | str | Timing path used for the row, such as `torch_cuda_events` or `jax_wall_block_until_ready` |
| `timing_method_real` | str | Added by EL for real-space timing paths |
| `timing_method_reciprocal` | str | Added by EL for reciprocal-space timing paths |
| `compile_policy` | str | Compile/warmup policy; shipped rows use `warmup_excluded` |
| `success` | bool | `False` rows are filtered by the plotter |
| `error` | str | Concise failure or skip message for `success=False` rows |
| `error_type` | str | Stable failure class, such as `OutOfMemoryError`, `SkippedByPolicy`, or `SkippedAfterOOM` |
| `cutoff` | float | Added by NL and D3 |
| `accuracy` | float | Added by EL |
| `time_d3_us_per_atom` | float | Added by D3 (excludes NL build time) |
| `time_real_us_per_atom` | float | Added by EL for real-space timing breakdowns |
| `time_reciprocal_us_per_atom` | float | Added by EL for reciprocal-space timing breakdowns |
| `backend_comparable` | bool | Added by NL to mark rows included in backend-comparison plots |
| `timing_scope` | str | Added by NL to separate backend-comparison rows from coverage-only rows |

When Torch and JAX runs share an output directory, each backend rerun replaces
only its own rows and preserves the other backend's rows. Failed, skipped, and
OOM cases are written directly into the main CSV with `success=False`; the
plotter filters those rows out. The suite no longer writes separate failure
files.

## Reproducing

Run from the repository root. Module-specific flags are documented on
each module's doc page; the flags below are common to all three.

```bash
python -m benchmarks.neighborlist.benchmark_neighborlist \
    --config benchmarks/neighborlist/benchmark_config.yaml \
    --output-dir docs/benchmarks/benchmark_results
```

Swap in `benchmarks.interactions.dispersion.benchmark_dftd3` or
`benchmarks.interactions.electrostatics.benchmark_electrostatics` for
the other modules, or invoke all three via the unified suite:

```bash
python -m benchmarks.benchmark_suite --benchmark all \
    --run-dir docs/benchmarks/benchmark_results
```

The docs CSVs are reportable benchmark outputs: they use the full configured
grid, 3 warmups, and 10 timed runs unless an explicit command-line filter is
shown. Reduced smoke runs should write to a separate output directory.

The shipped H100 CSVs were collected as scheduler shards with the same full
grids and timing protocol. CSV rows record per-benchmark timings, not scheduler
elapsed time; keep scheduler logs or `RUN_LOG.md` artifacts with any PR report
that quotes shard wall time. Queue time and environment setup are
site-dependent and should be reported separately.

For the JAX backend, pass `--backend jax`. The runner sets JAX/XLA defaults
before import unless you already configured them in the environment.
For D3 on offline clusters, pass `--d3-params-path` to a scratch-local
`dftd3_parameters.pt` file or pre-populate
`$XDG_CACHE_HOME/nvalchemiops/dftd3_parameters.pt`.

## Visualization

Sphinx's generate_plots hook reads the standardized NL/D3/EL suite CSVs and
the legacy dynamics CSVs it knows how to parse, then writes PNGs to
`../_static/`. The benchmark pages embed those images.
