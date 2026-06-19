# Toolkit-Ops Benchmark Branch Handoff

Created: 2026-06-18

## Branches

- Code branch: `origin/benchmarks-0.4-rc-clean`
- Base: `upstream/0.4.0-rc` at `3f69347`
- Code commit: `4755bbc525baf9fb50a8a4201608a4964cb7cd64`
- Patch artifact: `/tmp/0001-benchmarks-refresh-0.4-rc-suite.patch`
- Agent-only side branch: `origin/benchmarks-0.4-rc-clean-agent-handoff`

Do not merge the agent-only side branch upstream. It exists only so future
agents can pull the handoff/manual without putting them in the project PR.

## Suggested Skills

- `diagnose` for benchmark timing or correctness regressions.
- `cc:review` for ordinary external review.
- `cc:adversarial-review` for pre-merge scrutiny of design and performance
  assumptions.
- `handoff` when transferring this work to another agent.

## What The Code Branch Contains

The branch ports and refreshes the benchmarking work onto `0.4.0-rc`:

- Unified reportable benchmark suite for NL, D3, and electrostatics.
- Scratch-safe cluster runner at `benchmarks/run_reportable_suite.sh`.
- Shared benchmark config/plotting/system utilities under `benchmarks/`.
- Granular NL methods in Torch/JAX benchmark surfaces.
- Reportable CSVs under `docs/benchmarks/benchmark_results/`.
- Legacy/stale CSVs moved under `docs/benchmarks/benchmark_results/archive/`.
- `torch_dftd`/`torchpme` benchmark artifacts removed from active docs.
- Docs updated for the new benchmark suite and plot surfaces.
- Focused planning/coverage tests in `test/test_benchmark_planning.py`.
- Sphinx warning cleanup for stale docs references and malformed docstrings.

## Validation Already Run

Current push-turn validation:

- `git diff --check` passed.
- Docs build passed with:
  `PLOT_GALLERY=False RUN_STALE_EXAMPLES=False uv run sphinx-build -q -b html docs docs/_build-codex-docs-warning-clean-check3 -w /tmp/codex-sphinx-warnings3.log`
- Final warning log contained only offline intersphinx inventory warnings and
  Sphinx-Gallery `config.cache`; source-level docs warnings were cleaned.
- `python -m compileall -q` passed on touched Python files.
- Pinned Ruff `0.11.13` check passed on touched Python/example files.
- Pinned Ruff `0.11.13` format check passed on touched Python/example files.
- Touched-file pre-commit passed.

Earlier same-branch validation before this push:

- `uv run pytest test/test_benchmark_planning.py -q` passed with 124 tests.
- `docs/benchmarks/generate_plots.py` passed.
- Touched-file pre-commit passed for the benchmark-suite work.

## Known Caveats

- The final docs-clean pass did not launch a fresh full H100 reportable suite.
  The committed CSVs are the branch's reportable data artifacts and should be
  inspected against the generated plots before upstream merge.
- The docs build may show intersphinx warnings when offline or sandboxed.
- The Sphinx-Gallery `config.cache` warning is harmless local cache noise.
- Do not set hardware skip limits for reportable runs. OOMs/timeouts should be
  recorded as CSV rows with `success=False`, `error`, and `error_type`.

## Review Focus

- Confirm the branch really stays based on `0.4.0-rc`.
- Inspect benchmark configs for the intended 3 warmups + 10 timed runs.
- Confirm reportable runs do not use hardware-specific atom-count limits.
- Compare raw CSVs with generated plots, especially memory and throughput.
- Check Torch/JAX timing comparability and failure-row handling.
- Confirm docs mention new NL variants and no stale `torch_dftd`/`torchpme`
  active benchmark language remains.
- Confirm `benchmarks/run_reportable_suite.sh` keeps caches, venv, logs, D3
  parameter cache, and result dirs outside `/home`.
