# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark direct OMat energy/force/stress model calls on random batches."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.config import load_yaml_config  # noqa: E402
from benchmarks.dynamics.model_stacks import (  # noqa: E402
    _build_model_stack,
    _dtype_from_config,
    _sync_work,
)

TARGET_ATOMS = [
    512,
    1_024,
    2_048,
    5_000,
    10_000,
    20_000,
    40_000,
    60_000,
    80_000,
    100_000,
    130_000,
    160_000,
    200_000,
]


def _configure_torch_acceleration() -> None:
    precision = os.environ.get("OMAT_FLOAT32_MATMUL_PRECISION", "high")
    if precision:
        torch.set_float32_matmul_precision(precision)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = os.environ.get(
            "OMAT_ALLOW_TF32",
            "1",
        ).lower() in {"1", "true", "yes", "on"}
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32


def _load_archive(path: Path) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    archive = torch.load(path, map_location="cpu", weights_only=False)
    data = archive["data"]
    segments = archive["segment_lengths"]["atoms"].to(dtype=torch.long)
    starts = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(segments, 0)])
    return data, segments, starts


def _choose_segments(
    segments: torch.Tensor,
    target_atoms: int,
    *,
    seed: int,
    sampling_mode: str,
) -> list[int]:
    if sampling_mode == "repeat_largest":
        idx = int(torch.argmax(segments).item())
        n_atoms = int(segments[idx])
        repeats = max(1, math.ceil(target_atoms / n_atoms))
        return [idx] * repeats
    if sampling_mode != "random":
        raise ValueError(f"Unsupported sampling mode: {sampling_mode}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + target_atoms * 17)
    order = torch.randperm(int(segments.numel()), generator=generator).tolist()
    chosen: list[int] = []
    total = 0
    cursor = 0
    while total < target_atoms and cursor < len(order):
        idx = int(order[cursor])
        cursor += 1
        n_atoms = int(segments[idx])
        if total and total + n_atoms > target_atoms:
            continue
        chosen.append(idx)
        total += n_atoms
    if not chosen:
        smallest = int(torch.argmin(segments).item())
        chosen.append(smallest)
    return chosen


def _make_atomic_data(
    data: dict[str, torch.Tensor],
    starts: torch.Tensor,
    idx: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
):
    from nvalchemi.data import AtomicData

    start = int(starts[idx])
    end = int(starts[idx + 1])
    n_atoms = end - start
    charge = data.get("charge")
    return AtomicData(
        positions=data["coord"][start:end].to(device=device, dtype=dtype).clone(),
        atomic_numbers=data["numbers"][start:end].to(device=device, dtype=torch.long),
        charge=(
            charge[idx].to(device=device, dtype=dtype).reshape(1, 1)
            if isinstance(charge, torch.Tensor)
            else None
        ),
        forces=torch.zeros(n_atoms, 3, dtype=dtype, device=device),
        energy=torch.zeros(1, 1, dtype=dtype, device=device),
        velocities=torch.zeros(n_atoms, 3, dtype=dtype, device=device),
        cell=data["cell"][idx].to(device=device, dtype=dtype).reshape(1, 3, 3),
        pbc=torch.ones(1, 3, dtype=torch.bool, device=device),
    )


def _make_batch(
    data: dict[str, torch.Tensor],
    segments: torch.Tensor,
    starts: torch.Tensor,
    target_atoms: int,
    *,
    seed: int,
    sampling_mode: str,
    device: torch.device,
    dtype: torch.dtype,
):
    from nvalchemi.data import Batch

    indices = _choose_segments(
        segments,
        target_atoms,
        seed=seed,
        sampling_mode=sampling_mode,
    )
    data_list = [
        _make_atomic_data(data, starts, idx, device=device, dtype=dtype)
        for idx in indices
    ]
    batch = Batch.from_data_list(data_list, device=device)
    return batch, indices, int(sum(int(segments[idx]) for idx in indices))


def _make_source_batch(
    data: dict[str, torch.Tensor],
    segments: torch.Tensor,
    starts: torch.Tensor,
    *,
    dtype: torch.dtype,
):
    from nvalchemi.data import Batch

    cpu = torch.device("cpu")
    data_list = [
        _make_atomic_data(data, starts, idx, device=cpu, dtype=dtype)
        for idx in range(int(segments.numel()))
    ]
    return Batch.from_data_list(data_list, device=cpu)


def _target_prefix_count(segments: torch.Tensor, target_atoms: int) -> tuple[int, int]:
    total_available = int(segments.sum().item())
    repeats = max(1, math.ceil(target_atoms / total_available))
    systems_to_include = 0
    cumulative_atoms = 0

    for atom_count in segments.tolist() * repeats:
        atom_count = int(atom_count)
        if cumulative_atoms + atom_count <= target_atoms:
            cumulative_atoms += atom_count
            systems_to_include += 1
            continue
        if target_atoms - cumulative_atoms > cumulative_atoms + atom_count - target_atoms:
            cumulative_atoms += atom_count
            systems_to_include += 1
        break

    return max(systems_to_include, 1), cumulative_atoms


def _make_prefix_rollover_batch(
    source_batch,
    segments: torch.Tensor,
    target_atoms: int,
    *,
    device: torch.device,
):
    systems_to_include, total_atoms = _target_prefix_count(segments, target_atoms)
    repeats = max(1, math.ceil(systems_to_include / int(segments.numel())))
    batch = source_batch
    if repeats > 1:
        batch = source_batch.clone()
        for _ in range(repeats - 1):
            batch.append(source_batch.clone())
    subbatch = batch.index_select(slice(0, systems_to_include)).to(device)
    indices = [i % int(segments.numel()) for i in range(systems_to_include)]
    return subbatch, indices, int(total_atoms)


def _build_config(config_path: Path) -> dict[str, Any]:
    config = load_yaml_config(config_path)
    config.setdefault("model_stacks", {})
    config["model_stacks"]["position_perturbation"] = 0.0
    mace_cfg = config["model_stacks"].setdefault("mace", {})
    if "OMAT_MACE_COMPILE_MODEL" in os.environ:
        mace_cfg["compile_model"] = os.environ["OMAT_MACE_COMPILE_MODEL"].lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    return config


def _set_outputs(model: torch.nn.Module, outputs: set[str]) -> str:
    requested = set(outputs)
    try:
        model.model_config.active_outputs = requested
        return ",".join(sorted(requested))
    except Exception:
        fallback = {"energy", "forces"}
        model.model_config.active_outputs = fallback
        return ",".join(sorted(fallback))


def _time_model(
    model: torch.nn.Module,
    batch,
    *,
    warmup_runs: int,
    timing_runs: int,
    device: torch.device,
) -> float:
    for hook in model.make_neighbor_hooks():
        rebuild = getattr(hook, "_rebuild", None)
        if rebuild is not None:
            try:
                rebuild(batch)
            except TypeError as exc:
                if "shift_range_per_dimension" not in str(exc):
                    raise
                hook._buf_nl_kwargs = {}
                rebuild(batch)
    for _ in range(warmup_runs):
        model(batch)
    _sync_work(device)
    start = time.perf_counter()
    for _ in range(timing_runs):
        model(batch)
    _sync_work(device)
    return (time.perf_counter() - start) * 1000.0 / max(timing_runs, 1)


def _failure_row(
    *,
    method: str,
    target_atoms: int,
    error: Exception,
    outputs: str,
    sampling_mode: str,
) -> dict[str, Any]:
    return {
        "success": False,
        "method": method,
        "sampling_mode": sampling_mode,
        "target_atoms": target_atoms,
        "total_atoms": math.nan,
        "num_graphs": math.nan,
        "avg_atoms_per_graph": math.nan,
        "avg_call_time_ms": math.nan,
        "time_us_per_atom_call": math.nan,
        "throughput_atom_calls_per_s": math.nan,
        "outputs": outputs,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    _configure_torch_acceleration()
    config = _build_config(args.config)
    dtype = _dtype_from_config(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data, segments, starts = _load_archive(args.dataset)
    rows: list[dict[str, Any]] = []
    source_batch = None

    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print(f"dataset systems={int(segments.numel())} atoms={int(segments.sum())}")
    print(f"targets={args.target_atoms}")
    print(f"methods={args.methods}")
    print(f"sampling_mode={args.sampling_mode}")
    if args.sampling_mode == "prefix_rollover":
        source_batch = _make_source_batch(data, segments, starts, dtype=dtype)

    for method in args.methods:
        model = _build_model_stack(method, config, device, dtype)
        outputs = _set_outputs(model, set(args.outputs.split(",")))
        for target_atoms in args.target_atoms:
            batch = None
            try:
                if args.sampling_mode == "prefix_rollover":
                    batch, indices, total_atoms = _make_prefix_rollover_batch(
                        source_batch,
                        segments,
                        int(target_atoms),
                        device=device,
                    )
                else:
                    batch, indices, total_atoms = _make_batch(
                        data,
                        segments,
                        starts,
                        int(target_atoms),
                        seed=args.seed,
                        sampling_mode=args.sampling_mode,
                        device=device,
                        dtype=dtype,
                    )
                avg_ms = _time_model(
                    model,
                    batch,
                    warmup_runs=args.warmup_runs,
                    timing_runs=args.timing_runs,
                    device=device,
                )
                row = {
                    "success": True,
                    "method": method,
                    "sampling_mode": args.sampling_mode,
                    "target_atoms": int(target_atoms),
                    "total_atoms": int(total_atoms),
                    "num_graphs": len(indices),
                    "avg_atoms_per_graph": total_atoms / max(len(indices), 1),
                    "avg_call_time_ms": avg_ms,
                    "time_us_per_atom_call": avg_ms * 1000.0 / max(total_atoms, 1),
                    "throughput_atom_calls_per_s": total_atoms / (avg_ms / 1000.0),
                    "outputs": outputs,
                    "error_type": "",
                    "error": "",
                }
                print(
                    f"  {method} target={target_atoms} total={total_atoms} "
                    f"graphs={len(indices)}: {avg_ms:.4f} ms/call"
                )
            except torch.cuda.OutOfMemoryError as exc:
                row = _failure_row(
                    method=method,
                    target_atoms=int(target_atoms),
                    error=exc,
                    outputs=outputs,
                    sampling_mode=args.sampling_mode,
                )
                print(f"  {method} target={target_atoms}: OOM")
            except Exception as exc:
                row = _failure_row(
                    method=method,
                    target_atoms=int(target_atoms),
                    error=exc,
                    outputs=outputs,
                    sampling_mode=args.sampling_mode,
                )
                print(f"  {method} target={target_atoms}: FAILED - {exc}")
            finally:
                del batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--method", dest="methods", nargs="+", required=True)
    parser.add_argument("--target-atoms", type=int, nargs="+", default=TARGET_ATOMS)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--timing-runs", type=int, default=10)
    parser.add_argument("--outputs", default="energy,forces,stress")
    parser.add_argument(
        "--sampling-mode",
        choices=["random", "repeat_largest", "prefix_rollover"],
        default="random",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = run(args)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
