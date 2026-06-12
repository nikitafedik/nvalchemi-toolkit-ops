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

"""Chemical system generation and loading for benchmarks.

Systems supported:
- CsCl (cesium chloride): BCC-like crystal, 2 atoms/unit cell, programmatic
- NH3 (ammonia): Packmol-packed PBC boxes, loaded from PDB files
- OMat sample: packed PyTorch dataset, loaded from ``benchmarks/omat``

Each system provides: positions, atomic_numbers, cell, pbc, charges (optional),
and batching support via tiling/replication.

Supports both 'torch' and 'jax' backends via the ``backend`` parameter on
:func:`create_system`. The underlying system generation runs on numpy and is
converted to the requested framework at the end. Returned dictionaries have
the same keys for both backends; only the array types differ.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

__all__ = [
    "configs_for_mode",
    "cscl_actual_atoms",
    "create_cscl_batch",
    "create_cscl_system",
    "create_nh3_batch",
    "create_omat_batch",
    "create_system",
    "find_nh3_pdbs",
    "filter_configs_by_total_atoms",
    "get_constant_atoms_configs",
    "get_constant_total_configs",
    "get_system_size_configs",
    "omat_dataset_system_count",
    "omat_prefix_rollover_counts",
    "planned_atom_counts",
    "load_nh3_system",
    "load_omat_system",
    "parse_pdb",
    "resolve_nh3_dir",
]

# =============================================================================
# Constants
# =============================================================================

# Element atomic numbers
ELEMENT_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "Cs": 55, "Cl": 17}

# Partial charges for NH3 (neutral molecule: 3×0.3 + 1×(-0.9) = 0)
NH3_PARTIAL_CHARGES = {"H": 0.3, "N": -0.9}

# CsCl charges (ionic crystal)
CSCL_CHARGES = {"Cs": 1.0, "Cl": -1.0}

# CsCl lattice constant in Angstroms (Cs at corner, Cl at body center)
CSCL_LATTICE_CONSTANT = 4.119


def cscl_actual_atoms(n):
    """Return actual CsCl atom count for a target of n atoms.

    CsCl has 2 atoms per cubic unit cell, so valid sizes are 2*k^3.
    """
    n_cells = max(1, int(np.ceil((n / 2) ** (1 / 3))))
    return 2 * n_cells**3


# Default paths (relative to benchmarks/ directory)
SCRIPT_DIR = Path(__file__).parent
DEFAULT_NH3_DIR = SCRIPT_DIR / "nh3"
DEFAULT_OMAT_PATH = SCRIPT_DIR / "omat" / "omat24_sample.pt"
_OMAT_CACHE: dict[Path, dict] = {}


# =============================================================================
# PDB Parsing (NH3 systems)
# =============================================================================


def parse_pdb(path):
    """Parse PDB file with CRYST1 support.

    Parameters
    ----------
    path : str or Path
        Path to PDB file.

    Returns
    -------
    coords : np.ndarray, shape (N, 3), float32
        Atomic coordinates in Angstroms.
    atomic_numbers : np.ndarray, shape (N,), int32
        Atomic numbers.
    elements : list[str]
        Element symbols.
    cell : np.ndarray, shape (3, 3), float32
        Unit cell matrix (diagonal for cubic cells).
    """
    lines = Path(path).read_text().splitlines()
    coords, numbers, elements = [], [], []
    cell = None

    for line in lines:
        if line.startswith("CRYST1"):
            parts = line.split()
            cell = np.diag([float(parts[1]), float(parts[2]), float(parts[3])]).astype(
                np.float32
            )
        if line.startswith(("HETATM", "ATOM")):
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            # Element symbol: columns 77-78 or infer from atom name
            el = line[76:78].strip() if len(line) >= 78 else line[12:14].strip()[0]
            elements.append(el)
            numbers.append(ELEMENT_Z.get(el, 1))

    if cell is None:
        cell = np.eye(3, dtype=np.float32) * 10.0  # default fallback

    return np.asarray(coords, np.float32), np.asarray(numbers, np.int32), elements, cell


def find_nh3_pdbs(nh3_dir=None):
    """Find all NH3 PDB files sorted by atom count.

    Parameters
    ----------
    nh3_dir : str or Path, optional
        Directory containing ammonia_pbc_*.pdb files.

    Returns
    -------
    list[Path]
        Sorted PDB file paths (by atom count extracted from filename).
    """
    import re

    nh3_dir = Path(nh3_dir or DEFAULT_NH3_DIR)
    pdb_files = list(nh3_dir.glob("ammonia_pbc_*.pdb"))

    if not pdb_files:
        raise FileNotFoundError(
            f"No NH3 PDB files in {nh3_dir}. Run: cd nh3 && bash generate_pbc_pdbs.sh"
        )

    # Sort by atom count in filename (natural sort without natsort dependency)
    def _extract_count(p):
        m = re.search(r"ammonia_pbc_(\d+)\.pdb", p.name)
        return int(m.group(1)) if m else 0

    return sorted(pdb_files, key=_extract_count)


# =============================================================================
# Backend Converters
# =============================================================================


def _to_torch(np_data, device="cuda", dtype=torch.float32):
    """Convert a numpy system dict to torch tensors on device.

    Floating arrays become ``dtype`` (default float32); charges always float64;
    integer arrays int32; boolean stays bool. Non-array metadata is passed
    through unchanged.
    """
    out = {}
    for k, v in np_data.items():
        if not isinstance(v, np.ndarray):
            out[k] = v
            continue
        if v.dtype == bool:
            out[k] = torch.tensor(v, dtype=torch.bool, device=device)
        elif k == "charges":
            out[k] = torch.tensor(v, dtype=torch.float64, device=device)
        elif np.issubdtype(v.dtype, np.integer):
            out[k] = torch.tensor(v, dtype=torch.int32, device=device)
        else:
            out[k] = torch.tensor(v, dtype=dtype, device=device)
    return out


def _to_jax(np_data, dtype=None):
    """Convert a numpy system dict to jax arrays on the default device.

    Floating arrays become ``dtype`` if given (default float32); charges always
    float64; integers int32; bool stays bool. Imports jax lazily so torch-only
    paths do not require jax.
    """
    import jax.numpy as jnp

    float_dtype = dtype if dtype is not None else jnp.float32
    out = {}
    for k, v in np_data.items():
        if not isinstance(v, np.ndarray):
            out[k] = v
            continue
        if v.dtype == bool:
            out[k] = jnp.asarray(v, dtype=jnp.bool_)
        elif k == "charges":
            out[k] = jnp.asarray(v, dtype=jnp.float64)
        elif np.issubdtype(v.dtype, np.integer):
            out[k] = jnp.asarray(v, dtype=jnp.int32)
        else:
            out[k] = jnp.asarray(v, dtype=float_dtype)
    return out


def _dispatch_backend(np_data, backend, device, dtype):
    """Dispatch a numpy system dict to the requested backend."""
    if backend == "torch":
        return _to_torch(np_data, device=device, dtype=dtype)
    elif backend == "jax":
        # For JAX, convert torch dtype tokens to jax equivalents
        import jax.numpy as jnp

        if dtype is None or dtype is torch.float32:
            jdt = jnp.float32
        elif dtype is torch.float64:
            jdt = jnp.float64
        else:
            jdt = dtype  # assume already a jax dtype
        return _to_jax(np_data, dtype=jdt)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'torch' or 'jax'.")


# =============================================================================
# NH3 System Creation
# =============================================================================


def _build_nh3_single_numpy(pdb_path):
    """Build a single-system NH3 numpy dict (backend-agnostic)."""
    coords, numbers, elements, cell = parse_pdb(pdb_path)
    charges = np.array(
        [NH3_PARTIAL_CHARGES.get(el, 0.0) for el in elements], dtype=np.float64
    )
    return {
        "positions": coords.astype(np.float32),
        "atomic_numbers": numbers.astype(np.int32),
        "charges": charges,
        "cell": cell.astype(np.float32)[None],  # [1, 3, 3]
        "pbc": np.array([[True, True, True]], dtype=bool),
        "batch_idx": np.zeros(len(numbers), dtype=np.int32),
        "elements": elements,
        "atoms_per_system": len(numbers),
        "cell_size": float(np.diag(cell)[0]),
    }


def _build_nh3_batch_numpy(pdb_path, batch_size):
    """Build a batched NH3 numpy dict (backend-agnostic)."""
    coords, numbers, elements, cell = parse_pdb(pdb_path)
    n = len(numbers)
    charges = np.array(
        [NH3_PARTIAL_CHARGES.get(el, 0.0) for el in elements], dtype=np.float64
    )
    return {
        "positions": np.tile(coords, (batch_size, 1)).astype(np.float32),
        "atomic_numbers": np.tile(numbers, batch_size).astype(np.int32),
        "charges": np.tile(charges, batch_size),
        "cell": np.tile(cell[None], (batch_size, 1, 1)).astype(np.float32),
        "pbc": np.ones((batch_size, 3), dtype=bool),
        "batch_idx": np.repeat(np.arange(batch_size, dtype=np.int32), n),
        "elements": elements,
        "atoms_per_system": n,
        "total_atoms": n * batch_size,
        "batch_size": batch_size,
        "cell_size": float(np.diag(cell)[0]),
    }


def load_nh3_system(pdb_path, device="cuda", dtype=torch.float32, backend="torch"):
    """Load a single NH3 system from PDB file.

    Parameters
    ----------
    pdb_path : str or Path
        Path to PDB file.
    device : str, default='cuda'
        Torch device (only used for ``backend='torch'``).
    dtype : torch.dtype, default=torch.float32
        Floating-point precision.
    backend : str, default='torch'
        Framework backend: ``'torch'`` or ``'jax'``.

    Returns
    -------
    dict
        Keys: positions, atomic_numbers, charges, cell, pbc,
              elements, atoms_per_system, cell_size.
    """
    np_data = _build_nh3_single_numpy(pdb_path)
    return _dispatch_backend(np_data, backend, device, dtype)


def create_nh3_batch(
    pdb_path, batch_size, device="cuda", dtype=torch.float32, backend="torch"
):
    """Create a batched NH3 system by replicating a single PDB.

    Parameters
    ----------
    pdb_path : str or Path
        Path to NH3 PDB file.
    batch_size : int
        Number of replicas.
    device : str, default='cuda'
        Torch device (only used for ``backend='torch'``).
    dtype : torch.dtype, default=torch.float32
        Floating-point precision.
    backend : str, default='torch'
        Framework backend: ``'torch'`` or ``'jax'``.

    Returns
    -------
    dict
        Batched system with concatenated positions, tiled cells, batch_idx, etc.
    """
    np_data = _build_nh3_batch_numpy(pdb_path, batch_size)
    return _dispatch_backend(np_data, backend, device, dtype)


# =============================================================================
# CsCl System Creation
# =============================================================================


def _build_cscl_single_numpy(num_atoms):
    """Build a single-system CsCl numpy dict (backend-agnostic)."""
    a = CSCL_LATTICE_CONSTANT
    atoms_per_cell = 2  # Cs + Cl

    n_cells = max(1, int(np.ceil((num_atoms / atoms_per_cell) ** (1 / 3))))
    actual_atoms = cscl_actual_atoms(num_atoms)

    positions = []
    atomic_numbers = []
    charges = []

    for ix in range(n_cells):
        for iy in range(n_cells):
            for iz in range(n_cells):
                origin = np.array([ix, iy, iz], dtype=np.float32) * a
                # Cs at corner
                positions.append(origin)
                atomic_numbers.append(ELEMENT_Z["Cs"])
                charges.append(CSCL_CHARGES["Cs"])
                # Cl at body center
                positions.append(
                    origin + np.array([0.5, 0.5, 0.5], dtype=np.float32) * a
                )
                atomic_numbers.append(ELEMENT_Z["Cl"])
                charges.append(CSCL_CHARGES["Cl"])

    cell_size = n_cells * a
    cell = np.eye(3, dtype=np.float32) * cell_size

    return {
        "positions": np.asarray(positions[:actual_atoms], dtype=np.float32),
        "atomic_numbers": np.asarray(atomic_numbers[:actual_atoms], dtype=np.int32),
        "charges": np.asarray(charges[:actual_atoms], dtype=np.float64),
        "cell": cell[None],  # [1, 3, 3]
        "pbc": np.array([[True, True, True]], dtype=bool),
        "batch_idx": np.zeros(actual_atoms, dtype=np.int32),
        "atoms_per_system": actual_atoms,
        "total_atoms": actual_atoms,
        "batch_size": 1,
        "cell_size": cell_size,
    }


def _build_cscl_batch_numpy(num_atoms_per_system, batch_size):
    """Build a batched CsCl numpy dict (backend-agnostic)."""
    single = _build_cscl_single_numpy(num_atoms_per_system)
    n = single["atoms_per_system"]

    positions = np.tile(single["positions"], (batch_size, 1))
    atomic_numbers = np.tile(single["atomic_numbers"], batch_size)
    charges = np.tile(single["charges"], batch_size)
    cell = np.tile(single["cell"], (batch_size, 1, 1))
    pbc = np.tile(single["pbc"], (batch_size, 1))
    batch_idx = np.repeat(np.arange(batch_size, dtype=np.int32), n)

    return {
        "positions": positions.astype(np.float32),
        "atomic_numbers": atomic_numbers.astype(np.int32),
        "charges": charges,
        "cell": cell.astype(np.float32),
        "pbc": pbc,
        "batch_idx": batch_idx,
        "atoms_per_system": n,
        "total_atoms": n * batch_size,
        "batch_size": batch_size,
        "cell_size": single["cell_size"],
    }


def create_cscl_system(num_atoms, device="cuda", dtype=torch.float32, backend="torch"):
    """Create a CsCl supercell with approximately num_atoms atoms.

    CsCl is BCC-like: Cs at (0,0,0), Cl at (0.5,0.5,0.5) in fractional coords.
    2 atoms per unit cell.

    Parameters
    ----------
    num_atoms : int
        Target number of atoms (rounded to nearest even number).
    device : str, default='cuda'
        Torch device (only used for ``backend='torch'``).
    dtype : torch.dtype, default=torch.float32
        Floating-point precision.
    backend : str, default='torch'
        Framework backend: ``'torch'`` or ``'jax'``.

    Returns
    -------
    dict
        System with positions, atomic_numbers, charges, cell, pbc.
    """
    np_data = _build_cscl_single_numpy(num_atoms)
    return _dispatch_backend(np_data, backend, device, dtype)


def create_cscl_batch(
    num_atoms_per_system,
    batch_size,
    device="cuda",
    dtype=torch.float32,
    backend="torch",
):
    """Create a batched CsCl system by replicating a supercell.

    Parameters
    ----------
    num_atoms_per_system : int
        Atoms per individual CsCl supercell.
    batch_size : int
        Number of replicas.
    device : str, default='cuda'
        Torch device (only used for ``backend='torch'``).
    dtype : torch.dtype, default=torch.float32
        Floating-point precision.
    backend : str, default='torch'
        Framework backend: ``'torch'`` or ``'jax'``.

    Returns
    -------
    dict
        Batched system with concatenated positions, tiled cells, batch_idx.
    """
    np_data = _build_cscl_batch_numpy(num_atoms_per_system, batch_size)
    return _dispatch_backend(np_data, backend, device, dtype)


# =============================================================================
# OMat Sample Dataset Loading
# =============================================================================


def _load_omat_archive(dataset_path: str | Path | None = None) -> dict:
    """Load and cache the packed OMat sample archive on CPU."""
    path = Path(dataset_path or DEFAULT_OMAT_PATH).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR.parent / path
    path = path.resolve()
    if path not in _OMAT_CACHE:
        if not path.exists():
            raise FileNotFoundError(f"OMat sample dataset not found: {path}")
        _OMAT_CACHE[path] = torch.load(path, map_location="cpu", weights_only=False)
    return _OMAT_CACHE[path]


def _omat_atom_segments(dataset_path: str | Path | None = None) -> torch.Tensor:
    """Return per-system atom counts from a packed OMat sample archive."""
    archive = _load_omat_archive(dataset_path)
    return archive["segment_lengths"]["atoms"].to(dtype=torch.long)


def _target_prefix_count(segments: torch.Tensor, target_atoms: int) -> tuple[int, int]:
    """Return the OMat prefix length and actual atoms nearest ``target_atoms``."""
    total_available = int(segments.sum().item())
    repeats = max(1, int(np.ceil(target_atoms / max(total_available, 1))))
    systems_to_include = 0
    cumulative_atoms = 0

    for atom_count in segments.tolist() * repeats:
        atom_count = int(atom_count)
        if cumulative_atoms + atom_count <= target_atoms:
            cumulative_atoms += atom_count
            systems_to_include += 1
            continue
        if (
            target_atoms - cumulative_atoms
            > cumulative_atoms + atom_count - target_atoms
        ):
            cumulative_atoms += atom_count
            systems_to_include += 1
        break

    return max(systems_to_include, 1), cumulative_atoms


def omat_prefix_rollover_counts(
    dataset_path: str | Path | None,
    target_atoms: int,
) -> tuple[int, int]:
    """Return ``(num_systems, total_atoms)`` for OMat prefix-rollover batching."""
    segments = _omat_atom_segments(dataset_path)
    return _target_prefix_count(segments, int(target_atoms))


def omat_dataset_system_count(dataset_path: str | Path | None) -> int:
    """Return the number of systems in a packed OMat sample archive."""
    return int(_omat_atom_segments(dataset_path).numel())


def _omat_slice(dataset_path: str | Path | None, segment_index: int) -> dict:
    """Return one OMat system as a backend-neutral numpy dictionary."""
    archive = _load_omat_archive(dataset_path)
    data = archive["data"]
    segments = archive["segment_lengths"]["atoms"].to(dtype=torch.long)
    if segment_index < 0 or segment_index >= int(segments.numel()):
        raise IndexError(
            f"OMat segment_index {segment_index} out of range for {segments.numel()} systems"
        )
    starts = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(segments, 0)])
    start = int(starts[segment_index])
    end = int(starts[segment_index + 1])
    charge = data.get("charge")
    np_data = {
        "positions": data["coord"][start:end].numpy().astype(np.float32),
        "atomic_numbers": data["numbers"][start:end].numpy().astype(np.int32),
        "cell": data["cell"][segment_index].numpy().astype(np.float32),
        "pbc": np.ones(3, dtype=bool),
        "batch_idx": np.zeros(end - start, dtype=np.int32),
        "num_atoms": end - start,
        "total_atoms": end - start,
        "batch_size": 1,
        "segment_index": int(segment_index),
    }
    if charge is not None:
        np_data["charge"] = np.asarray([float(charge[segment_index])], dtype=np.float32)
    return np_data


def load_omat_system(
    dataset_path: str | Path | None,
    segment_index: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    backend: str = "torch",
) -> dict:
    """Load one system from a packed OMat sample archive.

    Parameters
    ----------
    dataset_path : str or Path
        Path to the packed OMat ``.pt`` sample.
    segment_index : int
        System index within the packed archive.
    device : str, default='cuda'
        PyTorch device (only used for ``backend='torch'``).
    dtype : torch.dtype, default=torch.float32
        Floating-point precision.
    backend : str, default='torch'
        Framework backend: ``'torch'`` or ``'jax'``.

    Returns
    -------
    dict
        System dictionary with positions, atomic numbers, cell, pbc, and charge.
    """
    np_data = _omat_slice(dataset_path, int(segment_index))
    return _dispatch_backend(np_data, backend, device, dtype)


def create_omat_batch(
    dataset_path: str | Path | None,
    segment_index: int,
    batch_size: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    backend: str = "torch",
) -> dict:
    """Create a replicated batch from one OMat sample system."""
    single = _omat_slice(dataset_path, int(segment_index))
    n = int(single["num_atoms"])
    np_data = {
        "positions": np.tile(single["positions"], (batch_size, 1)).astype(np.float32),
        "atomic_numbers": np.tile(single["atomic_numbers"], batch_size).astype(
            np.int32
        ),
        "cell": np.tile(single["cell"][None], (batch_size, 1, 1)).astype(np.float32),
        "pbc": np.tile(single["pbc"], (batch_size, 1)),
        "batch_idx": np.repeat(np.arange(batch_size, dtype=np.int32), n),
        "num_atoms": n,
        "total_atoms": n * batch_size,
        "batch_size": batch_size,
        "segment_index": int(segment_index),
    }
    if "charge" in single:
        np_data["charge"] = np.tile(single["charge"], batch_size).astype(np.float32)
    return _dispatch_backend(np_data, backend, device, dtype)


# =============================================================================
# Unified System Factory
# =============================================================================


def create_system(
    system_type: str,
    num_atoms: int | None = None,
    pdb_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
    segment_index: int | None = None,
    batch_size: int = 1,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    backend: str = "torch",
) -> dict:
    """Create a benchmark system (single or batched).

    Parameters
    ----------
    system_type : str
        'cscl', 'nh3', or 'omat'.
    num_atoms : int, optional
        Target atoms per system (required for CsCl, ignored for NH3).
    pdb_path : str or Path, optional
        PDB file path (required for NH3, ignored for CsCl).
    dataset_path : str or Path, optional
        Packed OMat sample path (required for OMat unless using the default).
    segment_index : int, optional
        OMat system index within ``dataset_path``.
    batch_size : int, default=1
        Number of system replicas.
    device : str, default='cuda'
        PyTorch device (only used for ``backend='torch'``).
    dtype : torch.dtype, default=torch.float32
        Floating-point precision.
    backend : str, default='torch'
        Framework backend: ``'torch'`` or ``'jax'``. The returned dict has the
        same keys for both; array types are backend-specific.

    Returns
    -------
    dict
        System dictionary with all required fields for benchmarking.
    """
    if system_type == "cscl":
        if num_atoms is None:
            raise ValueError("num_atoms required for CsCl systems")
        if batch_size == 1:
            return create_cscl_system(
                num_atoms, device=device, dtype=dtype, backend=backend
            )
        else:
            return create_cscl_batch(
                num_atoms, batch_size, device=device, dtype=dtype, backend=backend
            )

    elif system_type == "nh3":
        if pdb_path is None:
            raise ValueError("pdb_path required for NH3 systems")
        if batch_size == 1:
            return load_nh3_system(
                pdb_path, device=device, dtype=dtype, backend=backend
            )
        else:
            return create_nh3_batch(
                pdb_path, batch_size, device=device, dtype=dtype, backend=backend
            )

    elif system_type == "omat":
        if segment_index is None:
            raise ValueError("segment_index required for OMat systems")
        if batch_size == 1:
            return load_omat_system(
                dataset_path,
                segment_index,
                device=device,
                dtype=dtype,
                backend=backend,
            )
        else:
            return create_omat_batch(
                dataset_path,
                segment_index,
                batch_size,
                device=device,
                dtype=dtype,
                backend=backend,
            )

    else:
        raise ValueError(
            f"Unknown system type: {system_type}. Use 'cscl', 'nh3', or 'omat'."
        )


# =============================================================================
# Scaling Mode Helpers
# =============================================================================


def get_system_size_configs(system_type, atom_counts, nh3_dir=None, sys_config=None):
    """Generate configs for system-size scaling (batch=1, vary N).

    Parameters
    ----------
    system_type : str
        'cscl' or 'nh3'.
    atom_counts : list[int]
        Target atom counts.
    nh3_dir : str or Path, optional
        NH3 PDB directory.
    sys_config : dict, optional
        Full system config. Used by OMat for dataset path and sampling controls.

    Yields
    ------
    dict
        Config with 'num_atoms', 'pdb_path' (NH3 only), 'batch_size'=1.
    """
    if system_type == "nh3":
        pdb_files = find_nh3_pdbs(nh3_dir)
        for pdb in pdb_files:
            coords, _, _, _ = parse_pdb(pdb)
            n = len(coords)
            if atom_counts and n not in atom_counts:
                continue
            yield {"num_atoms": n, "pdb_path": pdb, "batch_size": 1}
    elif system_type == "omat":
        sys_config = sys_config or {}
        dataset_path = sys_config.get("dataset_path", DEFAULT_OMAT_PATH)
        segments = _omat_atom_segments(dataset_path)
        segment_indices = sys_config.get("segment_indices")
        max_per_count = int(sys_config.get("max_systems_per_atom_count", 1))
        seen_by_count: dict[int, int] = {}
        candidates = (
            [int(i) for i in segment_indices]
            if segment_indices is not None
            else range(int(segments.numel()))
        )
        rows = []
        for idx in candidates:
            n = int(segments[idx])
            if atom_counts and n not in atom_counts:
                continue
            seen = seen_by_count.get(n, 0)
            if segment_indices is None and max_per_count > 0 and seen >= max_per_count:
                continue
            seen_by_count[n] = seen + 1
            rows.append(
                {
                    "num_atoms": n,
                    "dataset_path": str(dataset_path),
                    "segment_index": idx,
                    "batch_size": 1,
                }
            )
        yield from sorted(rows, key=lambda row: (row["num_atoms"], row["segment_index"]))
    else:
        for n in atom_counts:
            yield {"num_atoms": n, "pdb_path": None, "batch_size": 1}


def get_constant_total_configs(system_type, target_atoms, nh3_dir=None):
    """Generate configs for constant-total-atoms scaling (128k batch).

    batch_size = target_atoms / atoms_per_system.

    Parameters
    ----------
    system_type : str
        'cscl' or 'nh3'.
    target_atoms : int
        Total atom target (e.g., 131072 = 128k).
    nh3_dir : str or Path, optional
        NH3 PDB directory.

    Yields
    ------
    dict
        Config with 'num_atoms', 'pdb_path', 'batch_size'.
    """
    if system_type == "nh3":
        pdb_files = find_nh3_pdbs(nh3_dir)
        for pdb in pdb_files:
            coords, _, _, _ = parse_pdb(pdb)
            n = len(coords)
            batch_size = target_atoms // n
            if batch_size < 1:
                continue
            yield {"num_atoms": n, "pdb_path": pdb, "batch_size": batch_size}
    else:
        for n in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
            actual = cscl_actual_atoms(n)
            batch_size = target_atoms // actual
            if batch_size < 1:
                continue
            yield {"num_atoms": n, "pdb_path": None, "batch_size": batch_size}


def get_constant_atoms_configs(
    system_type,
    atoms_per_system_sizes,
    max_total_atoms=131072,
    nh3_dir=None,
    sys_config=None,
    batch_sizes=None,
):
    """Generate configs for constant-atoms-per-system scaling (vary batch).

    Batch size grows in powers of 2 until total_atoms exceeds max_total_atoms.

    Parameters
    ----------
    system_type : str
        'cscl' or 'nh3'.
    atoms_per_system_sizes : list[int]
        Fixed atom counts to test (e.g., [256, 8192]).
    max_total_atoms : int, default=131072
        Maximum total atoms (atoms_per_system * batch_size).
    nh3_dir : str or Path, optional
        NH3 PDB directory.

    Yields
    ------
    dict
        Config with 'num_atoms', 'pdb_path', 'batch_size'.
    """
    # Default batch sizes are powers of 2. Callers can pass explicit sizes for
    # finer sweeps around OOM cliffs.
    all_batch_sizes = (
        [int(bs) for bs in batch_sizes]
        if batch_sizes is not None
        else [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    )

    if system_type == "nh3":
        pdb_files = find_nh3_pdbs(nh3_dir)
        for pdb in pdb_files:
            coords, _, _, _ = parse_pdb(pdb)
            n = len(coords)
            if n not in atoms_per_system_sizes:
                continue
            for bs in all_batch_sizes:
                if n * bs > max_total_atoms:
                    break  # stop growing batch for this atom size
                yield {"num_atoms": n, "pdb_path": pdb, "batch_size": bs}
    elif system_type == "omat":
        sys_config = sys_config or {}
        dataset_path = sys_config.get("dataset_path", DEFAULT_OMAT_PATH)
        batch_construction = str(
            sys_config.get("batch_construction", "repeat_segment")
        )
        if batch_construction == "prefix_rollover":
            atom_count_targets = sys_config.get("atom_count_targets")
            if atom_count_targets is None:
                atom_count_targets = [
                    int(n) * int(bs)
                    for n in atoms_per_system_sizes
                    for bs in all_batch_sizes
                ]
            for target_atoms in atom_count_targets:
                target_atoms = int(target_atoms)
                if target_atoms > max_total_atoms:
                    continue
                batch_size, total_atoms = omat_prefix_rollover_counts(
                    dataset_path,
                    target_atoms,
                )
                atoms_per_system = max(1, int(round(total_atoms / batch_size)))
                yield {
                    "num_atoms": atoms_per_system,
                    "target_atoms": target_atoms,
                    "dataset_path": str(dataset_path),
                    "segment_index": "",
                    "batch_size": batch_size,
                    "total_atoms": total_atoms,
                    "batch_construction": batch_construction,
                }
            return

        segments = _omat_atom_segments(dataset_path)
        requested_sizes = {int(n) for n in atoms_per_system_sizes}
        segment_indices = sys_config.get("segment_indices")
        max_per_count = int(sys_config.get("max_systems_per_atom_count", 1))
        seen_by_count: dict[int, int] = {}
        candidates = (
            [int(i) for i in segment_indices]
            if segment_indices is not None
            else range(int(segments.numel()))
        )
        for idx in candidates:
            n = int(segments[idx])
            if requested_sizes and n not in requested_sizes:
                continue
            seen = seen_by_count.get(n, 0)
            if segment_indices is None and max_per_count > 0 and seen >= max_per_count:
                continue
            seen_by_count[n] = seen + 1
            for bs in all_batch_sizes:
                if n * bs > max_total_atoms:
                    break
                yield {
                    "num_atoms": n,
                    "dataset_path": str(dataset_path),
                    "segment_index": idx,
                    "batch_size": bs,
                }
    else:
        for n in atoms_per_system_sizes:
            actual = cscl_actual_atoms(n)
            for bs in all_batch_sizes:
                if actual * bs > max_total_atoms:
                    break
                yield {"num_atoms": n, "pdb_path": None, "batch_size": bs}


# =============================================================================
# Runner Helpers (shared across NL/D3/EL run_from_config)
# =============================================================================


def resolve_nh3_dir(sys_config: dict) -> Path | None:
    """Resolve the NH3 PDB directory from a ``config['systems']['nh3']`` subtree.

    YAML key ``pdb_dir`` wins. Relative paths are resolved against the
    packaged ``benchmarks/nh3`` location — that way configs work regardless
    of which runner's directory invoked them. Returns ``None`` when no YAML
    override is set, in which case downstream :func:`find_nh3_pdbs` uses
    :data:`DEFAULT_NH3_DIR`.
    """
    pdb_dir = sys_config.get("pdb_dir")
    if not pdb_dir:
        return None
    pdb_dir = Path(pdb_dir)
    if pdb_dir.is_absolute():
        return pdb_dir
    candidate = SCRIPT_DIR / pdb_dir.name
    if candidate.exists():
        return candidate
    return DEFAULT_NH3_DIR


def configs_for_mode(
    mode_name: str,
    mode_config: dict,
    sys_name: str,
    sys_config: dict,
    nh3_dir: Path | None = None,
) -> list[dict]:
    """Dispatch to the right scaling-mode helper and return a concrete config list.

    Parameters
    ----------
    mode_name : str
        One of ``'system_size'``, ``'constant_workload'``, ``'batch_scaling'``.
        Unknown names return an empty list so callers can ``continue`` cleanly.
    mode_config : dict
        Subtree at ``config['scaling'][mode_name]``. Required keys per mode:
        ``target_atoms`` (constant_workload), ``max_total_atoms``
        (batch_scaling). Missing keys raise ``KeyError`` — YAML is
        authoritative.
    sys_name : str
        ``'cscl'`` or ``'nh3'``.
    sys_config : dict
        Subtree at ``config['systems'][sys_name]``. Uses ``atom_counts`` and
        ``constant_atoms_sizes``; both default to empty/``[1024, 8192]`` to
        support NH3 configs that omit them (NH3 discovers atom counts from
        PDB filenames instead).
    nh3_dir : Path, optional
        NH3 PDB directory (see :func:`resolve_nh3_dir`). Ignored for CsCl.

    Returns
    -------
    list[dict]
        Concrete configs with ``num_atoms``, ``batch_size``, and optionally
        ``pdb_path``.
    """
    atom_counts = sys_config.get("atom_counts", [])
    constant_atoms_sizes = sys_config.get("constant_atoms_sizes", [1024, 8192])
    if mode_name == "system_size":
        return list(get_system_size_configs(sys_name, atom_counts, nh3_dir, sys_config))
    if mode_name == "constant_workload":
        return list(
            get_constant_total_configs(sys_name, mode_config["target_atoms"], nh3_dir)
        )
    if mode_name == "batch_scaling":
        return list(
            get_constant_atoms_configs(
                sys_name,
                constant_atoms_sizes,
                mode_config["max_total_atoms"],
                nh3_dir,
                sys_config,
                mode_config.get("batch_sizes"),
            )
        )
    return []


def planned_atom_counts(sys_name: str, cfg: dict) -> tuple[int, int, int]:
    """Return ``(atoms_per_system, batch_size, total_atoms)`` without allocation."""
    batch_size = int(cfg["batch_size"])
    if sys_name == "omat" and cfg.get("batch_construction") == "prefix_rollover":
        return int(cfg["num_atoms"]), batch_size, int(cfg["total_atoms"])
    if sys_name == "cscl":
        atoms_per_system = cscl_actual_atoms(cfg["num_atoms"])
    else:
        atoms_per_system = int(cfg["num_atoms"])
    return atoms_per_system, batch_size, atoms_per_system * batch_size


def filter_configs_by_total_atoms(
    configs: list[dict],
    sys_name: str,
    max_total_atoms: int | None,
) -> tuple[list[dict], list[tuple[dict, int]]]:
    """Split configs into runnable and skipped rows using a total-atom cap."""
    if max_total_atoms is None:
        return configs, []
    kept = []
    skipped = []
    for cfg in configs:
        _, _, total_atoms = planned_atom_counts(sys_name, cfg)
        if total_atoms > max_total_atoms:
            skipped.append((cfg, total_atoms))
        else:
            kept.append(cfg)
    return kept, skipped
