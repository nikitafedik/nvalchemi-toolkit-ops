# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark-local TensorNet wrapper with single-gradient EFS."""

from __future__ import annotations

import types

import torch


class DirectSingleGradTensorNetModel(torch.nn.Module):
    """Run a MatGL TensorNetWrapper with one autograd call for forces and stress.

    This is intentionally benchmark-local. It does not patch MatGL or
    nvalchemi-toolkit. The wrapped object is the existing
    ``matgl.ext.alchmtk.TensorNetWrapper`` built by ``model_stacks``.
    This wrapper keeps the same MatGL model and Warp TensorNet layers, but uses
    one ``torch.autograd.grad`` over positions and an affine strain tensor.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.model_config = model.model_config
        self._neighbor_hooks = None

    def make_neighbor_hooks(self):
        if self._neighbor_hooks is None:
            self._neighbor_hooks = self.model.make_neighbor_hooks()
        return self._neighbor_hooks

    def forward(self, batch) -> dict[str, torch.Tensor]:
        from matgl.graph._compute import compute_pair_vector_and_distance

        active = self.model_config.active_outputs & self.model_config.outputs
        compute_forces = "forces" in active
        compute_stress = "stress" in active
        dtype = next(self.model.model.parameters()).dtype
        positions = batch.positions.to(dtype)
        cell_raw = getattr(batch, "cell", None)
        cell = cell_raw.to(dtype) if cell_raw is not None else None

        grad_inputs: list[torch.Tensor] = []
        coord = positions
        scaling = None
        model_positions = positions
        model_cell = cell
        if compute_forces or compute_stress:
            coord = positions.detach().clone().requires_grad_(True)
            model_positions = coord
            grad_inputs.append(coord)
        if compute_stress and cell is not None:
            scaling = (
                torch.eye(3, dtype=dtype, device=positions.device)
                .repeat(batch.num_graphs, 1, 1)
                .requires_grad_(True)
            )
            model_positions = torch.einsum(
                "ni,nij->nj",
                coord,
                scaling[batch.batch_idx],
            )
            model_cell = cell @ scaling
            grad_inputs.append(scaling)

        edge_index = batch.neighbor_list.T
        shifts_raw = getattr(batch, "neighbor_list_shifts", None)
        if shifts_raw is None:
            pbc_offshift = torch.zeros(
                edge_index.shape[1],
                3,
                dtype=dtype,
                device=positions.device,
            )
        else:
            shifts = shifts_raw.to(dtype)
            if model_cell is None:
                pbc_offshift = torch.zeros_like(shifts)
            else:
                src_idx = edge_index[0]
                cell_exp = torch.index_select(model_cell, 0, batch.batch_idx[src_idx])
                pbc_offshift = torch.einsum("bd,bdh->bh", shifts, cell_exp)

        node_type = self.model._z_to_type[batch.atomic_numbers]
        graph = types.SimpleNamespace(
            node_type=node_type,
            pos=model_positions,
            edge_index=edge_index,
            pbc_offshift=pbc_offshift,
            batch=batch.batch_idx,
            num_graphs=batch.num_graphs,
        )
        energy = self.model.model(g=graph)
        energy = self.model.data_std * energy + self.model.data_mean

        if self.model.repuls is not None:
            _, bond_dist = compute_pair_vector_and_distance(
                model_positions,
                edge_index,
                pbc_offshift,
            )
            graph.bond_dist = bond_dist
            energy = energy + self.model.repuls(self.model.model.element_types, graph)

        if energy.dim() == 0:
            energy = energy.unsqueeze(0)
        if energy.dim() == 1:
            energy = energy.unsqueeze(-1)

        out: dict[str, torch.Tensor] = {"energy": energy.detach()}
        if compute_forces or (compute_stress and scaling is not None):
            grads = torch.autograd.grad(
                energy.sum(),
                grad_inputs,
                create_graph=self.training,
                retain_graph=self.training,
            )
            if compute_forces:
                out["forces"] = (-grads[0]).detach()
            if compute_stress and scaling is not None and cell is not None:
                scaling_grad = grads[-1]
                volume = torch.det(cell).abs().view(-1, 1, 1)
                out["stress"] = (scaling_grad / volume).detach()

        if self.model._element_ref_offset is not None:
            atomic_offset = self.model._element_ref_offset[node_type]
            graph_offset = torch.zeros(
                batch.num_graphs,
                device=atomic_offset.device,
                dtype=atomic_offset.dtype,
            )
            graph_offset.scatter_add_(0, batch.batch_idx, atomic_offset)
            out["energy"] = out["energy"] + graph_offset.unsqueeze(-1)

        return out
