from __future__ import annotations
from typing import Optional
import itertools

import torch
import torch.distributed as dist
import numpy as np
from ase import Atoms

from qimpy import rc, log
from qimpy.io import log_config, fmt, CheckpointPath
from qimpy.algorithms import Minimize, MatrixArray
from neighbor_list import NeighborList


TOL = 1.0E-5


class Supercell:
    bulk: Atoms  #: bulk crystal structure
    structure: Atoms  #: super-structure: slab, wire, nanoparticle etc.
    rot: torch.Tensor  #: Cartesian rotation from bulk to slab
    i_atom: torch.Tensor  #: bulk atom index for each atom in structure
    cell_bulk: torch.Tensor  #: bulk cell offset for each atom in structure
    bulk_to_surf_frac: torch.Tensor  #: transform of vectors in fractional coordinates

    def __init__(
        self,
        bulk: Atoms,
        structure: Atoms,
        cutoff: float = 0.0,
        center: Optional[torch.Tensor] = None,
    ):
        self.bulk = bulk
        self.structure = structure
        
        # Construct neighbor lists:
        if not cutoff:
            # Use minimum lattice vector length as neighbor cutoff
            cutoff = np.linalg.norm(bulk.cell, axis=1).min() + 0.1
        nl_bulk = NeighborList(bulk, cutoff)
        max_neighbors = {
            (Zi, Zj): neighbors_ij.shape[1]
            for Zi, neighbors_i in nl_bulk.neighbors.items()
            for Zj, neighbors_ij in neighbors_i.items()
        }
        nl_structure = NeighborList(structure, cutoff, max_neighbors)
        
        # --- Find rotation between bulk and structure ---
        Zi, Zj = max(max_neighbors, key=max_neighbors.get)  # combination with most neighbors
        neighbors0_i = nl_bulk.neighbors[Zi]  # bulk reference neighborhood
        neighbors0_ij = neighbors0_i[Zj][0]  # bulk reference neighborhood
        
        # Reorder reference neighborhood so that largest area triplet with central atom comes first:
        n_neighbors = len(neighbors0_ij)
        areas = torch.cross(
            neighbors0_ij[:, None], neighbors0_ij[None], dim=-1
        ).norm(dim=-1)
        j_pair = areas.argmax().item()
        j1 = j_pair // n_neighbors
        j2 = j_pair - j1 * n_neighbors
        rot0 = rotation_from_axes(neighbors0_ij[j1], neighbors0_ij[j2])
        log.info(f"Using type-{Zi}-{Zj} neighborhood with area {areas[j1, j2]:.2f} for rotation referencing.")
        
        # Compare structure to be tested for all rotation pairs:
        neighbors_i = nl_structure.neighbors[Zi]
        neighbors_ij = neighbors_i[Zj]
        j_pairs = [(j1, j2) for j1 in range(n_neighbors) for j2 in range(n_neighbors) if j1 != j2]
        best_metric = None
        best_rot = None
        for j1, j2 in j_pairs:
            rot = rotation_from_axes(neighbors_ij[:, j1], neighbors_ij[:, j2]) @ rot0.T
            metrics = []
            for Zk, neighbors_ik in neighbors_i.items():
                distances = (neighbors_i[Zk] @ rot - neighbors0_i[Zk][0][:, None, None]).norm(dim=-1)
                metrics.append(distances.min(dim=0)[0].mean(dim=-1))
            metric = torch.stack(metrics).mean(dim=0)

            if best_metric is None:
                best_metric = metric
                best_rot = rot
            else:
                improved = torch.where(metric < best_metric)[0]
                best_metric[improved] = metric[improved]
                best_rot[improved] = rot[improved]
        
        i_exact = torch.where(best_metric < TOL)[0]
        n_exact = len(i_exact)
        log.info(f"  Found {n_exact} type-{Zi} atoms with exact neighborhood match")
        if not n_exact:
            log.warn("  WARNING: No surface atom neighborhood matches bulk exactly.")
            i_exact = torch.argmin(best_metric)[None]
            best_match = best_metric[i_exact].item()
            log.info(f"  Using best match with neighborhood error = {best_match:.2g}")
        
        i_ref = i_exact[0]  # index of type-Zi atom in structure used as reference
        pos_ref = nl_structure.pos[Zi][i_ref]
        rot = best_rot[i_ref]  # Cartesian rotation from bulk to structure
        
        # Make rotation commensurate along periodic directions (integers in fractional coordinates):
        surf_to_bulk_frac = nl_structure.RT @ rot @ torch.linalg.inv(nl_bulk.RT)  # for right multiply
        for i_dir, is_periodic_i in enumerate(structure.get_pbc()):
            if is_periodic_i:  # corresponding row must be integers
                surf_to_bulk_frac[i_dir] = torch.round(surf_to_bulk_frac[i_dir])
        rot = torch.linalg.inv(nl_structure.RT) @ surf_to_bulk_frac @ nl_bulk.RT
        
        # Wrap super-structure atoms around truncation center for non-periodic directions:
        pos = torch.from_numpy(structure.get_positions()).to(rc.device)
        if center is not None:
            dpos_frac = pos @ torch.linalg.inv(nl_structure.RT) - center
            for i_dir, is_periodic_i in enumerate(structure.get_pbc()):
                if not is_periodic_i:
                    dpos_frac[:, i_dir] -= torch.round(dpos_frac[:, i_dir])  # wrap to [-0.5, 0.5)
            pos = (center + dpos_frac) @ nl_structure.RT

        # Map positions in structure into bulk fractional coordinates:
        bulk_frac = torch.from_numpy(bulk.get_positions()).to(rc.device) @ torch.linalg.inv(nl_bulk.RT)
        shift = pos_ref - nl_bulk.pos[Zi][0] @ torch.linalg.inv(rot)  # Cartesian shift based on reference atom
        
        # Optimize transformations to best match atoms:
        transform_optimizer = TransformOptimizer(
            surf_to_bulk_frac, shift, nl_bulk.RT, nl_structure.RT, bulk_frac, pos, structure.get_pbc()
        )
        transform_optimizer.minimize()
        rot = transform_optimizer.rot
        shift = transform_optimizer.shift
        errors_mean = transform_optimizer.errors.mean()#.item()
        errors_max = transform_optimizer.errors.max()#.item()
        rot_err = (rot @ rot.T - torch.eye(3, device=rc.device)).norm()

        log.info(f"  Optimized rotation (with orthogonality error = {rot_err:.2g}):\n   {fmt(rot, prefix='   ')}")
        log.info(f"  Optimized shift = {fmt(shift)}")
        log.info(f"  Mapped atoms with error mean: {errors_mean:.2e} A, max: {errors_max:.2e} A")
        self.rot = rot
        self.i_atom = transform_optimizer.i_atom
        self.cell_bulk = transform_optimizer.cell_bulk
        self.bulk_to_surf_frac = nl_bulk.RT @ torch.linalg.inv(rot) @ torch.linalg.inv(nl_structure.RT)
        self.shift_frac = shift @ torch.linalg.inv(nl_structure.RT)
    
    def map_cells(
        self, cell_map_bulk: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[int, int, torch.Tensor, torch.Tensor]]]:
        """Compute super-structure cell map and atom mapping given bulk cell map.
        Returns cell map (Ncells x 3) for super-structure and a list of
        (i_atom_sup1, i_atom_sup2, i_cells_bulk, i_cells_sup), where i_cells_bulk
        and i_cells_sup are the corresponding indices of the bulk and super-structure
        cell map for that pair of super-structure atoms. (The bulk atom indices are
        not stored explicitly, since they can be obtained from self.i_atom.)"""
        i_atoms_sup = range(len(self.structure))
        result = []
        for i_atom_sup1, i_atom_sup2 in itertools.product(i_atoms_sup, i_atoms_sup):
            dcell = (
                cell_map_bulk.to(torch.float64)
                + self.cell_bulk[i_atom_sup1]
                - self.cell_bulk[i_atom_sup2]
            ) @ self.bulk_to_surf_frac
            # Above, cell_map_bulk contains fractional distance of cell2 - cell1 in bulk
            # For supercell, we want atom1 in original cell and atom2 in cell_map_sup
            # Next, atom1 -> cell_bulk1 and atom2 -> cell_bulk2 + to_bulk(cell_map_sup)
            # Hence, cell_map_bulk = cell_bulk2 + to_bulk(cell_map_sup) - cell_bulk1,
            # so that cell_map_sup = to_surf(cell_map_bulk + cell_bulk1 - cell_bulk2).
            # Here, to_bulk and to_surf transform between the two frac coordinates.
            
            # Select integer / zero dcell for periodic / non-periodic directions:
            is_commensurate = None
            for i_dir, (dcell_i, is_periodic) in enumerate(
                zip(dcell.T, self.structure.pbc)
            ):
                if is_periodic:
                    condition = (dcell_i - torch.round(dcell_i)).abs() < TOL
                else:
                    condition = dcell_i.abs() < TOL
                is_commensurate = (
                    condition
                    if (is_commensurate is None)
                    else torch.logical_and(is_commensurate, condition)
                )
            i_cells_bulk = torch.where(is_commensurate)[0]
            dcell_sel = torch.round(dcell[i_cells_bulk]).to(torch.int64)
            result.append((i_atom_sup1, i_atom_sup2, i_cells_bulk, dcell_sel))
        
        # Construct supercell map from unique set of dcells:
        dcell_all = torch.cat([entry[-1] for entry in result], dim=0)
        cell_map_sup, i_cells_sup_all = dcell_all.unique(dim=0, return_inverse=True)
        log.info(
            f"Identified {len(cell_map_sup)} super-cells"
            f" from {len(cell_map_bulk)} bulk cells"
        )
        
        # Replace dcell with index into cell_map_sup in result:
        i_cells_sup_list = i_cells_sup_all.split([len(entry[-1]) for entry in result])
        result = [
            (i_atom_sup1, i_atom_sup2, i_cells_bulk, i_cells_sup)
            for (i_atom_sup1, i_atom_sup2, i_cells_bulk, _), i_cells_sup
            in zip(result, i_cells_sup_list)
        ]
        return cell_map_sup, result

    def map_hamiltonian(
        self,
        Hbulk: torch.Tensor,
        basis_indices: list[torch.Tensor],
        cell_map_sup: torch.Tensor,
        atom_cell_maps: list[tuple[int, int, torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, list[slice]]:
        """Construct tigh-binding Hamiltonian for super-structure from bulk Hamiltonian,
        given bulk basis indices corresponding to each atom and the map_cells outputs.
        """
        # Initialize slices for the bases associated with each i_atom_sup:
        n_bases_sup = [len(basis_indices[i_atom_bulk]) for i_atom_bulk in self.i_atom]
        slice_edges = np.cumsum([0] + n_bases_sup)
        n_basis_sup = slice_edges[-1]
        basis_sup_slices = [
            slice(i_start, i_stop)
            for i_start, i_stop in zip(slice_edges[:-1], slice_edges[1:])
        ]
        
        # Populate super-structure Hamiltonian:
        Hsup = torch.zeros(
            (len(cell_map_sup), n_basis_sup, n_basis_sup),
            dtype=Hbulk.dtype,
            device=Hbulk.device,
        )
        for i_atom_sup1, i_atom_sup2, i_cells_bulk, i_cells_sup in atom_cell_maps:
            basis_bulk1 = basis_indices[self.i_atom[i_atom_sup1]]
            basis_bulk2 = basis_indices[self.i_atom[i_atom_sup2]]
            Hbulk12 = Hbulk[i_cells_bulk][:, basis_bulk1][..., basis_bulk2]
            basis_sup1 = basis_sup_slices[i_atom_sup1]
            basis_sup2 = basis_sup_slices[i_atom_sup2]
            Hsup[i_cells_sup, basis_sup1, basis_sup2] = Hbulk12
        return Hsup, basis_sup_slices


def rotation_from_axes(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Get rotation matrix that brings x-axis to v1, and xy-plane to v1-v2 plane."""
    x_hat = v1 / v1.norm(dim=-1).clip(min=TOL)[..., None]
    y_hat = v2 - torch.einsum("...i, ...i, ...j -> ...j", v2, x_hat, x_hat)
    y_hat = y_hat / y_hat.norm(dim=-1).clip(min=TOL)[..., None]
    z_hat = torch.cross(x_hat, y_hat, dim=-1)
    return torch.stack((x_hat, y_hat, z_hat), dim=-1)


class TransformOptimizer(Minimize[MatrixArray]):
    def __init__(
        self,
        surf_to_bulk_frac: torch.Tensor,
        shift: torch.Tensor,
        RTbulk: torch.Tensor,
        RT: torch.Tensor,
        bulk_frac: torch.Tensor,
        pos: torch.Tensor,
        pbc: tuple[bool, bool, bool],
    ) -> None:
        super().__init__(
            group=dist.group.WORLD,
            checkpoint_in=CheckpointPath(),
            name="TransformOptimize:",
            n_iterations=100,
            energy_threshold=1e-9,
            extra_thresholds={}, #"|grad|": 1e-8},
            n_consecutive=1,
            method="cg",
        )
        self.transform = MatrixArray(
            torch.cat((surf_to_bulk_frac, shift[None]), dim=0), group=self.group 
        )
        self.RTbulk = RTbulk
        self.RT = RT
        self.invRTbulk = torch.linalg.inv(RTbulk)
        self.invRT = torch.linalg.inv(RT)
        self.bulk_frac = bulk_frac
        self.pos = pos
        self.pbc = pbc

    @property
    def surf_to_bulk_frac(self) -> torch.Tensor:
        return self.transform.M[:-1]
    
    @property
    def shift(self) -> torch.Tensor:
        return self.transform.M[-1]

    @property
    def rot(self) -> torch.Tensor:
        return self.invRT @ self.surf_to_bulk_frac @ self.RTbulk
    
    def step(self, direction: MatrixArray, step_size: float) -> None:
        self.transform += step_size * direction

    def compute(self, state: MinimizeState[MatrixArray], energy_only: bool) -> None:
        if not energy_only:
            self.transform.M.requires_grad = True
            self.transform.M.grad = None

        pos_bulk_frac = (self.pos - self.shift) @ self.rot @ self.invRTbulk
        dpos_frac = pos_bulk_frac - self.bulk_frac[:, None, :]
        dpos_frac -= torch.round(dpos_frac)
        distances = (dpos_frac @ self.RTbulk).norm(dim=-1)
        errors, i_atom = distances.min(dim=0)
        cell_bulk = torch.round(pos_bulk_frac - self.bulk_frac[i_atom]).to(torch.int64)
        loss = errors.mean()
        state.energy["loss"] = loss
        
        self.errors = errors.detach()
        self.i_atom = i_atom.detach()
        self.cell_bulk = cell_bulk.detach()
        
        if not energy_only:
            loss.backward()
            state.gradient = MatrixArray(self.transform.M.grad, self.group)
            state.K_gradient = self.constrain(state.gradient)
            state.extra = torch.empty((0,), device=rc.device)
            self.transform.M.requires_grad = False
            self.transform.M.grad = None

    def constrain(self, x: MatrixArray) -> MatrixArray:
        result = MatrixArray(x.M.clone().detach(), x.group)
        for i_dir, is_periodic_i in enumerate(self.pbc):
            if is_periodic_i:  # corresponding row of sulk_to_bulk_frac is constrained
                result.M[i_dir].zero_()
        return result
        

if __name__ == "__main__":

    def main():
        import argparse
        import ase.io
        
        log_config()
        rc.init()
        
        parser = argparse.ArgumentParser()
        parser.add_argument("bulk", type=str, help="Bulk structure file (ASE-compatible)")
        parser.add_argument("structure", type=str, help="Super-structure file (ASE-compatible)")
        parser.add_argument("--pbc", type=int, nargs=3, help="Super-structure preiodicity", default=(1, 1, 1))
        parser.add_argument("--cutoff", type=float, help="Neighbor cutoff distance", default=0.0)
        parser.add_argument("--cellmap", type=str, help="Bulk cell map file", default="")
        args = parser.parse_args()
        
        bulk = ase.io.read(args.bulk)
        structure = ase.io.read(args.structure)
        structure.set_pbc([bool(periodic_i) for periodic_i in args.pbc])
        supercell = Supercell(bulk, structure, args.cutoff)
        
        if args.cellmap:
            cell_map = torch.from_numpy(
                np.loadtxt(args.cellmap, usecols=(0, 1, 2)).astype(int)
            ).to(rc.device)
            supercell.map_cells(cell_map)
        rc.free()

    main()
