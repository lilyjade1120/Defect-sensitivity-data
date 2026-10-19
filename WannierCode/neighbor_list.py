from __future__ import annotations

from ase import Atoms
import torch
import numpy as np

from qimpy import rc, log
from qimpy.io import log_config
from find_neighbors import FindNeighbors


class NeighborList:
    """Positions and neighbor lists by atom and neighbor type."""
    atoms: Atoms  # original atoms object
    cutoff: float
    pos: dict[int, torch.Tensor]  # positions (Nx3) for each type
    RT: torch.Tensor  # lattice vectors in rows (and in ASE units)
    neighbors: dict[int, dict[int, torch.Tensor]]  # neighbor displacements (NxNneighx3) by atom and neighbor type
    
    def __init__(
        self,
        atoms: Atoms,
        cutoff: float,
        max_neighbors_override: dict[tuple[int, int], int] = None,
        TOL: float = 1E-8,
    ) -> None:
        self.atoms = atoms
        self.cutoff = cutoff
        Z = atoms.numbers
        Zuniq = np.unique(Z)
        self.pos = {
            Zi: torch.from_numpy(atoms.positions[Z == Zi]).to(rc.device)
            for Zi in Zuniq
        }
        log.info(f"Constructing neighbor list for {len(Zuniq)} atom types:")
        
        # Calculate supercell required for cutoff:
        self.RT = RT = torch.from_numpy(atoms.cell[:]).to(rc.device)
        sup = torch.ceil(
            2 * cutoff * torch.linalg.inv(RT).norm(dim=0) + 1
        ).to(torch.int).tolist()
        log.info(f"  Using {sup} supercell to find neighbors within {cutoff = :.1f} A")
        
        # Search neighbors in supercell:
        self.neighbors = {Zi: {} for Zi in Zuniq}
        RsupT = torch.diag(torch.tensor(sup, dtype=torch.float64, device=rc.device)) @ RT
        invRsupT = torch.linalg.inv(RsupT)
        i_sup = torch.stack(
            torch.meshgrid(
                *[
                    torch.arange(sup_i, dtype=torch.float64, device=rc.device)
                    for sup_i in sup
                ],
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 3)
        find_neighbors = FindNeighbors(RsupT, cutoff)
        for Zj, pos_j in self.pos.items():
            pos_j_sup = ((i_sup @ RT)[:, None] + pos_j).reshape(-1, 3)
            for Zi, pos_i in self.pos.items():
                n_atoms = len(pos_i)
                index_i, index_j, neighbors0_ij = find_neighbors(pos_i, pos_j_sup)

                # Convert pair indices to neighbor index by atom:
                counts_i = torch.zeros(n_atoms, dtype=int, device=rc.device)
                counts_i.index_add_(0, index_i, torch.ones_like(index_i))
                max_neighbors = counts_i.max().item()
                first_i = torch.searchsorted(
                    index_i, torch.arange(n_atoms, dtype=int, device=rc.device)
                )
                index_n = torch.arange(len(index_i), dtype=int, device=rc.device) - first_i[index_i]
                
                # Collect displacements:
                displacements = torch.zeros((n_atoms, max_neighbors, 3), device=rc.device)
                displacements[index_i, index_n] = neighbors0_ij
                if max_neighbors_override is None:
                    log.info(f"  Found up to {max_neighbors} type-{Zj} neighbors for {len(pos_i)} type-{Zi} atoms.")
                else:
                    max_neighbors = max_neighbors_override[(Zi, Zj)]
                    distances = displacements.norm(dim=-1)
                    distances[distances < TOL] = np.inf  # push empty displacments to end of sorted list below
                    sort_index = distances.argsort(dim=-1)
                    i_atom = torch.arange(n_atoms, device=rc.device)
                    displacements = displacements[i_atom[:, None], sort_index[:, :max_neighbors]]
                    log.info(f"  Collected up to {max_neighbors} type-{Zj} neighbors for {len(pos_i)} type-{Zi} atoms.")
                self.neighbors[Zi][Zj] = displacements


if __name__ == "__main__":

    def main():
        import argparse
        import ase.io
        
        rc.init()
        log_config()
        
        parser = argparse.ArgumentParser()
        parser.add_argument("filename", type=str, help="Structure file (ASE-compatible)")
        parser.add_argument("cutoff", type=float, help="Neighbor cutoff distance")
        args = parser.parse_args()
        
        atoms = ase.io.read(args.filename)
        NeighborList(atoms, args.cutoff)

    main()
