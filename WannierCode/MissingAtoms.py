from __future__ import annotations
from typing import Union

import numpy as np
import torch

from qimpy import rc, log, io
from qimpy.io import log_config
from Wannier import Wannier


class MissingAtoms:
    """Tight-binding approximation to missing atoms / vacancies.
    Select wannier centers to remove from the Hamiltonian in one or more layers,
    and those entire rows and columns will be zeroed-out in the device Hamiltonian.
    """
    wannier: Wannier
    n_cells: int  #: Number of cells along transport direction in which to delete atoms
    i_delete: torch.Tensor  #: Flattened index of atoms to delete

    def __init__(
        self,
        *,
        wannier: Wannier,
        delete: list[list[int]]
    ) -> None:
        """
        Parameters
        ----------
        wannier
            Reference pristine Hamiltonian in Wanier basis.
        delete
            Indices of wannier centers within each cell to remove,
            with the outer list length setting the defect supercell size.
            Use visualization in MissingAtoms.py to design the defect for now.
            TODO: implement spatial location / atom-proximity-based selection later.
            This could be a  bit complicated as Wannier centers need not only be
            associated with atoms, but also the voids between them.
        """
        self.wannier = wannier
        self.n_cells = len(delete)
        n_bands = wannier.Hwannier.shape[-1]
        for delete_i in delete:
            if len(delete_i) > 0:
                assert (0 <= min(delete_i)) and (max(delete_i) < n_bands)
        self.i_delete = torch.from_numpy(np.concatenate(
            [
                i_cell * n_bands + np.array(delete_i, dtype=int)
                for i_cell, delete_i in enumerate(delete)
            ]
        )).to(rc.device)


    def zero_missing(self, H: torch.Tensor, n_pad: int) -> None:
        """Zero-out the missing Wannier centers in `H`.
        Here, n_pad sets the starting cell within device to perform deletion;
        the number of layers in pristine H will usually exceed the defect size."""
        i_delete_padded = n_pad * self.wannier.Hwannier.shape[-1] + self.i_delete
        H[i_delete_padded, :] = 0.0
        H[:, i_delete_padded] = 0.0


def main() -> None:
    import sys
    import argparse
    import matplotlib.pyplot as plt
    from itertools import groupby
    from Layered import Layered

    rc.init()
    log_config()
    
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file", type=str, help="YAML input file")
    args = parser.parse_args()

    # Load input parameters from YAML file:
    input_dict = io.dict.key_cleanup(io.yaml.load(args.input_file))
    input_dict = io.dict.remove_units(input_dict)
    n_tile = input_dict["n_tile"] if "n_tile" in input_dict else 1
    i_dir = input_dict["i_dir"]
    #delete = input_dict["delete"]
    missing_atoms_list = input_dict["missing_atoms"]
    wannier = Wannier(prefix=input_dict["prefix"], matrix_elements=[])
    layered = Layered(wannier=wannier, i_dir=i_dir)
    #H00, H01 = layered.to_principal(layered.Htilde(torch.zeros(3, device=rc.device)))
    tiled_i_max = n_tile * layered.i_max
    def plotH(H):
        plt.figure()
        plt.imshow(torch.log10(H.abs()).to(rc.cpu).numpy())
        plt.colorbar(orientation="horizontal")

    for m in missing_atoms_list:
        #missing_atoms = MissingAtoms(**m, wannier=wannier, delete=delete)
        missing_atoms = MissingAtoms(**m, wannier = wannier)
        n_pad = (tiled_i_max - missing_atoms.n_cells) // 2

        # 3D visualization of center locations
        centers = wannier.centers.repeat(tiled_i_max, 1, 1)
        centers[..., i_dir] += torch.arange(tiled_i_max, device=rc.device)[:, None]
        centers = centers.flatten(0, 1)
        centers_cart = (centers @ wannier.lattice.Rbasis.T).to(rc.cpu).numpy()
        is_deleted = np.full(len(centers_cart), False)
        is_deleted[n_pad * wannier.Hwannier.shape[-1] + missing_atoms.i_delete] = True

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(*centers_cart[~is_deleted].T, c="k")
        ax.scatter(*centers_cart[is_deleted].T, c="r")  #deleted centers in red
        ax.set_box_aspect(np.ptp(centers_cart, axis=0))
    plt.show()


if __name__ == "__main__":
    main()
