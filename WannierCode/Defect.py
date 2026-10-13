from __future__ import annotations

import numpy as np
import torch

from qimpy import rc, log
from qimpy.io import log_config
from Wannier import Wannier


class Defect:
    """Ab initio defect scattering matrix elements."""
    wannier: Wannier  #: wannier basis used for matrix elements
    i_dir: int
    name: str

    n_cells: int  #: number of cells along i_dir
    cell_map: torch.Tensor
    cell_map_perp: torch.Tensor
    sup: np.ndarray
    perp: np.ndarray
    Hdefect: torch.Tensor

    def __init__(
        self,
        *,
        wannier: Wannier,
        i_dir: int,
        name: str,
    ) -> None:
        self.wannier = wannier
        self.name = name
        self.i_dir = i_dir
        prefix = wannier.prefix
        n_bands = wannier.Hwannier.shape[-1]

        # Get defect supercell from wannier log:
        key = f"defect-supercell {name}"
        for line in open(f"{prefix}/wannier.out"):
            if line.startswith(key):
                self.sup = np.array([int(_l) for _l in line.split()[2:5]])
                break
        else:
            raise KeyError(
                f"Defect {name} not fond in {prefix}/wannier.out"
            )
        prod_sup = np.prod(self.sup)

        # Construct perp dimensions from k-point folding:
        self.perp = np.copy(wannier.kfold)
        self.perp[i_dir] = 1
        prod_perp = np.prod(self.perp)

        # Read cell map (must be entirely along i_dir):
        fname = f"{prefix}/wannier.mlwfCellMapD_{name}"
        cell_map = np.loadtxt(fname, usecols=[0, 1, 2]).astype(int)
        sup_stride = np.array([self.sup[1] * self.sup[2], self.sup[2], 1])
        i_reduced = np.dot(np.mod(cell_map, self.sup[None, :]), sup_stride)
        self.cell_map = torch.from_numpy(cell_map.astype(float)).to(rc.device)
        log.info(f"Read {fname}")
        assert (cell_map ** 2).sum() == (cell_map[:, self.i_dir] ** 2).sum()

        # Read corresponding cell weights:
        fname = f"{prefix}/wannier.mlwfCellWeightsD_{name}"
        self.n_cells = cell_map.shape[0]
        cell_map_expected = cell_map[0, self.i_dir] + np.arange(self.n_cells)
        assert np.all(cell_map[:, self.i_dir] == cell_map_expected)
        cell_weights = np.fromfile(fname).reshape(self.n_cells, n_bands)
        log.info(f"Read {fname}")

        # Read perp cell map (must not have any component along i_dir):
        fname = f"{prefix}/wannier.mlwfCellMapPerpD_{name}"
        cell_map_perp = np.loadtxt(fname, usecols=[0, 1, 2]).astype(int)
        perp_stride = np.array([self.perp[1] * self.perp[2], self.perp[2], 1])
        i_reduced_perp = np.dot(np.mod(cell_map_perp, self.perp[None, :]), perp_stride)
        self.cell_map_perp = torch.from_numpy(cell_map_perp.astype(float)).to(rc.device)
        log.info(f"Read {fname}")
        assert (cell_map_perp[:, self.i_dir] **2 ).sum() == 0

        # Read corresponding cell weights:
        fname = f"{prefix}/wannier.mlwfCellWeightsPerpD_{name}"
        n_cells_perp = cell_map_perp.shape[0]
        cell_weights_perp = np.fromfile(fname).reshape(
            n_cells_perp, n_bands, n_bands
        ).swapaxes(-2, -1)
        log.info(f"Read {fname}")

        # Read and unpack Defect Hamiltonian:
        fname_defect_hamiltonian = f"{prefix}/wannier.mlwfHD_{name}"
        Hdefect = np.fromfile(fname_defect_hamiltonian).reshape(
            (prod_sup, prod_sup, prod_perp, n_bands, n_bands)
        ).swapaxes(-1, -2)
        log.info(f"Read {fname_defect_hamiltonian}")
        # --- unpack cell 1
        Hdefect = Hdefect[i_reduced]
        Hdefect *= cell_weights[:, None, None, :, None]
        # --- unpack cell 2
        Hdefect = Hdefect[:, i_reduced]
        Hdefect *= cell_weights[None, :, None, None, :]
        # --- unpack cell_perp
        Hdefect = Hdefect[:, :, i_reduced_perp]
        Hdefect *= cell_weights_perp

        # Reshape Hamiltonian for perp Fourier transform
        Hdefect = torch.from_numpy(Hdefect).to(rc.device).permute(0, 3, 1, 4, 2)
        self.Hdefect = Hdefect.flatten(2, 3).flatten(0, 1).to(torch.complex128)

    def Htilde(self, k: torch.Tensor) -> torch.Tensor:
        """Defect Hamiltonian in transverse-Fourier form for wave-vector `k`.
        Output is (n_cells * n_bands) x (n_cells * n_bands)."""
        phase = torch.exp((2j * np.pi) * (self.cell_map_perp @ k))
        return self.Hdefect @ phase


def main() -> None:
    import argparse
    import matplotlib.pyplot as plt

    rc.init()
    log_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=str, help="Wannier prefix")
    parser.add_argument("name", type=str, help="Defect name")
    parser.add_argument("i_dir", type=int, help="Layering direction")
    parser.add_argument("k0", type=float, help="k_perp lattice component 0")
    parser.add_argument("k1", type=float, help="k_perp lattice component 1")
    parser.add_argument("k2", type=float, help="k_perp lattice component 2")
    args = parser.parse_args()

    i_dir = args.i_dir
    k_perp = torch.tensor([args.k0, args.k1, args.k2], device=rc.device)
    wannier = Wannier(prefix=args.prefix, matrix_elements=[])
    defect = Defect(wannier=wannier, name=args.name,  i_dir=i_dir)
    Hd = defect.Htilde(k_perp)
    print(Hd.shape)

    def plotH(H):
        plt.figure()
        plt.imshow(torch.log10(H.abs()).to(rc.cpu).numpy())
        plt.colorbar(orientation="horizontal")

    plotH(Hd)
    plt.show()


if __name__ == "__main__":
    main()
