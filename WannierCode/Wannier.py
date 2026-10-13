from __future__ import annotations
from typing import Sequence, Optional
import os
import copy

from ase import Atoms
import numpy as np
import torch

from qimpy import rc, log
from qimpy.io import log_config, Unit
from qimpy.lattice import Lattice
from supercell import Supercell

class Wannier:
    """Wannier interpolator of Hamiltonian and matrix elements."""

    prefix: str  #: Directory containing wannier files.
    spin_suffix: str  #: "Up" or "Dn" for spin-polarized cases, "" otherwise.

    kfold: tuple[int, ...]  #: k-point folding
    n_spins: int  #: number of spin channels
    n_spinor: int  #: spinor length
    spin_weight: float  #: integration weight due to spin degeneracy
    mlwf_type: np.dtype  #: data-type for Wannier functions
    mu: float  #: chemical potential

    lattice: Lattice
    atoms: Atoms

    cell_map: torch.Tensor  #: Lattice indices for Wannier Hamiltonian (N x 3)
    Hwannier: torch.Tensor  #: Wannier Hamiltonian (N x Nbands x Nbands)
    matrix_elements_wannier: dict[str, torch.Tensor]  #: Wannier matrix elements
    centers: torch.Tensor  #: Wannier center fractional coordinates (Nbands x 3)

    def __init__(
        self,
        *,
        prefix: str,
        spin_suffix: str = "",
        matrix_elements: Sequence[str] = tuple(),
    ) -> None:
        """Read Wannier Hamiltonian.

        Parameters
        ----------
        prefix
            Directory containing wannier files.
        spin_suffix
            "Up" or "Dn" for spin-polarized cases, "" otherwise.
        matrix_elements
            Name of the required matrix elements
        """
        self.prefix = prefix
        self.spin_suffix = spin_suffix
        self._read_totalE()
        self._read_wannier(matrix_elements)

    def _read_totalE(self) -> None:
        """Get mu/VBM, kfold, lattice vectors and spin type from totalE.out."""
        mu = np.nan
        init_done = False
        R = []
        symbols = []
        positions = []
        ref_line = -10
        Rdone = False
        pbc = [True, True, True]

        for i_line, line in enumerate(open(f'{self.prefix}/totalE.out')):

            if line.startswith('Initialization completed'):
                init_done = True
            
            if init_done and line.find('FillingsUpdate:') >= 0:
                self.mu = float(line.split()[2])
            
            if line.startswith('coords-type'):
                is_cartesian = (line.split()[-1].lower() == "cartesian")

            if line.startswith('coulomb-interaction'):
                pbc_type = line.split()[1].lower()
                if pbc_type == 'slab':
                    pbc = [(int(condition) == 0) for condition in line.split()[2]]
                elif pbc_type in {'wire', 'cylindrical'}:
                    pbc = [(int(condition) != 0) for condition in line.split()[2]]
                elif pbc_type in {'isolated', 'spherical'}:
                    pbc = [False, False, False]
                # else Periodic, which is the default
            
            if (not init_done) and line.startswith('spintype'):
                spin_key = line.split()[1]
                if spin_key == "vector-spin" or spin_key == "spin-orbit":
                    self.mlwf_type = np.complex128
                    n_spinor = 2
                else:
                    self.mlwf_type = np.float64
                    n_spinor = 1
                if spin_key == "z-spin":
                    n_spins = 2
                    assert self.spin_suffix in {"Up", "Dn"}
                else:
                    n_spins = 1
                    assert self.spin_suffix == ""
                self.spin_weight = 2.0 / (n_spins * n_spinor)
                self.n_spins = n_spins
                self.n_spinor = n_spinor
                log.info(f"{n_spins = }  {n_spinor = }")
            
            if (not init_done) and line.startswith('kpoint-folding'):
                self.kfold = tuple(int(tok) for tok in line.split()[1:4])
                log.info(f"kfold = {self.kfold}")
            
            if (not init_done) and line.startswith('n_electrons:'):
                n_electrons = float(line.split()[1])
                n_bands_dft = int(line.split()[3])
                n_states = int(line.split()[5])
                n_valence = int(n_electrons / 2)  # number of valence bands
                Edft = np.reshape(np.fromfile(
                    f'{self.prefix}/totalE.eigenvals'), (n_states, n_bands_dft)
                )
                self.mu = np.max(Edft[:, :n_valence])  # VBM
            
            if (not init_done) and line.startswith('ion '):
                symbols.append(line.split()[1])
                positions.append([float(pos) for pos in line.split()[2:-1]])
            
            if line.find('Initializing the Grid') >= 0:
                ref_line = i_line
            if not Rdone:
                row_num = i_line - (ref_line + 2)
                if row_num >= 0 and row_num < 3:
                    R.append([float(x) for x in line.split()[1:-1]])
                if row_num == 3:
                    Rdone = True
        
        if os.path.exists(f'{self.prefix}/WannierMu.out'):
            with open(f'{self.prefix}/WannierMu.out') as file:
                for line in file:

                    if line.startswith('Wannier mu'):
                        self.mu = float(line.split()[2])

        log.info(f"mu = {self.mu}")
        
        # Convert geometry:
        R = np.array(R)
        positions = np.array(positions).reshape(-1, 3)
        if is_cartesian:
            positions = positions @ np.linalg.inv(R.T)
        
        self.lattice = Lattice(Rbasis=R, periodic=pbc)
        self.atoms = Atoms(
            symbols, scaled_positions=positions, pbc=pbc, cell=R.T / Unit.MAP["Angstrom"]
        )
        log.info(f"Read atoms: {self.atoms} with fractional positions:")
        log.info(f"{self.atoms.get_scaled_positions(wrap=False)}")

    def _read_wannier(self, matrix_elements: Sequence[str]) -> None:
        """Read Wannier Hamiltonian and matrix elements."""
        # Read the wannier cell map:
        fname_cell_map = f"{self.prefix}/wannier.mlwfCellMap{self.spin_suffix}"
        cell_map = np.loadtxt(fname_cell_map, usecols=[0, 1, 2]).astype(int)
        self.cell_map = torch.from_numpy(cell_map.astype(float)).to(rc.device)
        log.info(f"Read {fname_cell_map}")

        # Read Wannier weights:
        fname_Wwannier = f"{self.prefix}/wannier.mlwfCellWeights{self.spin_suffix}"
        Wwannier = np.fromfile(fname_Wwannier)
        n_cells = cell_map.shape[0]
        n_bands = int(np.sqrt(Wwannier.shape[0] / n_cells))
        Wwannier = torch.from_numpy(Wwannier.reshape(
            n_cells, n_bands, n_bands
        ).swapaxes(-2, -1)).to(rc.device)
        log.info(f"Read {fname_Wwannier}")

        # Read and expand Wannier Hamiltonian:
        k_stride = np.array([self.kfold[1] * self.kfold[2], self.kfold[2], 1])
        i_reduced = np.mod(cell_map, self.kfold) @ k_stride
        self.Hwannier = (
                Wwannier *
                torch.from_numpy(
                    self._read_matrix("H", (-1, n_bands, n_bands))[i_reduced]
                ).to(rc.device)
        )

        # Read and expand requested matrix elements:
        n_matrix_elements = {"P": 3, "S": 3, "ImSigma_ePh": 5, "ImSigma_PePh": 5}
        self.matrix_elements_wannier = {}
        for matrix_element in matrix_elements:
            self.matrix_elements_wannier[matrix_element] = (
                    Wwannier[:, None] *
                    torch.from_numpy(
                        self._read_matrix(
                            matrix_element,
                            (-1, n_matrix_elements[matrix_element], n_bands, n_bands),
                        )[i_reduced],
                    ).to(rc.device)
            )
            if matrix_element == "P":
                self.matrix_elements_wannier[matrix_element] *= 1j

        # Read Wannier centers:
        fname = f"{self.prefix}/wannier.out"
        centers_started = False
        centers = []
        for line in open(fname):
            if centers_started:
                tokens = line.split()
                if (len(tokens) > 5) and tokens[5] == 'spread:':
                    centers.append([float(token) for token in tokens[1:4]])
                else:
                    break
            if line.startswith('Centers in '):
                is_cartesian = (line.split()[2].lower() == "cartesian")
                centers_started = True
        # --- ensure farctional coordinates
        self.centers = torch.tensor(centers, device=rc.device)
        if is_cartesian:
            self.centers = self.centers @ torch.linalg.inv(self.lattice.Rbasis.T)
        assert len(self.centers) == self.Hwannier.shape[-1]
        log.info(f"Read {len(self.centers)} centers from {fname} at t[s]: {rc.clock():.2f}")

    def _read_matrix(self, var_name: str, shape: tuple[int, ...]) -> np.ndarray:
        fname = f"{self.prefix}/wannier.mlwf{var_name}{self.spin_suffix}"
        data = np.fromfile(fname, dtype=self.mlwf_type).reshape(shape)
        log.info(f"Read {fname}")
        return data.swapaxes(-2, -1).astype(np.complex128)  # JDFTx to numpy order

    def interpolate(
            self, k: torch.Tensor, matrix_elements: Sequence[str] = tuple()
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Get energies and specified matrix elements at `k`."""
        phase = torch.exp((2j * np.pi) * (k @ self.cell_map.T))
        E, V = torch.linalg.eigh(
            torch.einsum("...r, rab -> ...ab", phase, self.Hwannier)
        )
        matrix_elements_real = {
            matrix_element: torch.einsum(
                "...ba, ...r, ribc, ...cd -> ...iad",
                V.conj(), phase, self.matrix_elements_wannier[matrix_element], V
            )
            for matrix_element in matrix_elements
        }
        return E, matrix_elements_real

    def plot_bandstructure(self, n_interp: int = 10) -> None:
        import os
        import re
        
        import matplotlib.pyplot as plt
        from scipy.interpolate import interp1d

        # Read band structure k-points and make finer:
        kpoints_in = np.loadtxt(
            f'{self.prefix}/bandstruct.kpoints', skiprows=2, usecols=(1, 2, 3)
        )
        nKin = kpoints_in.shape[0]
        xIn = np.arange(nKin)
        x = (1. / n_interp) * np.arange(1 + n_interp * (nKin - 1))
        kpoints = torch.from_numpy(interp1d(xIn, kpoints_in, axis=0)(x)).to(rc.device)

        # Calculate energies by Wannier interpolation:
        E, _ = self.interpolate(kpoints, [])
        
        # Read reference energies if available:
        self.DFTexists = False
        if os.path.exists(Edft_filename := f"{self.prefix}/bandstruct.eigenvals"):
            Edft = np.fromfile(Edft_filename).reshape(nKin, -1)
            self.DFTexists = True
            # Correct the mu for the TB case: 
            mu = None
            for line in open(f'{self.prefix}/totalE.out'):
                if line.find('FillingsUpdate:') >= 0:
                    mu = float(line.split()[2])
            if mu is not None:
                Edft += (self.mu - mu)  # shift energies to the scale in self
            plt.plot(np.arange(0, len(kpoints), n_interp), (Edft - self.mu) / Unit.MAP['eV'] , "k", lw=1, zorder = 1)

        # Plot:
        plt.plot((E.to(rc.cpu).numpy() - self.mu) / Unit.MAP['eV'], 'r:', lw = 1, zorder = 2)
        plt.xlim(0, len(kpoints) - 1)
        Emin = E.min().item()
        Emax = E.max().item()
        margin = 0.05 * (Emax - Emin)
        #plt.ylim(Emin - margin, Emax + margin)
        plt.ylim(-2, 2)
        plt.ylabel(r"E -E$_F$ (eV)")
        # --- read and add plot labels:
        for line in open(f'{self.prefix}/bandstruct.plot'):
            if line.startswith('set xtics'):
                tokens = re.split('[ ,"]+', line)
                xtic_labels = [
                    (r'$\Gamma$' if token == 'Gamma' else token) for token in tokens[3:-1:2]
                ]
                xtic_pos = [int(token) * n_interp for token in tokens[4:-1:2]]
                plt.xticks(xtic_pos, xtic_labels)
        # --- annotate mu or VBM as applicable:
        plt.axhline(0, linestyle='dashed', color='black', linewidth=1)

    def make_tight_binding(self, structure: Atoms, cutoff: float = 0) -> tuple[Wannier, list[list[int]]]:
        """Make a tight-binding model for `structure` using self as the bulk."""
        # Construct supercell mapping:
        supercell = Supercell(self.atoms, structure, cutoff, torch.zeros(3, device = rc.device)) # HACK ALERT!!!
        # TODO: make center settable by user
        cell_map_sup, atom_cell_maps = supercell.map_cells(self.cell_map)
        
        # Map centers to atoms:
        pos = torch.from_numpy(self.atoms.get_scaled_positions(wrap=False)).to(rc.device)
        dpos = pos[:, None, :] - self.centers
        dpos -= torch.round(dpos)  # minimum image convention
        # TODO: handle or raise error if centers are unit cells away from atoms
        distances = (dpos @ self.lattice.Rbasis.T).norm(dim=-1)
        i_closest_atom = distances.argmin(dim=0)  # closest atom index for each center
        center_indices = [
            torch.where(i_closest_atom == i_atom)[0] for i_atom in range(len(pos))
        ]
        
        # Construct supercell Hamiltonian:
        Hsup, basis_sup_slices = supercell.map_hamiltonian(
            self.Hwannier, center_indices, cell_map_sup, atom_cell_maps
        )
        
        # Wrap it within another Wannier object:
        wannier_sup = copy.copy(self)
        wannier_sup.lattice = Lattice(
            Rbasis=structure.cell[:].T * Unit.MAP["Angstrom"], periodic=structure.get_pbc()
        )
        wannier_sup.atoms = structure
        wannier_sup.cell_map = cell_map_sup.to(torch.float64)
        wannier_sup.Hwannier = Hsup
        wannier_sup.matrix_elements_wannier.clear()
        # TODO: better handle other wannier_sup entries that may be invalid 
        return wannier_sup, basis_sup_slices


def main() -> None:
    """Test Wannier read and interpolation with band structure."""
    import argparse
    import os

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import ase.io

    log_config()
    rc.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=str, help="Wannier prefix")
    parser.add_argument("--structure", type=str, help="Structure for tight-binding model", default="")
    parser.add_argument("--pbc", type=int, nargs=3, help="Super-structure preiodicity", default=(1, 1, 1))
    parser.add_argument("--cutoff", type=float, help="cutoff", default=0.0)
    args = parser.parse_args()
    
    plt.figure()
    wannier = Wannier(prefix=args.prefix)
    wannier.plot_bandstructure()
    plt.savefig('bulk.png')

    if args.structure:
        plt.figure()
        structure = ase.io.read(args.structure)
        structure.set_pbc([bool(periodic_i) for periodic_i in args.pbc])
        wannier_sup, basis_sup_slices = wannier.make_tight_binding(structure, args.cutoff)
        log.info(f"Basis slices by atom: {basis_sup_slices}")
        wannier_sup.prefix = os.path.dirname(args.structure)
        wannier_sup.plot_bandstructure()
        DFThandle = Line2D([], [], color = 'k', ls = 'solid', label = 'DFT')
        TBhandle = Line2D([], [], color = 'r', ls = ':', label = 'Tight Binding')
        handles = [TBhandle]
        if wannier_sup.DFTexists:
            handles.append(DFThandle)
        plt.legend(handles = handles, loc = 'upper right')
        plt.savefig('supercell.png')
        plt.savefig('supercell.pdf')
    plt.show()
    rc.free()


if __name__ == "__main__":
    main()
