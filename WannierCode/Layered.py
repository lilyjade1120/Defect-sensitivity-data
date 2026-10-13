from __future__ import annotations

import numpy as np
import torch

from qimpy import rc, log
from qimpy.io import log_config
from Wannier import Wannier


class Layered:
    """Wannier Hamiltonian layered along a specific direction.
    The perpendicular directions are transfored to reciprocal space."""

    wannier: Wannier  #: Original 3D Wannier representation
    i_dir: int  #: Lattice direction retained after layering (transport direction)
    i_max: int  #: Maximum index along i_dir = number of unit cells / principal layer
    
    _Hwannier: torch.Tensor  #: Subset of original Hwannier used for transform
    _i_cell_out: torch.Tensor  #: Output cell indices for partial Fourier transform
    _iR_perp: torch.Tensor  #: Perpendicular lattice coordinates for transform tensor
    
    def __init__(
        self,
        *,
        wannier: Wannier,
        i_dir: int,
        tol: float = 0.0,
    ) -> None:
        """Set up layering of `wannier` along `i_dir`.
        If non-zero, truncate Hamiltonian at relative tolerance `tol`."""
        self.wannier = wannier
        self.i_dir = i_dir
        
        # Prepare quantities required for sparse transform tensor:
        # Use Hermitian symmetry to only work with iR[i_dir] >= 0
        cell_map = wannier.cell_map
        Hmag = wannier.Hwannier.norm(dim=(-2, -1))
        Hmag_sel = (Hmag > tol * Hmag.max())
        i_cell_in = torch.where(torch.logical_and(cell_map[:, i_dir] >= 0, Hmag_sel))[0]
        self._Hwannier = wannier.Hwannier[i_cell_in]
        self._i_cell_out = cell_map[i_cell_in, i_dir].to(i_cell_in.dtype)
        self.i_max = int(self._i_cell_out.max().item())
        self._iR_perp = cell_map[i_cell_in].clone().detach()
        self._iR_perp[:, i_dir] = 0
        log.info(f"Constructed layered Hamiltonian with {self.i_max} layers at t[s]: {rc.clock():.2f}")

    def Htilde(self, k: torch.Tensor) -> torch.Tensor:
        """Layered Hamiltonian in transverse-Fourier form for wave-vector `k`.
        Output is (i_max + 1) x n_bands x n_bands, containing only the positive
        cell coefficients along i_dir, taking advantage of Hemritian symmetry."""
        phase = torch.exp((2j * np.pi) * (self._iR_perp @ k))
        terms = self._Hwannier * phase[:, None, None]  # terms in Fourier sum
        n_bands = terms.shape[-1]
        Hk = torch.zeros(
            (self.i_max + 1, n_bands, n_bands), dtype=terms.dtype, device=rc.device
        )
        Hk.index_add_(0, self._i_cell_out, terms)
        return Hk

    def to_principal(self, Hk: torch.Tensor) -> torch.Tensor:
        """Transform layered Hamiltonian (e.g. from `Htilde`) to principal layers.
        Output is 2 x n_bands_sup x n_bands_sup, where n_bands_sup = n_bands * i_max,
        representing the diagonal (H00) and super-diagonal (H01) block of the
        block-tridiagonal Hamiltonian in principal layer form."""
        n_bands = Hk.shape[-1]
        i_max = self.i_max
        Hsup = torch.zeros(
            (i_max, 2*i_max, n_bands, n_bands), dtype=Hk.dtype, device=rc.device
        )
        for i_row in range(i_max):
            Hsup[i_row, i_row : (i_row + i_max + 1)] = Hk
            Hsup[(i_row + 1) : i_max, i_row] = hermitian_conj(Hk[1 : (i_max - i_row)])
        Hsup = Hsup.unflatten(1, (2, i_max)).permute(1, 0, 3, 2, 4)
        return Hsup.flatten(3, 4).flatten(1, 2)


def hermitian_conj(matrix: torch.Tensor) -> torch.Tensor:
    """Hermitian conjugate along the last two dimensions."""
    return matrix.swapaxes(-2, -1).conj()


def main() -> None:
    import argparse
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d

    rc.init()
    log_config()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=str, help="Wannier prefix")
    parser.add_argument("i_dir", type=int, help="Layering direction")
    parser.add_argument("k0", type=float, help="k_perp lattice component 0")
    parser.add_argument("k1", type=float, help="k_perp lattice component 1")
    parser.add_argument("k2", type=float, help="k_perp lattice component 2")
    parser.add_argument(
        "--tol", type=float, help="H tolerance", default=0.0, required=False
    )
    args = parser.parse_args()
    
    wannier = Wannier(prefix=args.prefix)
    i_dir = args.i_dir
    k_perp = torch.tensor([args.k0, args.k1, args.k2], device=rc.device)
    layered = Layered(wannier=wannier, i_dir=i_dir, tol=args.tol)

    # Construct partial DOS from original Wannier functions:
    Nk = 1000
    i_max = layered.i_max
    Nk = (Nk // i_max) * i_max  # make multiple of i_max for exact comparisons below
    k_par = torch.arange(Nk) / Nk
    k = k_perp.repeat(Nk, 1)
    k[:, i_dir] = k_par
    E, _ = wannier.interpolate(k, {})
    
    # Plot DOS with gauss smoothing:
    dE = 0.001
    Emin = E.min().item() - 10*dE
    Emax = E.max().item() + 10*dE
    Ebins = torch.arange(Emin, Emax, dE, device=rc.device)
    Emid = (Ebins[:-1] + 0.5*dE).to(rc.cpu).numpy()
    
    def plot_dos(E, label):
        dos = gaussian_filter1d(torch.histogram(E, Ebins)[0].to(rc.cpu).numpy(), 3)
        plt.plot(Emid, dos, label=label)
    
    plot_dos(E, "Original")

    # DOS from layered Hamiltonian
    Htilde = layered.Htilde(k_perp)
    iRpos = 1 + torch.arange(layered.i_max, device=rc.device)  # iR > 0 only
    phase = torch.exp((2j * np.pi) * torch.outer(k_par, iRpos))
    Hk_pos = torch.einsum("kr, rab -> kab", phase, Htilde[1:])
    Hk = Htilde[0] + Hk_pos + hermitian_conj(Hk_pos)
    E = torch.linalg.eigvalsh(Hk)
    plot_dos(E, "Layered")

    # DOS from principal-layer version:
    Hprincipal = layered.to_principal(Htilde)
    Hk_pos = Hprincipal[1] * torch.exp((2j * np.pi) * k_par[::i_max])[:, None, None]
    Hk = Hprincipal[0] + Hk_pos + hermitian_conj(Hk_pos)
    E = torch.linalg.eigvalsh(Hk)
    plot_dos(E, "Principal-layered")
    herm_err = (Hk - hermitian_conj(Hk)).norm() / Hk.norm()
    log.info(f"{herm_err = }")

    plt.xlabel("$E$ ($E_h$)")
    plt.ylabel("$g(E)$ ($E_h^{-1}$)")
    plt.legend()
    
    # Visualize layered hamiltonian
    def plotH(H):
        Hrow = H.swapaxes(-2, -1).flatten(0, 1).abs().T
        plt.figure()
        plt.imshow(torch.log10(Hrow).to(rc.cpu).numpy())
        plt.colorbar(orientation="horizontal", label="log$_{10}|H|$")
    
    plotH(Htilde)
    plotH(Hprincipal)
    plt.show()


if __name__ == "__main__":
    main()
