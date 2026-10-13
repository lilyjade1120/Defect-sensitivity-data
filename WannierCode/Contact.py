from __future__ import annotations

import numpy as np
import torch

from qimpy import rc, log
from qimpy.io import log_config
from Layered import Layered, hermitian_conj


class Contact:
    """Construct contact self energies form layered Hamiltonian."""
    layered: Layered
    tol: float
    eta: float
    n_iter: int
    verbose: bool

    def __init__(
        self,
        *,
        layered: Layered,
        tol: float,
        eta: float,
        n_iter: int = 100,
        verbose: bool = False,
    ) -> None:
        """Set up to calculate contact Green's functions and self energies
        
        Parameters
        -----------
        layered
            Layered (1D-reduced) Hamiltonian of the contact.
        tol
            Convergence threshold (in Eh) of self energy in Sancho-Rubio iteration.
        eta
            Imaginary offset of energy for Green's function (Lorentzian broadening).
        n_iter
            Maximum iterations of the Sancho-Rubio iteration.
        verbose
            If True, report the converged errors and number of iterations taken.
        """
        self.layered = layered
        self.tol = tol
        self.eta = eta
        self.n_iter = n_iter
        self.verbose = verbose

    def get_self_energies(
        self, k: torch.Tensor, E: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the left and right contact self energies, SigmaL and SigmaR,
        at transverse wave-vector `k` and energy `E`.
        Also return the Hamiltonian H00 at the present `k`.
        """
        Htilde = self.layered.Htilde(k)
        H00, H01 = self.layered.to_principal(Htilde)
        H10 = H01.T.conj()
        gL00 = self.converge_sancho_rubio(H00, H10, E)
        gR00 = self.converge_sancho_rubio(H00, H01, E)
        SigmaL = H10 @ gL00 @ H01
        SigmaR = H01 @ gR00 @ H10
        return SigmaL, SigmaR, H00, H01

    def converge_sancho_rubio(
        self, H00: torch.Tensor, H01: torch.Tensor, E: float
    ) -> torch.Tensor:
        """
        Calculate surface Greens function at energy `E` by the Sancho-Rubio iteration.
        H00 and H01 are the diagonal and super-diagonal blocks in principal-layer form.
        """
        I = torch.eye(len(H00), dtype=H00.dtype, device=rc.device)
        EI = complex(E, self.eta) * I
        epsilonS = H00.clone().detach()  # in-place operations below
        epsilon = H00.clone().detach()  # in-place operations below
        alpha = H01  # reference; should not be used in place
        beta = H01.T.conj()  # possible reference; should not be used in place
        
        for i_iter in range(self.n_iter):
            g = torch.linalg.inv(EI - epsilon)
            g_alpha = g @ alpha
            g_beta = g @ beta
            alpha_g_beta = alpha @ g_beta
            # Update (order ensures that RHS uses previous iteration terms):
            epsilonS += alpha_g_beta
            epsilon += alpha_g_beta + beta @ g_alpha
            alpha = alpha @ g_alpha
            beta = beta @ g_beta
            # Check convergence:
            log.info(f'Iteration {i_iter}: ||alpha_g_beta|| = {alpha_g_beta.norm():.3e}')
            if alpha_g_beta.norm() < self.tol: # * len(H00):
                break
        else:
            raise ValueError(
                f"Greens function did not converge in {self.n_iter} iterations"
            )
        
        # Check and return final result:
        g = torch.linalg.inv(EI - epsilonS)
        if self.verbose:
            err = torch.linalg.norm((EI - H00 - H01 @ g @ H01.T.conj()) @ g - I).item()
            log.info(
                f"Sancho-Rubio method converged with {err = :.2e}"
                f" in {i_iter} iterations for {E = }."
                f"\n ||alpha_g_beta||  = {alpha_g_beta.norm():.3e} < {self.tol}. "
            )
        return g

def incoherentTransmission_fast(
    SigmaL: torch.Tensor, SigmaR: torch.Tensor, Harray: list[torch.Tensor], E: float
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Compute transmission at energy `E`, given contact self energies
    `SigmaL` and `SigmaR`, and device Hamiltonian `H`."""
    GammaL = 1j * (SigmaL - SigmaL.T.conj())
    GammaR = 1j * (SigmaR - SigmaR.T.conj())

    eigsGammaL, vecsGammaL = torch.linalg.eigh(GammaL)
    eigsGammaL = torch.clip(eigsGammaL, min = 0)

    GammaLhalf = (vecsGammaL @ torch.diag(torch.sqrt(eigsGammaL)).type(torch.complex128) @ (vecsGammaL).conj().T).to(rc.device)
    Targ = torch.zeros_like(Harray[0])
    for H in Harray:
        EI = E * torch.eye(len(H), dtype=H.dtype, device=rc.device)
        Gr = torch.linalg.inv(EI - H - SigmaL - SigmaR)  # retarded Greens function
        Ga = Gr.T.conj()  # advanced Greens function
        Targ += (GammaLhalf @ Gr @ GammaR @ Ga @ GammaLhalf) / len(Harray)
    T = torch.trace(Targ).item().real
    eigs = torch.linalg.eigvalsh(Targ)
    return T, eigs, None 

def incoherentTransmission(
    SigmaL: torch.Tensor, SigmaR: torch.Tensor, Harray: list[torch.Tensor], E: float
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Compute transmission at energy `E`, given contact self energies
    `SigmaL` and `SigmaR`, and device Hamiltonian `H`."""
    GammaL = 1j * (SigmaL - SigmaL.T.conj())
    GammaR = 1j * (SigmaR - SigmaR.T.conj())
    #eigsGammaL, vecsGammaL = torch.linalg.eig(GammaL)
    #GammaLhalf = vecsGammaL @ torch.diag(torch.sqrt(eigsGammaL)).type(torch.complex128) @ (vecsGammaL).conj().T
    #Targ = torch.zeros_like(Harray[0])
    Aarg = torch.zeros_like(Harray[0])
    Barg = torch.zeros_like(Harray[0])
    for H in Harray:
        EI = E * torch.eye(len(H), dtype=H.dtype, device=rc.device)
        Gr = torch.linalg.inv(EI - H - SigmaL - SigmaR)  # retarded Greens function
        Ga = Gr.T.conj()  # advanced Greens function
        Aarg += Ga @ GammaL @ Gr / len(Harray)
        Barg += Ga @ GammaR @ Gr / len(Harray)
        #Targ += (GammaLhalf @ Gr @ GammaR @ Ga @ GammaLhalf) / len(Harray)
    #T = torch.trace(Targ).item().real
    eigsA, vecsA = transformDevice(Aarg, GammaR)
    eigsB, vecsB = transformDevice(Barg, GammaL) # check Trace
    T = eigsA.sum().item()
    #Tb = eigsB.sum().item()
    keep = torch.where(eigsA > 0.1)[0]
    return T, eigsA, torch.hstack((vecsA[:, keep], vecsB[:, keep]))

def transformDevice(Aarg, Gamma):
    assert isHermitian(Aarg)
    lbda, U = torch.linalg.eigh(Aarg)
    lbda[torch.where(lbda<0)[0]] = 0
    V = U * torch.sqrt(lbda)[None,:] # not unitary anymore
    arg = V.T.conj() @ Gamma @ V
    assert isHermitian(arg)
    eigs, eigvT = torch.linalg.eigh(arg)
    vecs = V @ eigvT
    return eigs, vecs

def transmission(
    SigmaL: torch.Tensor, SigmaR: torch.Tensor, H: torch.Tensor, E: float
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Compute transmission at energy `E`, given contact self energies
    `SigmaL` and `SigmaR`, and device Hamiltonian `H`."""
    EI = E * torch.eye(len(H), dtype=H.dtype, device=rc.device)
    Gr = torch.linalg.inv(EI - H - SigmaL - SigmaR)  # retarded Greens function
    Ga = Gr.T.conj()  # advanced Greens function
    GammaL = 1j * (SigmaL - SigmaL.T.conj())
    GammaR = 1j * (SigmaR - SigmaR.T.conj())

    eigsGammaL, vecsGammaL = torch.linalg.eigh(GammaL)
    eigsGammaL = torch.clip(eigsGammaL, min = 0)

    GammaLhalf = vecsGammaL @ torch.diag(torch.sqrt(eigsGammaL)).type(torch.complex128) @ (vecsGammaL).conj().T
    GammaLhalf = GammaLhalf.to(device=rc.device)
    Targ = GammaLhalf @ Gr @ GammaR @ Ga @ GammaLhalf
    T = torch.trace(Targ).item().real
    eigs, vecs = torch.linalg.eigh(Targ)

    return T, eigs, vecs[:, torch.where(eigs>0.9)[0]].to(device=rc.device)

def isHermitian(matrix: torch.Tensor) -> bool:
    return torch.allclose(matrix, matrix.conj().transpose(0, 1))


def main() -> None:
    import argparse
    import matplotlib.pyplot as plt
    from Wannier import Wannier

    rc.init()
    log_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", type=str, help="Wannier prefix")
    parser.add_argument("i_dir", type=int, help="Layering direction")
    parser.add_argument("k0", type=float, help="k_perp lattice component 0")
    parser.add_argument("k1", type=float, help="k_perp lattice component 1")
    parser.add_argument("k2", type=float, help="k_perp lattice component 2")
    parser.add_argument("E", type=float, help="energy to evaluate g at")
    parser.add_argument("eta", type=float, help="broadening (i.e. Im(E))")
    parser.add_argument("tol_g", type=float, help="g convergence tolerance")
    parser.add_argument(
        "--tolH", type=float, help="H tolerance", default=0.0, required=False
    )
    parser.add_argument(
        "--Nk_par", type=int, default=0, required=False,
        help="# k-points along transport direction for v-based alternate calculation",
    )
    args = parser.parse_args()
    
    wannier = Wannier(prefix=args.prefix)
    i_dir = args.i_dir
    k_perp = torch.tensor([args.k0, args.k1, args.k2], device=rc.device)
    E = args.E
    layered = Layered(wannier=wannier, i_dir=i_dir, tol=args.tolH)
    contact = Contact(layered=layered, tol=args.tol_g, eta=args.eta, verbose=True)

    SigmaL, SigmaR, H00 = contact.get_self_energies(k_perp, E)
    T, _ = transmission(SigmaL, SigmaR, H00, E)
    log.info(f"Transmission {T = }")

    if args.Nk_par:
        # Compute equivalent ballistic transmission from velocity:
        Nk = args.Nk_par
        n_dir = wannier.lattice.Gbasis[:, i_dir]  # reciprocal lattice direction
        n_hat = n_dir / n_dir.norm()  # unit vector along it
        k_par = torch.arange(Nk) / Nk
        k = k_perp.repeat(Nk, 1)
        k[:, i_dir] = k_par
        eig, mat_elem = wannier.interpolate(k, ["P"])
        velocity_n = torch.einsum("kiaa -> kai", mat_elem["P"]).real @ n_hat
        weights = (1 / ((np.pi * Nk) * (eig - complex(E, args.eta)))).imag  # Lorentzian
        GL = (weights * velocity_n.abs()).sum().item() * 0.5  # Landauer formula
        Talt = GL * n_dir.norm().item()  # length factor from 2D vs 3D BZ integration
        log.info(f"Transmission {Talt = }")

    # Visualize Greens functions / self energies
    def plot_matrix(g, label):
        plt.figure()
        plt.imshow(torch.log10(g.abs().T).to(rc.cpu).numpy())
        plt.colorbar(orientation="horizontal", label="log$_{10}$|" + label + "|")
    
    plot_matrix(SigmaL, "$\Sigma_L$")
    plot_matrix(SigmaR, "$\Sigma_R$")
    
    plt.show()



if __name__ == "__main__":
    main()
