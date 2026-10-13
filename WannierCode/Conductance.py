from __future__ import annotations
from typing import Sequence, Optional, Union
import argparse
import os

import numpy as np
import torch
import torch.distributed as dist

from qimpy import rc, log, io
from qimpy.mpi import TaskDivision
from Wannier import Wannier
from Defect import Defect
from MissingAtoms import MissingAtoms
from Layered import Layered, hermitian_conj
from Contact import Contact, transmission, incoherentTransmission, incoherentTransmission_fast


class Conductance:
    """Calculate transmission and small-potential conductance using NEGF."""
    i_dir: int  #: Transport direction
    Nk: tuple(int, int, int)  #: k-point mehs dimensions
    dmu: float  #: Energy offset from neutral chemical potential.
    n_tile: int #:
    wannier: Wannier  #: Bulk Hamiltonian in Wannier basis
    defect: Optional[Defect]  #: Defect matrix elements in Wannier basis
    missing_atoms: list[MissingAtoms]  #: Tight-binding atom-removed approximation
    layered: Layered  #: Layered (1D-reduced hamiltonian)
    contact: Contact  #: Contact Greens function calculator

    def __init__(
        self,
        *,
        i_dir: int,
        Nk: Sequence[int],
        dmu: float = 0.0,
        n_tile: int = 1,
        wannier: Union[dict, Wannier],
        defect: Optional[dict] = None,
        missing_atoms: Optional[list[dict]] = None,
        layered: dict,
        contact: dict,
        checkpoint: str = "",
    ) -> None:
        """
        Parameters
        ----------
        i_dir
            Lattice direction number for transport (0-based)
        dmu
            Offset from neutral mu at which to calculate all properties.
        wannier
            Parameters to initialize Wannier hamiltonian.
            All arguments except matrix_elements may be specified (set to empty).
        defect
            Parameters to initialize defect matrix elements,
            except `wannier` and `i_dir` that are passed automatically.
        missing_atoms
            Parameters to initialize missing-atom tight-binding defect approximation,
            except `wannier` that is passed automatically.
        layered
            Parameters to initialize layered (1D-reduced) Hamiltonian,
            except `wannier` and `i_dir` that are passed automatically.
        contact
            Paramters to initialize contact Greens function calculator,
            except `layered` that is passed automatically.
        checkpoint
            If provided, write results to this HDF5 file.
        """
        
        log.info(f"torch version: {torch.__version__}")

        self.i_dir = i_dir
        assert len(Nk) == 3
        assert Nk[i_dir] == 1
        self.Nk = tuple(Nk)
        self.dmu = dmu
        self.n_tile = n_tile
        self.wannier = Wannier(**wannier) if isinstance(wannier, dict) else wannier
        self.layered = Layered(**layered, wannier=self.wannier, i_dir=i_dir)
        self.tiled_i_max = self.n_tile * self.layered.i_max
        if defect is None:
            self.defect = None
        else:
            self.defect = Defect(**defect, wannier=self.wannier, i_dir=i_dir)
            n_bands = self.wannier.Hwannier.shape[-1]
            i_defect_start = n_bands * ((self.tiled_i_max - self.defect.n_cells) // 2)
            assert i_defect_start >= 0  # i.e. i_max >= n_cells
            i_defect_stop = i_defect_start + self.defect.n_cells * n_bands
            self._defect_slice = slice(i_defect_start, i_defect_stop)
        self.missing_atoms = []
        if missing_atoms is not None:
            for m in missing_atoms:
                self.missing_atoms.append(MissingAtoms(**m, wannier=self.wannier))
                self._n_pad = (self.tiled_i_max - self.missing_atoms[-1].n_cells) // 2
                assert self._n_pad >= 0  # i.e. i_max >= n_cells
            log.info("Calculating incoherent sum of %d defect structures" %(len(missing_atoms)))
        self.contact = Contact(**contact, layered=self.layered)
        self.checkpoint = checkpoint

    def compute(self) -> tuple[float, torch.Tensor]:
        """
        Calculate the conductance and k-resolved transmission.
        Results are also saved to `self.checkpoint`, if specified.
        """
        # Create flattened list of k-points:
        k_grids_1d = [torch.arange(Nki, device=rc.device) / Nki for Nki in self.Nk]
        k = torch.stack(torch.meshgrid(*tuple(k_grids_1d), indexing="ij")).flatten(1).T
        k_div = TaskDivision(
            n_tot=len(k), n_procs=rc.n_procs, i_proc=rc.i_proc, name="k"
        )

        n_tile = self.n_tile
        n_bands_sup = self.wannier.Hwannier.shape[-1] * self.layered.i_max
        # Compute transmission for each k:
        E = self.wannier.mu + self.dmu
        log.info(f"Calculating dmu = {self.dmu}")
        T = torch.zeros(k_div.n_tot, device=rc.device)
        eigs = torch.zeros((k_div.n_mine, n_bands_sup * n_tile), dtype=torch.complex64, device=rc.device)
        T_pristine = torch.zeros_like(T)
        eigs_pristine = torch.zeros_like(eigs)
        vecs_sizes = torch.zeros(k_div.n_tot, dtype=int, device=rc.device)
        vecs = []
        vecs_pristine = []
        for ik in range(k_div.i_start, k_div.i_stop):
            # Compute Greens functions and transmission:
            SigmaL, SigmaR, H00, H01 = self.contact.get_self_energies(k[ik], E)
            if self.n_tile > 1:
                H = torch.zeros(
                    (n_tile, n_tile, n_bands_sup, n_bands_sup), dtype=H00.dtype, device=rc.device
                )
                for i_row in range(n_tile):
                    H[i_row, i_row] = H00
                    if i_row < n_tile -1:
                        H[i_row, i_row + 1] = H01
                    if i_row > 0:
                        H[i_row, i_row - 1] = hermitian_conj(H01)
                H = H.permute(0, 2, 1, 3)
                H = H.flatten(2, 3).flatten(0, 1)
                SigmaL_new = torch.zeros_like(H)
                SigmaR_new = torch.zeros_like(H)
                SigmaL_new[:n_bands_sup,:n_bands_sup] = SigmaL
                SigmaR_new[-n_bands_sup:, -n_bands_sup:] = SigmaR
                SigmaL = SigmaL_new
                SigmaR = SigmaR_new
            else:
                H = H00
            if self.defect is not None:
                Hd = self.defect.Htilde(k[ik])
                H[self._defect_slice, self._defect_slice] += Hd
            if self.missing_atoms:
                Harray = []
                for i_del in range(len(self.missing_atoms)):
                    Hcopy = H.clone()
                    self.missing_atoms[i_del].zero_missing(Hcopy, self._n_pad)
                    Harray.append(Hcopy)
                T[ik], _eig, _vec = incoherentTransmission_fast(SigmaL, SigmaR, Harray, E)
                #T[ik], _eig, _vec = incoherentTransmission(SigmaL, SigmaR, Harray, E)
                eigs[ik - k_div.i_start] = _eig
                #vecs_sizes[ik] = _vec.shape[1]
                #vecs.append(_vec)
            T_pristine[ik], _eig, _vec = transmission(SigmaL, SigmaR, H, E)
            eigs_pristine[ik - k_div.i_start] = _eig
            #vecs_pristine.append(_vec)
            # Report progress:
            ik_done = ik - k_div.i_start + 1
            percent_done = ik_done * 100.0 / k_div.n_mine
            log.info(
                f"Calculated {ik_done} of {k_div.n_mine} k-points ({percent_done:.2f}%)"
                f" at t[s]: {rc.clock():.2f}"
            )
        if rc.n_procs > 1:
            dist.all_reduce(T)
            dist.all_reduce(T_pristine)
            dist.all_reduce(vecs_sizes)
        T = T.reshape(self.Nk)
        T_pristine = T_pristine.reshape(self.Nk)
        eigs = torch.real(eigs)
        eigs_pristine = torch.real(eigs_pristine)
        # Compute conductance:
        Rbasis = self.wannier.lattice.Rbasis
        j_dir = (self.i_dir + 1) % 3
        k_dir = (self.i_dir + 2) % 3
        Area = torch.linalg.cross(Rbasis[:, j_dir], Rbasis[:, k_dir]).norm().item()
        Gprefac = self.wannier.spin_weight / (2 * np.pi * Area)
        G = Gprefac * torch.mean(T).item()
        G_pristine = Gprefac * torch.mean(T_pristine).item()
        log.info(f"Conductance, {G = } a.u.")
        log.info(f"Pristine Conductance, {G_pristine = } a.u.")

        # Write to checkpoint:
        if self.checkpoint:
            with io.Checkpoint(self.checkpoint, writable=True) as cp:
                cp.attrs["Rbasis"] = Rbasis.to(rc.cpu)
                cp.attrs["Gprefac"] = Gprefac
                cp.attrs["G"] = G
                cp.attrs["G_pristine"] = G_pristine
                cp["T"] = T.to(rc.cpu).numpy()
                cp["T_pristine"] = T_pristine.to(rc.cpu).numpy()
                for _name, _data in zip(["eigs", "eigs_pristine"], [eigs, eigs_pristine]):
                    dset = cp.create_dataset_real(_name, (k_div.n_tot, n_bands_sup * n_tile))
                    cp.write_slice(dset, (k_div.i_start, 0), _data)
                vset = [None] * k_div.n_tot 
                if len(vecs) != 0:
                    for k in range(k_div.n_tot):
                        vset[k] = cp.create_dataset_complex(f"vecs/{k}", 
                                                           (n_bands_sup * n_tile, vecs_sizes[k]))
                    for i,ik in enumerate(range(k_div.i_start, k_div.i_stop)):
                        print(len(vecs), i)
                        cp.write_slice_complex(vset[ik], (0, 0), vecs[i])
        return G, T


def main() -> None:
    io.log_config()
    rc.init()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="YAML input file")
    args = parser.parse_args()
    
    # Load input parameters from YAML file:
    input_dict = io.dict.key_cleanup(io.yaml.load(args.input_file))
    input_dict.setdefault( "checkpoint", os.path.splitext(args.input_file)[0] + ".h5")
    log.info(f"\n# Processed input:\n{io.yaml.dump(input_dict)}")
    input_dict = io.dict.remove_units(input_dict)
    Conductance(**input_dict).compute()
    rc.free()

if __name__ == "__main__":
    main()
