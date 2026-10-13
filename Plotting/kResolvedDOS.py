import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.ndimage import gaussian_filter1d

from colorbar import make_colormap

parser = argparse.ArgumentParser()
parser.add_argument("nk0", help="Number of k to sample along first lattice vector", type=int)
parser.add_argument("nk1", help="Number of k to sample along second lattice vector", type=int)
parser.add_argument("nk2", help="Number of k to sample along third lattice vector", type=int)
parser.add_argument("i_dir", help="0-based index of direction to resolve along", type=int)
parser.add_argument("--Emin_eV", help="start of energy range relative to VBM/mu in eV", type=float, default=-2.0)
parser.add_argument("--Emax_eV", help="start of energy range relative to VBM/mu in eV", type=float, default=+2.0)
parser.add_argument("--Esigma_eV", help="energy resolution in eV", type=float, default=0.01)
parser.add_argument("--dmu_H", help="Fermi level shift relative to neutral slab Fermi level in Hartree", type=float, default=0)
args = parser.parse_args()


#Get mu/VBM, kfold, lattice vectors and spin type from totalE.out
mu = np.nan
R = np.zeros((3,3))
iLine = 0
refLine = -10
initDone = False
Rdone = False

for iLine,line in enumerate(open('totalE.out')):
    if line.startswith('Initialization completed'):
        initDone = True
    if initDone and line.find('FillingsUpdate:')>=0:
        mu = float(line.split()[2])
        isVBM = False
    if (not initDone) and line.startswith('spintype'):
        spinKey = line.split()[1]
        if spinKey=="vector-spin" or spinKey=="spin-orbit":
            mlwfType = np.complex128
            spinWeight = 1
        else:
            mlwfType = np.float64
            spinWeight = 2
        assert spinKey != "z-spin"
    
    if line.find('Initializing the Grid') >= 0:
        refLine = iLine
    if not Rdone:
        rowNum = iLine - (refLine+2)
        if rowNum>=0 and rowNum<3:
            R[rowNum,:] = [ float(x) for x in line.split()[1:-1] ]
        if rowNum==3:
            Rdone = True
    
    if (not initDone) and line.startswith('kpoint-folding'):
        kfold = np.array([int(tok) for tok in line.split()[1:4]])
    if (not initDone) and line.startswith('nElectrons:'):
        nElectrons = float(line.split()[1])
        nBandsDFT = int(line.split()[3])
        nStates = int(line.split()[5])
        nValence = int(nElectrons/2) #number of valence bands
        Edft = np.reshape(np.fromfile('totalE.eigenvals'), (nStates, nBandsDFT))
        #mu = np.max(Edft[:,:nValence])
        isVBM = True

surfaceArea = np.linalg.norm(np.cross(R[:,0], R[:,1]))

mu += args.dmu_H
ErefName = 'VBM' if isVBM else "mu"
ErefLabel = 'VBM' if isVBM else "$\mu$"
print(f"{ErefName} = {mu}")
kfoldProd = np.prod(kfold)
kStride = np.array([kfold[1]*kfold[2], kfold[2], 1])

# Read in Wannier cell map, weights and Hamiltonian:
cellMapIn = np.loadtxt("wannier.mlwfCellMap")
cellMap = cellMapIn[:,:3].astype(int)
cellMapCart = cellMapIn[:,3:]
Wwannier = np.fromfile("wannier.mlwfCellWeights")
nCells = cellMap.shape[0]
nBands = int(np.sqrt(Wwannier.shape[0] / nCells))
Wwannier = Wwannier.reshape((nCells,nBands,nBands)).swapaxes(1,2)
# --- Read and expand Wannier Hamiltonian:
Hreduced = np.fromfile("wannier.mlwfH", dtype=mlwfType).reshape((kfoldProd,nBands,nBands)).swapaxes(1,2)
iReduced = np.dot(np.mod(cellMap, kfold[None,:]), kStride)
Hwannier = Wwannier * Hreduced[iReduced] 


eV = 1/27.211386
Angstrom = 1/0.529177249
Emin = mu + args.Emin_eV*eV
Emax = mu + args.Emax_eV*eV
nSubSample = 3   # histogram at finer resolution to retain info within grid points
nIntervals = int(np.ceil((Emax - Emin) / (args.Esigma_eV * eV))) * nSubSample
dbin = (Emax - Emin) / nIntervals
bins = np.linspace(Emin-0.5*dbin, Emax+0.5*dbin, nIntervals + 2)  # so that bin midpoints contain Emin and Emax exactly
hist_kwargs = dict(bins=bins, range=(bins[0], bins[-1]))

kmeshSize = (args.nk0, args.nk1, args.nk2)
kgrids1D = [(np.arange(nk)/nk - 0.5) for nk in kmeshSize]  # Gamma-centered mesh 
kpoints = np.stack(np.meshgrid(*kgrids1D, indexing="ij"), axis=-1)  # nk0 x nk1 x nk2 x 3
kpoints = kpoints.swapaxes(0, args.i_dir).reshape((kmeshSize[args.i_dir], -1, 3))  # nk_i x nk_rest x 3

DOS = []
prefac = spinWeight / (kpoints.shape[1] * dbin/eV)
for ik, kpoints_i in enumerate(kpoints):
    Hk = np.tensordot(np.exp((2j*np.pi)*(kpoints_i @ cellMap.T)), Hwannier, axes=1)
    Ek = np.linalg.eigvalsh(Hk) # has shape nk x nBands
    DOS.append(
        gaussian_filter1d(
            np.histogram(Ek, **hist_kwargs)[0] * prefac,
            args.Esigma_eV * eV / dbin,
        )[::nSubSample]
    )
    print(f"Completed {ik+1} of {len(kpoints)}")
DOS.append(DOS[0])  # repeat periodic
DOS = np.stack(DOS) / (surfaceArea * (Angstrom**2)) #normalize by surface area

np.savetxt(f'k{"xyz"[args.i_dir]}DOS{args.dmu_H}.txt', DOS)

# Plot
plt.figure(1, figsize = (6, 4))
plt.imshow(
    DOS.T,
    origin='lower',
    extent=[-0.5, 0.5, args.Emin_eV, args.Emax_eV],
    #cmap='Reds',
    #cmap = 'RdGy',
    cmap = 'bone_r', 
    norm=colors.SymLogNorm(np.median(DOS), linscale = 100, vmin=0, vmax=DOS.max()), 
    aspect="auto",
    interpolation="bilinear",
)

muDFT = -0.17079 # H
#dmu_slab = 0 
#dmu_slab = -0.1 # eV

# dmu_slab = 0 eV:
x_values1 = [-0.286, -0.087]

# dmu_slab = -0.1 eV
x_values2 = [-0.33, -0.16]

mu_slab = (muDFT - mu) / eV
for dmu_slab, xs, ls in zip([0, -0.1], [x_values1, x_values2], [(0, (3, 3)), (0, (2, 6))]):
    
    y = mu_slab + dmu_slab
    
    ls_double = np.array(ls[1])*2
    ls_double = (0, tuple(ls_double))
    
    plt.plot([xs[0], xs[1]], [y, y], marker = '|', ls = ls, lw = 2, color = 'w', ms = 6)
    plt.plot([xs[0], xs[1]], [y, y], marker = '|', ls = ls_double, lw = 1, color = 'k', ms = 3)
    
    plt.plot([-xs[0], -xs[1]], [y, y], marker = '|', ls = ls, lw = 2, color = 'w', ms = 6)
    plt.plot([-xs[0], -xs[1]], [y, y], marker = '|', ls = ls_double, lw = 1, color = 'k', ms = 3)

plt.axhline(0, color='k', linewidth=1, linestyle='dashed')
dir_name = "xyz"[args.i_dir]
plt.xlabel(rf'$k_{dir_name}$')
plt.xticks(ticks = [-0.5, 0, 0.5], labels = [r'$-\frac{\pi}{a}$', '0', r'$\frac{\pi}{a}$'])
#plt.ylabel(rf'$E$ - {ErefLabel} (eV)')
plt.ylabel(r'E - $\mu$')
plt.colorbar(label=r"DOS (eV$^{-1}${\AA}$^{-2}$)")
plt.savefig(f"k{dir_name}ResolvedDOS.png", bbox_inches='tight', dpi=300)
plt.show()
