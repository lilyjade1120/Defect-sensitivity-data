from scipy.ndimage import gaussian_filter1d
import numpy as np
import h5py

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import  matplotlib.gridspec as gridspec 
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, SymLogNorm

def readEigs(filename, ext=""):
    with h5py.File(filename) as fp:
        
        eigs = np.array(fp["eigs"+ext])
        T = np.array(fp["T"+ext])
        Nk = len(T.squeeze())
        periodic_dirs = [i_dir for i_dir, Nki in enumerate(T.shape) if (Nki > 1)]
        Rbasis = np.array(fp.attrs["Rbasis"])
        k = (np.arange(Nk + 1) / Nk) - 0.5  # -pi/a to pi/a, both inclusive
        eigs = np.concatenate((eigs[Nk//2:], eigs[:(Nk//2 + 1)]), axis=0) # reorders eigenvalues from [0, pi/a] to [-pi/a, pi/a] and repeats eig(k[0])
        
        idxZero = np.asarray(k == 0).nonzero()[0].item()
        eigsHalf = (eigs[idxZero:] + eigs[:idxZero + 1][::-1])/2
        kHalf = k[idxZero:]
        
        if "eigs_pristine" in fp.keys():
            pristine_exists = True
        else: 
            pristine_exists = False
    
    return kHalf, eigsHalf, pristine_exists

def plotDOS(DOS, ax):

    DOS = np.loadtxt(DOS)
    DOShalf = (DOS[len(DOS)//2:] +  DOS[:(len(DOS)//2)][::-1]) / 2

    img = ax.imshow(DOShalf.T,
                    origin = 'lower',
                    extent = [0, 0.5, Emin_eV, Emax_eV],
                    cmap = colormap,
                    norm = norm,
                    aspect = "auto",
                    interpolation = "bilinear"
                )
    ax.axhline(y = 0, ls = '--', color = 'k', lw = 1)

    return img

def plot_fermi_surface(path, ax):

    segmentMax = float(np.loadtxt(f'{path}/maxSegments.txt'))

    nSave = int(np.loadtxt(f'{path}/numSave.txt'))
    for weights, cmap in zip(['wm', 'wp'], [cmapM, cmapP]):
        for iSave in range(nSave):

            segments = np.loadtxt(f'{path}/segments{iSave}.txt').reshape(-1, 2, 2) / (segmentMax * 2) # map back to lattice coords
            segments = segments[:, :, ::-1] # swap x and y axes (not always necessary)
            edges = np.loadtxt(f'{path}/edges{iSave}.txt').astype(np.int64)
            w = np.loadtxt(f'{path}/{weights}{iSave}.txt')
            collection = LineCollection(segments, cmap = cmap, norm = plt.Normalize(0, 1))
            collection.set_array(w[np.sum(edges, axis = 1) // 2])
            ax.add_collection(collection)

# Construct custom colormaps:
t = np.linspace(0., 1., 256)[:,None]
color0 = np.array([0.,0.,0., 0.2]) #RGBA color corresponding to zero weight
colorP = np.array([0.,0.,1., 1.]) #RGBA color corresponding to full weight for +z surface
colorM = np.array([1.,0.,0., 1.]) #RGBA color corresponding to full weight for -z surface
cmapP = ListedColormap(color0[None,:]*(1.-t) + colorP[None,:]*t)
cmapM = ListedColormap(color0[None,:]*(1.-t) + colorM[None,:]*t)
cmaps = [cmapM, cmapP]

fig = plt.figure(figsize = (10, 8))

def annotate_axes(fig): # for testing
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False)

# Create plot grid like an animal, based heavily on: 
# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_customization.html

gs0 = gridspec.GridSpec(2, 1, figure=fig)
gs1 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs0[0], hspace = 0, height_ratios = [1.3, 1])
gs2 = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs0[1], hspace = 0, height_ratios = [1.3, 1])

ax1 = fig.add_subplot(gs1[0, 0])
ax2 = fig.add_subplot(gs1[0, 1])
ax3 = fig.add_subplot(gs1[0, 2])
ax4 = fig.add_subplot(gs1[1, 0])
ax5 = fig.add_subplot(gs1[1, 1])
ax6 = fig.add_subplot(gs1[1, 2])

ax7 = fig.add_subplot(gs2[0, 0])
ax8 = fig.add_subplot(gs2[0, 1])
ax9 = fig.add_subplot(gs2[0, 2])
ax10 = fig.add_subplot(gs2[1, 0])
ax11 = fig.add_subplot(gs2[1, 1])
ax12 = fig.add_subplot(gs2[1, 2])

FS_axes = [ax1, ax2, ax3, ax7, ax8, ax9]
DOS_axes = [ax4, ax5, ax6, ax10, ax11, ax12]

for idx, dmu_eV in enumerate([0.01, 0.02, 0.03, 0.04, 0.05, 0.06]):
    
    # --- Plot k-resolved transport on second axis --- #
    k, eigs, pristine_exists = readEigs(f'../NbAs001_SOC/FigureS3/miss_dmu{dmu_eV:.2f}.h5')

    dk = k[1] - k[0]
    eigs = np.sort(eigs, axis = 1)
    eigDiffFrom1 = (1. - eigs).T
    img = DOS_axes[idx].imshow(
            eigDiffFrom1,
            norm = colors.LogNorm(vmin = 0.001, vmax = 1, clip = True),
            extent = (0, 0.5, 0, len(eigs[0])),
            aspect = 'auto',
            cmap = 'afmhot',
            interpolation = 'nearest'
        ) 

    darkIdx = eigDiffFrom1[-1, :] < 1/100
    kDark = k[darkIdx] 

    if pristine_exists:

        k_p, eigs_p, _ = readEigs(f'../NbAs001_SOC/FigureS3/miss_dmu{dmu_eV:.2f}.h5', "_pristine")
        k_edges = np.append(k_p - dk/2, 0.5 + dk/2)
        DOS_axes[idx].stairs(eigs_p.sum(axis=1), k_edges, color = 'k')

    kwargs = dict(ls = '--', lw = 1, color = 'k')
    if len(kDark) > 0:
        for k in [kDark[0], kDark[-1]]:
            DOS_axes[idx].axvline(x = k, **kwargs)
            FS_axes[idx].axvline(x = k, **kwargs) 

    DOS_axes[idx].set_ylim([0, 24])
    DOS_axes[idx].set_xlim([0, 0.5])
    DOS_axes[idx].set_xticks(ticks = [0, 0.5], labels = ['0', r'$+\pi/a$'])
    DOS_axes[idx].set_yticks(np.arange(0, 24, 4))

    bbox = {'color': 'w', 'boxstyle': 'Square, pad=0'}
    DOS_axes[idx].text(
            x = 0.5, 
            y = 0.7,
            s = rf'$\Delta\mu = {dmu_eV - 0.04:.2f}$ eV',
            horizontalalignment = 'center',
            verticalalignment = 'center',
            bbox = bbox,
            transform = DOS_axes[idx].transAxes
        )

    # --- Plot Fermi surface on first axis --- #
    directory = f'{dmu_eV:.2f}'
    if dmu_eV < 0:
        directory = f'm{abs(dmu_eV):.2f}'
    path = f'../NbAs001_SOC/FermiSurface/{directory}'

    maxSegments = float(np.loadtxt(f'{path}/maxSegments.txt'))
    nSave = int(np.loadtxt(f'{path}/numSave.txt'))

    for iSave in range(nSave):

        segments = np.loadtxt(f'{path}/segments{iSave}.txt').reshape(-1, 2, 2) / (2 * maxSegments)
        edges = np.loadtxt(f'{path}/edges{iSave}.txt').astype(np.int64)
        for weights, cmap in zip(['wm', 'wp'], [cmapM, cmapP]):

            w = np.loadtxt(f'{path}/{weights}{iSave}.txt')
            collection = LineCollection(segments[:, :, ::-1], cmap = cmap, norm = plt.Normalize(0, 1))
            collection.set_array(w[np.sum(edges, axis = 1) // 2])
            FS_axes[idx].add_collection(collection)

    FS_axes[idx].set_xlim([0, 0.5])
    FS_axes[idx].set_ylim([0, 0.5])
    FS_axes[idx].set_yticks(ticks = [0, 0.5], labels = ['0', r'$+\pi/a$'])
    FS_axes[idx].set_xticks(ticks = [])
    FS_axes[idx].text(
        x = 0.01,
        y = 0.9, 
        s = f'({"abcdef"[idx]})',
        color = 'k',
        fontweight = 'bold',
        bbox = bbox,
        transform = FS_axes[idx].transAxes
    )

DOS_axes[0].set_ylabel(r'Transmission, $T$')
DOS_axes[3].set_ylabel(r'Transmission, $T$')

fig.savefig('figureS3.pdf', bbox_inches = 'tight')
