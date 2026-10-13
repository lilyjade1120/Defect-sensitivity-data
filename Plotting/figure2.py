import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import h5py

from scipy.ndimage import gaussian_filter1d
from matplotlib.collections import LineCollection 
from matplotlib.colors import ListedColormap, SymLogNorm
from scipy.interpolate import CubicSpline

#--- Make colorbar ---#
from colorbar import make_colormap
colormap, norm = make_colormap(gamma_pre = 1.2, gamma_post = 1)
# higher gamma_pre ==> darker overall
# higher gamma_post ==> more saturated overall

# --- Load data --- #
path001 = '../CoSi001/'
path111 = '../CoSi111/'

DOS_001 = f'{path001}/Figure2/kyDOS-0.007036895972044516.txt' # has shape nkx x 1/dE
transport1_001 = f'{path001}/Figure2/miss_dmu0.01.h5'
transport2_001 = f'{path001}/Figure2/miss_dmu-0.09.h5'

DOS_111 = f'{path111}/Figure2/kyDOS-0.006386805972044413.txt'
transport1_111 = f'{path111}/Figure2/miss_dmu0.03.h5'
transport2_111 = f'{path111}/Figure2/miss_dmu-0.07.h5'

plot_order = np.array([[DOS_001, DOS_111],
                      [transport2_001, transport2_111],
                      [transport1_001, transport1_111]]
                )
fermi_paths = [f'{path001}/FermiSurface/', f'{path111}/FermiSurface/']
# ---------------- #

# Constants
eV = 1/27.211

xmin = 0.08520179372197306 + 2/223
xmax = 0.273542600896861 + 1/223

# --- x-values for vertical bars on CoSi 001 plots --- #
# dmu_slab = 0 eV:
x_values1 = -np.array([xmin, xmax])

# dmu_slab = -0.1 eV
x_values2 = np.array([-0.331, -0.158])
# ---------------------------------------------------- #

# Shift Fermi levels appropriately
muDFT001 = -0.17079 # H
dmuDFTtoWannier001 = -0.007036895972044516 # H -- maps DFT neutral slab Fermi level to Wannier neutral bulk Fermi level
muWannierBulk001 = muDFT001 + dmuDFTtoWannier001

muDFT111 = -0.17699
dmuDFTtoWannier111 = -0.006386805972044413 
muWannierBulk111 = muDFT111 + dmuDFTtoWannier111

# y-axis range for k-resolved DOS
Emin_eV = -0.5
Emax_eV = 0.5

def h5read(filename):
    with h5py.File(filename) as f:
        
        T = np.array(f["T"]).squeeze()
        Nk = len(T)
        
        k = (np.arange(Nk + 1) / Nk) - 0.5  # -pi/a to pi/a, both inclusive
        idxZero = Nk//2 
        kHalf = k[idxZero:]
        
        if 'eigs' in f.keys():
            
            eigs = np.array(f["eigs"])
            eigs = np.concatenate((eigs[Nk//2:], eigs[:(Nk//2 + 1)]), axis=0) # reorders eigenvalues from [0, pi/a] to [-pi/a, pi/a] and repeats eig(k[0])
            
            eigsHalf = (eigs[idxZero:] + eigs[:idxZero + 1][::-1])/2
        
        else:
            eigsHalf = None
        
        if "eigs_pristine" in f.keys():
            
            eigs_p = np.array(f['eigs_pristine'])
            eigs_p = np.concatenate((eigs_p[Nk//2:], eigs_p[:(Nk//2 + 1)]), axis=0)
            
            eigsHalf_p = (eigs_p[idxZero:] + eigs_p[:idxZero + 1][::-1])/2
           
        else:
            eigsHalf_p = None
        
        T = np.concatenate((T[Nk//2:], T[:(Nk//2 + 1)])) 
        Thalf = (T[idxZero:] + T[:idxZero + 1][::-1])/2
    
    return kHalf, eigsHalf, eigsHalf_p, Thalf

def plotEigs(eigs, eigs_p, k, ax):
     
    eigs = np.sort(eigs, axis=1)
    eigDiffFrom1 = (1. - eigs).T
    img = ax.imshow(eigDiffFrom1,
               norm = colors.LogNorm(vmin=1e-3, vmax=1, clip=True),
               extent=(0, 0.5, 0, len(eigs[0])),
               aspect='auto',
               cmap='afmhot',
               interpolation='nearest'
            )

    dk = k[1] - k[0]
    kEdges = np.append(k - dk/2, k[-1] + dk/2)
    
    if type(eigs_p) != type(None) and len(eigs_p.shape) > 1: # eigs_p exists and needs to be summed over axis 1
        ax.stairs(eigs_p.sum(axis = 1), kEdges, color = 'k', ls = 'solid', lw = 1/2)
    
    elif type(eigs_p) != type(None) and len(eigs_p.shape) == 1: #eigs_p exist and are already summed over axis 1
        ax.stairs(eigs_p, kEdges, color = 'k', ls = 'solid', lw = 1/2)
    
    else:
        pass
    
    ax.set_ylim([0, 12])
    ax.set_xlim([0, 0.5])
    
    return img

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
    ax.set_ylim([-0.5, 0.5])
    ax.set_xlim([0, 0.5])
    
    return img

def plot_fermi_surface(path, ax):

    segmentMax = float(np.loadtxt(f'{path}/maxSegments.txt'))
    ticks = [-0.5, 0, 0.5]
    labels = ['-$\pi$/a', '0', '+$\pi$/a']

    nSave = int(np.loadtxt(f'{path}/numSave.txt'))
    for weights, cmap in zip(['wm', 'wp'], [cmapM, cmapP]):
        for iSave in range(nSave):

            segments = np.loadtxt(f'{path}/segments{iSave}.txt').reshape(-1, 2, 2) / (segmentMax * 2) # map back to lattice coords
            segments = segments[:, :, ::-1] # swap x and y axes (not always necessary)
            edges = np.loadtxt(f'{path}/edges{iSave}.txt').astype(np.int64)
            w = np.loadtxt(f'{path}/{weights}{iSave}.txt')
            collection = LineCollection(segments, linewidths = 1, cmap = cmap, norm = plt.Normalize(0, 1))
            collection.set_array(w[np.sum(edges, axis = 1) // 2])
            ax.add_collection(collection)
            ax.set_xlim([0, 0.5])
            ax.set_ylim([0, 0.5])
            ax.set_yticks(ticks = ticks[1:], labels = labels[1:])
            ax.set_xticks(ticks = ticks[1:], labels = labels[1:])

#Construct custom colormaps:
t = np.linspace(0., 1., 256)[:,None]
color0 = np.array([0.,0.,0., 0.2]) #RGBA color corresponding to zero weight
colorP = np.array([0.,0.,1., 1.]) #RGBA color corresponding to full weight for +z surface
colorM = np.array([1.,0.,0., 1.]) #RGBA color corresponding to full weight for -z surface
cmapP = ListedColormap(color0[None,:]*(1.-t) + colorP[None,:]*t)
cmapM = ListedColormap(color0[None,:]*(1.-t) + colorM[None,:]*t)
cmaps = [cmapM, cmapP]

fig, axes = plt.subplots(
    nrows = 4, 
    ncols = 2, 
    sharex = True, 
    sharey = 'row',
    figsize = (4,5),
    height_ratios = [1.3, 1, 1, 1],
    gridspec_kw = {'hspace': 0}
)

for idx, axRow in enumerate(axes):
    
    if idx == 1:
        plotDOS(plot_order[0, 0], axRow[0])
        imgDOS = plotDOS(plot_order[0, 1], axRow[1]) 
    elif idx > 1:
        for col in [0, 1]:
            if plot_order[idx - 1, col] != '':
                k, eigs, eigs_p, T = h5read(plot_order[idx - 1, col])
                imgEigs = plotEigs(eigs, eigs_p, k, axRow[col])
    else:
        for col in [0, 1]:
            plot_fermi_surface(fermi_paths[col], axRow[col])

axes[0,0].set_xlim(0, 0.5)
axes[0,0].set_xticks(ticks = [0, 0.5], labels = ['0', r'$\pi/a$'])
axes[0,0].set_yticks(ticks = [0, 0.5], labels = ['0', r'$\pi/a$'])
axes[1,0].set_ylabel(r'E - $\mu$')
axes[1,0].set_yticks(ticks = [-0.4, -0.2, 0, 0.2, 0.4], labels = ['-0.4', '-0.2', '0', '0.2', '0.4'])
axes[3,0].set_ylabel('Transmission, $T$', y = 1)
axes[2,0].set_yticks(ticks = [2, 4, 6, 8, 10], labels = ['4', '8', '12', '16', '20'])
axes[3,0].set_yticks(ticks = [2, 4, 6, 8, 10], labels = ['4', '8', '12', '16', '20'])

TransportColorAx = fig.add_axes((0.15, 0, 0.70, 0.03))
DOScolorAx = fig.add_axes((0.15, 0.95, 0.70, 0.03))

fig.colorbar(mappable = imgEigs,
             label = "Fraction scattered",
             cax = TransportColorAx,
             orientation = 'horizontal'
        )

DOSbar = fig.colorbar(mappable = imgDOS,
             label = r"DOS (eV$^{-1}\mathrm{\AA}^{-2}$)",
             cax = DOScolorAx,
             orientation = 'horizontal'
        )

DOSbar.ax.xaxis.set_ticks_position('top')
DOSbar.ax.xaxis.set_label_position('top')

muSlab001 = (muDFT001 - muWannierBulk001) / eV
muSlab111 = (muDFT111 - muWannierBulk111) / eV
dmu_wrt_bulk001 = np.array([])
dmu_wrt_bulk111 = np.array([])
for dmu_slab001, dmu_slab111, xs in zip([0.01, -0.09], [0.03, -0.07], [x_values1, x_values2]):
    
    y001 = muSlab001 + dmu_slab001
    y111 = muSlab111 + dmu_slab111
    dmu_wrt_bulk001 = np.append(dmu_wrt_bulk001, y001)
    dmu_wrt_bulk111 = np.append(dmu_wrt_bulk111, y111)
    
    axes[1,0].plot(-xs, [y001, y001], marker = '|', ls = 'solid', lw = 2, color = 'w', ms = 4, mew = 2)
    axes[1,0].plot(-xs, [y001, y001], marker = '|', ls = 'solid', lw = 1, color = 'k', ms = 3, mew = 1)
    
for idx, dmu in enumerate(zip(dmu_wrt_bulk001[::-1], dmu_wrt_bulk111[::-1])):
    for iCol, bbox in zip([0, 1], ({'color': 'w'}, None)):
        ax = axes[idx + 2, iCol]
        ax.text(0.5, 0.7, 
            rf'$\Delta\mu = {dmu[iCol]:.2f}$ eV', 
            horizontalalignment = 'center', 
            verticalalignment = 'center', 
            bbox = bbox,
            transform = ax.transAxes) 

#plot vertical bars
axes[0,0].vlines(x = -x_values1, ymin = 0, ymax = 0.5, color = 'k', lw = 0.5, linestyle = 'solid')
axes[1,0].vlines(x = -x_values2, ymin = -0.5, ymax = muSlab001 - 0.1, lw = 1/2, zorder = 1, color = 'w')
axes[1,0].vlines(x = -x_values1, ymin = muSlab001, ymax = 0.5, lw = 1/2, zorder = 1, color = 'w')
axes[2,0].vlines(x = -x_values2, ymin = 3, ymax = 12, lw = 1/2, zorder = 1, color = 'k')

# Add panel labels
for i_row, axRow in enumerate(axes):
    for i_col, ax in enumerate(axRow):
        
        if i_row == 1:
            color = 'w'
        else:
            color = 'k'

        ax.text(0.01, 0.99, f'({"abcdefgh"[i_col + 2*i_row]})', va="top", fontweight="bold", transform=ax.transAxes, color = color)

# Add column labels
axes[0,0].text(0.5, 1.05, 'CoSi(001)', va = 'bottom', ha = 'center', transform = axes[0,0].transAxes)
axes[0,1].text(0.5, 1.05, 'CoSi(111)', va = 'bottom', ha = 'center', transform = axes[0,1].transAxes)

# Add arrows for transport directions
axes[-1,0].annotate(
    text = '',
    xy = (0.45, 0.2),
    xytext = (0.25, 0.2),
    arrowprops = dict(arrowstyle = '->'),
    xycoords = 'figure fraction',
    zorder = 2
)

axes[-1,0].annotate(
    text = '',
    xy = (0.87, 0.2),
    xytext = (0.67, 0.2),
    arrowprops = dict(arrowstyle = '->'),
    xycoords = 'figure fraction',
    zorder = 2
)

# Label transport directions
axes[-1,0].text(0.5, -0.35, '[010]', va = 'bottom', ha = 'center', transform = axes[-1,0].transAxes)
axes[-1,1].text(0.5, -0.35, r'$\left[1\overline{1}0\right]$', va = 'bottom', ha = 'center', transform = axes[-1,1].transAxes)

fig.savefig('figure2.pdf', bbox_inches = 'tight', dpi = 300)
