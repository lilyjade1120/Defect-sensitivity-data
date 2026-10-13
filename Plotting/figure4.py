from scipy.ndimage import gaussian_filter1d
import numpy as np
import h5py

import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, SymLogNorm
from scipy.interpolate import CubicSpline

#--- Make colorbar ---#
from colorbar import make_colormap
colormap, norm = make_colormap(gamma_pre = 1.2, gamma_post = 1)
# higher gamma_pre ==> darker overall
# higher gamma_post ==> more saturated overall

# --- Load data --- #
base_path = '../'
SOC_path = f'{base_path}/NbAs001_SOC/Figure4/'
noSOC_path = f'{base_path}/NbAs001_noSOC/Figure4/'

DOS_noSOC = f'{noSOC_path}/kyDOS0.0014280609068551553.txt' # has shape nkx x 1/dE
transport1_noSOC = f'{noSOC_path}/miss_dmu0.04.h5'
transport2_noSOC = f'{noSOC_path}/miss_dmu0.24.h5'

DOS_SOC = f'{SOC_path}/kyDOS0.0014280609068551553.txt'
transport1_SOC = f'{SOC_path}/miss_dmu0.04.h5'
transport2_SOC = f'{SOC_path}/miss_dmu0.24.h5'


plot_order = np.array([[f'{noSOC_path}/../FermiSurface/0.04/', f'{SOC_path}/../FermiSurface/0.04/'],
                      [DOS_noSOC, DOS_SOC],
                      [transport1_noSOC, transport1_SOC]]
                )
# ---------------- #

# Constants
eV = 1/27.211

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
   
        k_dark = k[np.min(1 - eigs, axis = 1) < 5e-3]

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
    
    ax.set_ylim([0, 24])
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
            collection = LineCollection(segments, cmap = cmap, norm = plt.Normalize(0, 1))
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
    nrows = 3, 
    ncols = 2, 
    sharex = True, 
    sharey = 'row',
    figsize = (4,5),
    height_ratios = [1.115, 1, 1],
    gridspec_kw = {'hspace': 0}
)

for idx, axRow in enumerate(axes):
    
    if idx == 0:
        plot_fermi_surface(plot_order[0, 0], axRow[0])
        plot_fermi_surface(plot_order[0, 1], axRow[1])
    elif idx == 1:
        plotDOS(plot_order[1, 0], axRow[0])
        imgDOS = plotDOS(plot_order[1, 1], axRow[1]) 
    else:
        for col in [0, 1]:
            if plot_order[idx, col] != '':
                k, eigs, eigs_p, T = h5read(plot_order[idx, col])
                imgEigs = plotEigs(eigs, eigs_p, k, axRow[col])

dk = 0.00390625
darkrange = [0.3359375 - dk, 0.34765625 + dk] 
for idx in range(3):
    if idx == 0:
        ymin = 0
        ymax = 0.5
        color = 'k'
    elif idx == 1:
        ymin = -0.5
        ymax = 0.5
        color = 'w'
    else:
        ymin = 0
        ymax = 24
        color = 'k'

    axes[idx, 1].vlines(x = darkrange, ymin = ymin, ymax = ymax, color = color, ls = 'solid', lw = 0.5)

axes[0,0].set_xlim(0, 0.5)
axes[0,0].set_xticks(ticks = [0, 0.5], labels = ['0', r'$\pi/a$'])
axes[1,0].set_ylabel(r'E - $\mu$')
axes[1,0].set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
axes[2,0].set_ylabel('Transmission, $T$', fontsize = 'small')

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

for iRow, dmu in zip([2], [0.00]):
    for iCol in [0, 1]:
        ax = axes[iRow, iCol]
        bbox = {'color': 'w', 'boxstyle': 'Square, pad=0'}
        ax.text(0.5, 0.7, 
            rf'$\Delta\mu = {dmu:.2f}$ eV', 
            horizontalalignment = 'center', 
            verticalalignment = 'center', 
            bbox = bbox,
            transform = ax.transAxes) 

# Add panel labels
for i_row, axRow in enumerate(axes):
    for i_col, ax in enumerate(axRow):
        
        if i_row == 1:
            color = 'w'
        else:    
            color = 'k'
        
        ax.text(0.01, 0.99, f'({"abcdefgh"[i_col + 2*i_row]})', va="top", fontweight="bold", transform=ax.transAxes, color = color)

# Add column labels
axes[0,0].text(0.5, 1.05, 'NbAs(001) no SOC', va = 'bottom', ha = 'center', transform = axes[0,0].transAxes)
axes[0,1].text(0.5, 1.05, 'NbAs(001) SOC', va = 'bottom', ha = 'center', transform = axes[0,1].transAxes)

# Add arrows for transport directions
axes[-1,0].annotate(
    text = '',
    xy = (0.47, 0.2),
    xytext = (0.27, 0.2),
    arrowprops = dict(arrowstyle = '->'),
    xycoords = 'figure fraction',
    zorder = 2
)

axes[-1,0].annotate(
    text = '',
    xy = (0.9, 0.2),
    xytext = (0.67, 0.2),
    arrowprops = dict(arrowstyle = '->'),
    xycoords = 'figure fraction',
    zorder = 2
)

# Label transport directions
axes[-1,0].text(0.5, -0.25, '[010]', va = 'bottom', ha = 'center', transform = axes[-1,0].transAxes)
axes[-1,1].text(0.5, -0.25, '[010]', va = 'bottom', ha = 'center', transform = axes[-1,1].transAxes)

fig.savefig('figure4.pdf', bbox_inches = 'tight', dpi = 300)
