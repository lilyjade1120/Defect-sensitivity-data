import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import h5py

from scipy.ndimage import gaussian_filter1d

from matplotlib.colors import ListedColormap, SymLogNorm
from scipy.interpolate import CubicSpline

#--- Make colorbar ---#
from colorbar import make_colormap
colormap, norm = make_colormap(gamma_pre = 1.2, gamma_post = 1)
# higher gamma_pre ==> darker overall
# higher gamma_post ==> more saturated overall

# --- Load data --- #
pathSOC = '../NbAs010_SOC/FigureS5/'
pathNoSOC = '../NbAs010_noSOC/FigureS5/'

DOS_SOC = f'{pathSOC}/kyDOS0.txt'
transportSOC_1 = f'{pathSOC}/miss_dmu0.0.h5'
transportSOC_2 = f'{pathSOC}/miss_dmu0.2.h5'

DOS_NoSOC = f'{pathNoSOC}/kyDOS0.txt'
transportNoSOC_1 = f'{pathNoSOC}/miss_dmu0.0.h5'
transportNoSOC_2 = f'{pathNoSOC}/miss_dmu0.2.h5'

plot_order = np.array([[DOS_NoSOC, DOS_SOC],
                      [transportNoSOC_1, transportSOC_1],
                      [transportNoSOC_2, transportSOC_2]]
                )
# ---------------- #

# Constants
eV = 1/27.211

# Shift Fermi levels appropriately
muDFT_PdGa = -0.17079 # H
dmuDFTtoWannier_PdGa = -0.004349359928233376 # H -- maps DFT neutral slab Fermi level to Wannier neutral bulk Fermi level
muWannierBulk_PdGa = muDFT_PdGa + dmuDFTtoWannier_PdGa

muDFT_AlPt = -0.16841 # H 
dmuDFTtoWannier_AlPt = -0.006484355080120241 # H
muWannierBulk_AlPt = muDFT_AlPt + dmuDFTtoWannier_AlPt

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

def plotEigs(eigs, eigs_p, k, ax, soc = False):
     
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
    if soc:
        ax.set_ylim([0, 24])
    else:
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

fig, axes = plt.subplots(
    nrows = 3, 
    ncols = 2, 
    sharex = True, 
    figsize = (4,5), 
    gridspec_kw = {'hspace': 0}
)

for idx, axRow in enumerate(axes):
    
    if idx == 0:
        plotDOS(plot_order[0, 0], axRow[0])
        imgDOS = plotDOS(plot_order[0, 1], axRow[1]) 
    else:
        for col in [0, 1]:
            if plot_order[idx, col] != '':
                k, eigs, eigs_p, T = h5read(plot_order[idx, col])
               
                soc = False
                if col == 1:
                    soc = True
                imgEigs = plotEigs(eigs, eigs_p, k, axRow[col], soc = soc)

axes[0,0].set_xlim(0, 0.5)
axes[0,0].set_xticks(ticks = [0, 0.5], labels = ['0', r'$\pi/a$'])
axes[0,0].set_ylabel(r'E - $\mu$')
axes[2,0].set_ylabel('Transport Modes', y = 1)

axes[1,0].set_yticks(ticks = [2, 4, 6, 8, 10], labels = ['4', '8', '12', '16', '20'])
axes[2,0].set_yticks(ticks = [2, 4, 6, 8, 10], labels = ['4', '8', '12', '16', '20'])

axes[0,1].set_yticks(ticks = [-0.5, -0.25, 0.0, 0.25, 0.50]) 
axes[1,1].set_yticks(ticks = [4, 8, 12, 16, 20])
axes[2,1].set_yticks(ticks = [4, 8, 12, 16, 20])

axes[0,1].tick_params(axis = 'y', labelleft = False)
axes[1,1].tick_params(axis = 'y', labelleft = False)
axes[2,1].tick_params(axis = 'y', labelleft = False)

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

muSlab_PdGa = (muDFT_PdGa - muWannierBulk_PdGa) / eV
muSlab_AlPt = (muDFT_AlPt - muWannierBulk_AlPt) / eV
dmu_slabs = np.array([[-0.02, 0.08], [-0.08, 0.02]])

text_kwargs = {'horizontalalignment': 'center',
               'verticalalignment': 'center',
            }

# label delta mus
for iRow, dmu in zip([1, 2], [0.00, 0.20]):
    for iCol in [0, 1]:

            ax = axes[iRow, iCol]
            text_args = [0.5, 0.9]
            ax.text(*text_args, rf'$\Delta\mu = {dmu:.2f}$ eV', **text_kwargs, transform = ax.transAxes)


# Add panel labels
for i_row, axRow in enumerate(axes):
    for i_col, ax in enumerate(axRow):
        
        if i_row == 0:
            color = 'w'
        else:    
            color = 'k'
        
        ax.text(0.01, 0.99, f'({"abcdefgh"[i_col + 2*i_row]})', va="top", fontweight="bold", transform=ax.transAxes, color = color)

# Add column labels
axes[0,0].text(0.5, 1.05, 'NbAs(010) no SOC', va = 'bottom', ha = 'center', transform = axes[0,0].transAxes)
axes[0,1].text(0.5, 1.05, 'NbAs(010) with SOC', va = 'bottom', ha = 'center', transform = axes[0,1].transAxes)

# Add arrows for transport directions
axes[-1,0].annotate(
    text = '',
    xy = (0.475, 0.19),
    xytext = (0.275, 0.19),
    arrowprops = dict(arrowstyle = '->'),
    xycoords = 'figure fraction',
    zorder = 2
)

axes[-1,0].annotate(
    text = '',
    xy = (0.9, 0.19),
    xytext = (0.7, 0.19),
    arrowprops = dict(arrowstyle = '->'),
    xycoords = 'figure fraction',
    zorder = 2
)

# Label transport directions
axes[-1,0].text(0.5, -0.25, '[001]', va = 'bottom', ha = 'center', transform = axes[-1,0].transAxes)
axes[-1,1].text(0.5, -0.25, '[001]', va = 'bottom', ha = 'center', transform = axes[-1,1].transAxes)

fig.savefig('figureS5.pdf', bbox_inches = 'tight', dpi = 300)
