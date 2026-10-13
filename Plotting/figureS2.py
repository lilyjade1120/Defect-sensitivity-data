import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, SymLogNorm
from matplotlib.collections import LineCollection

#Construct custom colormaps:
t = np.linspace(0., 1., 256)[:,None]
color0 = np.array([0.,0.,0., 0.2]) #RGBA color corresponding to zero weight
colorP = np.array([0.,0.,1., 1.]) #RGBA color corresponding to full weight for +z surface
colorM = np.array([1.,0.,0., 1.]) #RGBA color corresponding to full weight for -z surface
cmapP = ListedColormap(color0[None,:]*(1.-t) + colorP[None,:]*t)
cmapM = ListedColormap(color0[None,:]*(1.-t) + colorM[None,:]*t)
cmaps = [cmapM, cmapP]

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

SOC = '../NbAs001_SOC/FermiSurface/'
noSOC = '../NbAs001_noSOC/FermiSurface/'

fig, axes = plt.subplots(
    nrows = 3,
    ncols = 2,
    sharex = True,
    sharey = True,
    figsize = (8,10),
    gridspec_kw = {'hspace': 0.2}
)

for axRow, directory in zip(axes, ['m0.01', '0.04', '0.09']):

    plot_fermi_surface(f'{noSOC}/{directory}/', axRow[0])
    plot_fermi_surface(f'{SOC}/{directory}/', axRow[1])

for iRow, dmu in zip([0, 1, 2], [-0.05, 0.00, 0.05]):
    for iCol in [0, 1]:
        ax = axes[iRow, iCol]
        bbox = {'color': 'w', 'boxstyle': 'Square, pad=0'}
        ax.text(0.5, 0.9, 
            rf'$\Delta\mu = {dmu:.2f}$ eV', 
            horizontalalignment = 'center', 
            verticalalignment = 'center',
            bbox = bbox,
            transform = ax.transAxes) 

# Add panel labels
for i_row, axRow in enumerate(axes):
    for i_col, ax in enumerate(axRow):
 
        bbox = {'color': 'w', 'boxstyle': 'Square, pad=0'}
        ax.text(0.01, 0.99, 
                f'({"abcdefgh"[i_col + 2*i_row]})', 
                va = "top", 
                fontweight = "bold",
                bbox = bbox,
                transform=ax.transAxes, 
                color = 'k')

# Add column labels
axes[0,0].text(0.5, 1.05, 'NbAs(001) no SOC', va = 'bottom', ha = 'center', transform = axes[0,0].transAxes)
axes[0,1].text(0.5, 1.05, 'NbAs(001) SOC', va = 'bottom', ha = 'center', transform = axes[0,1].transAxes)

# Add arrows for transport directions
axes[-1,0].annotate(
    text = '',
    xy = (0.4, 0.08),
    xytext = (0.2, 0.08),
    arrowprops = dict(arrowstyle = '->'),
    xycoords = 'figure fraction',
    zorder = 2
)

axes[-1,0].annotate(
    text = '',
    xy = (0.82, 0.08),
    xytext = (0.62, 0.08),
    arrowprops = dict(arrowstyle = '->'),
    xycoords = 'figure fraction',
    zorder = 2
)

# Label transport directions
axes[-1,0].text(0.5, -0.25, '[010]', va = 'bottom', ha = 'center', transform = axes[-1,0].transAxes)
axes[-1,1].text(0.5, -0.25, '[010]', va = 'bottom', ha = 'center', transform = axes[-1,1].transAxes)

plt.savefig('figureS2.pdf')
