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
            ax.set_xlim([-0.5, 0.5])
            ax.set_ylim([-0.5, 0.5])
            ax.set_yticks(ticks = ticks, labels = labels)
            ax.set_xticks(ticks = ticks, labels = labels)


fig, axes = plt.subplots(
    nrows = 1,
    ncols = 2,
    sharex = True,
    sharey = True,
    figsize = (8, 4),
)


plot_fermi_surface(f'../CoSi001/FermiSurface/', axes[0])
plot_fermi_surface(f'../CoSi111/FermiSurface/', axes[1])

# Add panel labels
for i_row, axRow in enumerate([axes]):
    for i_col, ax in enumerate(axRow):
 
        bbox = {'color': 'w', 'boxstyle': 'Square, pad=0'}
        ax.text(0.01, 0.99, 
                f'({"abcdefgh"[i_col + 2*i_row]})', 
                va = "top", 
                fontweight = "bold",
                bbox = bbox,
                transform=ax.transAxes, 
                color = 'k')

# Add arrows for transport directions
axes[0].annotate(
    text = '',
    xy = (0.35, 0.08),
    xytext = (0.15, 0.08),
    arrowprops = dict(arrowstyle = '->'),
    xycoords = 'figure fraction',
    zorder = 2
)

axes[1].annotate(
    text = '',
    xy = (0.78, 0.08),
    xytext = (0.56, 0.08),
    arrowprops = dict(arrowstyle = '->'),
    xycoords = 'figure fraction',
    zorder = 2
)

# Label transport directions
axes[0].text(0.5, -0.17, '[010]', va = 'bottom', ha = 'center', transform = axes[0].transAxes)
axes[1].text(0.5, -0.17, r'[1$\overline{1}$0]', va = 'bottom', ha = 'center', transform = axes[1].transAxes)


# Add column labels
axes[0].text(0.5, 1.05, 'CoSi(001)', va = 'bottom', ha = 'center', transform = axes[0].transAxes)
axes[1].text(0.5, 1.05, 'CoSi(111)', va = 'bottom', ha = 'center', transform = axes[1].transAxes)

plt.savefig('figureS6.pdf', bbox_inches = 'tight')
