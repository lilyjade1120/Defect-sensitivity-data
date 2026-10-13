import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

path = f'../NbAs001_SOC/FermiSurface/Momentum/'
fig, axes = plt.subplots(1, 3, sharex = True, sharey = True, figsize = (15, 5), gridspec_kw = {'wspace': -0.5})

segmentMax = float(np.loadtxt(f'{path}/maxSegments.txt'))
ticks = [-0.5, 0, 0.5]
labels = ['-$\pi$/a', '0', '+$\pi$/a']

nSave = int(np.loadtxt(f'{path}/numSave.txt'))
for idx in range(3):
    for iSave in range(nSave):

        segments = np.loadtxt(f'{path}/segments{iSave}.txt').reshape(-1, 2, 2) / (segmentMax * 2) # map back to lattice coords        
        segments = segments[:, :, ::-1] # swap x and y
        edges = np.loadtxt(f'{path}/edges{iSave}.txt').astype(np.int64) 
        w = np.loadtxt(f'{path}/p{iSave}.txt')[:, idx]
        collection = LineCollection(segments, cmap = 'RdBu', norm = plt.Normalize(-1/4, 1/4))
        collection.set_array(w[np.sum(edges, axis = 1) // 2]) 
        axes[idx].add_collection(collection)
        axes[idx].set_aspect('equal')

        # Add context
        axes[idx].text(
            0.5,
            1.05,
            rf'P$_{"xyz"[idx]}$',
            horizontalalignment = 'center',
            verticalalignment = 'center',
            transform = axes[idx].transAxes
        )

        axes[idx].annotate(
            text = '',
            xy = (0.7, -0.1),
            xytext = (0.3, -0.1),
            arrowprops = dict(arrowstyle = '->'),
            xycoords = 'axes fraction',
            zorder = 2
        )

        axes[idx].annotate(
        text = '',
        xy = (-0.12, 0.7),
        xytext = (-0.12, 0.3),
        arrowprops = dict(arrowstyle = '->'),
        xycoords = 'axes fraction',
        zorder = 2
        )
        
        # Label directions
        axes[idx].text(0.5, -0.15, '[010]', va = 'bottom', ha = 'center', transform = axes[idx].transAxes)
        axes[idx].text(-0.25, 0.5, '[100]', va = 'bottom', ha = 'center', transform = axes[idx].transAxes)

axes[0].set_xlim([0, 0.5])
axes[0].set_ylim([-0.5, 0.5])
axes[0].set_yticks(ticks = ticks, labels = labels)
axes[0].set_xticks(ticks = ticks[1:], labels = labels[1:])

fig.savefig('figureS4_momentum.pdf', bbox_inches = 'tight')
