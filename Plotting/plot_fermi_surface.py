import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

#Construct custom colormaps:
t = np.linspace(0., 1., 256)[:,None]
color0 = np.array([0.,0.,0., 0.2]) #RGBA color corresponding to zero weight
colorP = np.array([0.,0.,1., 1.]) #RGBA color corresponding to full weight for +z surface
colorM = np.array([1.,0.,0., 1.]) #RGBA color corresponding to full weight for -z surface
cmapP = ListedColormap(color0[None,:]*(1.-t) + colorP[None,:]*t)
cmapM = ListedColormap(color0[None,:]*(1.-t) + colorM[None,:]*t)
cmaps = [cmapM, cmapP]

fig, ax = plt.subplots()

segmentMax = float(np.loadtxt('maxSegments.txt'))
ticks = [-0.5, 0, 0.5]
labels = ['-$\pi$/a', '0', '+$\pi$/a']

nSave = int(np.loadtxt('numSave.txt'))
for weights, cmap in zip(['wm', 'wp'], [cmapM, cmapP]):
    for iSave in range(nSave):
         
        segments = np.loadtxt(f'segments{iSave}.txt').reshape(-1, 2, 2) / (segmentMax * 2) # map back to lattice coords
        segments = segments[:, :, ::-1] # swap x and y axes (not always necessary)
        edges = np.loadtxt(f'edges{iSave}.txt').astype(np.int64) 
        w = np.loadtxt(f'{weights}{iSave}.txt')
        collection = LineCollection(segments, cmap = cmap, norm = plt.Normalize(0, 1))
        collection.set_array(w[np.sum(edges, axis = 1) // 2]) 
        ax.add_collection(collection)
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_aspect('equal')
        ax.set_yticks(ticks = ticks, labels = labels)
        ax.set_xticks(ticks = ticks, labels = labels)

fig.savefig('fermi_surface.png', bbox_inches = 'tight')
