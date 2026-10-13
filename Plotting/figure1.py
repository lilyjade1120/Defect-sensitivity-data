import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import h5py

# Copper NEGF data
CuPristine = '../Cu001/pristine.h5'
CuCoherent = '../Cu001/one_device.h5'
CuIncoherent = '../Cu001/five_devices.h5'
CuSystematic = '../Cu001/many_devices.h5'

# CoSi NEGF data
CoSiPristine = '../CoSi001/Figure1/pristine.h5'
CoSiCoherent = '../CoSi001/Figure1/one_device.h5'
CoSiIncoherent = '../CoSi001/Figure1/five_devices.h5'
CoSiSystematic = '../CoSi001/Figure1/many_devices.h5'

names = ['coherent', 'incoherent', 'systematic']
CuNEGF = [CuCoherent, CuIncoherent, CuSystematic]
CoSiNEGF = [CoSiCoherent, CoSiIncoherent, CoSiSystematic] 

def h5read(filename):
    with h5py.File(filename) as f:
           
        T = np.array(f["T"]).squeeze()
        Nk = len(T)

        k = (np.arange(Nk + 1) / Nk) - 0.5  # -pi/a to pi/a, both inclusive
        idxZero = Nk//2 #np.asarray(k == 0).nonzero()[0].item()
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

def plotEigs(ax, eigs, kEdges, eigs_p): 
    
    eigs = np.sort(eigs, axis=1)
    eigDiffFrom1 = (1. - eigs).T
    img = ax.imshow(eigDiffFrom1,
               norm = colors.LogNorm(vmin = 1e-3, vmax = 1, clip = True),
               extent=(0, 0.5, 0, len(eigs[0])),
               aspect='auto',
               cmap='afmhot',
               interpolation='nearest'
            )
    
    if type(eigs_p) != type(None) and len(eigs_p.shape) > 1: # eigs_p exists and needs to be summed over axis 1
        ax.stairs(eigs_p.sum(axis = 1), kEdges, color = 'k', ls = 'solid')
    
    elif type(eigs_p) != type(None) and len(eigs_p.shape) == 1: #eigs_p exist and are already summed over axis 1
        ax.stairs(eigs_p, kEdges, color = 'k', ls = 'solid')
    
    else:
        pass
    
    ax.set_ylim([0, 12])
    ax.set_xlim([0, 0.5])
    
    return img
   
fig, axes = plt.subplots(
    ncols = 2,  
    nrows = 4, 
    sharex = True, 
    sharey = True, 
    figsize = (5, 6), 
    gridspec_kw = {'hspace': 0, 'wspace': 0.1}
)
axRow1, axRow2, axRow3, axRow4 = axes


kCuPristine, eigsCuPristine, _, _ = h5read(CuPristine)
kCoSiPristine, _, eigsCoSiPristine, _ = h5read(CoSiSystematic)

# Plot total transmission on row 1
CuPath = '../Cu001/NetTransmission/'
CoSiPath = '../CoSi001/Figure1/NetTransmission/'

for idx, path in enumerate([CuPath, CoSiPath]): # Plot transmission for individual incoherent devices
    for device in [1, 2, 3, 4, 5]:
        
        k, _, _, T = h5read(f'{path}/miss5-{device}.h5')
        if idx == 1 and device == 1:
            label = f'Each Defect'
        else:
            label = '_'
        axRow1[idx].plot(k, T, ls = 'solid', label = label, color = 'C0', lw = 1)
        
for idx, path in enumerate([CuCoherent, CoSiCoherent]): #Plot transmission for coherent devices
    
    k, _, _, T = h5read(path)
    if idx == 1:
        label = 'All Defects'
    else:
        label = '_'
    
    axRow1[idx].plot(k, T, ls = '--', color = 'k', label = label)

# Set up and plot transmission for pristine devices
dkCuPristine = kCuPristine[1] - kCuPristine[0]
kEdgesCuPristine = np.append(kCuPristine - dkCuPristine/2, kCuPristine[-1] + dkCuPristine/2)

dkCoSiPristine = kCoSiPristine[1] - kCoSiPristine[0]
kEdgesCoSiPristine = np.append(kCoSiPristine - dkCoSiPristine/2, kCoSiPristine[-1] + dkCoSiPristine/2)

axRow1[0].stairs(eigsCuPristine.sum(axis = 1), kEdgesCuPristine, ls = 'solid', color = 'k')
axRow1[1].stairs(eigsCoSiPristine.sum(axis = 1), kEdgesCoSiPristine, ls = 'solid', color = 'k', label = 'Pristine')    

axRow1[1].legend(labelspacing = 0.2)

# Plot transmission spectral analysis on rows 2 - 4
for Cu, CoSi, axRow, name in zip(CuNEGF, CoSiNEGF, (axRow2, axRow3, axRow4), names):
    
    kCu, eigsCu, _, TCu = h5read(Cu)
    kCoSi, eigsCoSi, _, TCoSi = h5read(CoSi)
    
    if name == 'systematic': 
        kCu, eigsCu, eigsCuPristine, _ = h5read(Cu)
        dkCu = kCu[1] - kCu[0]
        kEdgesCuPristine = np.append(kCu - dkCu/2, kCu[-1] + dkCu/2)
        
        kCoSi, eigsCoSi, eigsCoSiPristine, _ = h5read(CoSi)
        dkCoSi = kCoSi[1] - kCoSi[0]
        kEdgesCoSiPristine = np.append(kCoSi - dkCoSi/2, kCoSi[-1] + dkCoSi/2) 
    
    Cu = plotEigs(axRow[0], eigsCu, kEdgesCuPristine, eigsCuPristine)
    CoSi = plotEigs(axRow[1], eigsCoSi, kEdgesCoSiPristine, eigsCoSiPristine)

axRow1[0].set_ylabel(r'$T$') 
axRow3[0].set_ylabel(r'Transmission, $T$')

axRow4[0].set_xticks([0, 0.5], [0, r'$\pi/a$'])
axRow4[0].set_yticks(ticks = [0, 5, 10], labels = ['0', '10', '20'])
colorax = fig.add_axes((0.25, 0, 0.55, 0.03))
fig.colorbar(mappable = CoSi, 
             label="Fraction scattered",
             cax = colorax,
             orientation = 'horizontal'
        )

# Add panel labels
for i_row, axRow in enumerate(axes):
    for i_col, ax in enumerate(axRow):
        ax.text(0.01, 0.99, f'({"abcdefgh"[i_col + 2*i_row]})', va="top", fontweight="bold", transform=ax.transAxes)

# Add column labels
axRow1[0].text(0.5, 1.05, 'Cu(001)', va = 'bottom', ha = 'center', transform = axRow1[0].transAxes)
axRow1[1].text(0.5, 1.05, 'CoSi(001)', va = 'bottom', ha = 'center', transform = axRow1[1].transAxes)

# Add arrows for transport directions
axRow4[0].annotate('', 
    xy = (0.4, 0.175), 
    xytext = (0.2, 0.175), 
    arrowprops = dict(arrowstyle = '->'), 
    xycoords = 'figure fraction', 
    zorder = 2
)

axRow4[0].annotate('', 
    xy = (0.82, 0.175), 
    xytext = (0.62, 0.175), 
    arrowprops = dict(arrowstyle = '->'), 
    xycoords = 'figure fraction', 
    zorder = 2
)

# Label transport directions
axRow4[0].text(0.5, -0.3, '[010]', va = 'bottom', ha = 'center', transform = axRow4[0].transAxes)
axRow4[1].text(0.5, -0.3, '[010]', va = 'bottom', ha = 'center', transform = axRow4[1].transAxes)

fig.savefig('figure1.pdf', bbox_inches = 'tight')
