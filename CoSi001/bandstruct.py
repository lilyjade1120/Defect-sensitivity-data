import numpy as np
import matplotlib.pyplot as plt
eigvals = np.fromfile("bandstruct.eigenvals", count=-1)
kpoints = np.loadtxt("bandstruct.kpoints", skiprows=2, usecols=(1,2,3))
nRows = kpoints.shape[0]
nEigvals = eigvals.shape[0]
nCols = round(nEigvals/nRows)
eig = np.reshape(eigvals, (nRows, nCols))

mu_b = 0.60392 # Bulk Fermi level from Wannier
mu_s = -0.15944 # Slab Fermi level from Wannier

E0_b = -2.8810 # Bulk semicore bulk-band reference energy

E0_s = -3.6639 # Surface semicore bulk-band reference energy

mu_b_new = mu_b - E0_b + E0_s # Shifts bulk Fermi level to surface reference

# change the following accordingly
mu = mu_b_new  
eV  = 27.211386246
tics = ' "M" 0, "Gamma" 50, "Z" 100, "R" 150, "Gamma" 221 '
tics = tics.strip()
components = tics.split(',')
klabels = []
kvalues = []
for component in components:
    label, value = component.strip().split()
    label = label.strip('"')
    if label == 'Gamma':
        label = r'$\Gamma$'
    
    value = int(value)
    klabels.append(label)
    kvalues.append(value)

print(f'{mu_b_new = :.4f} H')
print(f'{mu_s = :.4f} H')
print(f'dmu = {mu_b_new - mu} H')
print(f'dmu = {(mu_b_new - mu)*27.211:.3f} eV')

one_eV = 1/27.211 # H

print(eig[0,:])
fig, ax = plt.subplots()
for i in range(nCols):
        ax.plot((eig[:,i] - mu) * eV, c = 'k')
ax.set_xticks(kvalues, klabels)
ax.set_ylabel(r'E - $\mu$ (eV)')
ax.set_xlim([0,nRows-1])
#ax.set_xlabel(r'k-point path $\left(\frac{2\pi}{a}\right)$')
ax.axhline(y = 0, c = 'k', lw = 1, ls = '--')
#ax.axhline(y = (mu_b_new - mu) * eV, c = 'b', lw = 1, ls = '--')
ax.set_ylim(bottom = -1, top = 1)
fig.savefig('bandstructure.png',bbox_inches='tight')

#plt.show()
