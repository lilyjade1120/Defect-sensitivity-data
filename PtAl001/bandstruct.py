import numpy as np
import matplotlib.pyplot as plt
eigvals = np.fromfile("bandstruct.eigenvals", count=-1)
kpoints = np.loadtxt("bandstruct.kpoints", skiprows=2, usecols=(1,2,3))
nRows = kpoints.shape[0]
nEigvals = eigvals.shape[0]
nCols = round(nEigvals/nRows)
eig = np.reshape(eigvals, (nRows, nCols))

print(min(eig.flatten()))

# change the following accordingly
vbm = -0.17278 # Hartree 
eV  = 27.211386246
tics=' "M" 0, "Gamma" 50, "Z" 100, "R" 150, "Gamma" 221 '
tics = tics.strip()
components = tics.split(',')
klabels = []
kvalues = []
for component in components:
    label, value = component.strip().split()
    label = label.strip('"')
    value = int(value)
    klabels.append(label)
    kvalues.append(value)
fig, ax = plt.subplots()
for i in range(nCols):
        ax.plot((eig[:,i] - vbm) * eV)
ax.set_xticks(kvalues, klabels)
ax.set_ylabel(r'E - $\mu$ (eV)')
ax.set_xlim([0,nRows-1])

ax.set_ylim(bottom=-2, top=2)
fig.savefig('bandstructure.png')

plt.show()
