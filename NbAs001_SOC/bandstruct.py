import numpy as np
import matplotlib.pyplot as plt
eigvals = np.fromfile("bandstruct.eigenvals", count=-1)
kpoints = np.loadtxt("bandstruct.kpoints", skiprows=2, usecols=(1,2,3))
nRows = kpoints.shape[0]
nEigvals = eigvals.shape[0]
nCols = round(nEigvals/nRows)
eig = np.reshape(eigvals, (nRows, nCols))

# change the following accordingly
vbm = 0.0
eV  = 1 # 27.211386246
tics=' "M" 0, "Y" 50, "Gamma" 100, "X" 150, "M" 200, "Gamma" 271 '
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
ax.set_ylabel('E (H)')
ax.set_xlim([0,nRows-1])

ax.set_ylim(bottom=-0.25, top=0)
fig.savefig('bandstructure.png')
#print(eig[:,0])
plt.show()
