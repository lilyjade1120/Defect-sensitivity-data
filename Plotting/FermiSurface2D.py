#!/usr/bin/env python
from __future__ import print_function

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
from scipy.linalg import eigh
import sys
import os
from skimage import measure
np.set_printoptions(linewidth=100, precision=5, suppress=True, threshold=sys.maxsize)

#Units
eV = 1/27.2114
fs = 1/0.0242
nm = 10/0.5291772

#Optional argument fo spin-polarized mode:
dmu = 0.      #offset in mu/VBM
nkPerDim = 128 #number of k-points per dimension
suffix = ""
nSpins = 1
iSpin = 0
if len(sys.argv)>1:
    if sys.argv[1]=='-h' or sys.argv[1]=='--help':
        print('''Usage: FermiSurface2D.py [<dmu_eV>=0] [<nK>=16] [<spin>=Up|Dn], where:
            <dmu_eV>: shift in Fermi level relative to neutral value (or VBM in absence of smearing) in eV
            <nK>: number of k-points per dimension in bounding box of Brillouin zone
            <spin>: select spin channel for spin-polarized calculations''')
    else:
        dmu = float(sys.argv[1])*eV
if len(sys.argv)>2:
    nkPerDim = ((int(sys.argv[2])+1)//2)*2 #ensure even
if len(sys.argv)>3:
    suffix = sys.argv[3]
    nSpins = 2
    if suffix=="Up":
        iSpin = 0
    elif suffix=="Dn":
        iSpin = 1
    else:
        print("<spin> must be 'Up' or 'Dn' and only specified for spin-polarized calculations.")
        exit(1)

np.set_printoptions(linewidth=250, precision=5)
nInterp = 10

#Get mu or VBM, kfold, spin type and lattice vectors from output file:
mu = np.nan
R = np.zeros((3,3))
iLine = 0
refLine = -10
initDone = False
Rdone = False
for line in open('totalE.out'):
    if line.startswith('Initialization completed'):
        initDone = True
    if initDone and line.find('FillingsUpdate:')>=0:
        mu = float(line.split()[2])
    if (not initDone) and line.startswith('spintype'):
        spinKey = line.split()[1]
        mlwfType = (np.complex128 if (spinKey=="vector-spin" or spinKey=="spin-orbit") else np.float64)
        if (spinKey == "z-spin") ^ (nSpins == 2):
            print("Optional argument <spin> should be specified only for z-spin calculations.")
            exit(1)
    if (not initDone) and line.startswith('kpoint-folding'):
        kfold = np.array([int(tok) for tok in line.split()[1:4]])
    if (not initDone) and line.startswith('nElectrons:'):
        nElectrons = float(line.split()[1])
        nValence = int(nElectrons/2) #number of valence bands
        nBandsDFT = int(line.split()[3])
        Edft = np.reshape(np.fromfile('totalE.eigenvals'), (nSpins,-1,nBandsDFT))[iSpin]
        mu = np.max(Edft[:,:nValence]) #VBM
    if line.find('Initializing the Grid') >= 0:
        refLine = iLine
    if not Rdone:
        rowNum = iLine - (refLine+2)
        if rowNum>=0 and rowNum<3:
            R[rowNum,:] = [ float(x) for x in line.split()[1:-1] ]
        if rowNum==3:
            Rdone = True
    iLine += 1
mu += dmu #include offset
kfoldProd = np.prod(kfold)
kStride = np.array([kfold[1]*kfold[2], kfold[2], 1])

#Check lattice vectors:
assert(np.linalg.norm(R[:2,2]) < 1e-6)
assert(np.linalg.norm(R[2,:2]) < 1e-6)

#Read the wannier cell map and weights:    
cellMap = np.loadtxt("wannier.mlwfCellMap"+suffix, usecols=[0,1,2]).astype(int)
Wwannier = np.fromfile("wannier.mlwfCellWeights"+suffix)
nCells = cellMap.shape[0]
nBands = int(np.sqrt(Wwannier.shape[0] / nCells))
Wwannier = Wwannier.reshape((nCells,nBands,nBands)).swapaxes(1,2)

#Read and expand Wannier Hamiltonian:
Hreduced = np.fromfile("wannier.mlwfH"+suffix, dtype=mlwfType).reshape((kfoldProd,nBands,nBands)).swapaxes(1,2)
iReduced = np.dot(np.mod(cellMap, kfold[None,:]), kStride)
Hwannier = Wwannier * Hreduced[iReduced]

#Read and expand Wannier weights:
WmWannier = Wwannier * np.fromfile("wannier.mlwfWm"+suffix, dtype=mlwfType).reshape((kfoldProd,nBands,nBands)).swapaxes(1,2)[iReduced]
WpWannier = Wwannier * np.fromfile("wannier.mlwfWp"+suffix, dtype=mlwfType).reshape((kfoldProd,nBands,nBands)).swapaxes(1,2)[iReduced]

#Construct 2D Brillouin zone:
Rpar = R[:2,:2] #in-plane lattice vectors
G = (2*np.pi) * np.linalg.inv(Rpar)

#--- Nearby reciprocal lattice vectors:
iGgrid = np.arange(-3,4)
iiG0, iiG1 = np.meshgrid(iGgrid, iGgrid, indexing='ij')
iG = np.array([iiG0.flatten(), iiG1.flatten()]).T
iG = iG[np.sum(iG**2, axis = 1) > 0]  #eliminate origin
Gnear = np.dot(iG, G)                   #nearby reciprocal lattice vectors

#--- Distance to BZ boundary (Cartesian coordinates in last axis): <0 == interior, >0 == exterior
def bzBoundaryDist(kCart):
    kMag = np.sqrt(np.sum(kCart**2, axis=-1))
    kNearDist =  np.sqrt(np.min(np.sum((kCart[...,None,:] - Gnear[(None,)*(len(kCart.shape)-1) + (Ellipsis,)])**2, axis=-1), axis=-1))
    return 0.5*(kMag - kNearDist)

#--- BZ boundary edge intersection
def bzEdgeIntersect(v0, v1):
    tol = 1e-6
    den = np.dot(v1-v0, Gnear.T)
    den = np.copysign(np.maximum(tol,np.abs(den)), den) #regularize division below
    alpha = (0.5*GnearSq[None,:] - np.dot(v0, Gnear.T)) / den
    vCut = v0[...,None] + alpha[:,None,:]*(v1-v0)[...,None] #Cuts with each plane
    vCutSq = np.where(np.logical_and(np.logical_and(alpha>-tol, alpha<1.+tol), np.abs(den)>tol), np.sum(vCut**2, axis=1), np.inf) #distance of each valid cut from origin
    nearestCut = vCutSq.argmin(axis=1) #plane intersection closest to origin for each edge
    alphaSel = alpha[(range(len(nearestCut)),nearestCut)]
    #Deal with pathological corner cases (eg. one or both points on plane):
    iBad = np.where(np.isinf(vCutSq[(range(len(nearestCut)),nearestCut)]))[0]
    if len(iBad)>0:
        vBdistMag = np.abs(bzBoundaryDist(np.array([v0[iBad],v1[iBad]])))
        alphaSel[iBad] = vBdistMag.argmin(axis=0) #pick the point closer to BZ in this case
    return v0 + alphaSel[:,None]*(v1-v0), alphaSel

#--- Keep only nearby G's that will contribute faces to BZ:
tol = 1e-3
Gnear = Gnear[bzBoundaryDist(0.5*Gnear) < tol]
GnearSq = np.sum(Gnear**2, axis=-1)

#--- Find BZ vertices:
vertCart = [] #vertex position in cartesian coordinates
for i1 in range(Gnear.shape[0]-1):
    for i2 in range(i1+1,Gnear.shape[0]):
        Lhs = 0.5*Gnear[(i1,i2),:]
        rhs = np.sum(Lhs**2, axis=-1)
        if np.abs(np.linalg.det(Lhs))>tol: #not singular
            vertCur = np.linalg.solve(Lhs, rhs)
            if bzBoundaryDist(vertCur)<tol:
                vertCart.append(vertCur)

vertCart = np.array(vertCart)

#--- Weld vertices:
vertMap = []
for i1 in range(vertCart.shape[0]):
    newVert = True
    for i2 in range(i1):
        if np.linalg.norm(vertCart[i2]-vertCart[i1]) < tol:
            newVert = False
            break
    if newVert:
        vertMap.append(i1)
vertCart = vertCart[vertMap]

#--- Sort vertices in order:
vertCart = vertCart[np.arctan2(vertCart[:,1],vertCart[:,0]).argsort()]
vertCartClosed = np.vstack((vertCart, vertCart[:1])) #close polygon
print('Constructed kx-ky BZ polygon with', vertCart.shape[0], 'edges and vertices')

#Uniform k-mesh on BZ bounding box:
kMax = np.max(np.abs(vertCart), axis=0)
kGrid = [None]*2 # surely there's a better way to do this
dk = np.zeros(2)
for iDim in range(2):
    dk[iDim] = 2*kMax[iDim]/nkPerDim
    kGrid[iDim] = np.linspace(-kMax[iDim]-dk[iDim], kMax[iDim]+dk[iDim], nkPerDim+3)
kx,ky = np.meshgrid(kGrid[0], kGrid[1], indexing='ij')  #Cartesian
kCart = np.concatenate((kx[...,None], ky[...,None]), axis=-1)
ikOut = np.where(bzBoundaryDist(kCart)>tol)

#Use symmetries to reduce number of computed k-points:
#--- Read symmetries:
symData = np.loadtxt('totalE.sym').reshape((-1,4,3))
symTrans = symData[:,3] #translations in lattice coordinates
sym = symData[:,:3] #rotations in lattice coordinates
assert(np.linalg.norm(sym[:,2,:2])< 1e-6) #symmetries should not connect in-plane to normal direction
assert(np.linalg.norm(sym[:,:2,2])< 1e-6) #symmetries should not connect in-plane to normal direction
sym = sym[:,:2,:2] #in-plane lattice symmetries
symCart = np.einsum('ab,sbc,cd->sad', Rpar, sym, np.linalg.inv(Rpar))
nSym = len(sym)

#--- Determine kCart mapping under symmetries:
meshDim = np.array(kCart.shape[:2])
kList = kCart.reshape((-1,2)) #flattened list
ikReduced = np.arange(kList.shape[0]) #index to unique k-point
kStride = np.array([meshDim[1], 1])
for invert in [-1,+1]:
    for rot in symCart:
        kRot = np.dot(kList, rot)*invert
        kRotMesh = (kRot - kList[0,None])/dk[None] #in bbox mesh coordinates
        ikRotMesh = np.round(kRotMesh).astype(int)
        exact = np.where(np.logical_and(
            np.linalg.norm(kRotMesh-ikRotMesh, axis=1) < 1e-3,
            np.logical_and(
                (ikRotMesh>=0).all(axis=1),
                (ikRotMesh<meshDim[None]).all(axis=1)
        )))[0] #exact match to a point on bbox mesh
        rotIndex = np.dot(ikRotMesh[exact], kStride)
        ikReduced[exact] = np.minimum(ikReduced[exact], rotIndex)

ikReducedUniq = np.unique(ikReduced)
kReduced = kList[ikReducedUniq]
nkReduced = len(kReduced)

#--- Replace ikReduced by index in reduced set (currently still points to full set):
nkMesh = len(kList)
ikReducedUniqInv = np.zeros(nkMesh, dtype=int) #map from full set to unique index
ikReducedUniqInv[ikReducedUniq] = np.arange(nkReduced)
ikReduced = ikReducedUniqInv[ikReduced]
print('Reduced', nkMesh, 'k-points in bbox k-mesh to', nkReduced, 'under symmetries', flush=True)

# Calculate eigenvalues:
blockSize = 100 #Maximum number of k-points computed at a time
nBlocks = int(np.ceil(nkReduced/blockSize))
Ek = np.zeros((nkReduced, nBands))
cellMap2D = cellMap[:,:2]

print('Computing', nBlocks, 'blocks: ', end='', flush=True)
for iBlock in range(nBlocks):
    ikStart = blockSize*iBlock
    ikStop = np.minimum(ikStart+blockSize, nkReduced)
    kpoints = np.dot(kReduced[ikStart:ikStop], np.linalg.inv(G)) #recip. lattice coordinates
    phase = np.exp((2j*np.pi)*np.dot(kpoints,cellMap2D.T))
    Ek[ikStart:ikStop] = np.linalg.eigvalsh(np.tensordot(phase, Hwannier, axes=1))
    print(iBlock+1, end=' ', flush=True)
print('done.')

#--- Expand to full set:
Ek = Ek[ikReduced].reshape(tuple(meshDim)+(nBands,))

#Draw BZ outline:
for iw in range(2):
    plt.figure(iw+1)
    plt.plot(vertCartClosed[:,0], vertCartClosed[:,1], 'k', lw=2)
    plt.gca().set_aspect('equal')
    
    # x-axis labels
    plt.text(min(vertCartClosed[:,0]) - 0.05, min(vertCartClosed[:,1]) - 0.05, '-$\pi/a$')
    plt.text(max(vertCartClosed[:,0]) - 0.05, min(vertCartClosed[:,1]) - 0.05, '+$\pi/a$')
    
    # y-axis labels
    plt.text(min(vertCartClosed[:,0]) - 0.1, min(vertCartClosed[:,1]), '-$\pi/a$')
    plt.text(min(vertCartClosed[:,0]) - 0.1, max(vertCartClosed[:,1]), '+$\pi/a$')
    
    plt.axis('off')

#Construct custom colormaps:
t = np.linspace(0., 1., 256)[:,None]
color0 = np.array([0.,0.,0., 0.05]) #RGBA color corresponding to zero weight
colorP = np.array([0.,0.,1., 1.]) #RGBA color corresponding to full weight for +z surface (Mo-end)
colorM = np.array([1.,0.,0., 1.]) #RGBA color corresponding to full weight for -z surface (P-end)
cmapP = ListedColormap(color0[None,:]*(1.-t) + colorP[None,:]*t)
cmapM = ListedColormap(color0[None,:]*(1.-t) + colorM[None,:]*t)
cmaps = [ cmapM, cmapP ]

#Compute isosurfaces of each band:
mesh = []
iSave = 0
fig, ax = plt.subplots()
maxSegments = 0
for iBand in range(nBands):
    Ei = Ek[...,iBand]
    Emin = np.min(Ei)
    Emax = np.max(Ei)
    if Emin < mu and Emax > mu: #band cuts Fermi level
        
        verts = measure.find_contours(Ei, mu)
        
        #Convert to a single indexed array:
        nVerts = 0
        edges = []
        for iv, v in enumerate(verts):
            verts[iv] = v * dk[None,:] + kCart[0,0,None,:] #convert to Cartesian k (from mesh coords)
            edges.append(np.array([np.arange(nVerts,nVerts+len(v)-1), np.arange(nVerts+1,nVerts+len(v))]).T)
            nVerts += len(v)
        verts = np.vstack(verts)
        edges = np.vstack(edges)
        
        #Identify vertices outside the BZ and drop/truncate edges by BZ:
        bzDist = bzBoundaryDist(verts)
        vertOut = bzDist>0.
        iSwap = np.where(bzDist[edges[:,0]] < bzDist[edges[:,1]])
        edges[iSwap] = edges[iSwap,::-1] #reorder so that first vertex is (more) outside
        
        #--- Drop edges that are completely outside:
        outside = np.logical_and(vertOut[edges[:, 0]], vertOut[edges[:, 1]])
        inside = np.logical_not(outside)
        edges = edges[inside]
        
        #--- Truncate edges that cut the BZ boundary:
        condition = np.logical_xor(vertOut[edges[:, 0]], vertOut[edges[:, 1]])
        iCut = np.where(condition)[0]
        vertsNew, _ = bzEdgeIntersect(verts[edges[:, 0]], verts[edges[:, 1]])
        edges[iCut,0] = nVerts + iCut
        verts = np.vstack([verts, vertsNew])
        nVerts = len(verts)
        
        #--- Drop degenerate edges:
        edgeLength = np.linalg.norm(verts[edges[:,0]] - verts[edges[:,1]], axis=1)
        edges = edges[edgeLength > 1e-3]
        
        #--- Remove unused vertices
        vertSel = np.unique(edges)
        vertSelInv = np.zeros(nVerts, dtype=int)
        vertSelInv[vertSel] = np.arange(len(vertSel), dtype=int)
        verts = verts[vertSel]; nVerts = verts.shape[0]
        edges = vertSelInv[edges]
        print('Fermi level cuts band', iBand+1, 'with', len(edges), 'edges and', len(verts), 'vertices. ', end='', flush=True)
        
        #Compute normals and tau if needed:
        nBlocks = int(np.ceil(nVerts/blockSize))
        wm = np.zeros(nVerts)
        wp = np.zeros(nVerts)
        print('Computing', nBlocks, 'blocks: ', end='', flush=True)
        for iBlock in range(nBlocks):
            ikStart = blockSize*iBlock
            ikStop = np.minimum(ikStart+blockSize, nVerts)
            kpoints = np.dot(verts[ikStart:ikStop], np.linalg.inv(G)) #recip. lattice coordinates
            phase = np.exp((2j*np.pi)*np.dot(kpoints,cellMap2D.T))
            Vi = np.vstack(
                    [ 
                        eigh(
                            H, 
                            subset_by_index = [iBand, iBand]
                        )[1].T for H in np.tensordot(phase, Hwannier, axes = 1) 
                    ]
                ) #only compute required eigenvector
            wm[ikStart:ikStop] = np.einsum("ka,kab,kb->k", Vi.conjugate(), np.tensordot(phase, WmWannier,  axes=1), Vi).real
            wp[ikStart:ikStop] = np.einsum("ka,kab,kb->k", Vi.conjugate(), np.tensordot(phase, WpWannier,  axes=1), Vi).real
            print(iBlock+1, end=' ', flush=True)
        print('done.')
                
        #Plot:
        segments = np.concatenate([verts[edges[:,0],None], verts[edges[:,1],None]], axis=1)
        
        if len(segments.flatten()) > 0 and max(segments.flatten()) > maxSegments:
            maxSegments = max(segments.flatten())

        segments_save = segments.reshape(-1, 2)
        if len(segments.flatten()) > 0:
            np.savetxt(f'segments{iSave}.txt', segments_save)
            np.savetxt(f'edges{iSave}.txt', edges)
            np.savetxt(f'wm{iSave}.txt', wm)
            np.savetxt(f'wp{iSave}.txt', wp)
            iSave += 1
        for iw, w in enumerate([wm, wp]):
            lc = LineCollection(segments, cmap=cmaps[iw], norm=plt.Normalize(0, 1))
            lc.set_array(0.5*(w[edges[:,0]]+w[edges[:,1]]))
            ax.add_collection(lc)
np.savetxt(f'maxSegments.txt', [maxSegments])
np.savetxt(f'numSave.txt', [iSave])
ax.set_xlim([-maxSegments, maxSegments])
ax.set_ylim([-maxSegments, maxSegments])
