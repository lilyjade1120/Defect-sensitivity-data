import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from typing import Tuple

# Unit definitions:
Angstrom = 1/0.5291772
eV = 1/27.2114


def main() -> None:
	# Get paths and correspnding material-name labels:
	if (len(sys.argv) < 3) or (len(sys.argv) % 2 != 1):
		print(
			"Usage: python supercellConvergence.py <dir1> <label1>"
			" [<dir2> <label2> ...]"
		)
		exit(1)
	prefixes = sys.argv[1::2]
	labels = sys.argv[2::2]
	nMaterials = len(prefixes)
	assert len(labels) == nMaterials
	
	# Determine what all to plot:
	collectFunctions = [
		collectFunction
		for collectFunction, filename in (
			(collectWannier, "wannier.mlwfCellMap"),
			(collectPhonon, "totalE.phononCellMap"),
			(collectEph, "wannier.mlwfCellMapPh"),
		)
		if anyHaveFile(prefixes, filename)
	]
	nQuantities = len(collectFunctions)
	
	fig, axes = plt.subplots(3, 2, sharex = True, sharey = 'row', squeeze = False, figsize = (8, 6)) 
	plt.subplots_adjust(hspace=0.2, wspace=0.2)
	prefixes = np.array(prefixes).reshape((3, 2))
	labels = np.array(labels).reshape((3, 2))
	collectFunction = collectFunctions[0]
	for axesRow, prefixRow, labelRow in zip(axes, prefixes, labels):
		for prefix, ax, label in zip(prefixRow, axesRow, labelRow):
			# Collect 2D histograms:
			hist = Histogram2D(29., 3E-7, 3.0, 100)
			ylabel, rpower = collectFunction(prefix, hist)
			data = hist.data / (
				hist.r[:, None] ** rpower
			)  # balance histogram probabilities from small to large r
			vmax = get_percentile(
				data[data != 0.], 95
			)  # make colormap scaling insensitive to outliers
			# Plot:
			plt.sca(ax)
			plt.imshow(
				data.T, vmax=vmax,
				extent=hist.extent, origin='lower',
				aspect='auto', cmap='Blues',
			)
			ax.text(
				0.5, 1.0, label, ha='center', va='bottom',
				transform=ax.transAxes, fontweight="bold",
			)
		axesRow[0].set_ylabel(ylabel)
		ytickpowers = [0, -2, -4, -6]
		axesRow[0].set_yticks(ytickpowers)
		axesRow[0].set_yticklabels(
			[(f"$10^{{{power}}}$" if power else "$0$") for power in ytickpowers]
		)

	for ax in axes[-1]:
		ax.set_xlabel("$r$ [$\AA$]")
	plt.savefig("supercellConvergence.pdf", bbox_inches='tight', dpi=300)
	plt.show()


class Histogram2D:
	"""Wraps numpy.histogram2d specialized for the matrix element plots."""
	
	def __init__(
		self, r_max: float, y_min: float, y_max: float, N: int
	) -> None:
		"""
		Set up to collect from 0 to r_max, and from y_min to y_max,
		with N intervals along each dimension.
		"""
		self.extent = (0., r_max, np.log10(y_min), np.log10(y_max))
		self.r_edges = np.linspace(*self.extent[:2], N+1)
		self.y_edges = np.logspace(*self.extent[2:], N+1)
		self.data = np.zeros((N, N))
	
	def add_events(self, r: np.ndarray, y: np.ndarray) -> None:
		self.data += np.histogram2d(
			r.flatten(), y.flatten(), (self.r_edges, self.y_edges)
		)[0]
	
	@property
	def r(self) -> np.ndarray:
		return 0.5*(self.r_edges[:-1] + self.r_edges[1:])
	
	@property
	def y(self) -> np.ndarray:
		return 0.5*(self.y_edges[:-1] + self.y_edges[1:])


def collectWannier(prefix: str, hist: Histogram2D) -> Tuple[str, int]:
	if not os.path.isfile(f"{prefix}/wannier.mlwfCellMap"):
		return
	print(f"Collecting Wannier for '{prefix}'")

	# Get the Wannier cell map and center locations:
	cellMap, cellMapCart, R = getCellMap(f"{prefix}/wannier.mlwfCellMap")
	centers = getCenters(f"{prefix}/wannier.out", R)
	nCells = cellMap.shape[0]
	nBands = centers.shape[0]

	# Read Wannier weights:
	Ww = np.fromfile(
		f"{prefix}/wannier.mlwfCellWeights"
	).reshape((nCells, nBands, nBands)).swapaxes(1, 2)

	# Read and expand Wannier Hamiltonian:
	kfold, mlwfType = getKfoldMlwfType(f"{prefix}/totalE.out")
	kfoldProd = np.prod(kfold)
	kStride = np.array([kfold[1]*kfold[2], kfold[2], 1])
	iReduced = np.mod(cellMap, kfold[None,:]) @ kStride
	absHw = Ww * abs(
		np.fromfile(f"{prefix}/wannier.mlwfH", dtype=mlwfType)
	).reshape(
		(kfoldProd, nBands, nBands)
	).swapaxes(1, 2)[iReduced]
	
	# Collect histogram:
	rDeltaMag = np.linalg.norm(
		cellMapCart[:, None, None, :]
		+ centers[None, None, :, :]
		- centers[None, :, None, :],
		axis=-1
	)  # spacing between all pairs of centers in cell map
	hist.add_events(rDeltaMag/Angstrom, absHw/absHw.max())
	return "$|H_w|$", 1



def collectPhonon(prefix: str, hist: Histogram2D) -> Tuple[str, int]:
	if not os.path.isfile(f"{prefix}/totalE.phononCellMap"):
		return
	print(f"Collecting phonon for '{prefix}'")

	# Get the phonon cell map and atomic geometry:    
	cellMap, cellMapCart, R = getCellMap(
		getCorrectedFilename(f"{prefix}/totalE.phononCellMap")
	)
	atpos = getAtpos(f"{prefix}/totalE.out", R)
	nCells = cellMap.shape[0]
	nAtoms = atpos.shape[0]
	nModes = nAtoms * 3
	
	# Read phonon force matrix (or corrected verion as appropriate)
	absHphonon = abs(np.fromfile(
		getCorrectedFilename(f"{prefix}/totalE.phononOmegaSq"),
		dtype=np.float64,
	)).reshape((nCells, nModes, nModes)).swapaxes(1, 2)
	
	# Collect histogram:
	rDeltaMag = np.repeat(np.repeat(
		np.linalg.norm(
			cellMapCart[:, None, None, :]
			+ atpos[None, None, :, :]
			- atpos[None, :, None, :],
			axis=-1
		),  # spacing between all pairs of atoms in cell map
		3, axis=1), 3, axis=2)  # repeated for each direction of atom motion
	hist.add_events(rDeltaMag/Angstrom, absHphonon/absHphonon.max())
	return "Dyn. matrix", 1


def collectEph(prefix: str, hist: Histogram2D) -> Tuple[str, int]:
	if not os.path.isfile(f"{prefix}/wannier.mlwfCellMapPh"):
		return
	print(f"Collecting e-ph for '{prefix}':", end=' ', flush=True)

	# Get the e-ph cell map, atomic geometry and center locations:
	cellMap, cellMapCart, R = getCellMap(f"{prefix}/wannier.mlwfCellMapPh")
	atpos = getAtpos(f"{prefix}/totalE.out", R)
	centers = getCenters(f"{prefix}/wannier.out", R)
	nCells = cellMap.shape[0]
	nAtoms = atpos.shape[0]
	nModes = nAtoms * 3
	nBands = centers.shape[0]
	
	# Read e-ph cell weights:
	cellWeights = np.repeat(
		np.fromfile(
			f"{prefix}/wannier.mlwfCellWeightsPh"
		).reshape((nCells, nBands, 1, nAtoms)).swapaxes(1, 3),
		3, axis=2).reshape((nCells, nModes, nBands)) #repeat for 3 directions
	
	#Read e-ph matrix elements and prepare quantities for expansion:
	kfold, mlwfType = getKfoldMlwfType(f"{prefix}/totalE.out")
	phononSup = getPhononSup(f"{prefix}/phonon.out")
	prodPhononSup = np.prod(phononSup)
	phononSupStride = np.array([phononSup[1] * phononSup[2], phononSup[2], 1])
	iReduced = np.mod(cellMap, phononSup[None,:]) @ phononSupStride
	absHePhW = abs(
		np.fromfile(f"{prefix}/wannier.mlwfHePh", dtype=mlwfType)
	).reshape(
		(prodPhononSup, prodPhononSup, nModes, nBands, nBands)
	).swapaxes(3, 4)
	absHePhWmax = absHePhW.max()
	
	def get_rmag(R):
		"""Get nModes x nBands atom-wannier center spacings at cell R."""
		return np.repeat(np.linalg.norm(
			R[None, None, :] + centers[None, :, :] - atpos[:, None, :],
			axis=-1
		), 3, axis=0)  # repeated for each direction of atom motion

	# Collect one pair of cells at a time to limit memory usage:
	nCellPairs = len(iReduced) ** 2
	iCellPairInterval = int(np.ceil(0.02 * nCellPairs))
	iCellPairsDone = 0
	for R1, w1, iReduced1 in zip(cellMapCart, cellWeights, iReduced):
		r1mag = get_rmag(R1)
		for R2, w2, iReduced2 in zip(cellMapCart, cellWeights, iReduced):
			r2mag = get_rmag(R2)
			rPairMag = r1mag[:, :, None] + r2mag[:, None, :]
			absHePhWcur = (
				absHePhW[iReduced1, iReduced2] * w1[:, :, None] * w2[:, None, :]
			)
			hist.add_events(rPairMag/Angstrom, absHePhWcur / absHePhWmax)
			# Report progress:
			iCellPairsDone += 1
			if iCellPairsDone % iCellPairInterval == 0:
				percentDone = int(np.round(iCellPairsDone * 100. / nCellPairs))
				print(f"{percentDone:d}%", end=' ', flush=True)
	print("done.", flush=True)
	return "$|g_{ep}|$", 3


def anyHaveFile(prefixes, filename):
	return any(os.path.isfile(f"{prefix}/{filename}") for prefix in prefixes)


def getCorrectedFilename(filename):
	if os.path.isfile(filenameCorr := f"{filename}Corr"):
		return filenameCorr  # Use corrected version if it exists
	else:
		return filename


def getCellMap(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Get fractional cell map, Cartesian cell map and lattice vectors."""
	data = np.loadtxt(filename)
	cellMap = data[:,:3].astype(int)
	cellMapCart = data[:,3:]
	R = np.linalg.lstsq(cellMap, cellMapCart, rcond=None)[0].T
	return cellMap, cellMapCart, R


def getAtpos(filename: str, R: np.ndarray) -> np.ndarray:
	"""Get atomic positions in Cartesian coordinates."""
	ionPosStarted = False
	ionPos = []
	for line in open(filename):
		if ionPosStarted:
			tokens = line.split()
			if len(tokens) and tokens[0] == 'ion':
				ionPos.append([float(token) for token in tokens[2:5]])
			else:
				break
		if line.startswith('# Ionic positions in'):
			isCartesian = (line.split()[4].lower() == "cartesian")
			ionPosStarted = True
	# Return in Cartesian coordinates:
	ionPos = np.array(ionPos)
	if not isCartesian:
		ionPos = ionPos @ R.T
	return ionPos


def getCenters(filename: str, R: np.ndarray) -> np.ndarray:
	"""Get Wannier centers in Cartesian coordinates."""
	centersStarted = False
	centers = []
	for line in open(filename):
		if centersStarted:
			tokens = line.split()
			if (len(tokens) > 5) and tokens[5] == 'spread:':
				centers.append([float(token) for token in tokens[1:4]])
			else:
				break
		if line.startswith('Centers in '):
			isCartesian = (line.split()[2].lower() == "cartesian")
			centersStarted = True
	# Return in Cartesian coordinates:
	centers = np.array(centers)
	if not isCartesian:
		centers = centers @ R.T
	return centers


def getKfoldMlwfType(filename: str) -> Tuple[np.ndarray, np.dtype]:
	"""Get kpoint-folding and dtype of wannier functions from totalE.out"""
	for line in open(filename):
		if line.startswith('Initialization completed'):
			break
		if line.startswith('spintype'):
			spinKey = line.split()[1]
			mlwfType = (
				np.complex128
				if (spinKey=="vector-spin" or spinKey=="spin-orbit")
				else np.float64
			)
		if line.startswith('kpoint-folding'):
			kfold = np.array([int(tok) for tok in line.split()[1:4]])
	return kfold, mlwfType


def getPhononSup(filename: str) -> np.ndarray:
	"""Get phonon supercell dimensions from phonon.out"""
	for line in open(filename):
		tokens = line.split()
		if len(tokens) == 5:
			if tokens[0] == 'supercell' and tokens[4] == '\\':
				return np.array([int(token) for token in tokens[1:4]])
	raise KeyError


def get_percentile(data: np.ndarray, percentile: float) -> float:
	"""Get value of data corresponding to specified percentile (out of 100)."""
	sorted_data = np.sort(data).flatten()
	return sorted_data[int(percentile * 0.01 * len(sorted_data))]


if __name__ == "__main__":
	main()

