from __future__ import annotations
import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import h5py


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_file", type=str, help="HDF5 output file from Conductance")
    parser.add_argument("--file2", type=str, help="HDF5 output file from Conductance", required=False)
    parser.add_argument("--file3", type=str, help="HDF5 output file from Conductance", required=False)
    parser.add_argument("--pristine", type=str, help="HDF5 output file from Conductance", required=False)
    parser.add_argument("--label1", type=str, help="HDF5 output file from Conductance", required=False)
    parser.add_argument("--label2", type=str, help="HDF5 output file from Conductance", required=False)
    parser.add_argument("--label3", type=str, help="HDF5 output file from Conductance", required=False)
    parser.add_argument('--threshtest', action=argparse.BooleanOptionalAction, required=False)
    parser.add_argument('--dmu', type=str, help='Chemical potential shift', required=False)
    args = parser.parse_args()
    #output_prefix = os.path.splitext(args.h5_file)[0]
    h5files = [args.h5_file, args.file2] if args.file2 is not None else [args.h5_file]
    h5files = [args.h5_file, args.file2, args.file3] if args.file3 is not None else h5files
    label1 = "pristine" if args.label1 is None else args.label1
    label2 = "defect 1" if args.label2 is None else args.label2
    label3 = "defect 2" if args.label3 is None else args.label3
    labels = [label1, label2, label3]
    dmu = float(args.dmu) if args.dmu is not None else 0.0
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if args.threshtest:
        k, eigs = readEigs(args.h5_file)
        thresholds = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4]
        tot = len(eigs.reshape(-1))
        print("|threshold|\tfraction ones: (of %d total eigenvalues)" % (tot))
        for threshold in thresholds:
            num_zero = np.count_nonzero(np.abs(eigs) < threshold)
            num_ones = np.count_nonzero(np.abs(eigs - 1.) < threshold)
            print(threshold, " |\t", num_ones/tot, "\tmissing:", tot-num_ones-num_zero)
    font = {'size': 12}
    matplotlib.rc('font', **font)
    plt.figure(figsize=(7.,5.5))
    for _h5, _label, _color in zip(h5files, labels, colors):
        k, eigs, pristine_exists = readEigs(_h5)
        eigs = np.sort(eigs, axis=1)
        eigDiffFrom1 = (1. - eigs).T
        plt.imshow(eigDiffFrom1, 
                   norm=LogNorm(vmin=0.001, vmax=1, clip=True), 
                   extent=(
                            k[0] - (k[1] - k[0])/2, k[-1] + (k[1] - k[0])/2, 0, len(eigs[0])
                        ),
                   aspect='auto', 
                   cmap='afmhot', 
                   interpolation='nearest'
                )
        plt.ylim([0,12])
        #print(f'{eigDiffFrom1 = }')
        #print(f'{len(k) = }')
        #print(np.sum(1/eigDiffFrom1-1)/len(k))
        print((eigs.sum()/(2*(eigDiffFrom1).sum())))
        plt.colorbar(label="Fraction scattered")
        #plt.colorbar(label="|1.0 - eigenvalue|")
        if pristine_exists:
            k_p, eigs_p, _ = readEigs(_h5, "_pristine")
            plt.plot(k_p, eigs_p.sum(axis=1), 'k--', label="pristine")
            plt.legend()
        if args.pristine:
            k_p, eigs_p, _ = readEigs(args.pristine)
            plt.plot(k_p, eigs_p.sum(axis=1), 'k--', label="pristine")
            plt.legend()
    #plt.ylabel("Eigenvalue #")
    plt.ylabel("Active transport modes")
    plt.xlabel("$k$ ($a_0^{-1}$)")
    plt.savefig(f'eigenAnalysis_{dmu}.png')
    #plt.show()

def readEigs(filename, ext=""):
    with h5py.File(filename) as fp:
        #print(list(fp.attrs))
        eigs = np.array(fp["eigs"+ext])
        print("max eig:", eigs.max(), np.greater(eigs, 1).sum())
        #assert np.greater(eigs, 1).sum() == 0
        T = np.array(fp["T"+ext])
        Nk = len(T.squeeze())
        periodic_dirs = [i_dir for i_dir, Nki in enumerate(T.shape) if (Nki > 1)]
        Rbasis = np.array(fp.attrs["Rbasis"])
        k = np.arange(-Nk // 2, Nk // 2) / Nk * (2 * np.pi / np.linalg.norm(Rbasis[:, periodic_dirs[0]]))
        if "eigs_pristine" in fp.keys():
            pristine_exists = True
        else: pristine_exists = False
    return k, np.roll(eigs, Nk//2, axis=0), pristine_exists


if __name__ == "__main__":
    main()
