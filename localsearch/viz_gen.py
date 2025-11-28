
from distance import edge_hamming_distance

import os.path as osp
import numpy as np
from matplotlib import pyplot as plt

import tsp
from draw.base import plot_tsp


root_path = osp.dirname(osp.dirname(tsp.__file__))


NSCALE = 10
SOLUTION_DATASET_PATH = osp.join(root_path, "datasets")
SOLUTION_DATASET_NAME = f"sol_{NSCALE}n_1280000t_0.npy"

ALGO = "2swap"
OPTIMA_DATASET_PATH = osp.join(root_path, "local_optima_datasets")
OPTIMA_DATASET_NAME = f"{ALGO}_2lo_{NSCALE}n_1280000t_0.npz"

VIZ_NUM = 20 


if __name__ == "__main__":
    with open(osp.join(SOLUTION_DATASET_PATH, SOLUTION_DATASET_NAME), "rb") as f:
        oracle_solutions = np.load(f)
    
    with open(osp.join(OPTIMA_DATASET_PATH, OPTIMA_DATASET_NAME), "rb") as f:
        npz = np.load(f)
        optima = npz["optima"]
        swaps = npz["swaps"]

    assert len(oracle_solutions) == len(optima) == len(swaps)

    pidxs = np.random.choice(len(oracle_solutions), size=VIZ_NUM, replace=False)

    sols = oracle_solutions[pidxs]
    opts = optima[pidxs]
    swps = swaps[pidxs]

    opt_per_prob = opts.shape[1]
    nscale = opts.shape[-1]

    fig, axes = plt.subplots(VIZ_NUM, 1 + opt_per_prob, sharey=True, figsize=(6 * (1 + opt_per_prob), 6 * VIZ_NUM))
    
    for ridx in range(VIZ_NUM):
        oracle_ax = axes[ridx][0]
        plot_tsp(sols[ridx], np.arange(nscale), oracle_ax, line_col="green")
        oracle_ax.set_xlabel("Oracle")
        oracle_ax.set_ylabel(f"Idx {pidxs[ridx]}")

        for oidx in range(opt_per_prob):
            opt_ax = axes[ridx][oidx+1]
            plot_tsp(sols[ridx], opts[ridx, oidx], opt_ax)
            res = edge_hamming_distance(opts[ridx, oidx], np.arange(nscale))
            opt_ax.set_xlabel(f"Cython {ALGO} | {swps[ridx, oidx]} swaps\n{res} edge-space residual")

    save_path = osp.join(osp.dirname(osp.dirname(tsp.__file__)), "residual/viz_gen.png")
    fig.savefig(save_path)
    