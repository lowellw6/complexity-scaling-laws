
import os.path as osp
import torch
import numpy as np
from matplotlib import pyplot as plt

import tsp
from draw.base import plot_tsp

from search import two_opt_search_python
from distance import edge_hamming_distance, tei_node_hamming_distance


N_SCALE = 50
DATASET_SIZE = 1_280_000
DATASET_IDX = 0



def grab_both_residuals(tour_a, tour_b):
    edge_residual = edge_hamming_distance(tour_a, tour_b)
    assert edge_residual == edge_hamming_distance(tour_b, tour_a)  # test vs reference tour swap shouldn't change outcome

    node_residual = tei_node_hamming_distance(tour_a, tour_b)
    assert node_residual == tei_node_hamming_distance(tour_b, tour_a)  # test vs reference tour swap shouldn't change outcome

    return edge_residual, node_residual



if __name__ == "__main__":
    # load an example pyconcorde dataset problem (later will take mean over many problems)
    data_dir = osp.join(osp.dirname(osp.dirname(tsp.__file__)), "datasets")
    assert osp.exists(data_dir), f"can't find dataset directory: {data_dir}"

    dpath = osp.join(data_dir, f"sol_{N_SCALE}n_{DATASET_SIZE}t_{DATASET_IDX}.npy")

    with open(dpath, "rb") as f:
        dataset = np.load(f)

    assert dataset.shape == (DATASET_SIZE, N_SCALE, 2)

    optimal_example = dataset[0]

    # generate a random tour
    random_tour = torch.randperm(N_SCALE).numpy()

    # compute local optimum with pure python 2-opt running until no improvement can be made
    local_optimal_tour, swaps = two_opt_search_python(optimal_example, random_tour)

    # compute distance (residual) between 2-opt local optimum and global optumum (loaded order)
    edge_residual, node_residual = grab_both_residuals(local_optimal_tour, np.arange(N_SCALE))
    
    # do same for random baseline
    rand_edge_residual, rand_node_residual = grab_both_residuals(random_tour, np.arange(N_SCALE))

    # plot tours along with residual
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(22, 6))
    rand_ax, two_opt_ax, oracle_ax = axes

    plot_tsp(optimal_example, random_tour, rand_ax, line_col="orange")
    plot_tsp(optimal_example, local_optimal_tour, two_opt_ax)
    plot_tsp(optimal_example, np.arange(N_SCALE), oracle_ax, line_col="green")

    rand_ax.set_xlabel(f"Random\n{rand_edge_residual} edge-space residual | {rand_node_residual} node-space residual")
    two_opt_ax.set_xlabel(f"2-OPT | {swaps} swaps\n{edge_residual} edge-space residual | {node_residual} node-space residual")
    oracle_ax.set_xlabel("Oracle")

    save_path = osp.join(osp.dirname(osp.dirname(tsp.__file__)), "residual/demo.png")
    fig.savefig(save_path)