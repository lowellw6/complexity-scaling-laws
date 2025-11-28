
import pyximport; pyximport.install(language_level=3)
from csearch import two_opt_search, batch_two_opt_search, batch_two_swap_search
from search import two_opt_search_python
from distance import edge_hamming_distance

from time import time

import os.path as osp
import torch
import numpy as np
from matplotlib import pyplot as plt

import tsp
from draw.base import plot_tsp

N_SCALE = 50
DATASET_SIZE = 1_280_000
DATASET_IDX = 0

TEST_BATCH = 10  # batch cython func test, use even value, set 0 to skip
PLOT_BATCH = True  # skips plotting batch test, set True if TEST_BATCH is more than a few



if __name__ == "__main__":
    # load an example pyconcorde dataset problem (later will take mean over many problems)
    data_dir = osp.join(osp.dirname(osp.dirname(tsp.__file__)), "datasets")
    assert osp.exists(data_dir), f"can't find dataset directory: {data_dir}"

    dpath = osp.join(data_dir, f"sol_{N_SCALE}n_{DATASET_SIZE}t_{DATASET_IDX}.npy")

    with open(dpath, "rb") as f:
        dataset = np.load(f)

    assert dataset.shape == (DATASET_SIZE, N_SCALE, 2)

    optimal_example = dataset[0].astype(np.double)

    # generate a random tour
    random_tour = torch.randperm(N_SCALE).numpy().astype(np.int32)

    # test python version and time
    before = time()
    python_result, python_swaps = two_opt_search_python(optimal_example, random_tour)
    python_time = time() - before

    # now cython version and time
    before = time()
    cython_result, cython_swaps = two_opt_search(optimal_example, random_tour)
    cython_time = time() - before

    assert np.all(python_result == cython_result), "Python and Cython 2-OPT results don't match!"

    # test batch cython version
    batch_examples = dataset[1:TEST_BATCH+1].astype(np.double)
    batch_rand_tours = np.stack([torch.randperm(N_SCALE).numpy().astype(np.int32) for _ in range(TEST_BATCH)], axis=0)

    before = time()
    batch_results, batch_swaps = batch_two_opt_search(batch_examples, batch_rand_tours)
    batch_time = time() - before
    avg_batch_time = batch_time / TEST_BATCH

    assert batch_results.shape == (TEST_BATCH, N_SCALE)
    assert batch_swaps.shape == (TEST_BATCH,)

    # test batch 2swap (aka 2exchange)
    before = time()
    exc_batch_results, exc_batch_swaps = batch_two_swap_search(batch_examples, batch_rand_tours)
    exc_batch_time = time() - before
    exc_avg_batch_time = exc_batch_time / TEST_BATCH

    assert exc_batch_results.shape == (TEST_BATCH, N_SCALE)
    assert exc_batch_swaps.shape == (TEST_BATCH,)

    # compute distance (residual) between 2-opt local optimum and global optumum (loaded order)
    python_dist = edge_hamming_distance(python_result, np.arange(N_SCALE))
    cython_dist = edge_hamming_distance(cython_result, np.arange(N_SCALE))
    rand_dist = edge_hamming_distance(random_tour, np.arange(N_SCALE))

    # plot tours along with residual
    batch_test_mult = int(np.ceil(TEST_BATCH / 2)) if PLOT_BATCH else 0
    fig, axes = plt.subplots(2 + 2 * batch_test_mult, 2, sharey=True, figsize=(12, 12 + 6 * 2 * batch_test_mult))
    (rand_ax, oracle_ax), (popt_ax, copt_ax) = axes[:2]

    plot_tsp(optimal_example, random_tour, rand_ax, line_col="orange")
    plot_tsp(optimal_example, np.arange(N_SCALE), oracle_ax, line_col="green")
    plot_tsp(optimal_example, python_result, popt_ax)
    plot_tsp(optimal_example, cython_result, copt_ax)

    rand_ax.set_xlabel(f"Random\n{rand_dist} edge-space residual")
    oracle_ax.set_xlabel("Oracle")

    popt_ax.set_xlabel(f"Python 2-OPT | {python_swaps} swaps | {python_time:.3} sec\n{python_dist} edge-space residual")
    copt_ax.set_xlabel(f"Cython 2-OPT | {cython_swaps} swaps | {cython_time:.3} sec\n{cython_dist} edge-space residual")

    if PLOT_BATCH:
        for row_idx in range(2, 2 + batch_test_mult):
            bt_idx = 2 * (row_idx - 2)
            ax1, ax2 = axes[row_idx]

            plot_tsp(batch_examples[bt_idx], batch_results[bt_idx], ax1)
            plot_tsp(batch_examples[bt_idx + 1], batch_results[bt_idx + 1], ax2)

            dist1 = edge_hamming_distance(batch_results[bt_idx], np.arange(N_SCALE))
            dist2 = edge_hamming_distance(batch_results[bt_idx + 1], np.arange(N_SCALE))

            ax1.set_xlabel(f"Cython 2-OPT | {batch_swaps[bt_idx]} swaps | {avg_batch_time:.3} sec avg\n{dist1} edge-space residual")
            ax2.set_xlabel(f"Cython 2-OPT | {batch_swaps[bt_idx + 1]} swaps | {avg_batch_time:.3} sec avg\n{dist2} edge-space residual")

        for row_idx in range(2 + batch_test_mult, 2 + 2 * batch_test_mult):
            bt_idx = 2 * (row_idx - batch_test_mult - 2)
            ax1, ax2 = axes[row_idx]

            plot_tsp(batch_examples[bt_idx], exc_batch_results[bt_idx], ax1)
            plot_tsp(batch_examples[bt_idx + 1], exc_batch_results[bt_idx + 1], ax2)

            dist1 = edge_hamming_distance(exc_batch_results[bt_idx], np.arange(N_SCALE))
            dist2 = edge_hamming_distance(exc_batch_results[bt_idx + 1], np.arange(N_SCALE))

            ax1.set_xlabel(f"Cython 2-SWAP | {exc_batch_swaps[bt_idx]} swaps | {exc_avg_batch_time:.3} sec avg\n{dist1} edge-space residual")
            ax2.set_xlabel(f"Cython 2-SWAP | {exc_batch_swaps[bt_idx + 1]} swaps | {exc_avg_batch_time:.3} sec avg\n{dist2} edge-space residual")

    print(f"Total batch 2opt Cython time: {batch_time:.3} sec")
    print(f"Avg batch 2opt Cython time: {avg_batch_time:.3} sec")

    print(f"Total batch 2swap Cython time: {exc_batch_time:.3} sec")
    print(f"Avg batch 2swap Cython time: {exc_avg_batch_time:.3} sec")

    save_path = osp.join(osp.dirname(osp.dirname(tsp.__file__)), "residual/test_cython.png")
    fig.savefig(save_path)