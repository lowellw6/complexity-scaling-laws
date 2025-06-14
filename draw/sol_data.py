"""
Draw solution datasets for qualitative validation and analysis

Inputs:
1) Problem dataset (coordinates) w/ shape (batch, nodes, dims)
2) Solution tour selection dataset (pointers) w/ shape (batch, sol_per_problem, nodes, dims)

Supports 2D and 3D TSP. Plots a random tour for comparison like 3d.py
"""

import argparse
from matplotlib import pyplot as plt
import torch
import numpy as np

three_d = __import__("3d")
from base import plot_tsp


FORMAT = "svg"

NODE_COLOR = "blue"
NODE_SIZE = 40

DATA_TOUR_COLOR = "green"
RANDOM_TOUR_COLOR = "orange"
ARROW_RATIO = 0.2  # smaller = shorter arrow head
TOUR_WIDTH = 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prob_path",
        help="path to .npy containing TSP problems",
    )
    parser.add_argument(
        "tour_path",
        help="path to .npy containing corresponding TSP tours",
    )
    parser.add_argument(
        "batch_range",
        type=int,
        nargs=2,
        help="batch indices (dim 0 slice) to plot"
    )
    args = parser.parse_args()

    probs = np.load(args.prob_path, mmap_mode="r")
    tours = np.load(args.tour_path, mmap_mode="r")

    start_idx, end_idx = args.batch_range
    num_tours = end_idx - start_idx

    probs_slice = probs[start_idx:end_idx]
    tours_slice = tours[start_idx:end_idx]

    assert len(probs_slice.shape) == 3 and len(tours_slice.shape) == 3
    assert probs_slice.shape[1] == tours_slice.shape[2]
    _, sol_per_prob, nodes = tours_slice.shape
    dims = probs_slice.shape[-1]

    subplot_kws = {'projection': "3d"} if dims == 3 else {}

    total_tours = sol_per_prob * num_tours
    flat_tours_slice = np.reshape(tours_slice, (total_tours, nodes))

    fig, axes = plt.subplots(total_tours, 2, figsize=(10, 5 * total_tours), subplot_kw=subplot_kws)
    for idx, (data_ax, rand_ax) in enumerate(axes):
        prob = probs_slice[idx // sol_per_prob]
        tour = flat_tours_slice[idx]

        rand_tour = tour[torch.randperm(len(tour))]

        if dims == 2:
            plot_tsp(prob, tour, data_ax, NODE_COLOR, line_col=DATA_TOUR_COLOR)
            plot_tsp(prob, rand_tour, rand_ax, NODE_COLOR, line_col=RANDOM_TOUR_COLOR)
        elif dims == 3:
            three_d.plot_tsp_3d(prob[tour], data_ax, NODE_SIZE, NODE_COLOR, DATA_TOUR_COLOR, ARROW_RATIO, TOUR_WIDTH)
            three_d.plot_tsp_3d(prob[rand_tour], rand_ax, NODE_SIZE, NODE_COLOR, RANDOM_TOUR_COLOR, ARROW_RATIO, TOUR_WIDTH)        
        else:
            raise Exception(f"Can only support 2D and 3D data, but data has '{dims}' dimensions")

    fig.suptitle("Dataset Solutions (Left) vs Random (Right)", fontsize=16)

    input_stub = args.tour_path.split(".")[0]
    if FORMAT in ("eps", "svg"):
        fig.savefig(input_stub + f".{FORMAT}", format=FORMAT)
    else:  # assuming non-vector with dpi
        fig.savefig(input_stub + f".{FORMAT}", format=FORMAT, dpi=300)
