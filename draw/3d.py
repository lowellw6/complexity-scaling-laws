"""
Plot 3d TSP tours saved as (batch, nodes, dimensions) shaped numpy array

Plots random tours as well on right for reference
"""

import argparse
from matplotlib import pyplot as plt
import torch
import numpy as np



FORMAT = "png"

NODE_COLOR = "blue"
NODE_SIZE = 40

DATA_TOUR_COLOR = "green"
RANDOM_TOUR_COLOR = "orange"
ARROW_RATIO = 0.2  # smaller = shorter arrow head
TOUR_WIDTH = 1


def plot_tsp_3d(tour, ax, node_size, node_color, tour_color, arrow_ratio, arrow_width):
    x, y, z = np.split(tour, 3, axis=-1)
    dx = np.roll(x, -1) - x
    dy = np.roll(y, -1) - y
    dz = np.roll(z, -1) - z

    d = np.sqrt(dx * dx + dy * dy + dz * dz)
    lengths = d.cumsum()

    ax.scatter(x, y, z, s=node_size, color=node_color)
    ax.quiver(x, y, z, dx, dy, dz, color=tour_color, arrow_length_ratio=arrow_ratio, linewidth=arrow_width)
    ax.set_title("{} nodes, total length {:.2f}".format(len(tour), lengths[-1]))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tour_path",
        help="path to .npy containing 3d TSP tours",
    )
    parser.add_argument(
        "batch_range",
        type=int,
        nargs=2,
        help="batch indices (dim 0 slice) to plot"
    )
    args = parser.parse_args()

    tour_data = np.load(args.tour_path)
    assert len(tour_data.shape) == 3 and tour_data.shape[-1] == 3

    start_idx, end_idx = args.batch_range
    num_tours = end_idx - start_idx

    fig, axes = plt.subplots(num_tours, 2, figsize=(10, 5 * num_tours), subplot_kw={'projection': "3d"})
    for offset, (data_ax, rand_ax) in enumerate(axes):
        tour = tour_data[start_idx + offset]
        plot_tsp_3d(tour, data_ax, NODE_SIZE, NODE_COLOR, DATA_TOUR_COLOR, ARROW_RATIO, TOUR_WIDTH)

        rand_tour = tour[torch.randperm(len(tour))]
        plot_tsp_3d(rand_tour, rand_ax, NODE_SIZE, NODE_COLOR, RANDOM_TOUR_COLOR, ARROW_RATIO, TOUR_WIDTH)

    fig.suptitle("Data (Left) vs Random (Right)", fontsize=16)

    input_stub = args.tour_path.split(".")[0]
    if FORMAT in ("eps", "svg"):
        fig.savefig(input_stub + f".{FORMAT}", format=FORMAT)
    else:  # assuming non-vector with dpi
        fig.savefig(input_stub + f".{FORMAT}", format=FORMAT, dpi=300)
