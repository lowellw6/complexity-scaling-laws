
import argparse
from importlib import import_module

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.text import Text
from mpl_toolkits.mplot3d.art3d import Text3D
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

from scipy.optimize import curve_fit
import numpy as np
import os
import os.path as osp
import mlflow
from mlflow import MlflowClient
from ast import literal_eval
from collections import defaultdict
import decimal

# from plot_utils import get_sol_eval, get_bind_eval, power_scaling_fit, save_fit_eq, clear_fits


FORMAT = "pdf"

import tsp
root_path = osp.dirname(osp.dirname(tsp.__file__))


def fetch_node_bind_data(n, b):
    bind_path = osp.join(root_path, "bindthem")

    dpath = f"node_scale_{n}n_2d.npz" if n != 20 else f"node_scale_{n}n_2d_plus_model_scale.npz"
    dpath = osp.join(bind_path, dpath)
    dset = np.load(dpath, mmap_mode="r")

    optimals = dset["global_optima"]
    
    _2opt_lo = dset["_2opt_local_optima"]
    _2opt_starts = dset["_2opt_starts"]
    _2opt_swaps = dset["_2opt_swaps"]

    _2swap_lo = dset["_2swap_local_optima"]
    _2swap_starts = dset["_2swap_starts"]
    _2swap_swaps = dset["_2swap_swaps"]

    drl_node_sol = dset["drl_node_scaling"]
    il_node_sol = dset["il_node_scaling"]

    return optimals[b], drl_node_sol[b]


def fetch_dim_bind_data(d, node_scale, b):
    bind_path = osp.join(root_path, "bindthem")
    
    dpath = f"dim_scale_{node_scale}n_{d}d.npz"
    dpath = osp.join(bind_path, dpath)
    dset = np.load(dpath, mmap_mode="r")

    proxy_optimals = dset["proxy_global_optima"]
    
    _2opt_lo = dset["_2opt_local_optima"]
    _2opt_starts = dset["_2opt_starts"]
    _2opt_swaps = dset["_2opt_swaps"]

    _2swap_lo = dset["_2swap_local_optima"]
    _2swap_starts = dset["_2swap_starts"]
    _2swap_swaps = dset["_2swap_swaps"]

    if d == 2:
        drl_dim_sol = dset["drl_node_scaling"]  # for 2d, drl solutions came from node scaling experiment
    else:
        drl_dim_sol = dset["drl_dim_scaling"]

    return proxy_optimals[b], drl_dim_sol[b]


def save_fig(fig, name, format):
    if format in ("eps", "svg"):
        fig.savefig(f"{name}.{format}", format=format, bbox_inches="tight")
    else:  # assuming non-vector with dpi
        fig.savefig(f"{name}.{format}", format=format, dpi=300, bbox_inches="tight")


def azel_to_xyz(elevation, azimuth, r):  # AI-generated
    elevation_rad = np.radians(elevation)
    azimuth_rad = np.radians(azimuth)
    x = r * np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = r * np.cos(elevation_rad) * np.sin(azimuth_rad)
    z = r * np.sin(elevation_rad)
    return (x, y, z) 


def plot_2d(axis, xy, tour, artcfg):
    """
    Adapted from base.plot_tsp()
    """
    line_col = artcfg["lc"]
    node_col = artcfg["nc"]

    quiv_width = artcfg["qw"]
    node_size = artcfg["ns"]

    xs, ys = np.split(xy[tour], 2, axis=-1)
    dx = np.roll(xs, -1) - xs
    dy = np.roll(ys, -1) - ys
    d = np.sqrt(dx * dx + dy * dy)
    lengths = d.cumsum()

    qv = axis.quiver(
        xs,
        ys,
        dx,
        dy,
        scale_units="xy",
        angles="xy",
        scale=1,
        width=quiv_width,
        headwidth=0,
        headlength=0,
        headaxislength=0,
        color=line_col,  # if None, uses a default blue
        zorder=0
    )

    if node_size > 0:
        sc = axis.scatter(xs, ys, s=node_size, color=node_col, zorder=1)
    else:
        sc = None

    return lengths, qv, sc


def plot_3d(axis, xyz, tour, artcfg):
    line_col = artcfg["lc"]
    node_col = artcfg["nc"]

    quiv_width = artcfg["qw"]
    node_size = artcfg["ns"]

    x, y, z = np.split(xyz[tour], 3, axis=-1)
    dx = np.roll(x, -1) - x
    dy = np.roll(y, -1) - y
    dz = np.roll(z, -1) - z

    d = np.sqrt(dx * dx + dy * dy + dz * dz)
    lengths = d.cumsum()

    qv = axis.quiver(
        x, 
        y, 
        z, 
        dx, 
        dy, 
        dz,
        linewidth=quiv_width,
        arrow_length_ratio=0, 
        color=line_col,
        zorder=0
    )

    if node_size > 0:
        px, py, pz = azel_to_xyz(axis.elev, axis.azim, 1.5)

        dpx = x - px
        dpy = y - py
        dpz = z - pz
        pd = np.sqrt(dpx * dpx + dpy * dpy + dpz * dpz)
        
        min_pd = pd.min()
        max_pd = pd.max()

        norm_pd = (pd - min_pd) / (max_pd - min_pd)
        color_pd = 0.75 * norm_pd

        sc = axis.scatter(x, y, z, s=node_size, c=color_pd, cmap=plt.cm.gray, vmin=0, vmax=1, depthshade=False, zorder=1)
        #axis.scatter(x, y, z, color="black", s=node_size, zorder=1)

    else:
        sc = None

    return lengths, qv, sc


    

def pinfo(id, xy, lengths):
    print(f"{id} : {len(xy)} nodes, total length {lengths[-1]:.5f}")



def make_cover_plots():
    # setup axes
    fig = plt.figure(figsize=(5.5, 1.5))

    n10_ax = fig.add_subplot(1, 3, 1)
    n20_ax = fig.add_subplot(1, 3, 2)
    d20_ax = fig.add_subplot(1, 3, 3, projection='3d', computed_zorder=False)

    fig.subplots_adjust(wspace=0.35)

    n10_ax.set_aspect("equal")
    n20_ax.set_aspect("equal")

    d20_ax.xaxis.set_rotate_label(False)  # disable automatic rotation
    d20_ax.yaxis.set_rotate_label(False)
    d20_ax.zaxis.set_rotate_label(False)
 
    axes = (n10_ax, n20_ax, d20_ax)

    # art configs
    sol_artcfg_2d = dict(lc="black", nc="black", qw=0.01, ns=15)  # "#0080FF"
    opt_artcfg_2d = dict(lc="lime", nc=None, qw=0.0275, ns=0)

    sol_artcfg_2d_bign = dict(lc="black", nc="black", qw=0.0075, ns=5)  # "#0080FF"
    opt_artcfg_2d_bign = dict(lc="lime", nc=None, qw=0.02, ns=0)

    sol_artcfg_3d = dict(lc="black", nc="black", qw=0.75, ns=15)  # "#0080FF"
    opt_artcfg_3d = dict(lc="lime", nc=None, qw=2.2, ns=0)  # old proxy orange col #FF8000

    # fetch 2D 5-node problem, oracle, and ppo solution
    n10_opt_xy, n10_drl_sol = fetch_node_bind_data(5, 8)   #fetch_node_bind_data(10, 4)  # 62, 65, 66

    lengths, opt_qv, _ = plot_2d(n10_ax, n10_opt_xy, np.arange(len(n10_opt_xy)), opt_artcfg_2d)
    pinfo("opt n5", n10_opt_xy, lengths)

    lengths, sol_qv, sol_sc = plot_2d(n10_ax, n10_opt_xy, n10_drl_sol[0], sol_artcfg_2d)
    pinfo("drl n5", n10_opt_xy, lengths)

    # fetch 2D 40-node problem, oracle, and ppo solution
    n20_opt_xy, n20_drl_sol = fetch_node_bind_data(40, 777)  #fetch_node_bind_data(20, 78)  # 2, 28, 48, 52, 53, 78, 91, 98

    lengths = plot_2d(n20_ax, n20_opt_xy, np.arange(len(n20_opt_xy)), opt_artcfg_2d_bign)[0]
    pinfo("opt n40", n20_opt_xy, lengths)

    lengths = plot_2d(n20_ax, n20_opt_xy, n20_drl_sol[0], sol_artcfg_2d_bign)[0]
    pinfo("drl n40", n20_opt_xy, lengths)

    # fetch 3D dim problem, proxy opt, and ppo solution
    d20_opt_xyz, d20_drl_sol = fetch_dim_bind_data(3, 10, 151)  #fetch_dim_bind_data(3, 20, 78)  # 11, 78

    lengths, proxyopt_qv, _ = plot_3d(d20_ax, d20_opt_xyz, np.arange(len(d20_opt_xyz)), opt_artcfg_3d)
    pinfo("opt 3d n10", d20_opt_xyz, lengths)

    lengths = plot_3d(d20_ax, d20_opt_xyz, d20_drl_sol[0], sol_artcfg_3d)[0]
    pinfo("drl 3d n10", d20_opt_xyz, lengths)

    # formatting and labelling
    label_size = 6
    tick_label_size = 4
    minor_tick_label_size = 4

    [ax.set_xlim(0, 1) for ax in (n10_ax, n20_ax, d20_ax)]
    [ax.set_ylim(0, 1) for ax in (n10_ax, n20_ax, d20_ax)]
    d20_ax.set_zlim(0, 1)

    d20_ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # makes background transparant
    d20_ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    d20_ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    grid_color = "whitesmoke"
    d20_ax.xaxis._axinfo["grid"]['color'] = grid_color
    d20_ax.yaxis._axinfo["grid"]['color'] = grid_color
    d20_ax.zaxis._axinfo["grid"]['color'] = grid_color
    d20_ax.set_box_aspect(None, zoom=1.3)

    [ax.set_xlabel("$x_1$", fontsize=label_size) for ax in (n10_ax, n20_ax)]
    [ax.set_ylabel("$x_2$", fontsize=label_size, rotation=0, labelpad=7) for ax in (n10_ax, n20_ax)]

    d20_ax.set_xlabel("$x_1$", fontsize=label_size, rotation=0, labelpad=-15)
    d20_ax.set_ylabel("$x_2$", fontsize=label_size, rotation=0, labelpad=-13)
    d20_ax.set_zlabel("$x_3$", fontsize=label_size, rotation=0, labelpad=-14)

    [ax.tick_params(labelsize=tick_label_size) for ax in (n10_ax, n20_ax)]

    d20_ax.tick_params(axis="x", labelsize=tick_label_size, pad=-6)
    d20_ax.tick_params(axis="y", labelsize=tick_label_size, pad=-5.5)
    d20_ax.tick_params(axis="z", labelsize=tick_label_size, pad=-5)

    # d20_yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    # d20_ylbls = [Text3D(0, t, 10, f"{t:.1f}") for t in d20_yticks]
    # d20_ax.set_yticks(d20_yticks, d20_ylbls)

    # legend
    leg = fig.legend([sol_sc, sol_qv, opt_qv], ["Node", "Model tour", "Optimal tour"], loc="upper center", prop={'size': label_size}, bbox_to_anchor=(0.5, -0.1), ncols=4, frameon=False)
    
    for handle in leg.legend_handles:
        if isinstance(handle, Line2D):
            handle.set_linewidth(1.75)
        elif isinstance(handle, mpatches.Rectangle):
            if handle.get_facecolor() == (0.0, 0.0, 0.0, 1.0):  # ppo black
                handle.set_height(1.25)
                handle.set_y(1.5)            
            else:
                handle.set_height(1.75)
                handle.set_y(1.2)
    
    # saving
    save_fig(fig, "2_cover", "pdf")
    save_fig(fig, "2_cover", "png")




if __name__ == "__main__":    
    make_cover_plots()
