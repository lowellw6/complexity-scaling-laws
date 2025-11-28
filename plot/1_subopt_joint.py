"""
Combining fitness w.r.t. parameter and compute scaling
"""

import argparse
from importlib import import_module

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.text import Text
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

import pandas as pd
from copy import deepcopy

from scipy.optimize import curve_fit
import numpy as np
import os
import os.path as osp
import mlflow
from mlflow import MlflowClient
from ast import literal_eval
from collections import defaultdict
import decimal

from plot_utils import get_sol_eval, get_bind_eval, power_scaling_fit, save_fit_eq, clear_fits, forceAspect, is_pareto_efficient_simple
get_metrics_and_params = getattr(import_module("1_compute_fitness"), "get_metrics_and_params")
get_compute_multiplier = getattr(import_module("1_compute_fitness"), "get_compute_multiplier")
# save_fit_eq = getattr(import_module("1_compute_fitness"), "save_fit_eq")
model_scaling_fit = getattr(import_module("1_model_loss"), "model_scaling_fit")
plot_instance = getattr(import_module("2_problem_fitness"), "plot_instance")


PLOT_MAIN = True
PLOT_TREND_BREAK = False


PARETO_RAW = True  # turn off pareto front generation (expensive)
PARETO_FIT = True  # turn off pareto fit generation (expensive); does nothing if PARETO_RAW = False

LINE_WIDTH = 1.5
PARETO_LINE_WIDTH = 1.5
PARETO_FIT_LINE_WIDTH = 1 #0.75
POINT_SIZE = 4
PARETO_SIZE = 2
POINT_MARKER = "-"
PARETO_MARKER = "-"
PARETO_DEVIATE_MARKER = ":"
PARETO_FIT_MARKER = "--"
C0_PARETO_FIT_MARKER = "-."
LABEL_SIZE = 6
TICK_LABEL_SIZE = 4
MINOR_TICK_LABEL_SIZE = 4

OPTIMAL_N20 = 3.831
RAND_N20 = 10.428

PARETO_COLOR = "hotpink"
PARETO_DEVIATE_COLOR = "hotpink"
PARETO_FIT_COLOR = "black"  #"dimgray"     # "#FF6969"  # analogous of hotpink      # "#69B4FF"  # triadic of hotpink

# hyperparam dict structure: (ewm_alpha, pareto_x_start_thresh, pareto_x_break)
# ewm_alpha --> (0, 1] where lower is more smoothed and 1 is no smoothing
# pareto_x_start_thresh --> where to start pareto frontier, in compute
# pareto_x_break --> where to stop pareto frontier (for fit), in compute; remainder highlighted in different color (where trend breaks down, likely due to forced early convergence of trailing models)

DRL_DICT = {
    "avg" : (1, 1e-6, 5e-5),
    "std" : (1, 1e-6, 5e-5)
    # "avg" : (1, 1e-6, 6e-6), # 2e-5
    # "std" : (1, 1e-6, 6e-6)
}
IL_DICT = {
    "avg": (1, 3e-7, None),
}
SMOOTH_COLOR_ALPHA = 0.1

OUTLIER_MODEL_ALPHA = 0.2

DRL_PARETO_TRUNC = 3  # top k runs to drop from pareto fit (matching model loss break)
IL_PARETO_TRUNC = 7

PFDAY_FLOPS = 1e15 * 24 * 60 * 60

MODEL_DIM = [240, 192, 160, 128, 104, 88, 72, 56, 48, 40, 32, 24]  # forgot to log in new checkgen script
MODEL_PARAMETERS = [61826, 108546, 168322, 241154, 537986, 327042, 801154, 1116546, 1687554, 2631682, 3784706, 5905922]  # convenience



def save_fig(fig, name, format):
    if format in ("eps", "svg"):
        fig.savefig(f"{name}.{format}", format=format, bbox_inches="tight")
    else:  # assuming non-vector with dpi
        fig.savefig(f"{name}.{format}", format=format, dpi=300, bbox_inches="tight")


def plot_compute_evals(ax, tups, p_bounds, algo_str, fit_file_str, id_str, abs=False, logscale_y=True, ewm_alpha=None, show_pareto=True, pareto_min_c=None, pareto_max_c=None, omit_non_pareto=True, subopt=True):
    plt_fn = ax.loglog if logscale_y else ax.semilogx

    min_p, max_p = p_bounds

    pareto_test_y = []
    pareto_test_c = []

    all_c = []
    
    idx_thresh = DRL_PARETO_TRUNC if algo_str == "drl" else IL_PARETO_TRUNC

    for idx, (y, x, p, w) in enumerate(tups[::-1]):  # reverse to plot larger models first (visually emphasizing smaller models on top which converged)
        if omit_non_pareto and idx < idx_thresh:
            continue
        
        if algo_str == "drl":
            x *= 4  # iteration --> gradient updates (IL already has iteration == gradient updates)

        compute = get_compute_multiplier(p, w) * x  # x steps measured in gradient updates here
        compute = compute / PFDAY_FLOPS  # FLOPs --> PF-days

        if subopt:
            y = (y - OPTIMAL_N20)  # NOTE mean raw tour length --> mean suboptimality (not RON)

        col = plt.get_cmap("viridis")((np.log(p) - np.log(min_p)) / (np.log(max_p) - np.log(min_p)))

        if abs:
            y = np.abs(y)

        if ewm_alpha is not None and ewm_alpha < 1:
            # faded non-smooth plot first
            plt_fn(compute, y, POINT_MARKER, alpha=SMOOTH_COLOR_ALPHA, color=col, markersize=POINT_SIZE, linewidth=LINE_WIDTH, zorder=1)

            y = pd.DataFrame(y).ewm(alpha=ewm_alpha).mean().to_numpy().squeeze()

        if subopt and algo_str == "drl" and p == MODEL_PARAMETERS[2]:  # 3rd model is significant outlier, shifting both mean and std RON fits
            plt_fn(compute, y, POINT_MARKER, alpha=OUTLIER_MODEL_ALPHA, color=col, markersize=POINT_SIZE, linewidth=LINE_WIDTH, zorder=2)
        else:
            plt_fn(compute, y, POINT_MARKER, color=col, markersize=POINT_SIZE, linewidth=LINE_WIDTH, zorder=2)            

        if idx >= idx_thresh and (not subopt or algo_str != "drl" or p != MODEL_PARAMETERS[2]):  # not adding outlier model to pareto fit
            pareto_test_y.append(y)
            pareto_test_c.append(compute)

        all_c.append(compute)

    # pareto frontier generation
    if PARETO_RAW and show_pareto:
        pareto_cat_y = np.concatenate(pareto_test_y, axis=0)
        pareto_cat_c = np.concatenate(pareto_test_c, axis=0)
        pareto_test = np.stack([pareto_cat_y, pareto_cat_c], axis=1)
        pareto_mask = is_pareto_efficient_simple(pareto_test)

        pareto_y = pareto_cat_y[pareto_mask]
        pareto_c = pareto_cat_c[pareto_mask]
        sort_idx = np.argsort(pareto_c)
        pareto_y = pareto_y[sort_idx]
        pareto_c = pareto_c[sort_idx]

        if pareto_min_c is not None:
            thresh_mask = pareto_c > pareto_min_c
            pareto_y = pareto_y[thresh_mask]
            pareto_c = pareto_c[thresh_mask]

        if pareto_max_c is not None:
            over_mask = pareto_c > pareto_max_c
            overp_y = pareto_y[over_mask]
            overp_c = pareto_c[over_mask]

            plt_fn(overp_c, overp_y, PARETO_DEVIATE_MARKER, color=PARETO_DEVIATE_COLOR, markersize=PARETO_SIZE, linewidth=PARETO_LINE_WIDTH, zorder=3)

            under_mask = pareto_c < pareto_max_c
            pareto_y = pareto_y[under_mask]
            pareto_c = pareto_c[under_mask]

        plt_fn(pareto_c, pareto_y, PARETO_MARKER, color=PARETO_COLOR, markersize=PARETO_SIZE, linewidth=PARETO_LINE_WIDTH, zorder=3)

        # pareto power fit
        if PARETO_FIT:
            all_c = np.concatenate(all_c, axis=0)
            c_bounds = all_c.min(), all_c.max()
            print(f"\n{id_str} DRL RON cost | Pareto pts {len(pareto_c)}:")
            pareto_fit_c, pareto_fit_y, pareto_fit_popt = power_scaling_fit(pareto_c, pareto_y, c_bounds, c0=0.0, c1=0.0, mode="decay", sign="positive")
            plt_fn(pareto_fit_c, pareto_fit_y, PARETO_FIT_MARKER, color=PARETO_FIT_COLOR, markersize=PARETO_SIZE, linewidth=PARETO_FIT_LINE_WIDTH, zorder=4)

            popt, x_scale, y_scale = pareto_fit_popt
            c0, c1, c, m = popt
 
            save_fit_eq(fit_file_str, id_str, c0, c1, c, m, "positive", y_scale, x_scale)

            return popt
        

def get_compute_param_arrs(tups, algo_str):
    pareto_test_p = []
    pareto_test_y = []
    pareto_test_c = []

    all_c = []
    
    idx_thresh = DRL_PARETO_TRUNC if algo_str == "drl" else IL_PARETO_TRUNC

    for idx, (y, x, p, w) in enumerate(tups[::-1]):  # reverse to plot larger models first (visually emphasizing smaller models on top which converged)
        if idx < idx_thresh:
            continue
        
        if algo_str == "drl":
            x *= 4  # iteration --> gradient updates (IL already has iteration == gradient updates)

        compute = get_compute_multiplier(p, w) * x  # x steps measured in gradient updates here
        compute = compute / PFDAY_FLOPS  # FLOPs --> PF-days

        y = (y - OPTIMAL_N20)  # NOTE mean raw tour length --> mean suboptimality (not RON)

        if idx >= idx_thresh and (algo_str != "drl" or p != MODEL_PARAMETERS[2]):  # not adding outlier model to pareto fit
            pareto_test_p.append(p)
            pareto_test_y.append(y)
            pareto_test_c.append(compute)

        all_c.append(compute)

    return all_c, pareto_test_c, pareto_test_y, pareto_test_p


def plot_opt_model_size(ax, tups, p_bounds, algo_str, fit_file_str, id_str, artcfg, pareto_min_c=None, pareto_max_c=None):
    _, carrs, sarrs, pvals = get_compute_param_arrs(tups, algo_str)

    pareto_cat_s = np.concatenate(sarrs, axis=0)
    pareto_cat_c = np.concatenate(carrs, axis=0)
    pareto_test = np.stack([pareto_cat_s, pareto_cat_c], axis=1)
    pareto_mask = is_pareto_efficient_simple(pareto_test)

    arr_lens = [len(sarrs[idx]) for idx in range(len(sarrs))]
    model_split = np.asarray([len(sarrs[0])] + [sum(arr_lens[:idx]) for idx in range(2, len(sarrs))])
    pareto_splits = np.split(pareto_mask, model_split, axis=0)
    
    fit_comp = []
    fit_pvals = []
    for pv, carr, mask in zip(pvals, carrs, pareto_splits):
        rel_comp = carr[mask]
        
        # bounding to trend region used to fit pareto front (omitting protracted leading and trailing experiments from small/large boundary models)
        if algo_str == "drl":
            minc, maxc = DRL_DICT["avg"][1:]
            clip_mask = np.logical_and(rel_comp >= minc, rel_comp <= maxc)
            rel_comp = rel_comp[clip_mask]
        if algo_str == "il":
            minc, _ = IL_DICT["avg"][1:]
            rel_comp = rel_comp[rel_comp >= minc]

        if len(rel_comp) == 0: # or (algo_str == "il" and pv == MODEL_PARAMETERS[1]):
            continue

        plot_instance(ax.loglog, rel_comp, np.asarray(len(rel_comp) * [pv]), slice(len(rel_comp)), None, False, False, "decay", "positive", fit_file_str, "TEMP", artcfg, powerfit=False)

        maxpts = 7
        while len(rel_comp) > maxpts:
            rel_comp = rel_comp[1:]
            rel_comp = rel_comp[:-1]
        fit_comp.append(np.mean(rel_comp))
        fit_pvals.append(pv)
        
    if algo_str == "drl":
        x_bounds = (1.25e-6, 4.5e-5)
    elif algo_str == "il":
        x_bounds = (3.75e-7, 5e-6)
    #x_bounds = None

    plot_instance(ax.loglog, np.asarray(fit_comp), np.asarray(fit_pvals), slice(len(fit_comp)), None, False, False, "grow", "positive", fit_file_str, id_str, artcfg, powerfit=True, plot_data=False, manual_fit_xbounds=x_bounds)



def make_plots(client):
    # setup axes
    fig = plt.figure(figsize=(5.5, 2.1))
    out_grid = GridSpec(1, 2, wspace=0.25, width_ratios=[4, 15])
    comp_grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=out_grid[1], wspace=0.05)
    param_grid = GridSpecFromSubplotSpec(2, 1, subplot_spec=out_grid[0], height_ratios=[0.6333, 0.3667])

    param_ax = plt.Subplot(fig, param_grid[0, 0])

    ppo_comp_ax = plt.Subplot(fig, comp_grid[0, 0])
    bc_comp_ax = plt.Subplot(fig, comp_grid[0, 1], sharey=ppo_comp_ax)
    # colorbar_ax = plt.Subplot(fig, comp_grid[0, 2])

    comp_axes = [ppo_comp_ax, bc_comp_ax]
    data_axes = comp_axes + [param_ax]
    axes = data_axes # + [colorbar_ax]

    # gather instance data
    ppo_comp_tups = get_metrics_and_params(client, "CHECKGEN_compute_scaling_drl", "eval_cost_avg")
    bc_comp_tups = get_metrics_and_params(client, "CHECKGEN_compute_scaling_il_2", "eval_cost_avg")

    ppo_param_y, ppo_param_scale = get_sol_eval(client, "SOL_model_scaling_drl_2", "width", f"eval_cost_avg")
    ppo_params_x, ppo_params_scale = get_sol_eval(client, "SOL_model_scaling_drl_2", "width", "non_embedding_parameters")

    bc_param_y, bc_param_scale = get_bind_eval(client, "EVAL_bind", "model", f"il_model_cost.mean")  # bind has non-teacher-forced cost evals
    bc_params_x, bc_params_scale = get_sol_eval(client, "SOL_model_scaling_il_patch", "width", "non_embedding_parameters")

    fit_file_str = "1_subopt_joint_fits.txt"
    clear_fits(fit_file_str)

    # compute plots
    drlp = list(zip(*ppo_comp_tups))[2]
    drl_pbounds = drlp[0], drlp[-DRL_PARETO_TRUNC-1]
    ewm_alpha, pareto_min, pareto_max = DRL_DICT["avg"]
    ppo_comp_popt = plot_compute_evals(ppo_comp_ax, deepcopy(ppo_comp_tups), drl_pbounds, "drl", fit_file_str, "ppo comp subopt avg", ewm_alpha=ewm_alpha, pareto_min_c=pareto_min, pareto_max_c=pareto_max)

    # bcp = list(zip(*bc_comp_tups))[2]
    # bc_pbounds = bcp[0], bcp[-1]
    ewm_alpha, pareto_min, pareto_max = IL_DICT["avg"]
    bc_comp_popt = plot_compute_evals(bc_comp_ax, deepcopy(bc_comp_tups), drl_pbounds, "il", fit_file_str, "bc comp subopt avg", ewm_alpha=ewm_alpha, pareto_min_c=pareto_min, pareto_max_c=pareto_max)

    # param plots
    bc_param_artcfg = dict(pm="s-", om="o-", fm="--", ps=2.5, dw=1, fw=1, dc="#00CFCF", fc="hotpink", fe=None)
    ppo_param_artcfg = dict(pm="o-", om="o-", fm="--", ps=2.5, dw=1, fw=1, dc="#00CF68", fc="black", fe=None)

    plot_instance(param_ax.loglog, bc_params_x, bc_param_y - OPTIMAL_N20, slice(5), None, False, False, "decay", "positive", fit_file_str, "bc param subopt avg", bc_param_artcfg, powerfit=False)
    bc_param_popt = plot_instance(param_ax.loglog, bc_params_x, bc_param_y - OPTIMAL_N20, slice(5), slice(4, 9), False, False, "decay", "positive", fit_file_str, "bc param subopt avg", bc_param_artcfg, plot_data=False, powerfit_zorder=2)
    
    ppo_param_popt = plot_instance(param_ax.loglog, ppo_params_x, ppo_param_y - OPTIMAL_N20, slice(9), None, False, False, "decay", "positive", fit_file_str, "ppo param subopt avg", ppo_param_artcfg)

    # labelling axes
    label_size = 6
    tick_label_size = 4
    minor_tick_label_size = 4

    param_ax.set_ylabel("Suboptimality gap", fontsize=label_size)
    ppo_comp_ax.set_ylabel("Suboptimality gap", fontsize=label_size)
    param_ax.set_xlabel("Parameters ($N$)", fontsize=LABEL_SIZE)
    [ax.set_xlabel("Compute (PF-days)", fontsize=LABEL_SIZE) for ax in comp_axes]

    ppo_comp_ax.set_title("Reinforcement learning (RL)", fontsize=label_size)
    bc_comp_ax.set_title("Supervised fine-tuning (SFT)", fontsize=label_size)

    [ax.tick_params(labelsize=tick_label_size) for ax in axes]
    [ax.tick_params(axis="y", which="minor", labelsize=minor_tick_label_size) for ax in axes]
    [ax.tick_params(axis="x", which="minor", labelsize=minor_tick_label_size) for ax in axes]

    bc_comp_ax.tick_params("y", which="both", labelleft=False)

    ppo_comp_ax.set_ylim([1.2e-2, 0.15])
    ppo_comp_ax.set_xlim([5e-7, 5e-4])
    bc_comp_ax.set_xlim([2e-7, 7e-6])

    param_ax.set_ylim([7e-3, 8e-2])
    param_ax.set_xlim([5e4, 2e6])
    # bc_param_ax.set_xlim([5e4, 5e5])

    # legend
    bc_line, bc_fit, ppo_line, ppo_fit = param_ax.get_lines()
    ppo_legend = param_ax.legend([ppo_line, bc_line, ppo_fit, bc_fit], ["RL", "SFT", fr"$s \propto N^{{{-abs(ppo_param_popt[0][-1]):.2f}}}$", fr"$s \propto N^{{{-abs(bc_param_popt[0][-1]):.2f}}}$"], loc="upper right",  bbox_to_anchor=(1.1, 1.2), prop={'size': tick_label_size+1}, ncols=1, columnspacing=0.75, frameon=True, framealpha=1)
    # bc_legend = param_ax.legend([bc_line, bc_fit], ["SFT", fr"$s \propto N^{{{-abs(bc_param_popt[0][-1]):.2f}}}$"], loc="upper right", prop={'size': label_size}, ncols=1, frameon=False)
    # param_ax.add_artist(ppo_legend)

    comp_lines = ppo_comp_ax.get_lines()
    pareto_trend_line = next(filter(lambda l: l.get_c() == PARETO_FIT_COLOR, comp_lines))
    pareto_front_line = next(filter(lambda l: l.get_c() == PARETO_COLOR and l.get_ls() == PARETO_MARKER, comp_lines))
    pareto_deviate_line = next(filter(lambda l: l.get_c() == PARETO_DEVIATE_COLOR and l.get_ls() == PARETO_DEVIATE_MARKER, comp_lines))
    ppo_comp_ax.legend([pareto_front_line, pareto_deviate_line, pareto_trend_line], ["frontier", "trend break", fr"$s \propto (C_{{\min}})^{{{ppo_comp_popt[-1]:.2f}}}$"], loc="lower left", prop={'size': label_size}, ncols=1, frameon=False)
    bc_comp_ax.legend([pareto_front_line, pareto_trend_line], ["frontier", fr"$s \propto (C_{{\min}})^{{{bc_comp_popt[-1]:.2f}}}$"], loc="lower left", prop={'size': label_size}, ncols=1, frameon=False)

    # alphas = [(ppo_param_ax, abs(ppo_param_popt[0][-1])), 
    #           (bc_param_ax, abs(bc_param_popt[0][-1])),
    #           (ppo_comp_ax, abs(ppo_comp_popt[-1])),
    #           (bc_comp_ax, abs(bc_comp_popt[-1]))]
    # [ax.text(0.95, 0.95, fr"$\alpha \approx {alpha:.2f}$", fontsize=label_size, ha='right', va='top', transform=ax.transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round')) for (ax, alpha) in alphas]


    # add labelled subplots
    # forceAspect(param_ax, aspect=1.5)
    # [forceAspect(ax, aspect=1) for ax in data_axes]
    [fig.add_subplot(ax) for ax in axes]

    # colorbar
    colorbar_ax = fig.add_axes([bc_comp_ax.get_position().x1 + 0.005, bc_comp_ax.get_position().y0 , 0.01, bc_comp_ax.get_position().y1 - bc_comp_ax.get_position().y0])

    min_p, max_p = drl_pbounds
    transform = lambda p: (np.log(p) - np.log(min_p)) / (np.log(max_p) - np.log(min_p))
    reverse_transform = lambda t: np.exp(t * (np.log(max_p) - np.log(min_p)) + np.log(min_p))
    mappable = ScalarMappable(cmap=plt.get_cmap("viridis"))
    cbar = fig.colorbar(mappable, cax=colorbar_ax, location="right", fraction=1, aspect=60)
    col_yticks = [transform(p) for p in MODEL_PARAMETERS[:-DRL_PARETO_TRUNC]]
    col_ylbls = [Text(0, t, f"{(reverse_transform(t) / 10**np.floor(np.log10(reverse_transform(t)))):.2f}$ \:\! \\times \:\! 10^{{{int(np.floor(np.log10(reverse_transform(t))))}}}$") for t in col_yticks]
    cbar.ax.set_yticks(col_yticks, col_ylbls, fontsize=tick_label_size)
    cbar.set_label("Parameters ($N$)", rotation=90, fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=tick_label_size)

    # optimal model size vs compute
    bc_optsize_artcfg = dict(pm="s", om="o-", fm="--", ps=1, dw=1, fw=1, dc="#00CFCF", fc="hotpink", fe=None)
    ppo_optsize_artcfg = dict(pm="o", om="o-", fm="--", ps=1, dw=1, fw=1, dc="#00CF68", fc="black", fe=None)

    left = param_ax.get_position().x0
    height = param_ax.get_position().y1 - param_ax.get_position().y0
    bottom = ppo_comp_ax.get_position().y0 - 0.4 * (ppo_comp_ax.get_position().y1 - ppo_comp_ax.get_position().y0)
    width = param_ax.get_position().x1 - param_ax.get_position().x0
    optax = fig.add_axes([left, bottom, width, height])

    hoffmann_powerlaw = lambda c: (c / 1.4e-18)**0.5  # language, more recent (and optimistic) than Kaplan curve
    henighan_powerlaw = lambda c: (c / 1.6e-13)**0.65  # image 32x32 generation
    hilton_powerlaw = lambda c: 1.148e7 * c**0.4822  # single-agent RL on Procgen Hard benchmark

    c_space = np.logspace(-8, -3, num=500)
    ext_fw = 1.0
    optax.loglog(c_space, hoffmann_powerlaw(c_space), "-", color="#9E8140", linewidth=ext_fw, zorder=1)
    optax.loglog(c_space, henighan_powerlaw(c_space), "-", color="#9E405D", linewidth=ext_fw, zorder=1)
    optax.loglog(c_space, hilton_powerlaw(c_space), "-", color="#52409E", linewidth=ext_fw, zorder=1)

    plot_opt_model_size(optax, deepcopy(bc_comp_tups), drl_pbounds, "il", fit_file_str, "bc opt model size", bc_optsize_artcfg, pareto_min_c=pareto_min, pareto_max_c=pareto_max)
    plot_opt_model_size(optax, deepcopy(ppo_comp_tups), drl_pbounds, "drl", fit_file_str, "ppo opt model size", ppo_optsize_artcfg, pareto_min_c=pareto_min, pareto_max_c=pareto_max)

    #optax.set_ylim([5e4, 1.5e6])
    optax.set_ylim([1e4, 6e6])
    optax.set_xlim([2e-7, 7e-5])

    optax.set_ylabel("Parameters ($N$)", fontsize=LABEL_SIZE)
    optax.set_xlabel("Compute (PF-days)", fontsize=LABEL_SIZE)

    optax.tick_params(labelsize=tick_label_size)
    optax.tick_params(axis="y", which="minor", labelsize=minor_tick_label_size)
    optax.tick_params(axis="x", which="minor", labelsize=minor_tick_label_size)

    # colorbar_ax.annotate("", xy=(3.9, 0.875), xycoords='axes fraction', xytext=(12.9, 0.875), arrowprops=dict(arrowstyle="<->", shrinkA=14.0, shrinkB=14.0, mutation_scale=3.0))
    # colorbar_ax.annotate("", xy=(3.9, 0.12), xycoords='axes fraction', xytext=(12.9, 0.12), arrowprops=dict(arrowstyle="<->", shrinkA=14.0, shrinkB=14.0, mutation_scale=3.0))
    optax.annotate("", xy=(0.975, 0.5), xycoords='axes fraction', xytext=(1.3, 0.31), arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', linewidth=1.0)) #, shrinkA=14.0, shrinkB=14.0, mutation_scale=3.0))

    # ### thanks chatgpt
    # arrow = mpatches.FancyArrowPatch(
    #     (0.6, 0.2),    # Start point
    #     (0.2, 0.8),    # End point
    #     connectionstyle="angle3,angleA=90,angleB=180",  # Up then left
    #     arrowstyle='->',
    #     # mutation_scale=20,
    #     linewidth=2,
    #     color='green'
    # )
    # optax.add_patch(arrow)
    # ###

    lines = optax.get_lines()
    language, image_gen, rl_procgen = lines[:3]
    bc, ppo = lines[3], lines[14]
    bc_trend, ppo_trend = lines[8], lines[16]
    optax.legend([ppo, bc, ppo_trend, bc_trend, language, image_gen, rl_procgen], ["RL", "SFT", r"$N = \left(\frac{C}{1.8 \times 10^{-12}}\right)^{0.82}$", r"$N = \left(\frac{C}{2.0 \times 10^{-14}}\right)^{0.66}$", "Language\n(Hoffmann et al.)", "Image 32x32\n(Henighan et al.)", "RL Procgen\n(Hilton et al.)"], loc="lower left", bbox_to_anchor=(1.001, -0.35), prop={'size': tick_label_size+1}, ncols=2, frameon=True) #, framealpha=1)

    # saving
    save_fig(fig, "1_subopt_joint", "pdf")
    save_fig(fig, "1_subopt_joint", "png")


def make_plots_trend_break(client):
    # setup axes
    fig = plt.figure(figsize=(5.5, 1.21))
    out_grid = GridSpec(1, 2, wspace=0.15)
    comp_grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=out_grid[1], wspace=0.1) #, width_ratios=(25, 25, 1))
    param_grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=out_grid[0], wspace=0.1)

    ppo_comp_ax = plt.Subplot(fig, comp_grid[0, 0])
    bc_comp_ax = plt.Subplot(fig, comp_grid[0, 1], sharey=ppo_comp_ax)
    # colorbar_ax = plt.Subplot(fig, comp_grid[0, 2])

    ppo_param_ax = plt.Subplot(fig, param_grid[0, 0])
    bc_param_ax = plt.Subplot(fig, param_grid[0, 1], sharey=ppo_param_ax, sharex=ppo_param_ax)

    comp_axes = [ppo_comp_ax, bc_comp_ax]
    param_axes = [ppo_param_ax, bc_param_ax]
    data_axes = comp_axes + param_axes
    axes = data_axes # + [colorbar_ax]

    # gather instance data
    ppo_comp_tups = get_metrics_and_params(client, "CHECKGEN_compute_scaling_drl", "eval_cost_avg")
    bc_comp_tups = get_metrics_and_params(client, "CHECKGEN_compute_scaling_il_2", "eval_cost_avg")

    ppo_param_y, ppo_param_scale = get_sol_eval(client, "SOL_model_scaling_drl_2", "width", f"eval_cost_avg")
    ppo_params_x, ppo_params_scale = get_sol_eval(client, "SOL_model_scaling_drl_2", "width", "non_embedding_parameters")

    bc_param_y, bc_param_scale = get_bind_eval(client, "EVAL_bind", "model", f"il_model_cost.mean")  # bind has non-teacher-forced cost evals
    bc_params_x, bc_params_scale = get_sol_eval(client, "SOL_model_scaling_il_patch", "width", "non_embedding_parameters")

    fit_file_str = "1_subopt_joint_fits_trend_break.txt"
    clear_fits(fit_file_str)

    # compute plots
    drlp = list(zip(*ppo_comp_tups))[2]
    drl_pbounds = drlp[0], drlp[-1]
    ewm_alpha, pareto_min, pareto_max = DRL_DICT["avg"]
    ppo_comp_popt = plot_compute_evals(ppo_comp_ax, deepcopy(ppo_comp_tups), drl_pbounds, "drl", fit_file_str, "ppo comp subopt avg", ewm_alpha=ewm_alpha, pareto_min_c=pareto_min, pareto_max_c=pareto_max, omit_non_pareto=False)

    # bcp = list(zip(*bc_comp_tups))[2]
    # bc_pbounds = bcp[0], bcp[-1]
    ewm_alpha, pareto_min, pareto_max = IL_DICT["avg"]
    bc_comp_popt = plot_compute_evals(bc_comp_ax, deepcopy(bc_comp_tups), drl_pbounds, "il", fit_file_str, "bc comp subopt avg", ewm_alpha=ewm_alpha, pareto_min_c=pareto_min, pareto_max_c=pareto_max, omit_non_pareto=False)

    # param plots
    bc_param_artcfg = dict(pm="s-", om="s-", fm="--", ps=3, dw=1, fw=0.75, dc="#00CFCF", fc="black", fe=None)
    ppo_param_artcfg = dict(pm="o-", om="o-", fm="--", ps=3, dw=1, fw=0.75, dc="#00CF68", fc="black", fe=None)

    bc_param_popt = plot_instance(bc_param_ax.loglog, bc_params_x, bc_param_y - OPTIMAL_N20, slice(5), slice(4, 12), False, False, "decay", "positive", fit_file_str, "bc param subopt avg", bc_param_artcfg, powerfit_zorder=2)
    # bc_param_popt = plot_instance(param_ax.loglog, bc_params_x, bc_param_y - OPTIMAL_N20, slice(5), slice(4, 9), False, False, "decay", "positive", fit_file_str, "bc param subopt avg", bc_param_artcfg, plot_data=False, powerfit_zorder=2)
    
    ppo_param_popt = plot_instance(ppo_param_ax.loglog, ppo_params_x, ppo_param_y - OPTIMAL_N20, slice(9), slice(8, 12), False, False, "decay", "positive", fit_file_str, "ppo param subopt avg", ppo_param_artcfg)

    # labelling axes
    label_size = 6
    tick_label_size = 4
    minor_tick_label_size = 4

    ppo_param_ax.set_ylabel("Suboptimality gap", fontsize=label_size)
    [ax.set_xlabel("Parameters ($N$)", fontsize=LABEL_SIZE) for ax in param_axes]
    [ax.set_xlabel("Compute (PF-days)", fontsize=LABEL_SIZE) for ax in comp_axes]

    ppo_comp_ax.set_title("RL", fontsize=label_size)
    bc_comp_ax.set_title("SFT", fontsize=label_size)

    [ax.tick_params(labelsize=tick_label_size) for ax in axes]
    [ax.tick_params(axis="y", which="minor", labelsize=minor_tick_label_size) for ax in axes]
    [ax.tick_params(axis="x", which="minor", labelsize=minor_tick_label_size) for ax in axes]

    bc_comp_ax.tick_params("y", which="both", labelleft=False)
    bc_param_ax.tick_params("y", which="both", labelleft=False)

    ppo_comp_ax.set_ylim([1e-2, 0.15])
    ppo_comp_ax.set_xlim([5e-7, 2e-3])
    bc_comp_ax.set_xlim([2e-7, 1.3e-4])

    ppo_param_ax.set_ylim([5e-3, 8e-2])
    # param_ax.set_xlim([5e4, 2e6])
    # bc_param_ax.set_xlim([5e4, 5e5])

    # legend
    ppo_line, ppo_break, ppo_fit = ppo_param_ax.get_lines()
    ppo_legend = ppo_param_ax.legend([ppo_line, ppo_break, ppo_fit], ["RL", "trend break", fr"$s \propto N^{{{-abs(ppo_param_popt[0][-1]):.2f}}}$"], loc="upper right", prop={'size': tick_label_size}, ncols=1, columnspacing=0.75, frameon=True)
    
    
    bc_line, bc_break, bc_fit = bc_param_ax.get_lines()
    bc_legend = bc_param_ax.legend([bc_line, bc_break, bc_fit], ["SFT", "trend break", fr"$s \propto N^{{{-abs(bc_param_popt[0][-1]):.2f}}}$"], loc="upper right", prop={'size': tick_label_size}, ncols=1, columnspacing=0.75, frameon=True)
    # bc_legend = param_ax.legend([bc_line, bc_fit], ["SFT", fr"$s \propto N^{{{-abs(bc_param_popt[0][-1]):.2f}}}$"], loc="upper right", prop={'size': label_size}, ncols=1, frameon=False)
    # param_ax.add_artist(ppo_legend)

    comp_lines = ppo_comp_ax.get_lines()
    pareto_trend_line = next(filter(lambda l: l.get_c() == PARETO_FIT_COLOR, comp_lines))
    pareto_front_line = next(filter(lambda l: l.get_c() == PARETO_COLOR and l.get_ls() == PARETO_MARKER, comp_lines))
    pareto_deviate_line = next(filter(lambda l: l.get_c() == PARETO_DEVIATE_COLOR and l.get_ls() == PARETO_DEVIATE_MARKER, comp_lines))
    ppo_comp_ax.legend([pareto_front_line, pareto_deviate_line, pareto_trend_line], ["frontier", "trend break", fr"$s \propto (C_{{\min}})^{{{ppo_comp_popt[-1]:.2f}}}$"], loc="lower left", prop={'size': tick_label_size-0.5}, ncols=1, frameon=False)
    bc_comp_ax.legend([pareto_front_line, pareto_trend_line], ["frontier", fr"$s \propto (C_{{\min}})^{{{bc_comp_popt[-1]:.2f}}}$"], loc="lower left", prop={'size': tick_label_size}, ncols=1, frameon=False)

    # alphas = [(ppo_param_ax, abs(ppo_param_popt[0][-1])), 
    #           (bc_param_ax, abs(bc_param_popt[0][-1])),
    #           (ppo_comp_ax, abs(ppo_comp_popt[-1])),
    #           (bc_comp_ax, abs(bc_comp_popt[-1]))]
    # [ax.text(0.95, 0.95, fr"$\alpha \approx {alpha:.2f}$", fontsize=label_size, ha='right', va='top', transform=ax.transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round')) for (ax, alpha) in alphas]


    # add labelled subplots
    # forceAspect(param_ax, aspect=1.5)
    [forceAspect(ax, aspect=1) for ax in data_axes]
    [fig.add_subplot(ax) for ax in axes]

    # colorbar
    colorbar_ax = fig.add_axes([bc_comp_ax.get_position().x1 + 0.01, bc_comp_ax.get_position().y0 , 0.01, bc_comp_ax.get_position().y1 - bc_comp_ax.get_position().y0])

    min_p, max_p = drl_pbounds
    transform = lambda p: (np.log(p) - np.log(min_p)) / (np.log(max_p) - np.log(min_p))
    reverse_transform = lambda t: np.exp(t * (np.log(max_p) - np.log(min_p)) + np.log(min_p))
    mappable = ScalarMappable(cmap=plt.get_cmap("viridis"))
    cbar = fig.colorbar(mappable, cax=colorbar_ax, location="right", fraction=1, aspect=60)
    col_yticks = [transform(p) for p in MODEL_PARAMETERS]
    col_ylbls = [Text(0, t, f"{(reverse_transform(t) / 10**np.floor(np.log10(reverse_transform(t)))):.2f}$ \:\! \\times \:\! 10^{{{int(np.floor(np.log10(reverse_transform(t))))}}}$") for t in col_yticks]
    cbar.ax.set_yticks(col_yticks, col_ylbls, fontsize=tick_label_size)
    cbar.set_label("Parameters ($N$)", rotation=90, fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=tick_label_size-1)

    # # optimal model size vs compute
    # bc_optsize_artcfg = dict(pm="s", om="o-", fm="--", ps=0.5, dw=1, fw=0.75, dc="#00CFCF", fc="hotpink", fe=None)
    # ppo_optsize_artcfg = dict(pm="o", om="o-", fm="--", ps=0.5, dw=1, fw=0.75, dc="#00CF68", fc="black", fe=None)

    # optax = fig.add_axes([colorbar_ax.get_position().x1 + 0.125, colorbar_ax.get_position().y0 , 0.6 * (bc_comp_ax.get_position().x1 - bc_comp_ax.get_position().x0), colorbar_ax.get_position().y1 - colorbar_ax.get_position().y0])
    
    # hoffmann_powerlaw = lambda c: (c / 1.4e-18)**0.5  # language, more recent (and optimistic) than Kaplan curve
    # henighan_powerlaw = lambda c: (c / 1.6e-13)**0.65  # image 32x32 generation
    # hilton_powerlaw = lambda c: 1.148e7 * c**0.4822  # single-agent RL on Procgen Hard benchmark

    # c_space = np.logspace(-8, -3, num=500)
    # ext_fw = 1.0
    # optax.loglog(c_space, hoffmann_powerlaw(c_space), "-", color="#9E8140", linewidth=ext_fw, zorder=1)
    # optax.loglog(c_space, henighan_powerlaw(c_space), "-", color="#9E405D", linewidth=ext_fw, zorder=1)
    # optax.loglog(c_space, hilton_powerlaw(c_space), "-", color="#52409E", linewidth=ext_fw, zorder=1)

    # plot_opt_model_size(optax, deepcopy(bc_comp_tups), drl_pbounds, "il", fit_file_str, "bc opt model size", bc_optsize_artcfg, pareto_min_c=pareto_min, pareto_max_c=pareto_max)
    # plot_opt_model_size(optax, deepcopy(ppo_comp_tups), drl_pbounds, "drl", fit_file_str, "ppo opt model size", ppo_optsize_artcfg, pareto_min_c=pareto_min, pareto_max_c=pareto_max)

    # #optax.set_ylim([5e4, 1.5e6])
    # optax.set_ylim([1e4, 6e6])
    # optax.set_xlim([2e-7, 7e-5])

    # optax.set_xlabel("Compute (PF-days)", fontsize=LABEL_SIZE)

    # optax.tick_params(labelsize=tick_label_size)
    # optax.tick_params(axis="y", which="minor", labelsize=minor_tick_label_size)
    # optax.tick_params(axis="x", which="minor", labelsize=minor_tick_label_size)

    # colorbar_ax.annotate("", xy=(3.9, 0.875), xycoords='axes fraction', xytext=(12.9, 0.875), arrowprops=dict(arrowstyle="<->", shrinkA=14.0, shrinkB=14.0, mutation_scale=3.0))
    # colorbar_ax.annotate("", xy=(3.9, 0.12), xycoords='axes fraction', xytext=(12.9, 0.12), arrowprops=dict(arrowstyle="<->", shrinkA=14.0, shrinkB=14.0, mutation_scale=3.0))

    # lines = optax.get_lines()
    # language, image_gen, rl_procgen = lines[:3]
    # bc, ppo = lines[3], lines[14]
    # bc_trend, ppo_trend = lines[8], lines[16]
    # fig.legend([ppo, bc, ppo_trend, bc_trend, language, image_gen, rl_procgen], ["RL", "SFT", r"$N = \left(\frac{C}{1.8 \times 10^{-12}}\right)^{0.82}$", r"$N = \left(\frac{C}{2.0 \times 10^{-14}}\right)^{0.66}$", "Language\n(Hoffmann et al.)", "Image 32x32\n(Henighan et al.)", "RL Procgen\n(Hilton et al.)"], loc="center left", bbox_to_anchor=(1.003, 0.5), prop={'size': tick_label_size-0.5}, ncols=1, frameon=True) #, framealpha=1)

    # saving
    save_fig(fig, "1_subopt_joint_trend_break", "pdf")
    save_fig(fig, "1_subopt_joint_trend_break", "png")




if __name__ == "__main__":
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    client = MlflowClient(tracking_uri)

    if PLOT_MAIN:
        make_plots(client)

    if PLOT_TREND_BREAK:
        make_plots_trend_break(client)
