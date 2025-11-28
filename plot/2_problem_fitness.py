import argparse
from importlib import import_module

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.text import Text
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from scipy.optimize import curve_fit
import numpy as np
import os
import os.path as osp
import mlflow
from mlflow import MlflowClient
from ast import literal_eval
from collections import defaultdict
import decimal

from plot_utils import get_sol_eval, get_bind_eval, power_scaling_fit, save_fit_eq, clear_fits, forceAspect, exp_scaling_fit, save_expfit_eq, quasi_scaling_fit, save_quasifit_eq, subexp_scaling_fit, save_subexpfit_eq, get_all_mlf_runs, numpify_metrics, numpify_steps


FORMAT = "png"

PLOT_JOINT_MAIN_NIPS = True  # v2 joint suboptimality scaling
PLOT_JOINT_MAIN_NIPS_TREND_BREAK = False  # supp fig
PLOT_JOINT_MAIN_NIPS_TEMPORAL = False  # suboptimality w.r.t. updates

PLOT_DRL_MAIN_NIPS = False  # just bounds now      #combo of PLOT_DRL_MAIN and PLOT_DECOMP_MAIN, with no norm cost or variance scaling, AND mean critic loss
PLOT_DRL_SUPP_NIPS = False  # norm cost and variance scaling exports from above (loss already in supp)

PLOT_DRL_MAIN = False
PLOT_DRL_SUPP = False
PLOT_IL_SUPP = False
PLOT_DECOMP_MAIN = False
PLOT_DECOMP_SUPP = False
PLOT_RAW_STD_SUPP = False
PLOT_RAW_STD_ISOLATION_SUPP = False
PLOT_BOUND_VS_UNBOUND_SUPP = False
PLOT_UPPER_BOUND_APPROACH = False

POWERFIT = True




def plot_instance(plt_fn, scale_x, loss_y, fit_slc, omit_slc, use_c0, use_c1, mode, sign, fit_file_str, id_str, artcfg, plot_data=True, powerfit=True, plot_powerfit=True, powerfit_zorder=3, omit_zorder=1, powerfit_full_xbounds=False, near_linear=False, expfit=False, data_zorder=2, quasifit=False, subexpfit=False, manual_fit_xbounds=None):
    """
    expfit == True will override to use exponential fitting
    quasifit == True will do the same for x^log_m(x) fitting
    """
    assert sum([expfit, subexpfit, quasifit]) <= 1

    point_marker = artcfg["pm"]
    omit_marker = artcfg["om"]
    fit_marker = artcfg["fm"]

    point_size = artcfg["ps"]
    
    data_width = artcfg["dw"]
    fit_width = artcfg["fw"]
    
    data_color = artcfg["dc"]
    fit_color = artcfg["fc"]

    fit_effects = artcfg["fe"]
    point_effects = artcfg["pe"] if "pe" in artcfg else None

    if plot_data:
        # used in fit
        plt_fn(scale_x[fit_slc], loss_y[fit_slc], point_marker, color=data_color, markersize=point_size, linewidth=data_width, zorder=data_zorder, path_effects=point_effects)
        
        # trailing points omitted from fit
        if omit_slc is not None:
            plt_fn(scale_x[omit_slc], loss_y[omit_slc], omit_marker, markerfacecolor="white", color=data_color, markersize=point_size, linewidth=data_width, zorder=omit_zorder)

    # power fit
    if POWERFIT and powerfit:
        print(f"\n{id_str}:")

        if quasifit:
            fit_fn = quasi_scaling_fit
            save_fn = save_quasifit_eq
        elif expfit:
            fit_fn = exp_scaling_fit
            save_fn = save_expfit_eq
        elif subexpfit:
            fit_fn = subexp_scaling_fit
            save_fn = save_subexpfit_eq
        else:
            fit_fn = power_scaling_fit
            save_fn = save_fit_eq        

        x_bounds = (scale_x.min(), scale_x.max()) if powerfit_full_xbounds else (scale_x.min(), scale_x[omit_slc].max() if omit_slc is not None else scale_x[fit_slc].max())
        if manual_fit_xbounds is not None:
            minbx, maxbx = x_bounds
            manmin, manmax = manual_fit_xbounds
            x_bounds = (manmin if manmin is not None else minbx, manmax if manmax is not None else maxbx)
        
        fit_x, fit_y, fit_popt = fit_fn(scale_x[fit_slc], loss_y[fit_slc], x_bounds, c0="fit" if use_c0 else 0, c1="fit" if use_c1 else 0, mode=mode, sign=sign, near_linear=near_linear)
        popt, x_scale, y_scale = fit_popt

        c0, c1, c, m = popt

        b = None
        if expfit or quasifit or subexpfit:
            b = 1.0 if mode == "grow" else -1.0

        save_fn(fit_file_str, id_str, c0, c1, c, m, sign, y_scale, x_scale, b)
        
        if plot_powerfit:
            plt_fn(fit_x, fit_y, fit_marker, color=fit_color, linewidth=fit_width, zorder=powerfit_zorder, path_effects=fit_effects)

        return fit_popt


def modcfg(artcfg, updates):
    new_artcfg = dict(artcfg)
    for k, v in updates.items():
        new_artcfg[k] = v
    return new_artcfg


def save_fig(fig, name, format):
    if format in ("eps", "svg"):
        fig.savefig(f"{name}.{format}", format=format, bbox_inches="tight")
    else:  # assuming non-vector with dpi
        fig.savefig(f"{name}.{format}", format=format, dpi=300, bbox_inches="tight")


def build_powerlaw(c0, c1, c, m, sign, y_scale, x_scale):
    def powerlaw(x):  # one to rule them all
        s = 1 if sign == "positive" else -1
        return y_scale * (c0 + s * (((x / x_scale) - c1) / c)**m)
    return powerlaw


def make_drl_main_plots(client):
    # setup axes
    fig = plt.figure(figsize=(5.5, 3.15))
    grid = GridSpec(2, 3, wspace=0.6, hspace=0.35)

    node_bound_ax = plt.Subplot(fig, grid[0, 0])
    node_ron_ax = plt.Subplot(fig, grid[0, 1])
    node_ronstd_ax = plt.Subplot(fig, grid[0, 2])

    dim_bound_ax = plt.Subplot(fig, grid[1, 0])
    dim_ron_ax = plt.Subplot(fig, grid[1, 1])
    dim_ronstd_ax = plt.Subplot(fig, grid[1, 2])

    axes = (node_bound_ax, node_ron_ax, node_ronstd_ax, dim_bound_ax, dim_ron_ax, dim_ronstd_ax)
    node_axes = axes[:3]
    dim_axes = axes[3:]

    # node_ron_ax.axhline(0, color="black", linewidth=plt.rcParams['xtick.major.width'])
    # dim_ron_ax.axhline(0, color="black", linewidth=plt.rcParams['xtick.major.width'])

    # gather instance data
    bind_exp_name = "EVAL_bind"

    ncost_mu, node_scale = get_bind_eval(client, bind_exp_name, "node", f"drl_node_cost.mean")
    ncost_std, scale2 = get_bind_eval(client, bind_exp_name, "node", f"drl_node_cost.std")
    ncost_opt, scale3 = get_bind_eval(client, bind_exp_name, "node", f"opt_cost.mean")
    ncost_rand, scale4 = get_bind_eval(client, bind_exp_name, "node", f"rand_cost.mean")
    assert all([np.allclose(node_scale, scale) for scale in (scale2, scale3, scale4)])

    nron_mu = (ncost_mu - ncost_opt) / (ncost_rand - ncost_opt)
    nron_std = ncost_std / (ncost_rand - ncost_opt)

    dcost_mu, dim_scale = get_bind_eval(client, bind_exp_name, "dim10", f"drl_dim_cost.mean")
    dcost_std, scale2 = get_bind_eval(client, bind_exp_name, "dim10", f"drl_dim_cost.std")
    dcost_opt, scale3 = get_bind_eval(client, bind_exp_name, "dim10", f"opt_cost.mean")
    dcost_rand, scale4 = get_bind_eval(client, bind_exp_name, "dim10", f"rand_cost.mean")
    assert all([np.allclose(dim_scale, scale) for scale in (scale2, scale3, scale4)])

    dron_mu = (dcost_mu - dcost_opt) / (dcost_rand - dcost_opt)
    dron_std = dcost_std / (dcost_rand - dcost_opt)

    node_fit_slc, node_omit_slc = slice(9), slice(8, 10)
    dim_fit_slc, dim_omit_slc = slice(11), None

    node_artcfg = dict(pm="o-", om="o-", fm="--", ps=3, dw=1, fw=0.75, dc="#FFB05F", fc="black", fe=None)  #[pe.Normal(), pe.Stroke(linewidth=0.5, foreground="white")])
    dim_artcfg = dict(pm="o-", om="o-", fm="--", ps=3, dw=1, fw=0.75, dc="#B05FFF", fc="black", fe=None) #[pe.Normal(), pe.Stroke(linewidth=0.5, foreground="white")]) 

    opt_artcfg = dict(pm="-", om="-", fm="-", ps=3, dw=1.5, fw=1.5, dc="lime", fc="lime", fe=None)
    rand_artcfg = dict(pm="-", om="-", fm="-", ps=3, dw=1.5, fw=1.5, dc="black", fc="black", fe=None)
    
    # plot instances
    fit_file_str = "2_problem_fitness_drl_main_fits.txt"
    clear_fits(fit_file_str)

    node_linspace = np.linspace(5, 50)
    dim_linspace = np.linspace(2, 12)
    #
    popt_opt, x_scale_opt, y_scale_opt = plot_instance(node_bound_ax.plot, node_scale, ncost_opt, slice(len(node_scale)), None, True, True, "grow", "positive", fit_file_str, "node opt avg", opt_artcfg, plot_data=False, plot_powerfit=True, powerfit_zorder=2)
    popt_rand, x_scale_rand, y_scale_rand = plot_instance(node_bound_ax.plot, node_scale, ncost_rand, slice(len(node_scale)), None, True, True, "grow", "positive", fit_file_str, "node rand avg", rand_artcfg, plot_data=False, plot_powerfit=True, powerfit_zorder=2)

    opt_pfn = build_powerlaw(*popt_opt, sign="positive", x_scale=x_scale_opt, y_scale=y_scale_opt)
    rand_pfn = build_powerlaw(*popt_rand, sign="positive", x_scale=x_scale_rand, y_scale=y_scale_rand) 
    node_fill = node_bound_ax.fill_between(node_linspace, opt_pfn(node_linspace), rand_pfn(node_linspace), color=node_artcfg["dc"], linewidth=0.75, zorder=1)
    #
    popt_opt, x_scale_opt, y_scale_opt = plot_instance(dim_bound_ax.plot, dim_scale, dcost_opt, slice(11), None, True, True, "grow", "positive", fit_file_str, "dim(10n) opt avg", opt_artcfg, plot_data=False, plot_powerfit=True, powerfit_zorder=2)
    popt_rand, x_scale_rand, y_scale_rand = plot_instance(dim_bound_ax.plot, dim_scale, dcost_rand, slice(11), None, True, True, "grow", "positive", fit_file_str, "dim(10n) rand avg", rand_artcfg, plot_data=False, plot_powerfit=True, powerfit_zorder=2)

    opt_pfn = build_powerlaw(*popt_opt, sign="positive", x_scale=x_scale_opt, y_scale=y_scale_opt)
    rand_pfn = build_powerlaw(*popt_rand, sign="positive", x_scale=x_scale_rand, y_scale=y_scale_rand) 
    dim_fill = dim_bound_ax.fill_between(dim_linspace, opt_pfn(dim_linspace), rand_pfn(dim_linspace), color=dim_artcfg["dc"], linewidth=0.75, zorder=1)
    #
    plot_instance(node_ron_ax.plot, node_scale, nron_mu, node_fit_slc, node_omit_slc, True, True, "decay", "negative", fit_file_str, "node RON avg (neg decay)", modcfg(node_artcfg, dict(fc="red", fm="-", fw=0.5)))
    plot_instance(node_ron_ax.plot, node_scale, nron_mu, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, "node RON avg (pos grow)", node_artcfg, plot_data=False)
    plot_instance(node_ronstd_ax.semilogy, node_scale, nron_std, node_fit_slc, node_omit_slc, True, True, "decay", "positive", fit_file_str, "node RON std", node_artcfg)

    plot_instance(dim_ron_ax.plot, dim_scale, dron_mu, dim_fit_slc, dim_omit_slc, True, True, "decay", "negative", fit_file_str, "dim(10n) RON avg (neg decay)", dim_artcfg)
    plot_instance(dim_ron_ax.plot, dim_scale, dron_mu, dim_fit_slc, dim_omit_slc, True, True, "grow", "positive", fit_file_str, "dim(10n) RON avg (pos grow)", dim_artcfg, plot_data=False, plot_powerfit=False)
    plot_instance(dim_ronstd_ax.plot, dim_scale, dron_std, dim_fit_slc, dim_omit_slc, True, True, "decay", "negative", fit_file_str, "dim(10n) RON std", dim_artcfg)

    # labelling axes
    label_size = 6
    tick_label_size = 4
    minor_tick_label_size = 4

    node_bound_ax.set_ylabel("Mean tour length (cost)", fontsize=label_size)
    dim_bound_ax.set_ylabel("Mean tour length (cost)", fontsize=label_size)
    node_ron_ax.set_ylabel("Mean normalized test cost", fontsize=label_size)
    dim_ron_ax.set_ylabel("Mean normalized test cost", fontsize=label_size)
    node_ronstd_ax.set_ylabel("SD normalized test cost", fontsize=label_size)
    dim_ronstd_ax.set_ylabel("SD normalized test cost", fontsize=label_size)

    [ax.set_xlabel("Nodes ($n$)", fontsize=label_size) for ax in node_axes]
    [ax.set_xlabel("Spatial dimensions ($x$)", fontsize=label_size) for ax in dim_axes]

    [ax.tick_params(labelsize=tick_label_size) for ax in axes]
    [ax.tick_params(axis="y", which="minor", labelsize=minor_tick_label_size) for ax in axes]

    node_bound_ax.set_ylim(bottom=0)
    dim_bound_ax.set_ylim(bottom=0)
    node_ron_ax.set_ylim(bottom=0)
    dim_ron_ax.set_ylim(bottom=0)

    [ax.set_xticks(list(range(5, 51, 15))) for ax in node_axes]
    [ax.set_xlim([0, 53]) for ax in node_axes[1:]]
    node_bound_ax.set_xlim([5, 50])
    [ax.set_xticks([2, 5, 8, 12]) for ax in dim_axes]
    [ax.set_xlim([0, 13]) for ax in dim_axes[1:]]
    dim_bound_ax.set_xlim([2, 12])

    node_ron_yticks = [0, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3]
    node_ron_ylbls = [Text(0, 0, "0")] + [Text(0, t, f"{t * 1e3:.1f}$ \:\! \\times \:\! 10^{{-3}}$") for t in node_ron_yticks[1:]]
    node_ron_ax.set_yticks(node_ron_yticks, node_ron_ylbls)

    dim_ron_yticks = [0, 0.5e-2, 1e-2, 1.5e-2, 2e-2]
    dim_ron_ylbls = [Text(0, 0, "0")] + [Text(0, t, f"{t * 1e2:.1f}$ \:\! \\times \:\! 10^{{-2}}$") for t in dim_ron_yticks[1:]]
    dim_ron_ax.set_yticks(dim_ron_yticks, dim_ron_ylbls)

    # legend
    node_opt_line, node_rand_line = node_bound_ax.get_lines()
    node_patch = mpatches.Patch(color=node_fill.get_facecolor(), linewidth=0)
    node_bound_ax.legend([node_rand_line, node_opt_line, node_patch], ["random (linear)", "optimal (sublinear)", "achievable"], loc="upper left", prop={'size': tick_label_size}, ncols=1, frameon=False)

    dim_opt_line, dim_rand_line = dim_bound_ax.get_lines()
    dim_patch = mpatches.Patch(color=dim_fill.get_facecolor(), linewidth=0)
    dim_bound_ax.legend([dim_rand_line, dim_opt_line, dim_patch], ["random (sublinear)", "optimal (sublinear)", "achievable"], loc="lower right", prop={'size': tick_label_size}, ncols=1, frameon=False)

    node_trend_line, node_break_line, node_red_fit, node_best_fit = node_ron_ax.get_lines()
    node_ron_ax.legend([node_trend_line, node_break_line, node_best_fit, node_red_fit], ["trend inputs", "trend break", "unbounded growth", "bounded growth"], loc="upper left", prop={'size': tick_label_size}, ncols=1, frameon=False)

    node_trend_line, node_break_line, node_fit = node_ronstd_ax.get_lines()
    node_ronstd_ax.legend([node_trend_line, node_break_line, node_fit], ["trend inputs", "trend break", "decay"], loc="upper right", prop={'size': tick_label_size}, ncols=1, frameon=False)

    dim_trend_line, dim_fit = dim_ron_ax.get_lines()
    dim_ron_ax.legend([dim_trend_line, dim_fit], ["trend inputs", "bounded growth"], loc="lower right", prop={'size': tick_label_size}, ncols=1, frameon=False)

    dim_trend_line, dim_fit = dim_ronstd_ax.get_lines()
    dim_ronstd_ax.legend([dim_trend_line, dim_fit], ["trend inputs", "bounded growth"], loc="lower right", prop={'size': tick_label_size}, ncols=1, frameon=False)

    # add labelled subplots
    [fig.add_subplot(ax) for ax in axes]

    # saving
    save_fig(fig, "2_ppo_main_fitness", FORMAT)


def make_drl_main_plots_nips(client):
    # setup axes
    bound_fig = plt.figure(figsize=(5.5, 1.21))
    out_grid = GridSpec(1, 2, wspace=0.25)
    bound_grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=out_grid[0], wspace=0.15)
    span_grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=out_grid[1], wspace=0.15)

    node_bound_ax = plt.Subplot(bound_fig, bound_grid[0, 0])
    dim_bound_ax = plt.Subplot(bound_fig, bound_grid[0, 1], sharey=node_bound_ax)

    node_span_ax = plt.Subplot(bound_fig, span_grid[0, 0], sharex=node_bound_ax)
    dim_span_ax = plt.Subplot(bound_fig, span_grid[0, 1], sharey=node_span_ax)

    perf_fig = plt.figure(figsize=(5.5, 1.61))
    outer_grid = GridSpec(1, 2, wspace=0.25)
    subopt_grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid[0], wspace=0.1)
    loss_grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid[1], wspace=0.1)

    node_subopt_ax = plt.Subplot(perf_fig, subopt_grid[0, 0])
    node_crt_ax = plt.Subplot(perf_fig, loss_grid[0, 0], sharex=node_subopt_ax)

    dim_subopt_ax = plt.Subplot(perf_fig, subopt_grid[0, 1], sharey=node_subopt_ax)
    dim_crt_ax = plt.Subplot(perf_fig, loss_grid[0, 1], sharex=dim_subopt_ax, sharey=node_crt_ax)

    axes = (node_bound_ax, node_span_ax, node_subopt_ax, node_crt_ax, dim_bound_ax, dim_span_ax, dim_subopt_ax, dim_crt_ax)
    bound_axes = (node_bound_ax, dim_bound_ax)
    span_axes = (node_span_ax, dim_span_ax)
    subopt_axes = (node_subopt_ax, dim_subopt_ax)
    crt_axes = (node_crt_ax, dim_crt_ax)
    node_axes = axes[:4]
    dim_axes = axes[4:]

    [ax.tick_params("y", labelleft=False) for ax in dim_axes]
    [ax.axhline(0, color="black", linewidth=plt.rcParams['xtick.major.width']) for ax in subopt_axes + crt_axes]

    # gather instance data
    bind_exp_name = "EVAL_bind"

    ncost_mu, node_scale = get_bind_eval(client, bind_exp_name, "node", f"drl_node_cost.mean")
    ncost_opt, scale3 = get_bind_eval(client, bind_exp_name, "node", f"opt_cost.mean")
    ncost_rand, scale4 = get_bind_eval(client, bind_exp_name, "node", f"rand_cost.mean")
    ncost_std_rand, scale6 = get_bind_eval(client, bind_exp_name, "node", f"rand_cost.std")
    assert all([np.allclose(node_scale, scale) for scale in (scale3, scale4, scale6)])

    dcost_mu, dim_scale = get_bind_eval(client, bind_exp_name, "dim10", f"drl_dim_cost.mean")
    dcost_opt, scale3 = get_bind_eval(client, bind_exp_name, "dim10", f"opt_cost.mean")
    dcost_rand, scale4 = get_bind_eval(client, bind_exp_name, "dim10", f"rand_cost.mean")
    dcost_std_rand, scale6 = get_bind_eval(client, bind_exp_name, "dim10", f"rand_cost.std")
    assert all([np.allclose(dim_scale, scale) for scale in (scale3, scale4, scale6)])

    d20cost_opt, _ = get_bind_eval(client, bind_exp_name, "dim20", f"opt_cost.mean")
    d20cost_rand, _ = get_bind_eval(client, bind_exp_name, "dim20", f"rand_cost.mean")
    d20cost_std_rand, _ = get_bind_eval(client, bind_exp_name, "dim20", f"rand_cost.std")

    ppo_node_subopt = ncost_mu - ncost_opt
    ppo_dim_subopt = dcost_mu - dcost_opt

    node_span = ncost_rand - ncost_opt
    dim_span = dcost_rand - dcost_opt
    dim20_span = d20cost_rand - d20cost_opt

    node_exp_name = "SOL_node_scaling_drl_2"
    node_crt_loss_y, node_crt_loss_scale = get_sol_eval(client, node_exp_name, "nodes", f"eval_critic_loss_avg")

    dim_exp_name = "SOL_10n_dim_scaling_drl_2"
    dim_crt_loss_y, dim_crt_loss_scale = get_sol_eval(client, dim_exp_name, "dims", f"eval_critic_loss_avg")

    ppo_node_fit_slc, ppo_node_omit_slc = slice(9), slice(8, 10)
    ppo_dim_fit_slc, ppo_dim_omit_slc = slice(11), None

    node_artcfg = dict(pm="o-", om="o-", fm=":", ps=3, dw=1, fw=0.75, dc="#FFB05F", fc="black", fe=None)
    node_span_artcfg = dict(pm="-", om="-", fm="-", ps=3, dw=1.5, fw=1.5, dc="#805830", fc="black", fe=None) #, pe=[pe.Normal(), pe.Stroke(linewidth=1, foreground="black")])  # #664726
    
    dim_artcfg = dict(pm="o-", om="o-", fm="-.", ps=3, dw=1, fw=0.75, dc="#605FFF", fc="black", fe=None)  # original dim10 color #B05FFF
    dim20_artcfg = dict(pm="o-", om="o-", fm="-.", ps=3, dw=1, fw=0.75, dc="#FF5FFE", fc="black", fe=None)
    dim_span_artcfg = dict(pm="-", om="-", fm="-", ps=3, dw=1.5, fw=1.5, dc="#303080", fc="black", fe=None) #, pe=[pe.Normal(), pe.Stroke(linewidth=1, foreground="black")])
    dim20_span_artcfg = dict(pm="-", om="-", fm="-", ps=3, dw=1.5, fw=1.5, dc="#80307f", fc="black", fe=None) 

    opt_artcfg = dict(pm="-", om="-", fm="-", ps=3, dw=2, fw=1.5, dc="lime", fc="lime", fe=None)
    rand_artcfg = dict(pm="-", om="-", fm="-", ps=3, dw=2, fw=1.5, dc="black", fc="black", fe=None)

    bound_buff_col = "whitesmoke"
    std_fill_col = "#b8b8b8"

    # plot instances
    fit_file_str = "2_problem_fitness_drl_main_fits_nips.txt"
    clear_fits(fit_file_str)

    node_linspace = np.linspace(5, 50)
    dim_linspace = np.linspace(2, 12)
    #
    node_std_fill = node_span_ax.fill_between(node_scale, node_span - ncost_std_rand, node_span + ncost_std_rand, color=std_fill_col, linewidth=1, zorder=1)
    plot_instance(node_span_ax.plot, node_scale, node_span, slice(len(node_scale)), None, True, True, "grow", "positive", fit_file_str, "span node", node_span_artcfg, powerfit=False)

    popt_opt, x_scale_opt, y_scale_opt = plot_instance(node_bound_ax.plot, node_scale, ncost_opt, slice(len(node_scale)), None, False, True, "grow", "positive", fit_file_str, "node opt avg", opt_artcfg, plot_data=True, plot_powerfit=False, powerfit_zorder=2)
    popt_rand, x_scale_rand, y_scale_rand = plot_instance(node_bound_ax.plot, node_scale, ncost_rand, slice(len(node_scale)), None, False, True, "grow", "positive", fit_file_str, "node rand avg", rand_artcfg, plot_data=True, plot_powerfit=False, powerfit_zorder=2)

    opt_pfn = build_powerlaw(*popt_opt, sign="positive", x_scale=x_scale_opt, y_scale=y_scale_opt)
    rand_pfn = build_powerlaw(*popt_rand, sign="positive", x_scale=x_scale_rand, y_scale=y_scale_rand) 
    node_fill = node_bound_ax.fill_between(node_linspace, opt_pfn(node_linspace), rand_pfn(node_linspace), color=node_artcfg["dc"], linewidth=1, zorder=1)
    
    # xmin, xmax = node_bound_ax.get_xlim()
    # ymin, ymax = node_bound_ax.get_ylim()
    # node_bound_ax.fill_between(np.linspace(xmin, 5), ymin, ymax, color=bound_buff_col, linewidth=0.1, zorder=2)
    # node_bound_ax.fill_between(np.linspace(50, xmax), ymin, ymax, color=bound_buff_col, linewidth=0.1, zorder=2)
    # node_bound_ax.set_xlim([xmin, xmax])
    # node_bound_ax.set_ylim([ymin, ymax])
    #
    ### 10n
    dim_std_fill = dim_span_ax.fill_between(dim_scale, (dim_span - dcost_std_rand), (dim_span + dcost_std_rand), color=std_fill_col, linewidth=0.75, zorder=1)
    plot_instance(dim_span_ax.plot, dim_scale, dim_span, slice(len(dim_scale)), None, True, True, "grow", "positive", fit_file_str, "span dim10", dim_span_artcfg, powerfit=False)

    popt_opt, x_scale_opt, y_scale_opt = plot_instance(dim_bound_ax.plot, dim_scale, dcost_opt, slice(11), None, False, True, "grow", "positive", fit_file_str, "dim(10n) opt avg", opt_artcfg, plot_data=True, plot_powerfit=False, powerfit_zorder=2)
    popt_rand, x_scale_rand, y_scale_rand = plot_instance(dim_bound_ax.plot, dim_scale, dcost_rand, slice(11), None, False, True, "grow", "positive", fit_file_str, "dim(10n) rand avg", rand_artcfg, plot_data=True, plot_powerfit=False, data_zorder=1, powerfit_zorder=2)

    opt_pfn = build_powerlaw(*popt_opt, sign="positive", x_scale=x_scale_opt, y_scale=y_scale_opt)
    rand_pfn = build_powerlaw(*popt_rand, sign="positive", x_scale=x_scale_rand, y_scale=y_scale_rand) 
    dim_fill = dim_bound_ax.fill_between(dim_linspace, opt_pfn(dim_linspace), rand_pfn(dim_linspace), color=dim_artcfg["dc"], linewidth=0.75, zorder=0)
    
    ### 20n
    dim20_std_fill = dim_span_ax.fill_between(dim_scale, (dim20_span - d20cost_std_rand), (dim20_span + d20cost_std_rand), color=std_fill_col, linewidth=0.75, zorder=1)
    plot_instance(dim_span_ax.plot, dim_scale, dim20_span, slice(len(dim_scale)), None, True, True, "grow", "positive", fit_file_str, "span dim20", dim20_span_artcfg, powerfit=False)

    popt_opt, x_scale_opt, y_scale_opt = plot_instance(dim_bound_ax.plot, dim_scale, d20cost_opt, slice(11), None, False, True, "grow", "positive", fit_file_str, "dim(20n) opt avg", opt_artcfg, plot_data=True, plot_powerfit=False, powerfit_zorder=2)
    popt_rand, x_scale_rand, y_scale_rand = plot_instance(dim_bound_ax.plot, dim_scale, d20cost_rand, slice(11), None, False, True, "grow", "positive", fit_file_str, "dim(20n) rand avg", rand_artcfg, plot_data=True, plot_powerfit=False, powerfit_zorder=2)

    opt_pfn = build_powerlaw(*popt_opt, sign="positive", x_scale=x_scale_opt, y_scale=y_scale_opt)
    rand_pfn = build_powerlaw(*popt_rand, sign="positive", x_scale=x_scale_rand, y_scale=y_scale_rand) 
    dim20_fill = dim_bound_ax.fill_between(dim_linspace, opt_pfn(dim_linspace), rand_pfn(dim_linspace), color=dim20_artcfg["dc"], linewidth=0.75, zorder=1)
    

    # xmin, xmax = dim_bound_ax.get_xlim()
    # ymin, ymax = dim_bound_ax.get_ylim()
    # dim_bound_ax.fill_between(np.linspace(xmin, 2), ymin, ymax, color=bound_buff_col, linewidth=0.1, zorder=2)
    # dim_bound_ax.fill_between(np.linspace(12, xmax), ymin, ymax, color=bound_buff_col, linewidth=0.1, zorder=2)
    # dim_bound_ax.set_xlim([xmin, xmax])
    # dim_bound_ax.set_ylim([ymin, ymax])
    #
    plot_instance(node_subopt_ax.plot, node_scale, ppo_node_subopt, ppo_node_fit_slc, ppo_node_omit_slc, True, True, "grow", "positive", fit_file_str, "subopt ppo node", node_artcfg)
    plot_instance(dim_subopt_ax.plot, dim_scale, ppo_dim_subopt, ppo_dim_fit_slc, ppo_dim_omit_slc, True, True, "decay", "negative", fit_file_str, "subopt ppo dim10", dim_artcfg)

    plot_instance(node_crt_ax.plot, node_crt_loss_scale, node_crt_loss_y, ppo_node_fit_slc, ppo_node_omit_slc, True, True, "grow", "positive", fit_file_str, "node critic avg", node_artcfg)
    plot_instance(dim_crt_ax.plot, dim_crt_loss_scale, dim_crt_loss_y, ppo_dim_fit_slc, ppo_dim_omit_slc, True, True, "decay", "negative", fit_file_str, "dim(10n) critic avg", dim_artcfg)

    # labelling axes
    label_size = 6
    tick_label_size = 4
    minor_tick_label_size = 4

    node_bound_ax.set_ylabel("Mean tour length (cost)", fontsize=label_size)
    node_span_ax.set_ylabel("Suboptimality gap", fontsize=label_size)
    node_subopt_ax.set_ylabel("Suboptimality gap", fontsize=label_size)
    node_crt_ax.set_ylabel("Mean critic test loss", fontsize=label_size)

    [ax.set_xlabel("Nodes ($n$)", fontsize=label_size) for ax in node_axes]
    [ax.set_xlabel("Spatial dimensions ($d$)", fontsize=label_size) for ax in dim_axes]

    [ax.tick_params(labelsize=tick_label_size) for ax in axes]
    [ax.tick_params(axis="y", which="minor", labelsize=minor_tick_label_size) for ax in axes]

    [ax.set_xticks(list(range(5, 51, 15))) for ax in node_axes]
    # [ax.set_xlim([0, 53]) for ax in node_axes[2:]]
    node_bound_ax.set_xlim([5, 50])
    dim_bound_ax.set_xticks([2, 5, 8, 12])
    # [ax.set_xlim([0, 13]) for ax in dim_axes[2:]]
    dim_bound_ax.set_xlim([2, 12])
    
    dim_span_ax.set_xlim([2, 100])
    dim_span_ax.set_xticks([2, 12, 30, 50, 100])

    [ax.set_ylim(bottom=0) for ax in bound_axes + span_axes]
    # dim_subopt_ax.set_ylim(bottom=0)
    node_subopt_ax.set_ylim(bottom=-0.003)
    node_crt_ax.set_ylim(bottom=-3e-4)

    # legend
    node_opt_line, node_rand_line = node_bound_ax.get_lines()
    node_patch = mpatches.Patch(color=node_fill.get_facecolor(), linewidth=0)
    node_bound_ax.legend([node_rand_line, node_patch, node_opt_line], ["random", "achievable", "optimal"], loc="upper left", prop={'size': tick_label_size+1}, ncols=1, frameon=False)

    dim_opt_line, dim_rand_line = dim_bound_ax.get_lines()[:2]
    dim_patch = mpatches.Patch(color=dim_fill.get_facecolor(), linewidth=0)
    dim20_patch = mpatches.Patch(color=dim20_fill.get_facecolor(), linewidth=0)
    dim_bound_ax.legend([dim20_patch, dim_patch], ["ach. ($n=20$)", "ach. ($n=10$)"], loc="upper left", bbox_to_anchor=(-0.15, 1.03), prop={'size': tick_label_size}, ncols=1, columnspacing=0.5, frameon=True, framealpha=1)

    node_span_line = node_span_ax.get_lines()[-1]
    node_std_patch = mpatches.Patch(color=node_std_fill.get_facecolor(), linewidth=0)
    node_span_ax.legend([node_span_line, node_std_patch], ["random", "random SD"], loc="upper left", prop={'size': tick_label_size+1}, ncols=1, frameon=False)

    dim_span_line, dim20_span_line = dim_span_ax.get_lines()
    dim_std_patch = mpatches.Patch(color=dim_std_fill.get_facecolor(), linewidth=0)
    dim_span_ax.legend([dim20_span_line, dim_span_line, dim_std_patch], ["random ($n=20$)", "random ($n=10$)", "random SD"], loc="upper left", prop={'size': tick_label_size+1}, ncols=1, frameon=False)

    _, node_trend, node_break, node_unbounded_fit = node_crt_ax.get_lines()
    _, dim_trend, dim_bounded_fit = dim_crt_ax.get_lines()
    empty_handle = mpatches.Rectangle((0,0), 1, 1, fill=False, edgecolor="none", visible=False)
    perf_fig.legend([node_trend, node_unbounded_fit, node_break, empty_handle, dim_trend, dim_bounded_fit], ["$n$ trend inputs", "$s - \\beta \propto (n - \\gamma)^{{\\alpha}}$", "$n$ trend break", " ", "$d$ trend inputs", "$s - \\beta \propto -(d - \\gamma)^{{-\\alpha}}$"], loc="upper center", prop={'size': label_size}, bbox_to_anchor=(0.5, -0.1), ncols=3, frameon=False)

    # add labelled subplots
    [forceAspect(ax, aspect=1) for ax in bound_axes + span_axes]
    [forceAspect(ax, aspect=0.75) for ax in subopt_axes + crt_axes]
    
    [bound_fig.add_subplot(ax) for ax in bound_axes + span_axes]
    [perf_fig.add_subplot(ax) for ax in subopt_axes + crt_axes]

    # saving
    save_fig(bound_fig, "2_ppo_main_bounds_nips", "png")
    save_fig(bound_fig, "2_ppo_main_bounds_nips", "pdf")
    
    # save_fig(perf_fig, "2_ppo_main_fitness_nips", "png")
    # save_fig(perf_fig, "2_ppo_main_fitness_nips", "pdf")


def make_joint_main_plots_nips(client):
    # setup axes
    perf_fig = plt.figure(figsize=(5.5, 2.0))
    grid = GridSpec(1, 2, wspace=-0.15, width_ratios=[4.5, 5.5])

    node_subopt_ax = plt.Subplot(perf_fig, grid[0, 0])
    dim_subopt_ax = plt.Subplot(perf_fig, grid[0, 1])

    axes = [node_subopt_ax, dim_subopt_ax]

    # [ax.tick_params("y", labelleft=False) for ax in dim_axes]
    [ax.axhline(0, color="black", linewidth=plt.rcParams['xtick.major.width']) for ax in axes]

    # gather instance data
    bind_exp_name = "EVAL_bind"

    ncost_ppo, node_scale = get_bind_eval(client, bind_exp_name, "node", f"drl_node_cost.mean")
    ncost_bc, scale_2 = get_bind_eval(client, bind_exp_name, "node", f"il_node_cost.mean")
    ncost_opt, scale3 = get_bind_eval(client, bind_exp_name, "node", f"opt_cost.mean")

    d10cost_ppo, dim10_scale = get_bind_eval(client, bind_exp_name, "dim10", f"drl_dim_cost.mean")
    d10cost_opt, scale3 = get_bind_eval(client, bind_exp_name, "dim10", f"opt_cost.mean")

    d20cost_ppo, dim20_scale = get_bind_eval(client, bind_exp_name, "dim20", f"drl_dim_cost.mean")
    d20cost_opt, scale7 = get_bind_eval(client, bind_exp_name, "dim20", f"opt_cost.mean")

    ppo_node_subopt = ncost_ppo- ncost_opt
    bc_node_subopt = ncost_bc - ncost_opt
    ppo_dim10_subopt = d10cost_ppo - d10cost_opt
    ppo_dim20_subopt = d20cost_ppo - d20cost_opt

    ppo_node_fit_slc, ppo_node_omit_slc = slice(9), None
    bc_node_fit_slc, bc_node_omit_slc = slice(6), None
    ppo_d10_fit_slc, ppo_d10_omit_slc = slice(11), None
    ppo_d20_fit_slc, ppo_d20_omit_slc = slice(9), None

    ppo_node_artcfg = dict(pm="o-", om="o-", fm=":", ps=3, dw=1, fw=0.75, dc="#FFB05F", fc="black", fe=None)
    bc_node_artcfg = dict(pm="s-", om="o-", fm=":", ps=3, dw=1, fw=0.75, dc="#FF605F", fc="black", fe=None)
    d10_artcfg = dict(pm="o-", om="o-", fm="-", ps=3, dw=1, fw=0.5, dc="#605FFF", fc="black", fe=None)  # original dim10 color #B05FFF
    d20_artcfg = dict(pm="o-", om="o-", fm="-", ps=3, dw=1, fw=0.5, dc="#FF5FFE", fc="black", fe=None)

    # plot instances
    fit_file_str = "2_joint_problem_fitness_fits_nips.txt"
    clear_fits(fit_file_str)

    plot_instance(node_subopt_ax.plot, node_scale, bc_node_subopt, bc_node_fit_slc, bc_node_omit_slc, False, True, "grow", "positive", fit_file_str, "subopt bc node", bc_node_artcfg, powerfit_zorder=2)
    
    plot_instance(node_subopt_ax.plot, node_scale, ppo_node_subopt, ppo_node_fit_slc, ppo_node_omit_slc, False, True, "grow", "positive", fit_file_str, "subopt ppo node", ppo_node_artcfg)
    
    plot_instance(dim_subopt_ax.plot, dim10_scale, ppo_dim10_subopt, ppo_d10_fit_slc, ppo_d10_omit_slc, False, True, "grow", "positive", fit_file_str, "subopt ppo dim10 (power growth)", modcfg(d10_artcfg, dict(fc="black", fm=":", fw=0.5)), plot_data=False, plot_powerfit=False)
    plot_instance(dim_subopt_ax.plot, dim10_scale, ppo_dim10_subopt, ppo_d10_fit_slc, ppo_d10_omit_slc, True, False, "decay", "negative", fit_file_str, "subopt ppo dim10 (power decay)", modcfg(d10_artcfg, dict(fc="black", fm="-.", fw=0.5)), plot_data=False, plot_powerfit=False)
    d10_popt = plot_instance(dim_subopt_ax.plot, dim10_scale, ppo_dim10_subopt, ppo_d10_fit_slc, ppo_d10_omit_slc, True, False, "decay", "negative", fit_file_str, "subopt ppo dim10 (exp decay)", d10_artcfg, expfit=True)
    
    plot_instance(dim_subopt_ax.plot, dim20_scale, ppo_dim20_subopt, ppo_d20_fit_slc, ppo_d20_omit_slc, False, True, "grow", "positive", fit_file_str, "subopt ppo dim20 (power growth)", modcfg(d20_artcfg, dict(fc="black", fm=":", fw=0.5)), plot_data=False)
    plot_instance(dim_subopt_ax.plot, dim20_scale, ppo_dim20_subopt, ppo_d20_fit_slc, ppo_d20_omit_slc, True, False, "decay", "negative", fit_file_str, "subopt ppo dim20 (power decay)", modcfg(d20_artcfg, dict(fc="black", fm="-.", fw=0.5)), plot_data=False)
    d20_popt = plot_instance(dim_subopt_ax.plot, dim20_scale, ppo_dim20_subopt, ppo_d20_fit_slc, ppo_d20_omit_slc, True, False, "decay", "negative", fit_file_str, "subopt ppo dim20 (exp decay)", d20_artcfg, expfit=True)

    beta_col = "slategrey"
    d10_beta = d10_popt[-1] * d10_popt[0][0]
    d20_beta = d20_popt[-1] * d20_popt[0][0]
    dim_subopt_ax.axhline(d10_beta, color=beta_col, linewidth=0.5, zorder=1)
    dim_subopt_ax.axhline(d20_beta, color=beta_col, linewidth=0.5, zorder=1)

    # labelling axes
    label_size = 6
    tick_label_size = 4
    minor_tick_label_size = 4

    node_subopt_ax.set_ylabel("Suboptimality gap", fontsize=label_size)
    dim_subopt_ax.set_ylabel("Suboptimality gap", fontsize=label_size)

    node_subopt_ax.set_xlabel("Nodes ($n$)", fontsize=label_size)
    dim_subopt_ax.set_xlabel("Spatial dimensions ($d$)", fontsize=label_size)

    [ax.tick_params(labelsize=tick_label_size) for ax in axes]
    [ax.tick_params(axis="y", which="minor", labelsize=minor_tick_label_size) for ax in axes]

    node_subopt_ax.set_xticks(list(range(5, 46, 5)))
    #node_subopt_ax.set_xticks(list(range(5, 51, 15)))
    # [ax.set_xlim([0, 53]) for ax in node_axes[2:]]
    dim_subopt_ax.set_xticks(list(range(2, 13, 1)))
    #dim_subopt_ax.set_xticks([2, 5, 8, 12])
    # [ax.set_xlim([0, 13]) for ax in dim_axes[2:]]

    node_subopt_ax.set_ylim(bottom=-0.002)
    dim_subopt_ax.set_ylim(bottom=-0.008)

    # legend
    forceAspect(node_subopt_ax, aspect=0.9)
    forceAspect(dim_subopt_ax, aspect=1.1)

    _, bc_node, _, ppo_node, node_fit = node_subopt_ax.get_lines()
    node_subopt_ax.legend([ppo_node, bc_node, node_fit], ["RL", "SFT", "$s \propto (n - \\gamma)^{{\\alpha}}$"], loc="upper left", prop={'size': label_size}, ncols=1, frameon=False)
    
    _, d10, _, unbound_fit, bound_fit, d20, expbound_fit, _, _ = dim_subopt_ax.get_lines()
    perf_fig.legend([d10, d20, unbound_fit, bound_fit, expbound_fit], ["RL $(n=10)$", "RL $(n=20)$", r"$s \propto (d - \gamma)^{\alpha}$", r"$s - \beta \propto -d^{-\alpha}$", r"$s - \beta \propto -\psi^{-d}$",], loc="center right", bbox_to_anchor=(0.827, 0.52), prop={'size': label_size}, ncols=1, frameon=True) #, framealpha=1)

    xmin, xmax = dim_subopt_ax.get_xlim()
    yeps = 0.002
    xeps = 0.1
    dim_subopt_ax.text(xmin+xeps, d20_beta-yeps, r"$\beta_{\psi ; n=20}$", color=beta_col, fontsize=label_size, ha='left', va='top')
    dim_subopt_ax.text(xmin+16*xeps, d10_beta-yeps, r"$\beta_{\psi ; n=10}$", color=beta_col, fontsize=label_size, ha='left', va='top')

    # add labelled subplots
    [perf_fig.add_subplot(ax) for ax in axes]

    # saving
    save_fig(perf_fig, "2_complexity_camera_ready_nips", "png")
    save_fig(perf_fig, "2_complexity_camera_ready_nips", "pdf")


def make_joint_main_plots_nips_trend_break(client):
    # setup axes
    perf_fig = plt.figure(figsize=(5.5, 1.21))
    grid = GridSpec(1, 3, wspace=0.25, width_ratios=[4, 3, 3])

    node_subopt_ax = plt.Subplot(perf_fig, grid[0, 0])
    dim_subopt_ax = plt.Subplot(perf_fig, grid[0, 1])
    full_dim_subopt_ax = plt.Subplot(perf_fig, grid[0, 2])

    axes = [node_subopt_ax, dim_subopt_ax, full_dim_subopt_ax]

    # [ax.tick_params("y", labelleft=False) for ax in dim_axes]
    [ax.axhline(0, color="black", linewidth=plt.rcParams['xtick.major.width']) for ax in axes]

    # gather instance data
    bind_exp_name = "EVAL_bind"

    ncost_ppo, node_scale = get_bind_eval(client, bind_exp_name, "node", f"drl_node_cost.mean")
    ncost_bc, scale_2 = get_bind_eval(client, bind_exp_name, "node", f"il_node_cost.mean")
    ncost_opt, scale3 = get_bind_eval(client, bind_exp_name, "node", f"opt_cost.mean")

    d10cost_ppo, dim10_scale = get_bind_eval(client, bind_exp_name, "dim10", f"drl_dim_cost.mean")
    d10cost_opt, scale3 = get_bind_eval(client, bind_exp_name, "dim10", f"opt_cost.mean")

    d20cost_ppo, dim20_scale = get_bind_eval(client, bind_exp_name, "dim20", f"drl_dim_cost.mean")
    d20cost_opt, scale7 = get_bind_eval(client, bind_exp_name, "dim20", f"opt_cost.mean")

    ppo_node_subopt = ncost_ppo- ncost_opt
    bc_node_subopt = ncost_bc - ncost_opt
    ppo_dim10_subopt = d10cost_ppo - d10cost_opt
    ppo_dim20_subopt = d20cost_ppo - d20cost_opt

    ppo_node_fit_slc, ppo_node_omit_slc = slice(9), slice(8, 10)
    bc_node_fit_slc, bc_node_omit_slc = slice(6), slice(5, 10)
    ppo_d10_fit_slc, ppo_d10_omit_slc = slice(11), slice(10, 13)
    ppo_d20_fit_slc, ppo_d20_omit_slc = slice(9), slice(8, 13)
    full_ppo_d10_omit_slc = slice(10, 17)
    full_ppo_d20_omit_slc = slice(8, 17)

    ppo_node_artcfg = dict(pm="o-", om="o-", fm=":", ps=3, dw=1, fw=0.75, dc="#FFB05F", fc="black", fe=None)
    bc_node_artcfg = dict(pm="s-", om="s-", fm="-", ps=3, dw=1, fw=0.5, dc="#FF605F", fc="olive", fe=None)
    d10_artcfg = dict(pm="o-", om="o-", fm="-", ps=2, dw=1, fw=0.5, dc="#605FFF", fc="black", fe=None)  # original dim10 color #B05FFF
    d20_artcfg = dict(pm="o-", om="o-", fm="-", ps=2, dw=1, fw=0.5, dc="#FF5FFE", fc="black", fe=None)

    # plot instances
    fit_file_str = "2_joint_problem_fitness_fits_nips_trend_break.txt"
    clear_fits(fit_file_str)

    plot_instance(node_subopt_ax.plot, node_scale, bc_node_subopt, bc_node_fit_slc, bc_node_omit_slc, False, True, "grow", "positive", fit_file_str, "subopt bc node", bc_node_artcfg, powerfit_zorder=2)
    
    plot_instance(node_subopt_ax.plot, node_scale, ppo_node_subopt, ppo_node_fit_slc, ppo_node_omit_slc, False, True, "grow", "positive", fit_file_str, "subopt ppo node", ppo_node_artcfg)
    
    #plot_instance(dim_subopt_ax.plot, dim10_scale, ppo_dim10_subopt, ppo_d10_fit_slc, ppo_d10_omit_slc, False, True, "grow", "positive", fit_file_str, "subopt ppo dim10 (power growth)", modcfg(d10_artcfg, dict(fc="black", fm=":", fw=0.5)), plot_data=False, plot_powerfit=False)
    #plot_instance(dim_subopt_ax.plot, dim10_scale, ppo_dim10_subopt, ppo_d10_fit_slc, ppo_d10_omit_slc, True, False, "decay", "negative", fit_file_str, "subopt ppo dim10 (power decay)", modcfg(d10_artcfg, dict(fc="black", fm="-.", fw=0.5)), plot_data=False, plot_powerfit=False)
    d10_popt = plot_instance(dim_subopt_ax.plot, dim10_scale, ppo_dim10_subopt, ppo_d10_fit_slc, ppo_d10_omit_slc, True, False, "decay", "negative", fit_file_str, "subopt ppo dim10 (exp decay)", d10_artcfg, expfit=True)
    
    #plot_instance(dim_subopt_ax.plot, dim20_scale, ppo_dim20_subopt, ppo_d20_fit_slc, ppo_d20_omit_slc, False, True, "grow", "positive", fit_file_str, "subopt ppo dim20 (power growth)", modcfg(d20_artcfg, dict(fc="black", fm=":", fw=0.5)), plot_data=False)
    #plot_instance(dim_subopt_ax.plot, dim20_scale, ppo_dim20_subopt, ppo_d20_fit_slc, ppo_d20_omit_slc, True, False, "decay", "negative", fit_file_str, "subopt ppo dim20 (power decay)", modcfg(d20_artcfg, dict(fc="black", fm="-.", fw=0.5)), plot_data=False)
    d20_popt = plot_instance(dim_subopt_ax.plot, dim20_scale, ppo_dim20_subopt, ppo_d20_fit_slc, ppo_d20_omit_slc, True, False, "decay", "negative", fit_file_str, "subopt ppo dim20 (exp decay)", d20_artcfg, expfit=True)

    plot_instance(full_dim_subopt_ax.plot, dim10_scale, ppo_dim10_subopt, ppo_d10_fit_slc, full_ppo_d10_omit_slc, True, False, "decay", "negative", fit_file_str, "subopt ppo dim10 (exp decay)", d10_artcfg, expfit=True)
    plot_instance(full_dim_subopt_ax.plot, dim20_scale, ppo_dim20_subopt, ppo_d20_fit_slc, full_ppo_d20_omit_slc, True, False, "decay", "negative", fit_file_str, "subopt ppo dim20 (exp decay)", d20_artcfg, expfit=True)

    # beta_col = "slategrey"
    # d10_beta = d10_popt[-1] * d10_popt[0][0]
    # d20_beta = d20_popt[-1] * d20_popt[0][0]
    # dim_subopt_ax.axhline(d10_beta, color=beta_col, linewidth=0.5, zorder=1)
    # dim_subopt_ax.axhline(d20_beta, color=beta_col, linewidth=0.5, zorder=1)

    # labelling axes
    label_size = 6
    tick_label_size = 4
    minor_tick_label_size = 4

    node_subopt_ax.set_ylabel("Suboptimality gap", fontsize=label_size)
    # dim_subopt_ax.set_ylabel("Suboptimality gap", fontsize=label_size)

    node_subopt_ax.set_xlabel("Nodes ($n$)", fontsize=label_size)
    dim_subopt_ax.set_xlabel("Spatial dimensions ($d$)", fontsize=label_size)
    full_dim_subopt_ax.set_xlabel("Spatial dimensions ($d$)", fontsize=label_size)

    [ax.tick_params(labelsize=tick_label_size) for ax in axes]
    [ax.tick_params(axis="y", which="minor", labelsize=minor_tick_label_size) for ax in axes]

    # node_subopt_ax.set_xticks(list(range(5, 46, 5)))
    node_subopt_ax.set_xticks([5, 20, 35, 50])
    # [ax.set_xlim([0, 53]) for ax in node_axes[2:]]
    dim_subopt_ax.set_xticks([2, 5, 10, 15, 20])
    #dim_subopt_ax.set_xticks([2, 5, 8, 12])
    # [ax.set_xlim([0, 13]) for ax in dim_axes[2:]]
    full_dim_subopt_ax.set_xticks([2, 10, 20, 30, 40, 50, 100])

    node_subopt_ax.set_ylim(bottom=-0.003)
    dim_subopt_ax.set_ylim(bottom=-0.012)
    full_dim_subopt_ax.set_ylim(bottom=-0.012)

    # legend
    # [forceAspect(ax, aspect=1.5) for ax in axes]

    _, bc_node, bc_node_break, bc_fit, ppo_node, ppo_node_break, node_fit = node_subopt_ax.get_lines()
    node_subopt_ax.legend([ppo_node, ppo_node_break, node_fit, bc_node, bc_node_break, bc_fit], ["RL", "trend break", "$s \propto (n - \\gamma_{{1}})^{{\\alpha_{{1}}}}$", "SFT", "trend break", "$s \propto (n - \\gamma_{{2}})^{{\\alpha_{{2}}}}$"], loc="upper left", prop={'size': tick_label_size}, ncols=2, frameon=True)
    
    _, d10, d10_break, _, d20, d20_break, expbound_fit = dim_subopt_ax.get_lines()
    dim_subopt_ax.legend([d10, d10_break, expbound_fit, d20, d20_break], ["RL $(n=10)$", "trend break", r"$s - \beta \propto -\psi^{-d}$", "RL $(n=20)$", "trend break"], loc="center right", bbox_to_anchor=(1.0, 0.4), prop={'size': tick_label_size-1}, ncols=2, columnspacing=0.75, frameon=True) #, framealpha=1)

    full_dim_subopt_ax.legend([d10, d10_break, d20, d20_break, expbound_fit], ["RL $(n=10)$", "trend break", "RL $(n=20)$", "trend break", r"$s - \beta \propto -\psi^{-d}$"], loc="center right", bbox_to_anchor=(1.0, 0.6), prop={'size': tick_label_size-1}, ncols=1, columnspacing=0.75, frameon=True) #, framealpha=1)

    # xmin, xmax = dim_subopt_ax.get_xlim()
    # yeps = 0.003
    # xeps = 0.1
    # dim_subopt_ax.text(xmin+xeps, d20_beta-yeps, r"$\beta_{\psi ; n=20}$", color=beta_col, fontsize=label_size, ha='left', va='top')
    # dim_subopt_ax.text(xmax-xeps, d10_beta+yeps, r"$\beta_{\psi ; n=10}$", color=beta_col, fontsize=label_size, ha='right', va='bottom')

    # add labelled subplots
    [perf_fig.add_subplot(ax) for ax in axes]

    # saving
    save_fig(perf_fig, "2_joint_problem_fitness_nips_trend_break", "png")
    save_fig(perf_fig, "2_joint_problem_fitness_nips_trend_break", "pdf")


def make_joint_main_plots_nips_temporal(client):
    # setup axes
    fig = plt.figure(figsize=(5.5, 2.5))
    out_grid = GridSpec(1, 2, wspace=0.15, width_ratios=[3.3, 1])
    lead_grid = GridSpecFromSubplotSpec(1, 3, subplot_spec=out_grid[0], wspace=0.15)

    ppo_node_ax = plt.Subplot(fig, lead_grid[0, 1])
    bc_node_ax = plt.Subplot(fig, lead_grid[0, 0], sharey=ppo_node_ax)

    ppo_dim10_ax = plt.Subplot(fig, lead_grid[0, 2], sharey=ppo_node_ax)
    ppo_dim20_ax = plt.Subplot(fig, out_grid[0, 1])

    node_axes = [ppo_node_ax, bc_node_ax]
    dim_axes = [ppo_dim10_ax, ppo_dim20_ax]
    axes = node_axes + dim_axes

    [ax.axvline(170_000, color="forestgreen", linewidth=1.5, zorder=1) for ax in dim_axes + [ppo_node_ax]]  # where cosine decay finishes, and slow linear decay begins

    # gather data
    def get_metrics_and_scales(client, exp_name, metric_key, scale_key):
        plot_runs = get_all_mlf_runs(client, exp_name)

        curves = []  # (y, x, params)

        for run in plot_runs:
            metrics = client.get_metric_history(run.info.run_id, metric_key)
            scale = int(run.data.params[scale_key])

            x_vals = next(numpify_steps(metrics))
            y_vals = next(numpify_metrics(metrics))

            sort_idx = np.argsort(x_vals)

            curves.append((y_vals[sort_idx], x_vals[sort_idx], scale))

        return sorted(curves, key=lambda x: x[-1])
    
    bc_node_curves = get_metrics_and_scales(client, "CHECKGEN_node_scaling_il", "eval_cost_avg", "nodes")
    ppo_node_curves = get_metrics_and_scales(client, "CHECKGEN_node_scaling_drl", "eval_cost_avg", "nodes")
    ppo_dim10_curves = get_metrics_and_scales(client, "CHECKGEN_10n_dim_scaling_drl", "eval_cost_avg", "dims")
    ppo_dim20_curves = get_metrics_and_scales(client, "CHECKGEN_20n_dim_scaling_drl", "eval_cost_avg", "dims")

    bind_exp_name = "EVAL_bind"
    ncost_opt, _ = get_bind_eval(client, bind_exp_name, "node", f"opt_cost.mean")
    d10cost_opt, _ = get_bind_eval(client, bind_exp_name, "dim10", f"opt_cost.mean")
    d20cost_opt, _ = get_bind_eval(client, bind_exp_name, "dim20", f"opt_cost.mean")

    # sft node plot
    for idx, (y, x, n) in enumerate(bc_node_curves):
        s = y - ncost_opt[idx]
        # print(n, ncost_opt[idx], y[-1])
        bc_node_ax.plot(x, s, "-", color="#FF605F" if n < 35 else "black", linewidth=0.75, zorder=2)
    
    bc_node_ax.set_xlim([0, 73_143])
    bc_node_ax.set_ylim([-0.0075, 0.5])

    bc_node_ticks = [0, 3.5e4, 7e4]
    bc_node_labels = [Text(0, 0, "0"), Text(0, 3.5e4, "$3.5 \:\! \\times \:\! 10^{{4}}$"), Text(0, 7e4, "$7 \:\! \\times \:\! 10^{{4}}$")]
    bc_node_ax.set_xticks(bc_node_ticks, bc_node_labels)

    # drl node plot 
    for idx, (y, x, n) in enumerate(ppo_node_curves):
        x *= 4
        s = y - ncost_opt[idx]
        # print(n, ncost_opt[idx], y[-1])
        ppo_node_ax.plot(x, s, "-", color="#FFB05F" if n < 50 else "black", linewidth=0.75, zorder=2)

    ppo_node_ax.set_xlim([0, 1e6])
    ppo_node_ax.set_ylim([-0.0075, 0.5])
    ppo_node_ax.tick_params("y", labelleft=False)

    ppo_node_ticks = [0, 5e5, 1e6]
    ppo_node_labels = [Text(0, 0, "0"), Text(0, 5e5, "$5 \:\! \\times \:\! 10^{{5}}$"), Text(0, 1e6, "$10^{{6}}$")]
    ppo_node_ax.set_xticks(ppo_node_ticks, ppo_node_labels)

    # drl dim10 plot 
    for idx, (y, x, d) in enumerate(ppo_dim10_curves):
        x *= 4
        s = y - d10cost_opt[idx]
        # print(d, d10cost_opt[idx], y[-1])
        ppo_dim10_ax.plot(x, s, "-", color="#605FFF" if d < 15 else "black", linewidth=0.75, zorder=2)

    ppo_dim10_ax.tick_params("y", labelleft=False)
    ppo_dim10_ax.set_xlim([0, 1e6])
    ppo_dim10_ax.set_ylim([-0.0075, 0.5])

    ppo_dim10_ticks = [0, 5e5, 1e6]
    ppo_dim10_labels = [Text(0, 0, "0"), Text(0, 5e5, "$5 \:\! \\times \:\! 10^{{5}}$"), Text(0, 1e6, "$10^{{6}}$")]
    ppo_dim10_ax.set_xticks(ppo_dim10_ticks, ppo_dim10_labels)

    # drl dim20 plot 
    for idx, (y, x, d) in enumerate(ppo_dim20_curves):
        x *= 4
        s = y - d20cost_opt[idx]
        # print(d, d20cost_opt[idx], y[-1])
        ppo_dim20_ax.plot(x, s, "-", color="#FF5FFE" if d < 11 else "black", linewidth=0.75, zorder=2)

    # ppo_dim20_ax.tick_params("y", labelleft=False)
    ppo_dim20_ax.set_xlim([0, 1e6])
    ppo_dim20_ax.set_ylim([0, 1.0])

    ppo_dim20_ticks = [0, 5e5, 1e6]
    ppo_dim20_labels = [Text(0, 0, "0"), Text(0, 5e5, "$5 \:\! \\times \:\! 10^{{5}}$"), Text(0, 1e6, "$10^{{6}}$")]
    ppo_dim20_ax.set_xticks(ppo_dim20_ticks, ppo_dim20_labels)

    # labelling axes
    label_size = 6
    tick_label_size = 4
    minor_tick_label_size = 4

    bc_node_ax.set_ylabel("Suboptimality gap", fontsize=label_size)

    [ax.set_xlabel("Model updates", fontsize=label_size) for ax in axes]

    [ax.tick_params(labelsize=tick_label_size) for ax in axes]
    [ax.tick_params(axis="y", which="minor", labelsize=minor_tick_label_size) for ax in axes]

    ppo_node_ax.set_title("$n$-RL", fontsize=label_size)
    bc_node_ax.set_title("$n$-SFT", fontsize=label_size)
    ppo_dim10_ax.set_title("$d$-RL $(n=10)$", fontsize=label_size)
    ppo_dim20_ax.set_title("$d$-RL $(n=20)$", fontsize=label_size)

    # legend
    lrd_line = ppo_node_ax.get_lines()[0]
    omit_line = ppo_node_ax.get_lines()[-1]
    bc_node_fit = bc_node_ax.get_lines()[1]
    ppo_node_fit = ppo_node_ax.get_lines()[1]
    ppo_dim10_fit = ppo_dim10_ax.get_lines()[1]
    ppo_dim20_fit = ppo_dim20_ax.get_lines()[1]

    bc_node_ax.legend([omit_line, bc_node_fit], ["excluded", "fitted"], loc="upper right", prop={'size': tick_label_size+1}, ncols=1, frameon=True)
    ppo_node_ax.legend([omit_line, ppo_node_fit, lrd_line], ["excluded", "fitted", r"$\eta = 10^{-5}$"], loc="upper right", prop={'size': tick_label_size+1}, ncols=1, frameon=True)
    ppo_dim10_ax.legend([omit_line, ppo_dim10_fit, lrd_line], ["excluded", "fitted", r"$\eta = 10^{-5}$"], loc="upper right", prop={'size': tick_label_size+1}, ncols=1, frameon=True)
    ppo_dim20_ax.legend([omit_line, ppo_dim20_fit, lrd_line], ["excluded", "fitted", r"$\eta = 10^{-5}$"], loc="upper right", prop={'size': tick_label_size+1}, ncols=1, frameon=True)

    # add labelled subplots
    [fig.add_subplot(ax) for ax in axes]

    # saving
    save_fig(fig, "2_joint_problem_fitness_nips_temporal", "png")
    save_fig(fig, "2_joint_problem_fitness_nips_temporal", "pdf")


def make_drl_supp_plots_nips(client):  # appendix export of ron and ron std from original main plot
    # setup axes
    fig = plt.figure(figsize=(3.67, 3.67))
    grid = GridSpec(2, 2, wspace=0.35, hspace=0.35)

    node_ron_ax = plt.Subplot(fig, grid[0, 0])
    node_ronstd_ax = plt.Subplot(fig, grid[0, 1])

    dim_ron_ax = plt.Subplot(fig, grid[1, 0])
    dim_ronstd_ax = plt.Subplot(fig, grid[1, 1])

    axes = (node_ron_ax, node_ronstd_ax, dim_ron_ax, dim_ronstd_ax)
    node_axes = axes[:2]
    dim_axes = axes[2:]

    # node_ron_ax.axhline(0, color="black", linewidth=plt.rcParams['xtick.major.width'])
    # dim_ron_ax.axhline(0, color="black", linewidth=plt.rcParams['xtick.major.width'])

    # gather instance data
    bind_exp_name = "EVAL_bind"

    ncost_mu, node_scale = get_bind_eval(client, bind_exp_name, "node", f"drl_node_cost.mean")
    ncost_std, scale2 = get_bind_eval(client, bind_exp_name, "node", f"drl_node_cost.std")
    ncost_opt, scale3 = get_bind_eval(client, bind_exp_name, "node", f"opt_cost.mean")
    ncost_rand, scale4 = get_bind_eval(client, bind_exp_name, "node", f"rand_cost.mean")
    assert all([np.allclose(node_scale, scale) for scale in (scale2, scale3, scale4)])

    nron_mu = (ncost_mu - ncost_opt) / (ncost_rand - ncost_opt)
    nron_std = ncost_std / (ncost_rand - ncost_opt)

    dcost_mu, dim_scale = get_bind_eval(client, bind_exp_name, "dim10", f"drl_dim_cost.mean")
    dcost_std, scale2 = get_bind_eval(client, bind_exp_name, "dim10", f"drl_dim_cost.std")
    dcost_opt, scale3 = get_bind_eval(client, bind_exp_name, "dim10", f"opt_cost.mean")
    dcost_rand, scale4 = get_bind_eval(client, bind_exp_name, "dim10", f"rand_cost.mean")
    assert all([np.allclose(dim_scale, scale) for scale in (scale2, scale3, scale4)])

    dron_mu = (dcost_mu - dcost_opt) / (dcost_rand - dcost_opt)
    dron_std = dcost_std / (dcost_rand - dcost_opt)

    node_fit_slc, node_omit_slc = slice(9), slice(8, 10)
    dim_fit_slc, dim_omit_slc = slice(11), None

    node_artcfg = dict(pm="o-", om="o-", fm="--", ps=3, dw=1, fw=0.75, dc="#FFB05F", fc="black", fe=None)  #[pe.Normal(), pe.Stroke(linewidth=0.5, foreground="white")])
    dim_artcfg = dict(pm="o-", om="o-", fm="--", ps=3, dw=1, fw=0.75, dc="#B05FFF", fc="black", fe=None) #[pe.Normal(), pe.Stroke(linewidth=0.5, foreground="white")]) 

    opt_artcfg = dict(pm="-", om="-", fm="-", ps=3, dw=1.5, fw=1.5, dc="lime", fc="lime", fe=None)
    rand_artcfg = dict(pm="-", om="-", fm="-", ps=3, dw=1.5, fw=1.5, dc="black", fc="black", fe=None)
    
    # plot instances
    fit_file_str = "2_problem_fitness_drl_supp_fits_nips.txt"
    clear_fits(fit_file_str)

    plot_instance(node_ron_ax.plot, node_scale, nron_mu, node_fit_slc, node_omit_slc, True, True, "decay", "negative", fit_file_str, "node RON avg (neg decay)", modcfg(node_artcfg, dict(fc="red", fm="-", fw=0.5)))
    plot_instance(node_ron_ax.plot, node_scale, nron_mu, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, "node RON avg (pos grow)", node_artcfg, plot_data=False)
    plot_instance(node_ronstd_ax.semilogy, node_scale, nron_std, node_fit_slc, node_omit_slc, True, True, "decay", "positive", fit_file_str, "node RON std", node_artcfg)

    plot_instance(dim_ron_ax.plot, dim_scale, dron_mu, dim_fit_slc, dim_omit_slc, True, True, "decay", "negative", fit_file_str, "dim(10n) RON avg (neg decay)", dim_artcfg)
    plot_instance(dim_ron_ax.plot, dim_scale, dron_mu, dim_fit_slc, dim_omit_slc, True, True, "grow", "positive", fit_file_str, "dim(10n) RON avg (pos grow)", dim_artcfg, plot_data=False, plot_powerfit=False)
    plot_instance(dim_ronstd_ax.plot, dim_scale, dron_std, dim_fit_slc, dim_omit_slc, True, True, "decay", "negative", fit_file_str, "dim(10n) RON std", dim_artcfg)

    # labelling axes
    label_size = 6
    tick_label_size = 4
    minor_tick_label_size = 4

    node_ron_ax.set_ylabel("Mean normalized test cost", fontsize=label_size)
    dim_ron_ax.set_ylabel("Mean normalized test cost", fontsize=label_size)
    node_ronstd_ax.set_ylabel("SD normalized test cost", fontsize=label_size)
    dim_ronstd_ax.set_ylabel("SD normalized test cost", fontsize=label_size)

    [ax.set_xlabel("Nodes ($n$)", fontsize=label_size) for ax in node_axes]
    [ax.set_xlabel("Spatial dimensions ($d$)", fontsize=label_size) for ax in dim_axes]

    [ax.tick_params(labelsize=tick_label_size) for ax in axes]
    [ax.tick_params(axis="y", which="minor", labelsize=minor_tick_label_size) for ax in axes]

    node_ron_ax.set_ylim(bottom=0)
    dim_ron_ax.set_ylim(bottom=0)

    [ax.set_xticks(list(range(5, 51, 15))) for ax in node_axes]
    [ax.set_xlim([0, 53]) for ax in node_axes]
    [ax.set_xticks([2, 5, 8, 12]) for ax in dim_axes]
    [ax.set_xlim([0, 13]) for ax in dim_axes]

    node_ron_yticks = [0, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3]
    node_ron_ylbls = [Text(0, 0, "0")] + [Text(0, t, f"{t * 1e3:.1f}$ \:\! \\times \:\! 10^{{-3}}$") for t in node_ron_yticks[1:]]
    node_ron_ax.set_yticks(node_ron_yticks, node_ron_ylbls)

    dim_ron_yticks = [0, 0.5e-2, 1e-2, 1.5e-2, 2e-2]
    dim_ron_ylbls = [Text(0, 0, "0")] + [Text(0, t, f"{t * 1e2:.1f}$ \:\! \\times \:\! 10^{{-2}}$") for t in dim_ron_yticks[1:]]
    dim_ron_ax.set_yticks(dim_ron_yticks, dim_ron_ylbls)

    # legend
    node_trend_line, node_break_line, node_red_fit, node_best_fit = node_ron_ax.get_lines()
    node_ron_ax.legend([node_trend_line, node_break_line, node_best_fit, node_red_fit], ["trend inputs", "trend break", "unbounded growth", "bounded growth"], loc="upper left", prop={'size': tick_label_size}, ncols=1, frameon=False)

    node_trend_line, node_break_line, node_fit = node_ronstd_ax.get_lines()
    node_ronstd_ax.legend([node_trend_line, node_break_line, node_fit], ["trend inputs", "trend break", "decay"], loc="upper right", prop={'size': tick_label_size}, ncols=1, frameon=False)

    dim_trend_line, dim_fit = dim_ron_ax.get_lines()
    dim_ron_ax.legend([dim_trend_line, dim_fit], ["trend inputs", "bounded growth"], loc="lower right", prop={'size': tick_label_size}, ncols=1, frameon=False)

    dim_trend_line, dim_fit = dim_ronstd_ax.get_lines()
    dim_ronstd_ax.legend([dim_trend_line, dim_fit], ["trend inputs", "bounded growth"], loc="lower right", prop={'size': tick_label_size}, ncols=1, frameon=False)

    # add labelled subplots
    [forceAspect(ax) for ax in axes]
    [fig.add_subplot(ax) for ax in axes]

    # saving
    save_fig(fig, "2_ppo_supp_fitness_nips", "png")
    save_fig(fig, "2_ppo_supp_fitness_nips", "pdf")



def make_drl_supp_plots(client):  # full high-d 10n dim scaling and 20n dim scaling
    # setup axes
    fig = plt.figure(figsize=(5.5, 3.15))
    grid = GridSpec(2, 3, wspace=0.6, hspace=0.35)

    n10_bound_ax = plt.Subplot(fig, grid[0, 0])
    n10_ron_ax = plt.Subplot(fig, grid[0, 1])
    n10_ronstd_ax = plt.Subplot(fig, grid[0, 2])

    n20_bound_ax = plt.Subplot(fig, grid[1, 0])
    n20_ron_ax = plt.Subplot(fig, grid[1, 1])
    n20_ronstd_ax = plt.Subplot(fig, grid[1, 2])

    axes = (n10_bound_ax, n10_ron_ax, n10_ronstd_ax, n20_bound_ax, n20_ron_ax, n20_ronstd_ax)
    n10_axes = axes[:3]
    n20_axes = axes[3:]

    # node_ron_ax.axhline(0, color="black", linewidth=plt.rcParams['xtick.major.width'])
    # dim_ron_ax.axhline(0, color="black", linewidth=plt.rcParams['xtick.major.width'])

    # gather instance data
    bind_exp_name = "EVAL_bind"

    n10cost_mu, n10_scale = get_bind_eval(client, bind_exp_name, "dim10", f"drl_dim_cost.mean")
    n10cost_std, scale2 = get_bind_eval(client, bind_exp_name, "dim10", f"drl_dim_cost.std")
    n10cost_opt, scale3 = get_bind_eval(client, bind_exp_name, "dim10", f"opt_cost.mean")
    n10cost_rand, scale4 = get_bind_eval(client, bind_exp_name, "dim10", f"rand_cost.mean")
    assert all([np.allclose(n10_scale, scale) for scale in (scale2, scale3, scale4)])

    n10ron_mu = (n10cost_mu - n10cost_opt) / (n10cost_rand - n10cost_opt)
    n10ron_std = n10cost_std / (n10cost_rand - n10cost_opt)

    n20cost_mu, n20_scale = get_bind_eval(client, bind_exp_name, "dim20", f"drl_dim_cost.mean")
    n20cost_std, scale2 = get_bind_eval(client, bind_exp_name, "dim20", f"drl_dim_cost.std")
    n20cost_opt, scale3 = get_bind_eval(client, bind_exp_name, "dim20", f"opt_cost.mean")
    n20cost_rand, scale4 = get_bind_eval(client, bind_exp_name, "dim20", f"rand_cost.mean")
    assert all([np.allclose(n20_scale, scale) for scale in (scale2, scale3, scale4)])

    n20ron_mu = (n20cost_mu - n20cost_opt) / (n20cost_rand - n20cost_opt)
    n20ron_std = n20cost_std / (n20cost_rand - n20cost_opt)

    n10_fit_slc, n10_omit_slc = slice(11), slice(10, 17)
    n20_fit_slc, n20_omit_slc = slice(9), slice(8, 17)

    dim_artcfg = dict(pm="o-", om="o-", fm="--", ps=3, dw=1, fw=0.75, dc="#B05FFF", fc="black", fe=None) #[pe.Normal(), pe.Stroke(linewidth=0.5, foreground="white")]) 
    opt_artcfg = dict(pm=".", om="-", fm="-", ps=2, dw=1, fw=1, dc="red", fc="lime", fe=None)
    rand_artcfg = dict(pm=".", om="-", fm="-", ps=2, dw=1, fw=1, dc="red", fc="black", fe=None)
    
    # plot instances
    fit_file_str = "2_problem_fitness_drl_dim_supp_fits.txt"
    clear_fits(fit_file_str)
    
    dim_linspace = np.linspace(2, 100)
    #
    popt_opt, x_scale_opt, y_scale_opt = plot_instance(n10_bound_ax.plot, n10_scale, n10cost_opt, slice(len(n10_scale)), None, True, True, "grow", "positive", fit_file_str, "dim(10n) opt avg", opt_artcfg, plot_data=False, plot_powerfit=True, powerfit_zorder=2)
    popt_rand, x_scale_rand, y_scale_rand = plot_instance(n10_bound_ax.plot, n10_scale, n10cost_rand, slice(len(n10_scale)), None, True, True, "grow", "positive", fit_file_str, "dim(10n) rand avg", rand_artcfg, plot_data=False, plot_powerfit=True, powerfit_zorder=2)
    
    opt_pfn = build_powerlaw(*popt_opt, sign="positive", x_scale=x_scale_opt, y_scale=y_scale_opt)
    rand_pfn = build_powerlaw(*popt_rand, sign="positive", x_scale=x_scale_rand, y_scale=y_scale_rand)
    n10_fill = n10_bound_ax.fill_between(dim_linspace, opt_pfn(dim_linspace), rand_pfn(dim_linspace), color=dim_artcfg["dc"], linewidth=0.75, zorder=1)
    #
    popt_opt, x_scale_opt, y_scale_opt = plot_instance(n20_bound_ax.plot, n20_scale, n20cost_opt, slice(len(n20_scale)), None, True, True, "grow", "positive", fit_file_str, "dim(20n) opt avg", opt_artcfg, plot_data=False, plot_powerfit=True, powerfit_zorder=2)
    popt_rand, x_scale_rand, y_scale_rand = plot_instance(n20_bound_ax.plot, n20_scale, n20cost_rand, slice(len(n20_scale)), None, True, True, "grow", "positive", fit_file_str, "dim(20n) rand avg", rand_artcfg, plot_data=False, plot_powerfit=True, powerfit_zorder=2)

    opt_pfn = build_powerlaw(*popt_opt, sign="positive", x_scale=x_scale_opt, y_scale=y_scale_opt)
    rand_pfn = build_powerlaw(*popt_rand, sign="positive", x_scale=x_scale_rand, y_scale=y_scale_rand)
    n20_fill = n20_bound_ax.fill_between(dim_linspace, opt_pfn(dim_linspace), rand_pfn(dim_linspace), color=dim_artcfg["dc"], linewidth=0.75, zorder=1)
    #
    #plot_instance(n10_ron_ax.loglog, n10_scale, n10ron_mu, n10_fit_slc, n10_omit_slc, True, True, "grow", "positive", fit_file_str, "dim(10n) RON avg (pos grow)", modcfg(dim_artcfg, dict(fc="red", fm="-", fw=0.5)))
    plot_instance(n10_ron_ax.loglog, n10_scale, n10ron_mu, n10_fit_slc, n10_omit_slc, True, True, "decay", "negative", fit_file_str, "dim(10n) RON avg (neg decay)", dim_artcfg)
    plot_instance(n10_ronstd_ax.loglog, n10_scale, n10ron_std, n10_fit_slc, n10_omit_slc, True, True, "decay", "negative", fit_file_str, "dim(10n) RON std", dim_artcfg)

    #plot_instance(n20_ron_ax.loglog, n20_scale, n20ron_mu, n20_fit_slc, n20_omit_slc, True, True, "grow", "positive", fit_file_str, "dim(20n) RON avg (pos grow)", modcfg(dim_artcfg, dict(fc="red", fm="-", fw=0.5)))
    plot_instance(n20_ron_ax.loglog, n20_scale, n20ron_mu, n20_fit_slc, n20_omit_slc, True, True, "decay", "negative", fit_file_str, "dim(20n) RON avg (neg decay)", dim_artcfg)
    plot_instance(n20_ronstd_ax.loglog, n20_scale, n20ron_std, n20_fit_slc, n20_omit_slc, True, True, "decay", "negative", fit_file_str, "dim(20n) RON std", dim_artcfg)

    # labelling axes
    label_size = 6
    tick_label_size = 4
    minor_tick_label_size = 4

    n10_bound_ax.set_ylabel("Mean tour length (cost)", fontsize=label_size)
    n20_bound_ax.set_ylabel("Mean tour length (cost)", fontsize=label_size)
    n10_ron_ax.set_ylabel("Mean normalized test cost", fontsize=label_size)
    n20_ron_ax.set_ylabel("Mean normalized test cost", fontsize=label_size)
    n10_ronstd_ax.set_ylabel("SD normalized test cost", fontsize=label_size)
    n20_ronstd_ax.set_ylabel("SD normalized test cost", fontsize=label_size)

    [ax.set_xlabel("Spatial dimensions ($x$)", fontsize=label_size) for ax in axes]

    [ax.tick_params(labelsize=tick_label_size) for ax in axes]
    [ax.tick_params(axis="y", which="minor", labelsize=minor_tick_label_size) for ax in axes]

    # [ax.set_xticks([2, 5, 8, 12]) for ax in axes]
    [ax.set_xlim(left=1) for ax in n10_axes[1:] + n20_axes[1:]]
    n10_bound_ax.set_xlim([2, 100])
    n20_bound_ax.set_xlim([2, 100])

    # # legend
    n10_opt_line, n10_rand_line = n10_bound_ax.get_lines()
    n10_patch = mpatches.Patch(color=n10_fill.get_facecolor(), linewidth=0)
    n10_bound_ax.legend([n10_rand_line, n10_opt_line, n10_patch], ["random (sublinear)", "optimal (sublinear)", "achievable"], loc="lower right", prop={'size': tick_label_size}, ncols=1, frameon=False)

    n20_opt_line, n20_rand_line = n20_bound_ax.get_lines()
    n20_patch = mpatches.Patch(color=n20_fill.get_facecolor(), linewidth=0)
    n20_bound_ax.legend([n20_rand_line, n20_opt_line, n20_patch], ["random (sublinear)", "optimal (sublinear)", "achievable"], loc="lower right", prop={'size': tick_label_size}, ncols=1, frameon=False)

    n10_trend_line, n10_trend_break, n10_fit = n10_ron_ax.get_lines()
    n10_ron_ax.legend([n10_trend_line, n10_trend_break, n10_fit], ["trend inputs", "trend break", "bounded growth"], loc="lower right", prop={'size': tick_label_size}, ncols=1, frameon=False)

    n20_trend_line, n20_trend_break, n20_fit = n20_ron_ax.get_lines()
    n20_ron_ax.legend([n20_trend_line, n20_trend_break, n20_fit], ["trend inputs", "trend break", "bounded growth"], loc="lower right", prop={'size': tick_label_size}, ncols=1, frameon=False)

    n10_trend_line, n10_trend_break, n10_fit = n10_ronstd_ax.get_lines()
    n10_ronstd_ax.legend([n10_trend_line, n10_trend_break, n10_fit], ["trend inputs", "trend break", "bounded growth"], loc="lower right", prop={'size': tick_label_size}, ncols=1, frameon=False)

    n20_trend_line, n20_trend_break, n20_fit = n20_ronstd_ax.get_lines()
    n20_ronstd_ax.legend([n20_trend_line, n20_trend_break, n20_fit], ["trend inputs", "trend break", "bounded growth"], loc="lower right", prop={'size': tick_label_size}, ncols=1, frameon=False)

    [ax.text(0.05, 0.95, "$n=10$", fontsize=label_size, ha='left', va='top', transform=ax.transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round')) for ax in n10_axes]
    [ax.text(0.05, 0.95, "$n=20$", fontsize=label_size, ha='left', va='top', transform=ax.transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round')) for ax in n20_axes]

    # add labelled subplots
    [fig.add_subplot(ax) for ax in axes]

    # saving
    save_fig(fig, "2_ppo_dim_supp_fitness", FORMAT)


def make_il_supp_plots(client):  # node scaling without repeating bounds (which are the same as main drl fig)
    # setup axes
    fig = plt.figure(figsize=(5.0, 2.1))
    grid = GridSpec(1, 2, wspace=0.4, hspace=0.35)

    ron_ax = plt.Subplot(fig, grid[0, 0])
    ronstd_ax = plt.Subplot(fig, grid[0, 1])
 
    axes = (ron_ax, ronstd_ax)

    # ron_ax.axhline(0, color="black", linewidth=plt.rcParams['xtick.major.width'])

    # gather instance data
    bind_exp_name = "EVAL_bind"

    ncost_mu, node_scale = get_bind_eval(client, bind_exp_name, "node", f"il_node_cost.mean")
    ncost_std, scale2 = get_bind_eval(client, bind_exp_name, "node", f"il_node_cost.std")
    ncost_opt, scale3 = get_bind_eval(client, bind_exp_name, "node", f"opt_cost.mean")
    ncost_rand, scale4 = get_bind_eval(client, bind_exp_name, "node", f"rand_cost.mean")
    assert all([np.allclose(node_scale, scale) for scale in (scale2, scale3, scale4)])

    nron_mu = (ncost_mu - ncost_opt) / (ncost_rand - ncost_opt)
    nron_std = ncost_std / (ncost_rand - ncost_opt)

    fit_slc, omit_slc = slice(6), slice(5, 10)

    node_artcfg = dict(pm="o-", om="o-", fm="--", ps=3, dw=1, fw=0.75, dc="#FFB05F", fc="black", fe=None)

    # plot instances
    fit_file_str = "2_problem_fitness_bc_supp_fits.txt"
    clear_fits(fit_file_str)

    plot_instance(ron_ax.plot, node_scale, nron_mu, fit_slc, omit_slc, True, True, "decay", "negative", fit_file_str, "node RON avg (neg decay)", modcfg(node_artcfg, dict(fc="red", fm="-", fw=0.5)))
    plot_instance(ron_ax.plot, node_scale, nron_mu, fit_slc, omit_slc, True, True, "grow", "positive", fit_file_str, "node RON avg (pos grow)", node_artcfg, plot_data=False)
    plot_instance(ronstd_ax.semilogy, node_scale, nron_std, fit_slc, omit_slc, True, True, "decay", "positive", fit_file_str, "node RON std", node_artcfg)

    # labelling axes
    label_size = 6
    tick_label_size = 4
    minor_tick_label_size = 4

    ron_ax.set_ylabel("Mean normalized test cost", fontsize=label_size)
    ronstd_ax.set_ylabel("SD normalized test cost", fontsize=label_size)

    [ax.set_xlabel("Nodes ($n$)", fontsize=label_size) for ax in axes]

    [ax.tick_params(labelsize=tick_label_size) for ax in axes]
    [ax.tick_params(axis="y", which="minor", labelsize=minor_tick_label_size) for ax in axes]

    ron_ax.set_ylim(bottom=0)

    [ax.set_xticks(list(range(5, 51, 15))) for ax in axes]
    [ax.set_xlim([0, 53]) for ax in axes]

    # node_ron_yticks = [t * 1e-4 for t in range(0, 36, 5)]
    # node_ron_ylbls = [Text(0, 0, "0")] + [Text(0, t, f"{t * 1e3:.1f}$ \:\! \\times \:\! 10^{{-3}}$") for t in node_ron_yticks[1:]]
    # ron_ax.set_yticks(node_ron_yticks, node_ron_ylbls)
    ron_ax.ticklabel_format(axis="y", style="sci", scilimits=(-1, 1))
    ron_ax.yaxis.get_offset_text().set_size(label_size)

    # legend
    node_trend_line, node_break_line, node_red_fit, node_best_fit = ron_ax.get_lines()
    ron_ax.legend([node_trend_line, node_break_line, node_best_fit, node_red_fit], ["trend inputs", "trend break", "unbounded growth", "bounded growth"], loc="upper left", prop={'size': tick_label_size}, ncols=1, frameon=False)

    node_trend_line, node_break_line, node_fit = ronstd_ax.get_lines()
    ronstd_ax.legend([node_trend_line, node_break_line, node_fit], ["trend inputs", "trend break", "decay"], loc="upper right", prop={'size': tick_label_size}, ncols=1, frameon=False)

    # add labelled subplots
    [fig.add_subplot(ax) for ax in axes]

    # saving
    save_fig(fig, "2_bc_supp_fitness", FORMAT)


def make_decomp_main_plots(client):  # decomposition of suboptimality numerator and achievable performance span denominator for mean ron
    # setup axes
    fig = plt.figure(figsize=(5.5, 1.21))
    outer_grid = GridSpec(1, 2, wspace=0.25)  #, width_ratios=(50, 1))
    subopt_grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid[0], wspace=0.1)
    span_grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid[1], wspace=0.1)

    node_subopt_ax = plt.Subplot(fig, subopt_grid[0, 0])
    d10_subopt_ax = plt.Subplot(fig, subopt_grid[0, 1], sharey=node_subopt_ax)

    node_span_ax = plt.Subplot(fig, span_grid[0, 0])
    d10_span_ax = plt.Subplot(fig, span_grid[0, 1], sharey=node_span_ax)

    d10_subopt_ax.tick_params("y", labelleft=False)
    d10_span_ax.tick_params("y", labelleft=False)

    axes = (node_subopt_ax, d10_subopt_ax, node_span_ax, d10_span_ax)
    subopt_axes = (node_subopt_ax, d10_subopt_ax)
    span_axes = (node_span_ax, d10_span_ax)
    node_axes = (node_subopt_ax, node_span_ax)
    dim_axes = (d10_subopt_ax, d10_span_ax)

    node_subopt_ax.axhline(0, color="black", linewidth=plt.rcParams['xtick.major.width'])
    d10_subopt_ax.axhline(0, color="black", linewidth=plt.rcParams['xtick.major.width'])

    # gather instance data
    bind_exp_name = "EVAL_bind"

    ncost_ppo, node_scale = get_bind_eval(client, bind_exp_name, "node", f"drl_node_cost.mean")
    ncost_opt, scale3 = get_bind_eval(client, bind_exp_name, "node", f"opt_cost.mean")
    ncost_rand, scale4 = get_bind_eval(client, bind_exp_name, "node", f"rand_cost.mean")
    assert all([np.allclose(node_scale, scale) for scale in (scale3, scale4)])

    d10cost_ppo, dim10_scale = get_bind_eval(client, bind_exp_name, "dim10", f"drl_dim_cost.mean")
    d10cost_opt, scale5 = get_bind_eval(client, bind_exp_name, "dim10", f"opt_cost.mean")
    d10cost_rand, scale6 = get_bind_eval(client, bind_exp_name, "dim10", f"rand_cost.mean")
    assert all([np.allclose(dim10_scale, scale) for scale in (scale5, scale6)])

    ppo_node_subopt = ncost_ppo - ncost_opt
    ppo_d10_subopt = d10cost_ppo - d10cost_opt

    node_span = ncost_rand - ncost_opt
    d10_span = d10cost_rand - d10cost_opt

    ppo_node_fit_slc, ppo_node_omit_slc = slice(9), None
    ppo_d10_fit_slc, ppo_d10_omit_slc = slice(11), None 

    node_artcfg = dict(pm="o-", om="o-", fm="--", ps=3, dw=1, fw=0.75, dc="#FFB05F", fc="black", fe=None)  #[pe.Normal(), pe.Stroke(linewidth=0.5, foreground="white")])
    d10_artcfg = dict(pm="o-", om="o-", fm="--", ps=3, dw=1, fw=0.75, dc="#B05FFF", fc="black", fe=None) #[pe.Normal(), pe.Stroke(linewidth=0.5, foreground="white")]) 

    # plot instances
    fit_file_str = "2_problem_fitness_decomp_main_fits.txt"
    clear_fits(fit_file_str)

    # (first just get white curve for legend reasons)
    plot_instance(node_subopt_ax.plot, node_scale, ppo_node_subopt, ppo_node_fit_slc, ppo_node_omit_slc, True, True, "grow", "positive", fit_file_str, "WHITE CURVE", modcfg(node_artcfg, dict(dc="white", fc="white")), powerfit=False)

    plot_instance(node_subopt_ax.plot, node_scale, ppo_node_subopt, ppo_node_fit_slc, ppo_node_omit_slc, True, True, "grow", "positive", fit_file_str, "subopt ppo node", node_artcfg)
    plot_instance(d10_subopt_ax.plot, dim10_scale, ppo_d10_subopt, ppo_d10_fit_slc, ppo_d10_omit_slc, True, True, "decay", "negative", fit_file_str, "subopt ppo dim10", d10_artcfg)
    plot_instance(node_span_ax.plot, node_scale, node_span, ppo_node_fit_slc, None, True, True, "grow", "positive", fit_file_str, "span node", node_artcfg)
    plot_instance(d10_span_ax.plot, dim10_scale, d10_span, ppo_d10_fit_slc, None, True, True, "grow", "positive", fit_file_str, "span dim10", d10_artcfg, powerfit=False)

    # labelling axes
    label_size = 6
    tick_label_size = 4
    minor_tick_label_size = 4

    node_subopt_ax.set_ylabel("Suboptimality gap", fontsize=label_size)
    node_span_ax.set_ylabel("Achievable span", fontsize=label_size)

    [ax.set_xlabel("Nodes ($n$)", fontsize=label_size) for ax in node_axes]
    [ax.set_xlabel("Spatial dimensions ($x$)", fontsize=label_size) for ax in dim_axes]

    [ax.tick_params(labelsize=tick_label_size) for ax in axes]
    [ax.tick_params(axis="y", which="minor", labelsize=minor_tick_label_size) for ax in axes]

    [ax.set_xticks(list(range(5, 46, 10))) for ax in node_axes]
    [ax.set_xticks([2, 5, 8, 12]) for ax in (d10_subopt_ax, d10_span_ax)]

    [ax.set_ylim(bottom=0) for ax in axes]
    node_subopt_ax.set_ylim(bottom=-0.002)

    [ax.set_xlim([0, 48]) for ax in node_axes]
    [ax.set_xlim([0, 13]) for ax in dim_axes]

    # legend
    _, white, trend_inputs, trend = node_subopt_ax.get_lines()
    node_subopt_ax.legend([trend, trend_inputs], ["unbounded growth", "trend inputs"], loc="upper left", prop={'size': tick_label_size}, ncols=1, frameon=False)

    _, trend_inputs, trend = d10_subopt_ax.get_lines()
    d10_subopt_ax.legend([trend, trend_inputs], ["bounded growth", "trend inputs"], loc="upper left", prop={'size': tick_label_size}, ncols=1, frameon=False)

    trend_inputs, trend = node_span_ax.get_lines()
    node_span_ax.legend([trend, trend_inputs], ["near-linear growth", "trend inputs"], loc="upper left", prop={'size': tick_label_size}, ncols=1, frameon=False)

    # add labelled subplots
    [fig.add_subplot(ax) for ax in axes]

    # saving
    save_fig(fig, "2_fitness_decomp_main", FORMAT)


def make_decomp_supp_plots(client):  # ditto for bc node scaling and ppo n=20 dim scaling (also shows last trend breaking points of ppo suboptimality)
    # setup axes
    fig = plt.figure(figsize=(3.2, 3.2))
    grid = GridSpec(2, 2, wspace=0.5, hspace=0.5)

    ppo_subopt_ax = plt.Subplot(fig, grid[0, 0])
    bc_subopt_ax = plt.Subplot(fig, grid[0, 1], sharey=ppo_subopt_ax)
    d20_subopt_ax = plt.Subplot(fig, grid[1, 0])
    d20_span_ax = plt.Subplot(fig, grid[1, 1])

    axes = (ppo_subopt_ax, bc_subopt_ax, d20_subopt_ax, d20_span_ax)
    subopt_axes = (ppo_subopt_ax, bc_subopt_ax, d20_subopt_ax)
    node_axes = (ppo_subopt_ax, bc_subopt_ax)
    dim_axes = (d20_subopt_ax, d20_span_ax)

    ppo_subopt_ax.axhline(0, color="black", linewidth=plt.rcParams['xtick.major.width'])
    bc_subopt_ax.axhline(0, color="black", linewidth=plt.rcParams['xtick.major.width'])

    # gather instance data
    bind_exp_name = "EVAL_bind"

    ncost_ppo, node_scale = get_bind_eval(client, bind_exp_name, "node", f"drl_node_cost.mean")
    ncost_bc, scale_2 = get_bind_eval(client, bind_exp_name, "node", f"il_node_cost.mean")
    ncost_opt, scale3 = get_bind_eval(client, bind_exp_name, "node", f"opt_cost.mean")
    assert all([np.allclose(node_scale, scale) for scale in (scale3, scale_2)])

    d20cost_ppo, dim20_scale = get_bind_eval(client, bind_exp_name, "dim20", f"drl_dim_cost.mean")
    d20cost_opt, scale7 = get_bind_eval(client, bind_exp_name, "dim20", f"opt_cost.mean")
    d20cost_rand, scale8 = get_bind_eval(client, bind_exp_name, "dim20", f"rand_cost.mean")
    assert all([np.allclose(dim20_scale, scale) for scale in (scale7, scale8)])

    ppo_node_subopt = ncost_ppo - ncost_opt
    bc_node_subopt = ncost_bc - ncost_opt
    ppo_d20_subopt = d20cost_ppo - d20cost_opt

    d20_span = d20cost_rand - d20cost_opt

    ppo_node_fit_slc, ppo_node_omit_slc = slice(9), slice(8, 10)
    bc_node_fit_slc, bc_node_omit_slc = slice(6), slice(5, 10)
    ppo_d20_fit_slc, ppo_d20_omit_slc = slice(9), None

    node_artcfg = dict(pm="o-", om="o-", fm="--", ps=3, dw=1, fw=0.75, dc="#FFB05F", fc="black", fe=None)  #[pe.Normal(), pe.Stroke(linewidth=0.5, foreground="white")])
    d10_artcfg = dict(pm="o-", om="o-", fm="--", ps=3, dw=1, fw=0.75, dc="#B05FFF", fc="black", fe=None) #[pe.Normal(), pe.Stroke(linewidth=0.5, foreground="white")]) 
    d20_artcfg = d10_artcfg  #modcfg(d10_artcfg, dict(pm="D-", ps=2.5))

    # plot instances
    fit_file_str = "2_problem_fitness_decomp_supp_fits.txt"
    clear_fits(fit_file_str)

    plot_instance(ppo_subopt_ax.plot, node_scale, ppo_node_subopt, ppo_node_fit_slc, ppo_node_omit_slc, True, True, "grow", "positive", fit_file_str, "subopt ppo node", node_artcfg)
    plot_instance(bc_subopt_ax.plot, node_scale, bc_node_subopt, bc_node_fit_slc, bc_node_omit_slc, True, True, "grow", "positive", fit_file_str, "subopt bc node", node_artcfg)
    
    plot_instance(d20_subopt_ax.plot, dim20_scale, ppo_d20_subopt, ppo_d20_fit_slc, ppo_d20_omit_slc, True, True, "grow", "positive", fit_file_str, "subopt ppo dim20 (unbounded)", modcfg(d20_artcfg, dict(fc="red", fm="-", fw=0.5)), plot_data=False)
    plot_instance(d20_subopt_ax.plot, dim20_scale, ppo_d20_subopt, ppo_d20_fit_slc, ppo_d20_omit_slc, True, True, "decay", "negative", fit_file_str, "subopt ppo dim20 (bounded)", d20_artcfg)
    
    plot_instance(d20_span_ax.plot, dim20_scale, d20_span, ppo_d20_fit_slc, None, True, True, "grow", "positive", fit_file_str, "span dim20", d20_artcfg, powerfit=False)

    # labelling axes
    label_size = 6
    tick_label_size = 4
    minor_tick_label_size = 4

    [ax.set_ylabel("Suboptimality gap", fontsize=label_size) for ax in subopt_axes]
    d20_span_ax.set_ylabel("Achievable span", fontsize=label_size)

    [ax.set_xlabel("Nodes ($n$)", fontsize=label_size) for ax in node_axes]
    [ax.set_xlabel("Spatial dimensions ($x$)", fontsize=label_size) for ax in dim_axes]

    [ax.tick_params(labelsize=tick_label_size) for ax in axes]
    [ax.tick_params(axis="y", which="minor", labelsize=minor_tick_label_size) for ax in axes]

    [ax.set_xticks(list(range(5, 51, 15))) for ax in node_axes]
    [ax.set_xticks([2, 4, 6, 8, 10]) for ax in (d20_subopt_ax, d20_span_ax)]

    [ax.set_ylim(bottom=0) for ax in axes]
    bc_subopt_ax.set_ylim(bottom=-0.002)
    d20_span_ax.set_ylim(top=19)
  
    # legend
    _, trend_inputs, trend_break, trend = ppo_subopt_ax.get_lines()
    ppo_subopt_ax.legend([trend_inputs, trend_break, trend], ["trend inputs", "trend break", "unbounded growth"], loc="upper left", prop={'size': tick_label_size}, ncols=1, frameon=False)

    _, trend_inputs, trend_break, trend = bc_subopt_ax.get_lines()
    bc_subopt_ax.legend([trend_inputs, trend_break, trend], ["trend inputs", "trend break", "unbounded growth"], loc="upper left", prop={'size': tick_label_size}, ncols=1, frameon=False)

    unbounded_trend, trend_inputs, bounded_trend = d20_subopt_ax.get_lines()
    d20_subopt_ax.legend([trend_inputs, bounded_trend, unbounded_trend], ["trend inputs", "bounded growth", "unbounded growth"], loc="lower right", prop={'size': tick_label_size}, ncols=1, frameon=False)

    # d20_span_ax.legend([trend_inputs], ["evaluations"], loc="lower right", prop={'size': tick_label_size}, ncols=1, frameon=False)

    [ax.text(0.95, 0.07, "PPO", fontsize=label_size, ha='right', va='bottom', transform=ax.transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round')) for ax in (ppo_subopt_ax,)]
    [ax.text(0.95, 0.07, "BC", fontsize=label_size, ha='right', va='bottom', transform=ax.transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round')) for ax in (bc_subopt_ax,)]
    [ax.text(0.05, 0.95, "$n=20$", fontsize=label_size, ha='left', va='top', transform=ax.transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round')) for ax in (d20_subopt_ax, d20_span_ax)]

    # add labelled subplots
    [fig.add_subplot(ax) for ax in axes]

    # saving
    save_fig(fig, "2_fitness_decomp_supp", FORMAT)


def make_raw_std_supp_plots(client):  # raw stds of different algos, showing how near-optimal solvers approach 0 while nearer-to-random diverges
    # setup axes
    fig = plt.figure(figsize=(3.67, 3.67))
    grid = GridSpec(2, 2, wspace=0.35, hspace=0.35)

    node_grow_ax = plt.Subplot(fig, grid[0, 0])
    node_decay_ax = plt.Subplot(fig, grid[1, 0])
    d10_ax = plt.Subplot(fig, grid[0, 1])
    d20_ax = plt.Subplot(fig, grid[1, 1])

    axes = (node_grow_ax, node_decay_ax, d10_ax, d20_ax)
    node_axes = (node_grow_ax, node_decay_ax)
    dim_axes = (d10_ax, d20_ax)

    # gather instance data
    bind_exp_name = "EVAL_bind"

    ncost_std_ppo, node_scale = get_bind_eval(client, bind_exp_name, "node", f"drl_node_cost.std")
    ncost_std_bc, scale2 = get_bind_eval(client, bind_exp_name, "node", f"il_node_cost.std")
    ncost_std_2opt, scale3 = get_bind_eval(client, bind_exp_name, "node", f"_2opt_cost.std")
    ncost_std_2exc, scale4 = get_bind_eval(client, bind_exp_name, "node", f"_2swap_cost.std")
    ncost_std_opt, scale5 = get_bind_eval(client, bind_exp_name, "node", f"opt_cost.std")
    ncost_std_rand, scale6 = get_bind_eval(client, bind_exp_name, "node", f"rand_cost.std")
    assert all([np.allclose(node_scale, scale) for scale in (scale2, scale3, scale4, scale5, scale6)])

    d10cost_std_ppo, dim10_scale = get_bind_eval(client, bind_exp_name, "dim10", f"drl_dim_cost.std")
    d10cost_std_2opt, scale7 = get_bind_eval(client, bind_exp_name, "dim10", f"_2opt_cost.std")
    d10cost_std_2exc, scale8 = get_bind_eval(client, bind_exp_name, "dim10", f"_2swap_cost.std")
    d10cost_std_opt, scale9 = get_bind_eval(client, bind_exp_name, "dim10", f"opt_cost.std")
    d10cost_std_rand, scale10 = get_bind_eval(client, bind_exp_name, "dim10", f"rand_cost.std")
    assert all([np.allclose(dim10_scale, scale) for scale in (scale7, scale8, scale9, scale10)])
    # 2D local search experiments (2opt and 2exchange) both have corrupted, inflated results in EVAL_bind for 10-node version, where 20-node version does not, and I'm not sure why
    # code below substitutes values from earlier evals done on same proxy opt problems, where issue does not show up
    _2OPT_2D_COST_STD = 0.3463
    _2SWAP_2D_COST_STD = 0.3699
    d10cost_std_2opt[0] = _2OPT_2D_COST_STD
    d10cost_std_2exc[0] = _2SWAP_2D_COST_STD

    d20cost_std_ppo, dim20_scale = get_bind_eval(client, bind_exp_name, "dim20", f"drl_dim_cost.std")
    d20cost_std_2opt, scale11 = get_bind_eval(client, bind_exp_name, "dim20", f"_2opt_cost.std")
    d20cost_std_2exc, scale12 = get_bind_eval(client, bind_exp_name, "dim20", f"_2swap_cost.std")
    d20cost_std_opt, scale13 = get_bind_eval(client, bind_exp_name, "dim20", f"opt_cost.std")
    d20cost_std_rand, scale14 = get_bind_eval(client, bind_exp_name, "dim20", f"rand_cost.std")
    assert all([np.allclose(dim20_scale, scale) for scale in (scale11, scale12, scale13, scale14)])

    ppo_node_fit_slc, ppo_node_omit_slc = slice(9), slice(8, 10)
    bc_node_fit_slc, bc_node_omit_slc = slice(6), slice(5, 10)
    ppo_d10_fit_slc, ppo_d10_omit_slc = slice(11), slice(10, 17) 
    ppo_d20_fit_slc, ppo_d20_omit_slc = slice(9), slice(8, 17)

    node_artcfg = dict(pm="-", om=":", fm="--", ps=3, dw=1, fw=0.75, dc="#FFB05F", fc="black", fe=None)  #[pe.Normal(), pe.Stroke(linewidth=0.5, foreground="white")])
    d10_artcfg = dict(pm="-", om=":", fm="--", ps=3, dw=1, fw=0.75, dc="#B05FFF", fc="black", fe=None) #[pe.Normal(), pe.Stroke(linewidth=0.5, foreground="white")]) 
    d20_artcfg = d10_artcfg  #modcfg(d10_artcfg, dict(pm="D-", ps=2.5))

    rand_col = "black"
    opt_col = "lime"
    _2exc_col = "#2BD4B4"
    _2opt_col = "#B42BD4"
    bc_col = "#FF605F"  #"#2728D6"
    ppo_col = "#FFB05F" #"#27D6D5"

    # plot instances
    fit_file_str = "2_problem_fitness_rawstd_supp_fits.txt"
    gen_powerfits = False
    if gen_powerfits: clear_fits(fit_file_str)

    plot_instance(node_grow_ax.plot, node_scale, ncost_std_rand, slice(len(node_scale)), None, True, False, "grow", "positive", fit_file_str, "rand node", modcfg(node_artcfg, dict(dc=rand_col)), powerfit=gen_powerfits)
    plot_instance(node_grow_ax.plot, node_scale, ncost_std_2exc, slice(len(node_scale)), None, True, True, "grow", "positive", fit_file_str, "2exc node", modcfg(node_artcfg, dict(dc=_2exc_col)), powerfit=False)

    plot_instance(node_decay_ax.plot, node_scale, ncost_std_2opt, slice(len(node_scale)), None, True, True, "decay", "positive", fit_file_str, "2opt node", modcfg(node_artcfg, dict(dc=_2opt_col)), powerfit=False)
    plot_instance(node_decay_ax.plot, node_scale, ncost_std_bc, bc_node_fit_slc, bc_node_omit_slc, True, True, "decay", "positive", fit_file_str, "bc node", modcfg(node_artcfg, dict(dc=bc_col)), powerfit=gen_powerfits) # modcfg(node_artcfg, dict(dc="white", fc="white")), powerfit_zorder=0)
    plot_instance(node_decay_ax.plot, node_scale, ncost_std_ppo, ppo_node_fit_slc, ppo_node_omit_slc, True, True, "decay", "positive", fit_file_str, "ppo node", modcfg(node_artcfg, dict(dc=ppo_col)), powerfit=gen_powerfits)
    plot_instance(node_decay_ax.plot, node_scale, ncost_std_opt, slice(len(node_scale)), None, True, True, "decay", "positive", fit_file_str, "opt node", modcfg(node_artcfg, dict(dc=opt_col)), powerfit=gen_powerfits)

    plot_instance(d10_ax.semilogx, dim10_scale, d10cost_std_rand, slice(len(dim10_scale)), None, True, True, "decay", "negative", fit_file_str, "rand dim10", modcfg(d10_artcfg, dict(dc=rand_col)), powerfit=False)
    plot_instance(d10_ax.semilogx, dim10_scale, d10cost_std_2exc, slice(len(dim10_scale)), None, True, True, "decay", "negative", fit_file_str, "2exc dim10", modcfg(d10_artcfg, dict(dc=_2exc_col)), powerfit=gen_powerfits)

    plot_instance(d10_ax.semilogx, dim10_scale, d10cost_std_2opt, slice(len(dim10_scale)), None, True, True, "decay", "negative", fit_file_str, "2opt dim10", modcfg(d10_artcfg, dict(dc=_2opt_col)), powerfit=gen_powerfits)
    plot_instance(d10_ax.semilogx, dim10_scale, d10cost_std_ppo, ppo_d10_fit_slc, ppo_d10_omit_slc, True, True, "decay", "negative", fit_file_str, "ppo dim10", modcfg(d10_artcfg, dict(dc=ppo_col)), powerfit=gen_powerfits, omit_zorder=3)
    plot_instance(d10_ax.semilogx, dim10_scale, d10cost_std_opt, slice(len(dim10_scale)), None, True, True, "decay", "negative", fit_file_str, "opt dim10", modcfg(d10_artcfg, dict(dc=opt_col)), powerfit=gen_powerfits)

    plot_instance(d20_ax.semilogx, dim20_scale, d20cost_std_rand, slice(len(dim20_scale)), None, True, True, "decay", "negative", fit_file_str, "rand dim20", modcfg(d20_artcfg, dict(dc=rand_col)), powerfit=False)
    plot_instance(d20_ax.semilogx, dim20_scale, d20cost_std_2exc, slice(len(dim20_scale)), None, True, True, "decay", "negative", fit_file_str, "2exc dim20", modcfg(d20_artcfg, dict(dc=_2exc_col)), powerfit=gen_powerfits)

    plot_instance(d20_ax.semilogx, dim20_scale, d20cost_std_2opt, slice(len(dim20_scale)), None, True, True, "decay", "negative", fit_file_str, "2opt dim20", modcfg(d20_artcfg, dict(dc=_2opt_col)), powerfit=gen_powerfits)
    plot_instance(d20_ax.semilogx, dim20_scale, d20cost_std_ppo, ppo_d20_fit_slc, ppo_d20_omit_slc, True, True, "decay", "negative", fit_file_str, "ppo dim20", modcfg(d20_artcfg, dict(dc=ppo_col)), powerfit=gen_powerfits, omit_zorder=3)
    plot_instance(d20_ax.semilogx, dim20_scale, d20cost_std_opt, slice(len(dim20_scale)), None, True, True, "decay", "negative", fit_file_str, "opt dim20", modcfg(d20_artcfg, dict(dc=opt_col)), powerfit=gen_powerfits)

    # labelling axes
    label_size = 6
    tick_label_size = 4
    minor_tick_label_size = 4

    [ax.set_ylabel("SD test cost", fontsize=label_size) for ax in axes]

    [ax.set_xlabel("Nodes ($n$)", fontsize=label_size) for ax in node_axes]
    [ax.set_xlabel("Spatial dimensions ($d$)", fontsize=label_size) for ax in dim_axes]

    [ax.tick_params(labelsize=tick_label_size) for ax in axes]
    [ax.tick_params(axis="y", which="minor", labelsize=minor_tick_label_size) for ax in axes]

    [ax.set_xticks(list(range(5, 51, 15))) for ax in node_axes]

    [ax.set_xlim([5, 50]) for ax in node_axes]
    [ax.set_xlim([2, 100]) for ax in dim_axes]

    node_grow_ax.set_ylim(top=2)
    # d10_ax.set_ylim(top=0.9)

    # legend
    if not gen_powerfits:
        rand, _2exc = node_grow_ax.get_lines()
        _2opt, bc, bc_break, ppo, ppo_break, opt = node_decay_ax.get_lines()
        fig.legend([rand, opt, _2exc, _2opt, bc, ppo, bc_break, ppo_break], ["Random", "Optimal", "2-exchange", "2-opt", "SFT", "RL", "SFT trend break", "RL trend break"], loc="upper center", prop={'size': label_size}, bbox_to_anchor=(0.5, 0), ncols=4, frameon=False)

    d10_ax.text(0.95, 0.05, "$n=10$", fontsize=label_size, ha='right', va='bottom', transform=d10_ax.transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))
    d20_ax.text(0.95, 0.05, "$n=20$", fontsize=label_size, ha='right', va='bottom', transform=d20_ax.transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))

    # add labelled subplots
    [forceAspect(ax) for ax in axes]
    [fig.add_subplot(ax) for ax in axes]

    # saving
    save_fig(fig, "2_fitness_rawstd_supp", "pdf")
    save_fig(fig, "2_fitness_rawstd_supp", "png")


def make_isolated_rawstd_fits(client):  # close up of trending (and decreasing) raw SD over node scale (PPO, BC, optimal)
    # setup axes
    fig = plt.figure(figsize=(3.67, 3.67))
    grid = GridSpec(2, 2, wspace=0.45, hspace=0.35)

    ppo_ax = plt.Subplot(fig, grid[0, 0])
    bc_ax = plt.Subplot(fig, grid[0, 1], sharey=ppo_ax)
    opt_ax = plt.Subplot(fig, grid[1, 0], sharey=ppo_ax)
    optnorm_ax = plt.Subplot(fig, grid[1, 1])

    axes = (ppo_ax, bc_ax, opt_ax, optnorm_ax)

    # gather instance data
    bind_exp_name = "EVAL_bind"

    ncost_std_ppo, node_scale = get_bind_eval(client, bind_exp_name, "node", f"drl_node_cost.std")
    ncost_std_bc, scale2 = get_bind_eval(client, bind_exp_name, "node", f"il_node_cost.std")
    ncost_std_opt, scale5 = get_bind_eval(client, bind_exp_name, "node", f"opt_cost.std")
    ncost_opt, scale6 = get_bind_eval(client, bind_exp_name, "node", f"opt_cost.mean")
    ncost_rand, scale7 = get_bind_eval(client, bind_exp_name, "node", f"rand_cost.mean")
    assert all([np.allclose(node_scale, scale) for scale in (scale2, scale5, scale6, scale7)])

    normstd_opt = ncost_std_opt / (ncost_rand - ncost_opt)

    ppo_node_fit_slc, ppo_node_omit_slc = slice(9), slice(8, 10)
    bc_node_fit_slc, bc_node_omit_slc = slice(6), slice(5, 10)

    node_artcfg = dict(pm="o-", om="o-", fm="--", ps=3, dw=1, fw=0.75, dc="#FFB05F", fc="black", fe=None)  #[pe.Normal(), pe.Stroke(linewidth=0.5, foreground="white")])

    opt_col = "lime"
    bc_col = "#2728D6"
    ppo_col = "#27D6D5"

    # plot instances
    fit_file_str = "2_problem_fitness_rawstd_isolated_fits.txt"
    clear_fits(fit_file_str)

    plot_instance(bc_ax.semilogy, node_scale, ncost_std_bc, bc_node_fit_slc, bc_node_omit_slc, True, True, "decay", "positive", fit_file_str, "bc node", modcfg(node_artcfg, dict(dc=bc_col)))
    plot_instance(ppo_ax.semilogy, node_scale, ncost_std_ppo, ppo_node_fit_slc, ppo_node_omit_slc, True, True, "decay", "positive", fit_file_str, "ppo node", modcfg(node_artcfg, dict(dc=ppo_col)))
    
    plot_instance(opt_ax.semilogy, node_scale, ncost_std_opt, slice(len(node_scale)), None, True, True, "decay", "positive", fit_file_str, "opt node", modcfg(node_artcfg, dict(dc=opt_col)))
    plot_instance(optnorm_ax.semilogy, node_scale, normstd_opt, slice(len(node_scale)), None, True, True, "decay", "positive", fit_file_str, "normalized opt node", modcfg(node_artcfg, dict(dc=opt_col)))

    # labelling axes
    label_size = 6
    tick_label_size = 4
    minor_tick_label_size = 4

    [ax.set_ylabel("SD test cost", fontsize=label_size) for ax in (ppo_ax, bc_ax)]
    opt_ax.set_ylabel("SD cost", fontsize=label_size)
    optnorm_ax.set_ylabel("SD normalized cost", fontsize=label_size)

    [ax.set_xlabel("Nodes ($n$)", fontsize=label_size) for ax in axes]

    [ax.tick_params(labelsize=tick_label_size) for ax in axes]
    [ax.tick_params(axis="y", which="minor", labelsize=minor_tick_label_size) for ax in axes]

    [ax.set_xticks(list(range(5, 51, 15))) for ax in axes]

    # [ax.set_xlim([5, 50]) for ax in node_axes]

    # legend
    trend_inputs, trend_break, trend = ppo_ax.get_lines()
    ppo_ax.legend([trend_inputs, trend_break, trend], ["trend inputs", "trend break", "decay"], loc="upper right", prop={'size': tick_label_size}, ncols=1, frameon=False)

    trend_inputs, trend_break, trend = bc_ax.get_lines()
    bc_ax.legend([trend_inputs, trend_break, trend], ["trend inputs", "trend break", "decay"], loc="upper right", prop={'size': tick_label_size}, ncols=1, frameon=False)

    trend_inputs, trend = opt_ax.get_lines()
    opt_ax.legend([trend_inputs, trend], ["trend inputs", "decay"], loc="upper right", prop={'size': tick_label_size}, ncols=1, frameon=False)

    trend_inputs, trend = optnorm_ax.get_lines()
    optnorm_ax.legend([trend_inputs, trend], ["trend inputs", "decay"], loc="upper right", prop={'size': tick_label_size}, ncols=1, frameon=False)

    ppo_ax.text(0.05, 0.05, "PPO", fontsize=label_size, ha='left', va='bottom', transform=ppo_ax.transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))
    bc_ax.text(0.05, 0.05, "BC", fontsize=label_size, ha='left', va='bottom', transform=bc_ax.transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))
    [ax.text(0.05, 0.05, "Optimal", fontsize=label_size, ha='left', va='bottom', transform=ax.transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round')) for ax in (opt_ax, optnorm_ax)]

    # add labelled subplots
    [fig.add_subplot(ax) for ax in axes]

    # saving
    save_fig(fig, "2_fitness_rawstd_isolated_supp", FORMAT)


def make_bound_vs_unbound_supp(client):  # for drl dim scaling
    # setup axes
    fig = plt.figure(figsize=(3.2, 3.2))
    grid = GridSpec(2, 2, wspace=0.5, hspace=0.5)

    n10_axes = [plt.Subplot(fig, grid[idx, 0]) for idx in range(2)]
    n20_axes = [plt.Subplot(fig, grid[idx, 1]) for idx in range(2)]

    axes = n10_axes + n20_axes
    
    # routine
    def plot_col(axes, run_name, fit_slc, omit_slc, bound_artcfg, unbound_artcfg, fit_file_str):
        # gather instance data
        bind_exp_name = "EVAL_bind"

        cost_mu, scale = get_bind_eval(client, bind_exp_name, run_name, f"drl_dim_cost.mean")
        cost_std, scale2 = get_bind_eval(client, bind_exp_name, run_name, f"drl_dim_cost.std")
        cost_opt, scale3 = get_bind_eval(client, bind_exp_name, run_name, f"opt_cost.mean")
        cost_rand, scale4 = get_bind_eval(client, bind_exp_name, run_name, f"rand_cost.mean")
        assert all([np.allclose(scale, s) for s in (scale2, scale3, scale4)])

        ron_mu = (cost_mu - cost_opt) / (cost_rand - cost_opt)
        ron_std = cost_std / (cost_rand - cost_opt)

        # plot instances
        ron_ax, ronstd_ax = axes
        nscale = 10 if run_name == "dim10" else 20

        plot_instance(ron_ax.plot, scale, ron_mu, fit_slc, omit_slc, True, True, "grow", "positive", fit_file_str, f"{nscale}n RON avg unbounded", unbound_artcfg)
        plot_instance(ron_ax.plot, scale, ron_mu, fit_slc, omit_slc, True, True, "decay", "negative", fit_file_str, f"{nscale}n RON avg bounded", bound_artcfg, plot_data=False)
        
        plot_instance(ronstd_ax.plot, scale, ron_std, fit_slc, omit_slc, True, True, "grow", "positive", fit_file_str, f"{nscale}n RON std unbounded", unbound_artcfg)
        plot_instance(ronstd_ax.plot, scale, ron_std, fit_slc, omit_slc, True, True, "decay", "negative", fit_file_str, f"{nscale}n RON std bounded", bound_artcfg, plot_data=False)
        
        # labelling axes
        label_size = 6
        tick_label_size = 4
        minor_tick_label_size = 4

        ron_ax.set_ylabel("Mean normalized test cost", fontsize=label_size)
        ronstd_ax.set_ylabel("SD normalized test cost", fontsize=label_size)

        [ax.set_xlabel("Spatial dimensions ($x$)", fontsize=label_size) for ax in axes]

        [ax.tick_params(labelsize=tick_label_size) for ax in axes]
        [ax.tick_params(axis="y", which="minor", labelsize=minor_tick_label_size) for ax in axes]

        [ax.set_xticks([2, 4, 6, 8, 10, 12, 15] if nscale == 10 else [2, 4, 6, 8, 10, 12]) for ax in axes]

        [ax.ticklabel_format(axis="y", style="sci", scilimits=(-1, 1)) for ax in axes]
        [ax.yaxis.get_offset_text().set_size(tick_label_size) for ax in axes]

    # plot columns
    bound_artcfg = dict(pm="o-", om="o-", fm="--", ps=3, dw=1, fw=0.75, dc="#B05FFF", fc="black", fe=None)
    unbound_artcfg = modcfg(bound_artcfg, dict(fc="red", fm="-", fw=0.5))
    
    fit_file_str = "2_problem_fitness_drl_dim_bound_vs_unbound_supp_fits.txt"
    clear_fits(fit_file_str)

    plot_col(n10_axes, "dim10", slice(11), slice(10, 12), bound_artcfg, unbound_artcfg, fit_file_str)
    plot_col(n20_axes, "dim20", slice(9), slice(8, 11), bound_artcfg, unbound_artcfg, fit_file_str)

    # add labelled subplots
    [fig.add_subplot(ax) for ax in axes]

    # legend
    dim_trend, dim_break, dim_unbounded_fit, dim_bounded_fit = axes[0].get_lines()
    fig.legend([dim_trend, dim_break, dim_bounded_fit, dim_unbounded_fit], ["$x$ trend inputs", "$x$ trend break", "$L - \\beta \propto -(x - \\gamma)^{{-\\alpha}}$", "$L - \\beta \propto (x - \\gamma)^{{\\alpha}}$"], loc="upper center", prop={'size': 6}, bbox_to_anchor=(0.5, 0.0), ncols=2, frameon=False)

    [ax.text(0.05, 0.95, "$n=10$", fontsize=6, ha='left', va='top', transform=ax.transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round')) for ax in n10_axes]
    [ax.text(0.05, 0.95, "$n=20$", fontsize=6, ha='left', va='top', transform=ax.transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round')) for ax in n20_axes]

    # saving
    save_fig(fig, "2_ppo_fitness_dim_bound_vs_unbound_supp", FORMAT)


def make_upper_bound_approach_supp(client):  # showing mean random cost approaches proven upper bound w.r.t. dimensions
    # setup axes
    fig = plt.figure(figsize=(4.0, 2.0))
    ax = fig.gca()

    # gather instance data
    bind_exp_name = "EVAL_bind"

    n10cost_rand, n10_scale = get_bind_eval(client, bind_exp_name, "dim10", f"rand_cost.mean")
    n20cost_rand, n20_scale = get_bind_eval(client, bind_exp_name, "dim20", f"rand_cost.mean")
    assert np.allclose(n10_scale, n20_scale)

    n10bound = (10 / np.sqrt(6)) * np.sqrt(n10_scale)
    n20bound = (20 / np.sqrt(6)) * np.sqrt(n20_scale)

    n10gap = n10bound - n10cost_rand
    n20gap = n20bound - n20cost_rand

    n10_artcfg = dict(pm="-o", om="-", fm="--", ps=4, dw=1.5, fw=1.5, dc="#605FFF", fc="black", fe=None)
    n20_artcfg = dict(pm="-o", om="-", fm="--", ps=4, dw=1.5, fw=1.5, dc="#FF5FFE", fc="black", fe=None)

    # plot instances
    fit_file_str = "2_rand_bound_approach_supp_fits.txt"
    clear_fits(fit_file_str)
    
    plot_instance(ax.loglog, n10_scale, n10gap, slice(len(n10gap)), None, False, False, "decay", "positive", fit_file_str, "10n gap", n10_artcfg, powerfit=True)
    plot_instance(ax.loglog, n20_scale, n20gap, slice(len(n20gap)), None, False, False, "decay", "positive", fit_file_str, "20n gap", n20_artcfg, powerfit=True)

    # labelling axes
    label_size = 6
    tick_label_size = 4
    minor_tick_label_size = 4

    ax.set_ylabel("Bound gap for mean random tour length", fontsize=label_size)
    ax.set_xlabel("Spatial dimensions ($d$)", fontsize=label_size)

    ax.tick_params(labelsize=tick_label_size)
    ax.tick_params(axis="y", which="minor", labelsize=minor_tick_label_size)

    ax.set_xlim([2, 100])
    # ax.set_ylim([0, 1])

    # legend
    n10_line, trend, n20_line, _ = ax.get_lines()
    ax.legend([n10_line, n20_line, trend], ["$n=10$", "$n=20$", "trend$\propto d^{{-0.53}}$"], loc="upper right", prop={'size': label_size}, ncols=1, frameon=True, framealpha=1)

    # saving
    save_fig(fig, "2_rand_bound_approach_supp_fitness", "pdf")
    save_fig(fig, "2_rand_bound_approach_supp_fitness", "png")


if __name__ == "__main__":
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    client = MlflowClient(tracking_uri)

    if PLOT_JOINT_MAIN_NIPS:
        make_joint_main_plots_nips(client)

    if PLOT_JOINT_MAIN_NIPS_TREND_BREAK:
        make_joint_main_plots_nips_trend_break(client)

    if PLOT_JOINT_MAIN_NIPS_TEMPORAL:
        make_joint_main_plots_nips_temporal(client)

    if PLOT_DRL_MAIN_NIPS:
        make_drl_main_plots_nips(client)

    if PLOT_DRL_SUPP_NIPS:
        make_drl_supp_plots_nips(client)
    
    if PLOT_DRL_MAIN:
        make_drl_main_plots(client)

    if PLOT_DRL_SUPP:
        make_drl_supp_plots(client)

    if PLOT_IL_SUPP:
        make_il_supp_plots(client)

    if PLOT_DECOMP_MAIN:
        make_decomp_main_plots(client)

    if PLOT_DECOMP_SUPP:
        make_decomp_supp_plots(client)

    if PLOT_RAW_STD_SUPP:
        make_raw_std_supp_plots(client)

    if PLOT_RAW_STD_ISOLATION_SUPP:
        make_isolated_rawstd_fits(client)
    
    if PLOT_BOUND_VS_UNBOUND_SUPP:
        make_bound_vs_unbound_supp(client)

    if PLOT_UPPER_BOUND_APPROACH:
        make_upper_bound_approach_supp(client)
