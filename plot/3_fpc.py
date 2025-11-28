import argparse
from importlib import import_module

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.text import Text
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.cm import ScalarMappable

from scipy.optimize import curve_fit
import numpy as np
import os
import os.path as osp
import mlflow
from mlflow import MlflowClient
from ast import literal_eval
from collections import defaultdict
import decimal

from plot_utils import get_sol_eval, get_bind_eval, power_scaling_fit, save_fit_eq, clear_fits, numpify_metrics, numpify_steps, forceAspect, partial_cmap, exp_scaling_fit
from plot_data import berlinColormap
plot_instance = getattr(import_module("2_problem_fitness"), "plot_instance")
modcfg = getattr(import_module("2_problem_fitness"), "modcfg")
get_proxy10n_data = getattr(import_module("A_proxyval_bigk"), "get_proxy10n_data")
get_proxy20n_data = getattr(import_module("A_proxyval_bigk"), "get_proxy20n_data")


FORMAT = "png"

PLOT_SEARCH_PROBLEM_SCALING_NIPS = True
PLOT_SEARCH_PROBLEM_SCALING_NIPS_2EXC = False

PLOT_JOINT_RES_ISO = False  # residual == natural performance metric (baseline isolation plots with residual Y)
PLOT_SOL_RES_MAIN_NIPS = False  # subopt vs res correlation for deep models, with summary of local search comparison below

PLOT_SOL_RES_MAIN = False
PLOT_SOL_RES_SUPP = False
PLOT_UNLIM_LO_RES_MAIN = False
PLOT_UNLIM_LO_RES_SUPP = False
PLOT_SEARCHCAP_MAIN = False
PLOT_SEARCHCAP_SUPP = False
PLOT_NODE_PHASE_MAIN = False
PLOT_NODE_PHASE_SUPP = False
PLOT_DIM_PHASE_SUPP = False
PLOT_UNEXPECTED_SUPP = False




def plot_heterochromia_instance(ax, plt_fn, scale_x, val_y, colors, fit_slc, omit_slc, use_c0, use_c1, mode, sign, fit_file_str, id_str, artcfg, plot_data=True, powerfit=True, plot_powerfit=True, powerfit_zorder=3, omit_zorder=1, segment_alpha=1.0):
    point_marker = artcfg["pm"]
    omit_marker = artcfg["om"]
    fit_marker = artcfg["fm"]

    point_size = artcfg["ps"]
    
    data_width = artcfg["dw"]
    fit_width = artcfg["fw"]
    
    data_color = artcfg["dc"]
    fit_color = artcfg["fc"]

    if plot_data:
        # used in fit
        plt_fn(scale_x[fit_slc], val_y[fit_slc], point_marker, color=data_color, markersize=point_size, linewidth=data_width, zorder=2, alpha=segment_alpha)
        ax.scatter(scale_x[fit_slc], val_y[fit_slc], c=colors[fit_slc], s=point_size, marker="o", zorder=2)
        
        # trailing points omitted from fit
        if omit_slc is not None:
            plt_fn(scale_x[omit_slc], val_y[omit_slc], omit_marker, markerfacecolor="white", color=data_color, markersize=point_size, linewidth=data_width, zorder=omit_zorder)
            #ax.scatter(scale_x[omit_slc], val_y[omit_slc], facecolors="none", edgecolors=colors[omit_slc], marker="o", s=point_size, zorder=omit_zorder)
            ax.scatter(scale_x[omit_slc], val_y[omit_slc], c=colors[omit_slc], marker="x", linewidths=0.2 * point_size, s=point_size, zorder=omit_zorder)

    # power fit
    if powerfit:
        print(f"\n{id_str}:")
        
        x_bounds = (scale_x[fit_slc].min(), scale_x[fit_slc].max())
        fit_x, fit_y, fit_popt = power_scaling_fit(scale_x[fit_slc], val_y[fit_slc], x_bounds, c0="fit" if use_c0 else 0, c1="fit" if use_c1 else 0, mode=mode, sign=sign)
        popt, x_scale, y_scale = fit_popt

        c0, c1, c, m = popt

        save_fit_eq(fit_file_str, id_str, c0, c1, c, m, sign, y_scale, x_scale)
        
        if plot_powerfit:
            plt_fn(fit_x, fit_y, fit_marker, color=fit_color, linewidth=fit_width, zorder=powerfit_zorder)

        return fit_popt


def flatten(iterable):
    retv = []
    for itr in iterable:
        if type(itr) in (list, tuple):
            retv += flatten(itr)
        else:
            retv.append(itr)
    return retv


def split_by_triphase(data_arrs, cap_probs, low_thresh, up_thresh, low_buff, up_buff):
    if type(data_arrs) is np.ndarray:
        data_len = len(data_arrs)
        single_arr = True
    else:
        assert all([len(arr) == len(data_arrs[0]) for arr in data_arrs[1:]])
        data_len = len(data_arrs[0])
        single_arr = False
            
    partial_mask = np.logical_and(cap_probs > low_thresh, cap_probs < up_thresh) 
    partial_mask = np.pad(partial_mask, (low_buff, up_buff))
    surround_mask = np.copy(partial_mask)

    offsets = list(range(-low_buff, 0)) + list(range(1, up_buff+1))
    for offset in offsets:
        surround_mask = np.logical_or(surround_mask, np.roll(partial_mask, offset))
    surround_mask = surround_mask[low_buff:]
    if up_buff > 0: surround_mask = surround_mask[:-up_buff]

    low_mask = np.arange(data_len) < np.nonzero(partial_mask)[0][0]
    up_mask = np.arange(data_len) > np.nonzero(partial_mask)[-1][-1]

    if single_arr:  # unconstrained, partially constrained, fully constrained
        return (data_arrs[low_mask], data_arrs[surround_mask], data_arrs[up_mask])
    else:
        return [(arr[low_mask], arr[surround_mask], arr[up_mask]) for arr in data_arrs]


def split_by_cap(client, exp_name, run_name, val_key, cap_key):
    values, val_scale = get_bind_eval(client, exp_name, run_name, val_key)
    caps, cap_scale = get_bind_eval(client, exp_name, run_name, cap_key)
    assert np.allclose(val_scale, cap_scale)

    unique_caps = np.unique(caps)
    split_vals = [values[caps == uc] for uc in unique_caps]

    return split_vals, unique_caps


def save_fig(fig, name, format):
    if format in ("eps", "svg"):
        fig.savefig(f"{name}.{format}", format=format, bbox_inches="tight")
    else:  # assuming non-vector with dpi
        fig.savefig(f"{name}.{format}", format=format, dpi=300, bbox_inches="tight")


def get_bigk_data(client, run_prefix, y_metric_key):
    x_metric_key = "nscale" if run_prefix.startswith("node") else "dscale"
    exp_name = "bigK_local_optima"

    input_exp = client.get_experiment_by_name(exp_name)
    exp_runs = client.search_runs(experiment_ids=[input_exp.experiment_id])  # WARNING this will be too slow if experiment is too large to view via MLflow UI
    plot_runs = list(filter(lambda run: run.info.run_name.startswith(run_prefix), exp_runs))

    x_vals = []
    y_vals = []

    for run in plot_runs:
        if run.info.run_name != "dim_scaling_20n_2swap_14":  # 40n, didn't complete for some reason
            x_vals.append(client.get_metric_history(run.info.run_id, x_metric_key)[0].value)
            y_vals.append(client.get_metric_history(run.info.run_id, y_metric_key)[0].value)

    x_vals = np.asarray(x_vals)
    y_vals = np.asarray(y_vals)

    sort_idx = np.argsort(x_vals)

    return y_vals[sort_idx], x_vals[sort_idx]


def get_proxynode_data(client, y_metric_key, algo="2opt"):
    exp_name = "APPROX_global_optima"
    run_names = [f"node_scaling_{idx}" for idx in range(6)] if algo == "2opt" else [f"node_scaling_2swap_{idx}" for idx in range(9)]

    input_exp = client.get_experiment_by_name(exp_name)
    exp_runs = client.search_runs(experiment_ids=[input_exp.experiment_id])  # WARNING this will be too slow if experiment is too large to view via MLflow UI
    plot_runs = list(filter(lambda run: run.info.run_name in run_names, exp_runs))

    x_vals = []
    y_vals = []

    for run in plot_runs:
        if algo == "2swap" or run.info.run_name in ("node_scaling_0", "node_scaling_1"):
            x_vals.append(client.get_metric_history(run.info.run_id, "nscale")[0].value)
            y_vals.append(client.get_metric_history(run.info.run_id, y_metric_key)[0].value)
        
        else:
            metrics = client.get_metric_history(run.info.run_id, y_metric_key)
            x_vals += list(next(numpify_steps(metrics)))
            y_vals += list(next(numpify_metrics(metrics)))

    x_vals = np.asarray(x_vals)
    y_vals = np.asarray(y_vals)

    sort_idx = np.argsort(x_vals)

    return y_vals[sort_idx], x_vals[sort_idx]



def make_search_complexity_plots_nips(client):
    # setup axes
    fig = plt.figure(figsize=(5.5, 3.0))
    outer_grid = GridSpec(2, 1, hspace=0.4, height_ratios=[0.4, 0.25])
    main_grid = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_grid[0, 0], wspace=0.2, width_ratios=[8, 4, 4])   
    
    dim_ax = plt.Subplot(fig, main_grid[0, 0])
    node_unc_ax = plt.Subplot(fig, main_grid[0, 1])  # unconstrained
    node_ax = plt.Subplot(fig, main_grid[0, 2])

    trials = 5
    phase_grid = GridSpecFromSubplotSpec(1, trials, subplot_spec=outer_grid[1, 0], wspace=0.1)
    phase_axes = [plt.Subplot(fig, phase_grid[0, 0])]
    phase_axes += [plt.Subplot(fig, phase_grid[0, col_idx], sharex=phase_axes[0], sharey=phase_axes[0]) for col_idx in range(1, trials)]

    main_axes = [dim_ax, node_unc_ax, node_ax]
    axes = main_axes + phase_axes

    [ax.axhline(0, color="black", linewidth=plt.rcParams['xtick.major.width']) for ax in [dim_ax, node_ax] + phase_axes]
    node_unc_ax.axhline(0, color="black", linewidth=plt.rcParams['xtick.major.width'], zorder=1)

    # gather instance data
    constrained_exp_name = "EVAL_constrained_local_optima"
    bind_exp_name = "EVAL_bind"

    d10cost_opt, dim_scale = get_bind_eval(client, bind_exp_name, "dim10", f"opt_cost.mean")
    d20cost_opt, _ = get_bind_eval(client, bind_exp_name, "dim20", f"opt_cost.mean")

    d10costs, _ = get_bind_eval(client, bind_exp_name, "dim10", f"_2opt_cost.mean")
    d10costs[0] = 2.8857 # 2D local search experiments (2opt and 2exchange) both have corrupted, inflated results in EVAL_bind for 10-node version, where 20-node version does not, and I'm not sure why; this substitutes values from earlier evals done on same proxy opt problems, where issue does not show up
    d20costs, _ = get_bind_eval(client, bind_exp_name, "dim20", f"_2opt_cost.mean")
    
    d10cost_ppo, _ = get_bind_eval(client, bind_exp_name, "dim10", f"drl_dim_cost.mean")
    d20cost_ppo, _ = get_bind_eval(client, bind_exp_name, "dim20", f"drl_dim_cost.mean")

    d10_subopt = (d10costs - d10cost_opt)
    d20_subopt = (d20costs - d20cost_opt)
    ppo_dim10_subopt = d10cost_ppo - d10cost_opt
    ppo_dim20_subopt = d20cost_ppo - d20cost_opt

    ncost_opt, _ = get_bind_eval(client, bind_exp_name, "node", f"opt_cost.mean")
    ncost_rand, _ = get_bind_eval(client, bind_exp_name, "node", f"rand_cost.mean")

    ncosts, node_scale = get_bind_eval(client, bind_exp_name, "node", "_2opt_cost.mean")
    ncost_ppo, _ = get_bind_eval(client, bind_exp_name, "node", f"drl_node_cost.mean")

    ncosts_constrained, ncap_ref = split_by_cap(client, constrained_exp_name, "plus_node_scaling_2opt", "2opt_local_opt.costs_avg", "search_cap")
    ncap_probs_constrained, _ = split_by_cap(client, constrained_exp_name, "plus_node_scaling_2opt", "edge.2opt_local_opt.caps_avg", "search_cap")

    node_subopt = ncosts - ncost_opt
    node_span = ncost_rand - ncost_opt
    nsubopts_constrained = [ncc - ncost_opt for ncc in ncosts_constrained]
    ppo_node_subopt = ncost_ppo- ncost_opt

    dim_fit_slc, dim_omit_slc = slice(17), None
    node_fit_slc, node_omit_slc = slice(10), None

    ppo_node_fit_slc, ppo_node_omit_slc = slice(9), slice(8, 10)
    ppo_d10_fit_slc, ppo_d10_omit_slc = slice(11), slice(10, 17)
    ppo_d20_fit_slc, ppo_d20_omit_slc = slice(9), slice(8, 17)

    base_artcfg = dict(pm="o-", om="o-", fm="--", ps=2.0, dw=1, fw=0.75, dc="#FFB05F", fc="black", fe=None)
    rand_artcfg = modcfg(base_artcfg, dict(dc="black"))
    unlim_artcfg = modcfg(base_artcfg, dict(dc="#A0C850", fc="#C85064", fm="-", fw=0.4))  # #FFEE00 #F7F800

    d10_artcfg = dict(pm="o-", om="o-", fm="-", ps=2.5, dw=1, fw=1, dc="#5FAEFF", fc="black", fe=None)  # original dim10 color #B05FFF
    d20_artcfg = dict(pm="o-", om="o-", fm="-", ps=2.5, dw=1, fw=1, dc="#FF5FAE", fc="black", fe=None)

    ppo_node_artcfg = dict(pm="-", om="o-", fm="-", ps=3, dw=1, fw=1.5, dc="#FFB05F", fc="#FFB05F", fe=None)
    ppo_d10_artcfg = dict(pm="-", om="o-", fm="-", ps=3, dw=1, fw=1, dc="#605FFF", fc="#605FFF", fe=None)  # original dim10 color #B05FFF
    ppo_d20_artcfg = dict(pm="-", om="o-", fm="-", ps=3, dw=1, fw=1, dc="#FF5FFE", fc="#FF5FFE", fe=None)

    # plot instances
    powerfit = True
    fit_file_str = "3_joint_search_complexity_nips_fits.txt"
    clear_fits(fit_file_str)

    # RL references
    plot_instance(dim_ax.plot, dim_scale, ppo_dim10_subopt, ppo_d10_fit_slc, ppo_d10_omit_slc, True, False, "decay", "negative", fit_file_str, "subopt ppo dim10 (exp decay)", ppo_d10_artcfg, expfit=True, plot_data=False, plot_powerfit=True, powerfit_zorder=1)
    
    plot_instance(dim_ax.plot, dim_scale, ppo_dim20_subopt, ppo_d20_fit_slc, ppo_d20_omit_slc, True, False, "decay", "negative", fit_file_str, "subopt ppo dim20 (exp decay)", ppo_d20_artcfg, expfit=True, plot_data=False, plot_powerfit=True, powerfit_zorder=1)

    plot_instance(node_unc_ax.plot, node_scale, ppo_node_subopt, ppo_node_fit_slc, ppo_node_omit_slc, False, True, "grow", "positive", fit_file_str, "subopt ppo node", ppo_node_artcfg, plot_data=False, plot_powerfit=True, powerfit_zorder=1)

    ## main dim plots
    # plot_instance(dim_ax.plot, dim_scale, d10_subopt, dim_fit_slc, dim_omit_slc, False, True, "grow", "positive", fit_file_str, "dim10 subopt (power growth)", modcfg(d10_artcfg, dict(fc="black", fm=":", fw=0.5)), plot_data=False, plot_powerfit=False)
    plot_instance(dim_ax.plot, dim_scale, d10_subopt, dim_fit_slc, dim_omit_slc, True, False, "decay", "negative", fit_file_str, "dim10 subopt (power decay)", modcfg(d10_artcfg, dict(fc="black", fm="-.", fw=0.5)), plot_data=False, plot_powerfit=False)
    plot_instance(dim_ax.plot, dim_scale, d10_subopt, dim_fit_slc, dim_omit_slc, True, False, "decay", "negative", fit_file_str, "dim10 subopt (exp decay)", d10_artcfg, expfit=True, plot_data=False, plot_powerfit=False)
    d10_popt = plot_instance(dim_ax.plot, dim_scale, d10_subopt, dim_fit_slc, dim_omit_slc, True, True, "decay", "negative", fit_file_str, "dim10 subopt (subexp decay)", d10_artcfg, subexpfit=True)
    
    # plot_instance(dim_ax.plot, dim_scale, d20_subopt, dim_fit_slc, dim_omit_slc, False, True, "grow", "positive", fit_file_str, "dim20 subopt (power growth)", modcfg(d20_artcfg, dict(fc="black", fm=":", fw=0.5)), plot_data=False)
    plot_instance(dim_ax.plot, dim_scale, d20_subopt, dim_fit_slc, dim_omit_slc, True, False, "decay", "negative", fit_file_str, "dim20 subopt (power decay)", modcfg(d20_artcfg, dict(fc="black", fm="-.", fw=0.5)), plot_data=False, plot_powerfit=False)
    plot_instance(dim_ax.plot, dim_scale, d20_subopt, dim_fit_slc, dim_omit_slc, True, False, "decay", "negative", fit_file_str, "dim20 subopt (exp decay)", d20_artcfg, expfit=True, plot_data=False, plot_powerfit=False)
    d20_popt = plot_instance(dim_ax.plot, dim_scale, d20_subopt, dim_fit_slc, dim_omit_slc, True, True, "decay", "negative", fit_file_str, "dim20 subopt (subexp decay)", d20_artcfg, subexpfit=True)

    beta_col = "slategrey"
    d10_beta = d10_popt[-1] * d10_popt[0][0]
    dim_ax.axhline(d10_beta, color=beta_col, linewidth=1, zorder=1)
    d20_beta = d20_popt[-1] * d20_popt[0][0]
    dim_ax.axhline(d20_beta, color=beta_col, linewidth=1, zorder=1)

    # main unconstrained node plot
    plot_instance(node_unc_ax.plot, node_scale, node_subopt, node_fit_slc, node_omit_slc, False, True, "grow", "positive", fit_file_str, "node subopt unlim", modcfg(unlim_artcfg, dict(ps=3, fw=1)), powerfit=powerfit)

    ## main node constraint sweep
    cap_map = partial_cmap(plt.get_cmap("inferno"), 0.4, 0.9)
    mod_col = lambda x, v: modcfg(x, dict(dc=cap_map((v - 5) / 20)))

    plot_instance(node_ax.plot, node_scale, node_span, node_fit_slc, node_omit_slc, False, True, "grow", "positive", fit_file_str, "node achievable span", rand_artcfg, powerfit=False)
    plot_instance(node_ax.plot, node_scale, node_subopt, node_fit_slc, node_omit_slc, False, True, "grow", "positive", fit_file_str, "node subopt", unlim_artcfg, powerfit=False)
    [plot_instance(node_ax.plot, node_scale, nsc, node_fit_slc, node_omit_slc, False, True, "grow", "positive", fit_file_str, f"node {int(cap)}M subopt", mod_col(base_artcfg, cap), powerfit=False) for nsc, cap in list(zip(nsubopts_constrained, ncap_ref))[:-1][::-1]]

    ## node phase plots
    for trial_idx, capv in enumerate(ncap_ref[:-1]):
        nsc = nsubopts_constrained[trial_idx]
        cap_probs = ncap_probs_constrained[trial_idx]

        phase_artcfg = modcfg(base_artcfg, dict(pm="-", dc="white", fm=":", ps=8, fw=1.0))
        cap_colors = berlinColormap(cap_probs)

        low_thresh = 0.3
        high_thresh = 0.6
        low_buff = 0
        high_buff = 0
        nx_phased, nsc_phased, cap_col_phased = split_by_triphase([node_scale, nsc, cap_colors], cap_probs, low_thresh, high_thresh, low_buff, high_buff)

        ux, px, fx = nx_phased 
        usubopt, psubopt, fsubopt = nsc_phased
        ucols, pcols, fcols = cap_col_phased

        trial_idx = len(phase_axes) - trial_idx - 1  # reverse order of plots
        plot_heterochromia_instance(phase_axes[trial_idx], phase_axes[trial_idx].plot, ux, usubopt, ucols, slice(len(ux)), None, False, True, "grow", "positive", fit_file_str, f"node unconstrained {int(capv)}M subopt", phase_artcfg, powerfit=False, segment_alpha=0.0)
        plot_heterochromia_instance(phase_axes[trial_idx], phase_axes[trial_idx].plot, px, psubopt, pcols, slice(len(px)), None, False, True, "grow", "positive", fit_file_str, f"node partially-constrained {int(capv)}M subopt", phase_artcfg, powerfit=False, segment_alpha=0.0)
        plot_heterochromia_instance(phase_axes[trial_idx], phase_axes[trial_idx].plot, fx, fsubopt, fcols, slice(len(fx)), None, False, True, "grow", "positive", fit_file_str, f"node fully-constrained {int(capv)}M subopt", phase_artcfg, powerfit=powerfit and len(fx) > 1, segment_alpha=0.0)


    # labelling axes
    label_size = 6
    tick_label_size = 4
    minor_tick_label_size = 4

    dim_ax.set_ylabel("Suboptimality gap", fontsize=label_size)
    phase_axes[0].set_ylabel("Suboptimality gap", fontsize=label_size)

    dim_ax.set_xlabel("Spatial dimensions ($d$)", fontsize=label_size)
    [ax.set_xlabel("Nodes ($n$)", fontsize=label_size) for ax in (node_unc_ax, node_ax)]
    [ax.set_xlabel("Nodes ($n$)", fontsize=label_size) for ax in phase_axes]

    dim_ax.set_xlim(left=0)
    dim_ax.set_ylim([0, 0.28])
    node_unc_ax.set_ylim(bottom=-0.01)
    node_ax.set_ylim(bottom=-0.5)

    dim_ax.set_xticks([2, 10, 20, 30, 40, 50, 100])
    node_unc_ax.set_xticks(list(range(5, 51, 15)))
    node_ax.set_xticks(list(range(5, 51, 15)))
    [ax.set_xticks(list(range(5, 51, 15))) for ax in phase_axes]

    [ax.tick_params(labelsize=tick_label_size) for ax in axes]
    [ax.tick_params(which="minor", labelsize=minor_tick_label_size) for ax in axes]

    [ax.tick_params(which="both", labelleft=False) for ax in phase_axes[1:]]

    # phase_axes[0].set_yticks([0.001, 0.01, 0.1, 1, 10])

    # [forceAspect(ax) for ax in phase_axes]

    # legend
    leg_lbl_size = 4.75

    _, ppo_d10, ppo_d20, d10, subexp_fit, d20, _, beta_line, _ = dim_ax.get_lines()
    eh = mpatches.Rectangle((0,0), 1, 1, fill=False, edgecolor="none", visible=False)
    dim_ax.legend([eh, ppo_d10, d10, eh, ppo_d20, d20, eh, eh, subexp_fit], ["$n=10$", "RL trend", "2-opt", "$n=20$", "RL trend", "2-opt", " ", " ", r"$s - \beta \propto -\psi^{-d^{\phi}}$"], loc="best", prop={'size': leg_lbl_size}, ncols=3, columnspacing=0.6, frameon=True) #, framealpha=1)
    
    xmin, xmax = dim_ax.get_xlim()
    xeps = 5
    dim_ax.text(xmin+xeps, d20_beta+0.002, r"2-opt $\beta_{n=20}$", color=beta_col, fontsize=label_size, ha='left', va='bottom')
    dim_ax.text(xmin+xeps, d10_beta+0.003, r"2-opt $\beta_{n=10}$", color=beta_col, fontsize=label_size, ha='left', va='bottom')

    _, ppo_node, unlim_line, pow_fit = node_unc_ax.get_lines()
    node_unc_ax.legend([ppo_node, unlim_line, pow_fit], [r"RL trend ($\alpha \approx 1.86$)", r"2-opt ($M=\infty$)", "$s \propto (n - \\gamma)^{{\\alpha}}$"], loc="upper left", bbox_to_anchor=(-0.03, 1.03), prop={'size': leg_lbl_size}, ncols=1, frameon=False, framealpha=1)

    rand_line = node_ax.get_lines()[1]
    unlim_line = node_ax.get_lines()[2]
    node_ax.legend([rand_line, unlim_line], ["random", r"2-opt ($M=\infty$)"], loc="upper left", prop={'size': leg_lbl_size}, ncols=1, frameon=False, framealpha=1)

    power_fit = phase_axes[-2].get_lines()[-1]
    alphas = [1.28, 1.49, 1.67, 1.82, 2.09][::-1]
    gammas = [10.8, 14.8, 18.1, 21.0, 22.3][::-1]
    [ax.legend([power_fit], [fr"$s \propto (n - {gam:.1f})^{{{alph:.2f}}}$"], loc="upper left", bbox_to_anchor=(-0.025, 0.925), prop={'size': leg_lbl_size-1 if alph < 1.3 else leg_lbl_size}, ncols=1, frameon=False, framealpha=1) for alph, gam, ax in zip(alphas, gammas, phase_axes)]

    [ax.text(0.0, 1.0, f"$M={int(capv)}$", fontsize=5, ha='left', va='top', transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round')) for ax, capv in zip(phase_axes, ncap_ref[:-1][::-1])]

    # add labelled subplots
    [fig.add_subplot(ax) for ax in axes]

    # colorbars
    splash_cbar_ax = fig.add_axes([node_ax.get_position().x1 + 0.015, node_ax.get_position().y0 , 0.01, node_ax.get_position().y1 - node_ax.get_position().y0])
    phase_cbar_ax = fig.add_axes([phase_axes[-1].get_position().x1 + 0.015, phase_axes[-1].get_position().y0 , 0.01, phase_axes[-1].get_position().y1 - phase_axes[-1].get_position().y0])

    splash_cbar = fig.colorbar(ScalarMappable(cmap=cap_map), cax=splash_cbar_ax, location="right", fraction=1, aspect=40)
    splash_cbar_ticks = (ncap_ref - 5) / 20
    splash_cbar.ax.set_yticks(splash_cbar_ticks[:-1], [Text(0, t, f"{int(v)}") for t, v in list(zip(splash_cbar_ticks, ncap_ref))[:-1]])
    splash_cbar.set_label("Search capacity ($M$)", rotation=90, fontsize=label_size)
    splash_cbar.ax.tick_params(labelsize=tick_label_size) 

    phase_cbar = fig.colorbar(ScalarMappable(cmap=berlinColormap), cax=phase_cbar_ax, location="right", fraction=1, aspect=40)
    phase_cbar_ticks = [0, 0.25, 0.5, 0.75, 1]
    phase_cbar.ax.set_yticks(phase_cbar_ticks, [Text(0, t, f"{int(100 * t)}") for t in phase_cbar_ticks])
    phase_cbar.set_label("Max depth %", rotation=90, fontsize=label_size)
    phase_cbar.ax.tick_params(labelsize=tick_label_size) 

    # saving
    save_fig(fig, "3_joint_search_complexity_nips", "pdf")
    save_fig(fig, "3_joint_search_complexity_nips", "png")


def make_search_complexity_plots_nips_2exc(client):
    # setup axes
    fig = plt.figure(figsize=(5.5, 2.17))
    outer_grid = GridSpec(2, 1, hspace=0.5, height_ratios=[0.4, 0.25])
    main_grid = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_grid[0, 0], wspace=0.2, width_ratios=[8, 4, 4])   
    
    dim_ax = plt.Subplot(fig, main_grid[0, 0])
    node_unc_ax = plt.Subplot(fig, main_grid[0, 1])  # unconstrained
    node_ax = plt.Subplot(fig, main_grid[0, 2])

    trials = 5
    phase_grid = GridSpecFromSubplotSpec(1, trials, subplot_spec=outer_grid[1, 0], wspace=0.1)
    phase_axes = [plt.Subplot(fig, phase_grid[0, 0])]
    phase_axes += [plt.Subplot(fig, phase_grid[0, col_idx], sharex=phase_axes[0], sharey=phase_axes[0]) for col_idx in range(1, trials)]

    main_axes = [dim_ax, node_unc_ax, node_ax]
    axes = main_axes + phase_axes

    [ax.axhline(0, color="black", linewidth=plt.rcParams['xtick.major.width']) for ax in [dim_ax, node_ax] + phase_axes]
    node_unc_ax.axhline(0, color="black", linewidth=plt.rcParams['xtick.major.width'], zorder=1)

    # gather instance data
    constrained_exp_name = "EVAL_constrained_local_optima"
    bind_exp_name = "EVAL_bind"

    d10cost_opt, dim_scale = get_bind_eval(client, bind_exp_name, "dim10", f"opt_cost.mean")
    d20cost_opt, _ = get_bind_eval(client, bind_exp_name, "dim20", f"opt_cost.mean")

    d10costs, _ = get_bind_eval(client, bind_exp_name, "dim10", f"_2swap_cost.mean")
    d10costs[0] = 2.9238 # 2D local search experiments (2opt and 2exchange) both have corrupted, inflated results in EVAL_bind for 10-node version, where 20-node version does not, and I'm not sure why; this substitutes values from earlier evals done on same proxy opt problems, where issue does not show up
    d20costs, _ = get_bind_eval(client, bind_exp_name, "dim20", f"_2swap_cost.mean")

    d10cost_ppo, _ = get_bind_eval(client, bind_exp_name, "dim10", f"drl_dim_cost.mean")
    d20cost_ppo, _ = get_bind_eval(client, bind_exp_name, "dim20", f"drl_dim_cost.mean")

    d10_subopt = (d10costs - d10cost_opt)
    d20_subopt = (d20costs - d20cost_opt)
    ppo_dim10_subopt = d10cost_ppo - d10cost_opt
    ppo_dim20_subopt = d20cost_ppo - d20cost_opt

    ncost_opt, _ = get_bind_eval(client, bind_exp_name, "node", f"opt_cost.mean")
    ncost_rand, _ = get_bind_eval(client, bind_exp_name, "node", f"rand_cost.mean")

    ncosts, node_scale = get_bind_eval(client, bind_exp_name, "node", "_2swap_cost.mean")
    ncost_ppo, _ = get_bind_eval(client, bind_exp_name, "node", f"drl_node_cost.mean")

    ncosts_constrained, ncap_ref = split_by_cap(client, constrained_exp_name, "plus_node_scaling_2swap", "2swap_local_opt.costs_avg", "search_cap")
    ncap_probs_constrained, _ = split_by_cap(client, constrained_exp_name, "plus_node_scaling_2swap", "edge.2swap_local_opt.caps_avg", "search_cap")

    node_subopt = ncosts - ncost_opt
    node_span = ncost_rand - ncost_opt
    nsubopts_constrained = [ncc - ncost_opt for ncc in ncosts_constrained]
    ppo_node_subopt = ncost_ppo- ncost_opt

    dim_fit_slc, dim_omit_slc = slice(17), None
    node_fit_slc, node_omit_slc = slice(10), None

    ppo_node_fit_slc, ppo_node_omit_slc = slice(9), slice(8, 10)
    ppo_d10_fit_slc, ppo_d10_omit_slc = slice(11), slice(10, 17)
    ppo_d20_fit_slc, ppo_d20_omit_slc = slice(9), slice(8, 17)

    base_artcfg = dict(pm="o-", om="o-", fm="--", ps=1.05, dw=1, fw=0.75, dc="#FFB05F", fc="black", fe=None)
    rand_artcfg = modcfg(base_artcfg, dict(dc="black"))
    unlim_artcfg = modcfg(base_artcfg, dict(dc="#A0C850", fc="#C85064", fm="-", fw=0.4))  # #FFEE00 #F7F800

    d10_artcfg = dict(pm="o-", om="o-", fm="-", ps=2, dw=1, fw=0.5, dc="#5FAEFF", fc="black", fe=None)  # original dim10 color #B05FFF
    d20_artcfg = dict(pm="o-", om="o-", fm="-", ps=2, dw=1, fw=0.5, dc="#FF5FAE", fc="black", fe=None)

    ppo_node_artcfg = dict(pm="-", om="o-", fm="-", ps=3, dw=1, fw=1, dc="#FFB05F", fc="#FFB05F", fe=None)
    ppo_d10_artcfg = dict(pm="-", om="o-", fm="-", ps=3, dw=1, fw=0.75, dc="#605FFF", fc="#605FFF", fe=None)  # original dim10 color #B05FFF
    ppo_d20_artcfg = dict(pm="-", om="o-", fm="-", ps=3, dw=1, fw=0.75, dc="#FF5FFE", fc="#FF5FFE", fe=None)

    # plot instances
    powerfit = True
    fit_file_str = "3_joint_search_complexity_nips_fits_2exc.txt"
    clear_fits(fit_file_str)

    # RL references
    plot_instance(dim_ax.plot, dim_scale, ppo_dim10_subopt, ppo_d10_fit_slc, ppo_d10_omit_slc, True, False, "decay", "negative", fit_file_str, "subopt ppo dim10 (exp decay)", ppo_d10_artcfg, expfit=True, plot_data=False, plot_powerfit=True, powerfit_zorder=1)
    
    plot_instance(dim_ax.plot, dim_scale, ppo_dim20_subopt, ppo_d20_fit_slc, ppo_d20_omit_slc, True, False, "decay", "negative", fit_file_str, "subopt ppo dim20 (exp decay)", ppo_d20_artcfg, expfit=True, plot_data=False, plot_powerfit=True, powerfit_zorder=1)

    plot_instance(node_unc_ax.plot, node_scale, ppo_node_subopt, ppo_node_fit_slc, ppo_node_omit_slc, False, True, "grow", "positive", fit_file_str, "subopt ppo node", ppo_node_artcfg, plot_data=False, plot_powerfit=True, powerfit_zorder=1)

    ## main dim plots
    # plot_instance(dim_ax.plot, dim_scale, d10_subopt, dim_fit_slc, dim_omit_slc, False, True, "grow", "positive", fit_file_str, "dim10 subopt (power growth)", modcfg(d10_artcfg, dict(fc="black", fm=":", fw=0.5)), plot_data=False, plot_powerfit=False)
    plot_instance(dim_ax.plot, dim_scale, d10_subopt, dim_fit_slc, dim_omit_slc, True, False, "decay", "negative", fit_file_str, "dim10 subopt (power decay)", modcfg(d10_artcfg, dict(fc="black", fm="-.", fw=0.5)), plot_data=False, plot_powerfit=False)
    plot_instance(dim_ax.plot, dim_scale, d10_subopt, dim_fit_slc, dim_omit_slc, True, True, "decay", "negative", fit_file_str, "dim10 subopt (subexp decay)", d10_artcfg, subexpfit=True, plot_data=False, plot_powerfit=False)
    d10_popt = plot_instance(dim_ax.plot, dim_scale, d10_subopt, dim_fit_slc, dim_omit_slc, True, False, "decay", "negative", fit_file_str, "dim10 subopt (exp decay)", d10_artcfg, expfit=True)

    # plot_instance(dim_ax.plot, dim_scale, d20_subopt, dim_fit_slc, dim_omit_slc, False, True, "grow", "positive", fit_file_str, "dim20 subopt (power growth)", modcfg(d20_artcfg, dict(fc="black", fm=":", fw=0.5)), plot_data=False)
    plot_instance(dim_ax.plot, dim_scale, d20_subopt, dim_fit_slc, dim_omit_slc, True, False, "decay", "negative", fit_file_str, "dim20 subopt (power decay)", modcfg(d20_artcfg, dict(fc="black", fm="-.", fw=0.5)), plot_data=False, plot_powerfit=False)
    plot_instance(dim_ax.plot, dim_scale, d20_subopt, dim_fit_slc, dim_omit_slc, True, True, "decay", "negative", fit_file_str, "dim20 subopt (subexp decay)", d20_artcfg, subexpfit=True, plot_data=False, plot_powerfit=False)
    d20_popt = plot_instance(dim_ax.plot, dim_scale, d20_subopt, dim_fit_slc, dim_omit_slc, True, False, "decay", "negative", fit_file_str, "dim20 subopt (exp decay)", d20_artcfg, expfit=True)

    beta_col = "slategrey"
    d10_beta = d10_popt[-1] * d10_popt[0][0]
    dim_ax.axhline(d10_beta, color=beta_col, linewidth=0.5, zorder=1)
    d20_beta = d20_popt[-1] * d20_popt[0][0]
    dim_ax.axhline(d20_beta, color=beta_col, linewidth=0.5, zorder=1)

    # main unconstrained node plot
    plot_instance(node_unc_ax.plot, node_scale, node_subopt, node_fit_slc, node_omit_slc, False, True, "grow", "positive", fit_file_str, "node subopt unlim", modcfg(unlim_artcfg, dict(ps=2)), powerfit=powerfit, plot_powerfit=False)

    ## main node constraint sweep
    cap_map = partial_cmap(plt.get_cmap("inferno"), 0.4, 0.9)
    mod_col = lambda x, v: modcfg(x, dict(dc=cap_map((v - 5) / 20)))

    plot_instance(node_ax.plot, node_scale, node_span, node_fit_slc, node_omit_slc, False, True, "grow", "positive", fit_file_str, "node achievable span", rand_artcfg, powerfit=False)
    plot_instance(node_ax.plot, node_scale, node_subopt, node_fit_slc, node_omit_slc, False, True, "grow", "positive", fit_file_str, "node subopt", unlim_artcfg, powerfit=False)
    [plot_instance(node_ax.plot, node_scale, nsc, node_fit_slc, node_omit_slc, False, True, "grow", "positive", fit_file_str, f"node {int(cap)}M subopt", mod_col(base_artcfg, cap), powerfit=False) for nsc, cap in list(zip(nsubopts_constrained, ncap_ref))[:-1][::-1]]

    ## node phase plots
    for trial_idx, capv in enumerate(ncap_ref[:-1]):
        nsc = nsubopts_constrained[trial_idx]
        cap_probs = ncap_probs_constrained[trial_idx]

        phase_artcfg = modcfg(base_artcfg, dict(pm="-", dc="white", fm=":", ps=3))
        cap_colors = berlinColormap(cap_probs)

        low_thresh = 0.25
        high_thresh = 0.6
        low_buff = 0
        high_buff = 0
        nx_phased, nsc_phased, cap_col_phased = split_by_triphase([node_scale, nsc, cap_colors], cap_probs, low_thresh, high_thresh, low_buff, high_buff)

        ux, px, fx = nx_phased 
        usubopt, psubopt, fsubopt = nsc_phased
        ucols, pcols, fcols = cap_col_phased

        trial_idx = len(phase_axes) - trial_idx - 1  # reverse order of plots
        plot_heterochromia_instance(phase_axes[trial_idx], phase_axes[trial_idx].plot, ux, usubopt, ucols, slice(len(ux)), None, False, True, "grow", "positive", fit_file_str, f"node unconstrained {int(capv)}M subopt", phase_artcfg, powerfit=False, segment_alpha=0.0)
        plot_heterochromia_instance(phase_axes[trial_idx], phase_axes[trial_idx].plot, px, psubopt, pcols, slice(len(px)), None, False, True, "grow", "positive", fit_file_str, f"node partially-constrained {int(capv)}M subopt", phase_artcfg, powerfit=False, segment_alpha=0.0)
        plot_heterochromia_instance(phase_axes[trial_idx], phase_axes[trial_idx].plot, fx, fsubopt, fcols, slice(len(fx)), None, False, True, "grow", "positive", fit_file_str, f"node fully-constrained {int(capv)}M subopt", phase_artcfg, powerfit=powerfit and len(fx) > 1, segment_alpha=0.0)


    # labelling axes
    label_size = 6
    tick_label_size = 4
    minor_tick_label_size = 4

    dim_ax.set_ylabel("Suboptimality gap", fontsize=label_size)
    phase_axes[0].set_ylabel("Suboptimality gap", fontsize=label_size)

    dim_ax.set_xlabel("Spatial dimensions ($d$)", fontsize=label_size)
    [ax.set_xlabel("Nodes ($n$)", fontsize=label_size) for ax in (node_unc_ax, node_ax)]
    [ax.set_xlabel("Nodes ($n$)", fontsize=label_size) for ax in phase_axes]

    dim_ax.set_xlim(left=0)
    dim_ax.set_ylim([0, 1.13])
    node_unc_ax.set_ylim(bottom=-0.1)
    node_ax.set_ylim(bottom=-0.5)

    dim_ax.set_xticks([2, 10, 20, 30, 40, 50, 100])
    node_unc_ax.set_xticks(list(range(5, 51, 15)))
    node_ax.set_xticks(list(range(5, 51, 15)))
    [ax.set_xticks(list(range(5, 51, 15))) for ax in phase_axes]

    [ax.tick_params(labelsize=tick_label_size) for ax in axes]
    [ax.tick_params(which="minor", labelsize=minor_tick_label_size) for ax in axes]

    [ax.tick_params(which="both", labelleft=False) for ax in phase_axes[1:]]

    # phase_axes[0].set_yticks([0.001, 0.01, 0.1, 1, 10])

    # [forceAspect(ax) for ax in phase_axes]

    # legend
    leg_lbl_size = 4

    _, ppo_d10, ppo_d20, d10, subexp_fit, d20, _, beta_line, _ = dim_ax.get_lines()
    eh = mpatches.Rectangle((0,0), 1, 1, fill=False, edgecolor="none", visible=False)
    dim_ax.legend([eh, ppo_d10, d10, eh, ppo_d20, d20, eh, eh, subexp_fit], ["$n=10$", "RL trend", "2-exc", "$n=20$", "RL trend", "2-exc", " ", " ", r"$s - \beta \propto -\psi^{-d}$"], loc="best", prop={'size': leg_lbl_size}, ncols=3, columnspacing=0.6, frameon=True) #, framealpha=1)
    
    xmin, xmax = dim_ax.get_xlim()
    xeps = 1
    dim_ax.text(xmin+xeps, d20_beta+0.03, r"2-exc $\beta_{n=20}$", color=beta_col, fontsize=tick_label_size, ha='left', va='bottom')
    dim_ax.text(xmin+xeps, d10_beta+0.05, r"2-exc $\beta_{n=10}$", color=beta_col, fontsize=tick_label_size, ha='left', va='bottom')

    _, ppo_node, unlim_line = node_unc_ax.get_lines()
    node_unc_ax.legend([ppo_node, unlim_line], [r"RL trend ($\alpha \approx 1.86$)", r"2-exc ($M=\infty$)"], loc="upper left", prop={'size': leg_lbl_size}, ncols=1, frameon=False, framealpha=1)

    rand_line = node_ax.get_lines()[1]
    unlim_line = node_ax.get_lines()[2]
    node_ax.legend([rand_line, unlim_line], ["random", r"2-exc ($M=\infty$)"], loc="upper left", prop={'size': leg_lbl_size}, ncols=1, frameon=False, framealpha=1)

    power_fit = phase_axes[-2].get_lines()[-1]
    alphas = [1.36, 1.52, 1.64, 1.81, 2.02][::-1]
    gammas = [9.0, 9.7, 9.1, 6.5, 2.1][::-1]
    [ax.legend([power_fit], [fr"$s \propto (n - {gam:.1f})^{{{alph:.2f}}}$"], loc="upper left", bbox_to_anchor=(0.0, 0.8), prop={'size': leg_lbl_size-1 if alph < 1.4 else leg_lbl_size}, ncols=1, frameon=False, framealpha=1) for alph, gam, ax in zip(alphas, gammas, phase_axes)]

    [ax.text(0.05, 0.925, f"$M={int(capv)}$", fontsize=5, ha='left', va='top', transform=ax.transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round')) for ax, capv in zip(phase_axes, ncap_ref[:-1][::-1])]

    # add labelled subplots
    [fig.add_subplot(ax) for ax in axes]

    # colorbars
    splash_cbar_ax = fig.add_axes([node_ax.get_position().x1 + 0.015, node_ax.get_position().y0 , 0.01, node_ax.get_position().y1 - node_ax.get_position().y0])
    phase_cbar_ax = fig.add_axes([phase_axes[-1].get_position().x1 + 0.015, phase_axes[-1].get_position().y0 , 0.01, phase_axes[-1].get_position().y1 - phase_axes[-1].get_position().y0])

    splash_cbar = fig.colorbar(ScalarMappable(cmap=cap_map), cax=splash_cbar_ax, location="right", fraction=1, aspect=40)
    splash_cbar_ticks = (ncap_ref - 5) / 20
    splash_cbar.ax.set_yticks(splash_cbar_ticks[:-1], [Text(0, t, f"{int(v)}") for t, v in list(zip(splash_cbar_ticks, ncap_ref))[:-1]])
    splash_cbar.set_label("Search capacity ($M$)", rotation=90, fontsize=label_size)
    splash_cbar.ax.tick_params(labelsize=tick_label_size) 

    phase_cbar = fig.colorbar(ScalarMappable(cmap=berlinColormap), cax=phase_cbar_ax, location="right", fraction=1, aspect=40)
    phase_cbar_ticks = [0, 0.25, 0.5, 0.75, 1]
    phase_cbar.ax.set_yticks(phase_cbar_ticks, [Text(0, t, f"{int(100 * t)}") for t in phase_cbar_ticks])
    phase_cbar.set_label("Max depth %", rotation=90, fontsize=label_size)
    phase_cbar.ax.tick_params(labelsize=tick_label_size) 

    # saving
    save_fig(fig, "3_joint_search_complexity_nips_2exc", "pdf")
    save_fig(fig, "3_joint_search_complexity_nips_2exc", "png")








def make_sol_res_main_plots(client):
    # setup axes
    fig = plt.figure(figsize=(5.0, 3.1))
    grid = GridSpec(2, 1, hspace=0.35) #, height_ratios=[0.4, 0.6])
    res_grid = GridSpecFromSubplotSpec(1, 3, subplot_spec=grid[0, 0], wspace=0.5)
    perf_grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=grid[1, 0], wspace=0.25)

    res_axes = [plt.Subplot(fig, res_grid[0, idx]) for idx in range(3)]
    subopt_ax = plt.Subplot(fig, perf_grid[0, 0])
    loss_ax = plt.Subplot(fig, perf_grid[0, 1], sharex=subopt_ax)

    axes = res_axes + [subopt_ax] + [loss_ax]

    # subopt_ax.tick_params("x", which="both", labelbottom=False)

    # gather instance data
    dnodes = 10

    node_exp_name = "SOL_node_scaling_drl_2"
    node_crt_loss_y, node_crt_loss_scale = get_sol_eval(client, node_exp_name, "nodes", f"eval_critic_loss_avg")

    dim_exp_name = f"SOL_{dnodes}n_dim_scaling_drl_2"
    dim_crt_loss_y, dim_crt_loss_scale = get_sol_eval(client, dim_exp_name, "dims", f"eval_critic_loss_avg")

    param_exp_name = "SOL_model_scaling_drl_2"
    param_crt_loss_y, param_crt_loss_scale = get_sol_eval(client, param_exp_name, "width", f"eval_critic_loss_avg")
    params_x, params_scale = get_sol_eval(client, param_exp_name, "width", "non_embedding_parameters")

    bind_exp_name = "EVAL_bind"
    nres, nres_scale = get_bind_eval(client, bind_exp_name, "node", f"drl_node_res.edge.mean")
    dres, dres_scale = get_bind_eval(client, bind_exp_name, f"dim{dnodes}", f"drl_dim_res.edge.mean")
    pres, pres_scale = get_bind_eval(client, bind_exp_name, "model", f"drl_model_res.edge.mean")

    nres /= 2  # converts XOR to num mismatched edges (base stat is always even)
    dres /= 2
    pres /= 2

    ###
    node_cost_y, _ = get_bind_eval(client, bind_exp_name, "node", f"drl_node_cost.mean")
    dim_cost_y, _ = get_bind_eval(client, bind_exp_name, f"dim{dnodes}", f"drl_dim_cost.mean")
    param_cost_y, _ = get_bind_eval(client, bind_exp_name, "model", f"drl_model_cost.mean")

    ncost_opt, _ = get_bind_eval(client, bind_exp_name, "node", f"opt_cost.mean")
    ncost_rand, _ = get_bind_eval(client, bind_exp_name, "node", f"rand_cost.mean")
    dcost_opt, _ = get_bind_eval(client, bind_exp_name, f"dim{dnodes}", f"opt_cost.mean")
    dcost_rand, _ = get_bind_eval(client, bind_exp_name, f"dim{dnodes}", f"rand_cost.mean")

    OPTIMAL_N20 = 3.831
    RAND_N20 = 10.428

    node_subopt = (node_cost_y - ncost_opt) # / (ncost_rand - ncost_opt)
    dim_subopt = (dim_cost_y - dcost_opt)
    param_subopt = (param_cost_y - OPTIMAL_N20)
    ###

    node_fit_slc, node_omit_slc = slice(9), None # slice(8, 10)
    dim_fit_slc, dim_omit_slc = slice(11), None
    param_fit_slc, param_omit_slc = slice(9), None #slice(8, 12)

    node_artcfg = dict(pm="o-", om="o-", fm="--", ps=3, dw=1, fw=0.75, dc="#FFB05F", fc="black", fe=None)
    dim_artcfg = dict(pm="o-", om="o-", fm="--", ps=3, dw=1, fw=0.75, dc="#B05FFF", fc="black", fe=None)
    param_artcfg = dict(pm="o-", om="o-", fm="--", ps=3, dw=1, fw=0.75, dc="#00CF68", fc="black", fe=None)

    mod_unbounded = lambda x: modcfg(x, dict(fm=":"))
    mod_bounded = lambda x: modcfg(x, dict(fm="-."))
    
    # plot instances
    powerfit = True
    fit_file_str = "3_fpc_sol_res_fits.txt"
    clear_fits(fit_file_str)

    plot_instance(res_axes[0].loglog, params_x, pres, param_fit_slc, param_omit_slc, True, True, "decay", "positive", fit_file_str, "param res", param_artcfg, powerfit=powerfit)
    plot_instance(res_axes[1].loglog, nres_scale, nres, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, "node res", mod_unbounded(node_artcfg), powerfit=powerfit)
    plot_instance(res_axes[2].plot, dres_scale, dres, dim_fit_slc, dim_omit_slc, True, True, "decay", "negative", fit_file_str, "dim10 res", mod_bounded(dim_artcfg), powerfit=powerfit)

    plot_instance(subopt_ax.loglog, nres, node_subopt, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, "node subopt v res", mod_unbounded(node_artcfg), powerfit=powerfit)
    plot_instance(subopt_ax.loglog, pres, param_subopt, param_fit_slc, param_omit_slc, True, True, "grow", "positive", fit_file_str, "param subopt v res", mod_unbounded(param_artcfg), powerfit=powerfit)
    plot_instance(subopt_ax.loglog, dres, dim_subopt, dim_fit_slc, dim_omit_slc, True, True, "grow", "positive", fit_file_str, "dim10 subopt v res", mod_unbounded(dim_artcfg), powerfit=powerfit)

    plot_instance(loss_ax.loglog, nres, node_crt_loss_y, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, "node loss v res", mod_unbounded(node_artcfg), powerfit=powerfit)
    plot_instance(loss_ax.loglog, pres, param_crt_loss_y, param_fit_slc, param_omit_slc, True, True, "grow", "positive", fit_file_str, "param loss v res", mod_unbounded(param_artcfg), powerfit=powerfit)
    plot_instance(loss_ax.loglog, dres, dim_crt_loss_y, dim_fit_slc, dim_omit_slc, True, True, "grow", "positive", fit_file_str, "dim10 loss v res", mod_unbounded(dim_artcfg), powerfit=powerfit)


    # labelling axes
    label_size = 6
    tick_label_size = 4
    minor_tick_label_size = 4

    [ax.set_ylabel("Mean edge residual", fontsize=label_size) for ax in res_axes]
    loss_ax.set_ylabel("Mean critic test loss", fontsize=label_size)
    subopt_ax.set_ylabel("Suboptimality gap", fontsize=label_size)

    res_axes[0].set_xlabel("Parameters ($N$)", fontsize=label_size)
    res_axes[1].set_xlabel("Nodes ($n$)", fontsize=label_size)
    res_axes[2].set_xlabel("Spatial dimensions ($d$)", fontsize=label_size)
    [ax.set_xlabel("Mean edge residual", fontsize=label_size) for ax in (subopt_ax, loss_ax)]

    subopt_ax.set_xlim([0.15, 7])
    subopt_ax.set_ylim(bottom=5e-4)
    loss_ax.set_ylim(bottom=4e-5)

    res_axes[2].set_xticks([2, 5, 8, 12])
    
    [ax.tick_params(labelsize=tick_label_size) for ax in axes]
    [ax.tick_params(which="minor", labelsize=minor_tick_label_size) for ax in axes]

    res_axes[0].yaxis.set_minor_formatter(mticker.ScalarFormatter())

    [forceAspect(ax) for ax in res_axes]

    # legend
    param_trend, decay_fit = res_axes[0].get_lines()
    node_trend, unbounded_fit = res_axes[1].get_lines()
    dim_trend, bounded_fit = res_axes[2].get_lines()
    fig.legend([param_trend, decay_fit, node_trend, unbounded_fit, dim_trend, bounded_fit], ["$N$ trend inputs", "Decay", "$n$ trend inputs", "Unbounded growth", "$d$ trend inputs", "Bounded growth"], loc="upper center", prop={'size': label_size}, bbox_to_anchor=(0.5, 0.0), ncols=3, frameon=False)

    # add labelled subplots
    [fig.add_subplot(ax) for ax in axes]

    # saving
    save_fig(fig, "3_fpc_sol_res", FORMAT)


def make_sol_res_supp_plots(client):  # dim20, no cuts, full loss std scaling, and seperate figure for omit isolations for subopt vs res curves
    # setup axes
    fig_rel = plt.figure(figsize=(5.5, 4.0))
    grid_rel = GridSpec(3, 1, hspace=0.45)
    res_grid = GridSpecFromSubplotSpec(1, 3, subplot_spec=grid_rel[0, 0], wspace=0.5)
    loss_grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=grid_rel[2, 0], wspace=0.25)

    res_axes = [plt.Subplot(fig_rel, res_grid[0, col_idx]) for col_idx in range(3)]
    subopt_ax = plt.Subplot(fig_rel, grid_rel[1, 0])
    stdloss_ax = plt.Subplot(fig_rel, loss_grid[0, 0])
    crtloss_ax = plt.Subplot(fig_rel, loss_grid[0, 1])

    rel_axes = res_axes + [subopt_ax, stdloss_ax, crtloss_ax]

    fig_iso = plt.figure(figsize=(5.5, 7.0))
    grid_iso = GridSpec(1, 3, wspace=0.45)
    subopt_iso_grid = GridSpecFromSubplotSpec(6, 1, subplot_spec=grid_iso[0, 0], hspace=0.3)
    stdloss_iso_grid = GridSpecFromSubplotSpec(6, 1, subplot_spec=grid_iso[0, 1], hspace=0.3)
    crtloss_iso_grid = GridSpecFromSubplotSpec(6, 1, subplot_spec=grid_iso[0, 2], hspace=0.3)

    subopt_iso_axes = [plt.Subplot(fig_iso, subopt_iso_grid[idx, 0]) for idx in range(6)]
    stdloss_iso_axes = [plt.Subplot(fig_iso, stdloss_iso_grid[idx, 0]) for idx in range(6)]
    crtloss_iso_axes = [plt.Subplot(fig_iso, crtloss_iso_grid[idx, 0]) for idx in range(6)]

    stdloss_iso_axes = stdloss_iso_axes[:3] + stdloss_iso_axes[4:]  # remove unused bc node scaling total loss std
    iso_axes = subopt_iso_axes + stdloss_iso_axes + crtloss_iso_axes

    axes = rel_axes + iso_axes

    # subopt_ax.tick_params("x", which="both", labelbottom=False)

    # gather instance data
    node_crt_loss_y, node_crt_loss_scale = get_sol_eval(client, "SOL_node_scaling_drl_2", "nodes", f"eval_critic_loss_avg")
    bc_node_crt_loss_y, _ = get_sol_eval(client, "SOL_node_scaling_il_patch", "nodes", f"eval_critic_loss_avg")

    dim10_crt_loss_y, dim10_crt_loss_scale = get_sol_eval(client, "SOL_10n_dim_scaling_drl_2", "dims", f"eval_critic_loss_avg")
    dim20_crt_loss_y, dim20_crt_loss_scale = get_sol_eval(client, "SOL_20n_dim_scaling_drl_2", "dims", f"eval_critic_loss_avg")

    param_crt_loss_y, param_crt_loss_scale = get_sol_eval(client, "SOL_model_scaling_drl_2", "width", f"eval_critic_loss_avg")
    params_x, params_scale = get_sol_eval(client, "SOL_model_scaling_drl_2", "width", "non_embedding_parameters")
    bc_param_crt_loss_y, _ = get_sol_eval(client, "SOL_model_scaling_il_patch", "width", f"eval_critic_loss_avg")
    bc_params_x, _ = get_sol_eval(client, "SOL_model_scaling_il_patch", "width", "non_embedding_parameters")

    node_totstd_loss_y, node_totstd_loss_scale = get_sol_eval(client, "SOL_node_scaling_drl_2", "nodes", f"eval_total_loss_std")
    bc_node_crtstd_loss_y, _ = get_sol_eval(client, "SOL_node_scaling_il_patch", "nodes", f"eval_critic_loss_std")  # NOTE actor loss variance doesn't trend
    dim10_totstd_loss_y, dim10_totstd_loss_scale = get_sol_eval(client, "SOL_10n_dim_scaling_drl_2", "dims", f"eval_total_loss_std")
    dim20_totstd_loss_y, dim20_totstd_loss_scale = get_sol_eval(client, "SOL_20n_dim_scaling_drl_2", "dims", f"eval_total_loss_std")
    param_totstd_loss_y, param_totstd_loss_scale = get_sol_eval(client, "SOL_model_scaling_drl_2", "width", f"eval_total_loss_std")
    bc_param_totstd_loss_y, _ = get_sol_eval(client, "SOL_model_scaling_il_patch", "width", f"eval_total_loss_std")

    bind_exp_name = "EVAL_bind"
    nres, nres_scale = get_bind_eval(client, bind_exp_name, "node", f"drl_node_res.edge.mean")
    d10res, d10res_scale = get_bind_eval(client, bind_exp_name, f"dim10", f"drl_dim_res.edge.mean")
    d20res, d20res_scale = get_bind_eval(client, bind_exp_name, f"dim20", f"drl_dim_res.edge.mean")
    pres, pres_scale = get_bind_eval(client, bind_exp_name, "model", f"drl_model_res.edge.mean")
    bc_nres, _ = get_bind_eval(client, bind_exp_name, "node", f"il_node_res.edge.mean")
    bc_pres, _ = get_bind_eval(client, bind_exp_name, "model", f"il_model_res.edge.mean")

    nres /= 2  # converts XOR to num mismatched edges (base stat is always even)
    d10res /= 2
    d20res /= 2
    pres /= 2
    bc_nres /= 2
    bc_pres /= 2

    ###
    node_cost_y, _ = get_bind_eval(client, bind_exp_name, "node", f"drl_node_cost.mean")
    dim10_cost_y, _ = get_bind_eval(client, bind_exp_name, f"dim10", f"drl_dim_cost.mean")
    dim20_cost_y, _ = get_bind_eval(client, bind_exp_name, f"dim20", f"drl_dim_cost.mean")
    param_cost_y, _ = get_bind_eval(client, bind_exp_name, "model", f"drl_model_cost.mean")
    bc_node_cost_y, _ = get_bind_eval(client, bind_exp_name, "node", f"il_node_cost.mean")
    bc_param_cost_y, _ = get_bind_eval(client, bind_exp_name, "model", f"il_model_cost.mean")

    ncost_opt, _ = get_bind_eval(client, bind_exp_name, "node", f"opt_cost.mean")
    ncost_rand, _ = get_bind_eval(client, bind_exp_name, "node", f"rand_cost.mean")
    d10cost_opt, _ = get_bind_eval(client, bind_exp_name, f"dim10", f"opt_cost.mean")
    d10cost_rand, _ = get_bind_eval(client, bind_exp_name, f"dim10", f"rand_cost.mean")
    d20cost_opt, _ = get_bind_eval(client, bind_exp_name, f"dim20", f"opt_cost.mean")
    d20cost_rand, _ = get_bind_eval(client, bind_exp_name, f"dim20", f"rand_cost.mean")

    OPTIMAL_N20 = 3.831
    RAND_N20 = 10.428

    node_subopt = (node_cost_y - ncost_opt)
    dim10_subopt = (dim10_cost_y - d10cost_opt)
    dim20_subopt = (dim20_cost_y - d20cost_opt)
    param_subopt = (param_cost_y - OPTIMAL_N20)
    bc_node_subopt = (bc_node_cost_y - ncost_opt)
    bc_param_subopt = (bc_param_cost_y - OPTIMAL_N20)
    ###

    node_fit_slc, node_omit_slc = slice(9), slice(8, 10)
    dim10_fit_slc, dim10_omit_slc = slice(11), slice(10, 17)
    dim20_fit_slc, dim20_omit_slc = slice(9), slice(8, 17)
    param_fit_slc, param_omit_slc = slice(9), slice(8, 12)
    bc_node_fit_slc, bc_node_omit_slc = slice(6), slice(5, 10)
    bc_param_fit_slc, bc_param_omit_slc = slice(5), slice(4, 12)

    base_ps = 2
    node_artcfg = dict(pm="o-", om="o-", fm=":", ps=base_ps, dw=1, fw=0.75, dc="#FFB05F", fc="black", fe=None)
    dim10_artcfg = dict(pm="o-", om="o-", fm=":", ps=base_ps, dw=1, fw=0.75, dc="#605FFF", fc="black", fe=None)
    dim20_artcfg = dict(pm="o-", om="o-", fm=":", ps=base_ps, dw=1, fw=0.75, dc="#FF5FFE", fc="black", fe=None)
    param_artcfg = dict(pm="o-", om="o-", fm=":", ps=base_ps, dw=1, fw=0.75, dc="#00CF68", fc="black", fe=None)
    bc_node_artcfg = dict(pm="o-", om="o-", fm=":", ps=base_ps, dw=1, fw=0.75, dc="#FF605F", fc="black", fe=None)
    bc_param_artcfg = dict(pm="o-", om="o-", fm=":", ps=base_ps, dw=1, fw=0.75, dc="#00CFCF", fc="black", fe=None)

    mod_decay = lambda x: modcfg(x, dict(fm="--", ps=3))
    mod_bounded = lambda x: modcfg(x, dict(fm="-.", ps=3))
    
    # plot instances
    powerfit = True
    fit_file_str = "3_fpc_sol_res_supp_fits.txt"
    clear_fits(fit_file_str)

    ## res row
    plot_instance(res_axes[0].loglog, bc_params_x, bc_pres, bc_param_fit_slc, None, True, True, "decay", "positive", fit_file_str, "bc param res", mod_decay(bc_param_artcfg), powerfit=powerfit)
    plot_instance(res_axes[1].loglog, nres_scale, bc_nres, bc_node_fit_slc, None, True, True, "grow", "positive", fit_file_str, "bc node res", modcfg(bc_node_artcfg, dict(ps=3)), powerfit=powerfit)
    plot_instance(res_axes[2].plot, d20res_scale, d20res, dim20_fit_slc, None, True, True, "decay", "negative", fit_file_str, "ppo dim20 res", mod_bounded(dim20_artcfg), powerfit=powerfit)

    ## relational subopt v res
    plot_instance(subopt_ax.loglog, nres, node_subopt, node_fit_slc, None, True, True, "grow", "positive", fit_file_str, "ppo node subopt v res", (node_artcfg), powerfit=False)
    plot_instance(subopt_ax.loglog, bc_nres, bc_node_subopt, bc_node_fit_slc, None, True, True, "grow", "positive", fit_file_str, "bc node subopt v res", (bc_node_artcfg), powerfit=False)
    plot_instance(subopt_ax.loglog, d10res, dim10_subopt, dim10_fit_slc, None, True, True, "grow", "positive", fit_file_str, "ppo dim10 subopt v res", (dim10_artcfg), powerfit=False)
    plot_instance(subopt_ax.loglog, d20res, dim20_subopt, dim20_fit_slc, None, True, True, "grow", "positive", fit_file_str, "ppo dim20 subopt v res", (dim20_artcfg), powerfit=False)
    plot_instance(subopt_ax.loglog, pres, param_subopt, param_fit_slc, None, True, True, "grow", "positive", fit_file_str, "ppo param subopt v res", (param_artcfg), powerfit=False)
    plot_instance(subopt_ax.loglog, bc_pres, bc_param_subopt, bc_param_fit_slc, None, True, True, "grow", "positive", fit_file_str, "bc param subopt v res", (bc_param_artcfg), powerfit=False)

    plot_instance(stdloss_ax.loglog, nres, node_totstd_loss_y, node_fit_slc, None, True, True, "grow", "positive", fit_file_str, "ppo node totstdloss v res", (node_artcfg), powerfit=False)
    plot_instance(stdloss_ax.loglog, d10res, dim10_totstd_loss_y, dim10_fit_slc, None, True, True, "grow", "positive", fit_file_str, "ppo dim10 totstdloss v res", (dim10_artcfg), powerfit=False)
    plot_instance(stdloss_ax.loglog, d20res, dim20_totstd_loss_y, dim20_fit_slc, None, True, True, "grow", "positive", fit_file_str, "ppo dim20 totstdloss v res", (dim20_artcfg), powerfit=False)
    plot_instance(stdloss_ax.loglog, pres, param_totstd_loss_y, param_fit_slc, None, True, True, "grow", "positive", fit_file_str, "ppo param totstdloss v res", (param_artcfg), powerfit=False)
    plot_instance(stdloss_ax.loglog, bc_pres, bc_param_totstd_loss_y, bc_param_fit_slc, None, True, True, "grow", "positive", fit_file_str, "bc param totstdloss v res", (bc_param_artcfg), powerfit=False)

    plot_instance(crtloss_ax.loglog, nres, node_crt_loss_y, node_fit_slc, None, True, True, "grow", "positive", fit_file_str, "ppo node crtloss v res", (node_artcfg), powerfit=False)
    plot_instance(crtloss_ax.loglog, bc_nres, bc_node_crt_loss_y, bc_node_fit_slc, None, True, True, "grow", "positive", fit_file_str, "bc node crtloss v res", (bc_node_artcfg), powerfit=False)
    plot_instance(crtloss_ax.loglog, d10res, dim10_crt_loss_y, dim10_fit_slc, None, True, True, "grow", "positive", fit_file_str, "ppo dim10 crtloss v res", (dim10_artcfg), powerfit=False)
    plot_instance(crtloss_ax.loglog, d20res, dim20_crt_loss_y, dim20_fit_slc, None, True, True, "grow", "positive", fit_file_str, "ppo dim20 crtloss v res", (dim20_artcfg), powerfit=False)
    plot_instance(crtloss_ax.loglog, pres, param_crt_loss_y, param_fit_slc, None, True, True, "grow", "positive", fit_file_str, "ppo param crtloss v res", (param_artcfg), powerfit=False)
    plot_instance(crtloss_ax.loglog, bc_pres, bc_param_crt_loss_y, bc_param_fit_slc, None, True, True, "grow", "positive", fit_file_str, "bc param crtloss v res", (bc_param_artcfg), powerfit=False)

    ## isolation subopt v res and fits
    plot_instance(subopt_iso_axes[2].loglog, nres, node_subopt, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, "ppo node subopt v res", (node_artcfg), powerfit=powerfit)
    plot_instance(subopt_iso_axes[3].loglog, bc_nres, bc_node_subopt, bc_node_fit_slc, bc_node_omit_slc, True, True, "grow", "positive", fit_file_str, "bc node subopt v res", (bc_node_artcfg), powerfit=powerfit)
    plot_instance(subopt_iso_axes[4].loglog, d10res, dim10_subopt, dim10_fit_slc, dim10_omit_slc, True, True, "grow", "positive", fit_file_str, "ppo dim10 subopt v res", (dim10_artcfg), powerfit=powerfit)
    plot_instance(subopt_iso_axes[5].loglog, d20res, dim20_subopt, dim20_fit_slc, dim20_omit_slc, True, True, "grow", "positive", fit_file_str, "ppo dim20 subopt v res", (dim20_artcfg), powerfit=powerfit)
    plot_instance(subopt_iso_axes[0].loglog, pres, param_subopt, param_fit_slc, param_omit_slc, True, True, "grow", "positive", fit_file_str, "ppo param subopt v res", (param_artcfg), powerfit=powerfit, powerfit_full_xbounds=True)
    plot_instance(subopt_iso_axes[1].loglog, bc_pres, bc_param_subopt, bc_param_fit_slc, bc_param_omit_slc, True, True, "grow", "positive", fit_file_str, "bc param subopt v res", (bc_param_artcfg), powerfit=powerfit, powerfit_full_xbounds=True)

    plot_instance(stdloss_iso_axes[2].loglog, nres, node_totstd_loss_y, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, "ppo node totstdloss v res", (node_artcfg), powerfit=powerfit)
    #plot_instance(stdloss_iso_axes[3].loglog, bc_nres, bc_node_crtstd_loss_y, bc_node_fit_slc, bc_node_omit_slc, True, True, "grow", "positive", fit_file_str, "bc node crtstdloss v res", (bc_node_artcfg), powerfit=powerfit)
    plot_instance(stdloss_iso_axes[3].loglog, d10res, dim10_totstd_loss_y, dim10_fit_slc, dim10_omit_slc, True, True, "grow", "positive", fit_file_str, "ppo dim10 totstdloss v res", (dim10_artcfg), powerfit=powerfit)
    plot_instance(stdloss_iso_axes[4].loglog, d20res, dim20_totstd_loss_y, dim20_fit_slc, dim20_omit_slc, True, True, "grow", "positive", fit_file_str, "ppo dim20 totstdloss v res", (dim20_artcfg), powerfit=powerfit)
    plot_instance(stdloss_iso_axes[0].loglog, pres, param_totstd_loss_y, param_fit_slc, param_omit_slc, True, True, "grow", "positive", fit_file_str, "ppo param totstdloss v res", (param_artcfg), powerfit=powerfit, powerfit_full_xbounds=True)
    plot_instance(stdloss_iso_axes[1].loglog, bc_pres, bc_param_totstd_loss_y, bc_param_fit_slc, bc_param_omit_slc, True, True, "grow", "positive", fit_file_str, "bc param totstdloss v res", (bc_param_artcfg), powerfit=powerfit, powerfit_full_xbounds=True)

    plot_instance(crtloss_iso_axes[2].loglog, nres, node_crt_loss_y, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, "ppo node crtloss v res", (node_artcfg), powerfit=powerfit)
    plot_instance(crtloss_iso_axes[3].loglog, bc_nres, bc_node_crt_loss_y, bc_node_fit_slc, bc_node_omit_slc, True, True, "grow", "positive", fit_file_str, "bc node crtloss v res", (bc_node_artcfg), powerfit=powerfit)
    plot_instance(crtloss_iso_axes[4].loglog, d10res, dim10_crt_loss_y, dim10_fit_slc, dim10_omit_slc, True, True, "grow", "positive", fit_file_str, "ppo dim10 crtloss v res", (dim10_artcfg), powerfit=powerfit)
    plot_instance(crtloss_iso_axes[5].loglog, d20res, dim20_crt_loss_y, dim20_fit_slc, dim20_omit_slc, True, True, "grow", "positive", fit_file_str, "ppo dim20 crtloss v res", (dim20_artcfg), powerfit=powerfit)
    plot_instance(crtloss_iso_axes[0].loglog, pres, param_crt_loss_y, param_fit_slc, param_omit_slc, True, True, "grow", "positive", fit_file_str, "ppo param crtloss v res", (param_artcfg), powerfit=powerfit, powerfit_full_xbounds=True)
    plot_instance(crtloss_iso_axes[1].loglog, bc_pres, bc_param_crt_loss_y, bc_param_fit_slc, bc_param_omit_slc, True, True, "grow", "positive", fit_file_str, "bc param crtloss v res", (bc_param_artcfg), powerfit=powerfit, powerfit_full_xbounds=True)

    # labelling axes
    label_size = 6
    tick_label_size = 4
    minor_tick_label_size = 4

    [ax.set_ylabel("Mean edge residual", fontsize=label_size) for ax in res_axes]
    stdloss_ax.set_ylabel("SD test loss", fontsize=label_size)
    crtloss_ax.set_ylabel("Mean critic test loss", fontsize=label_size)
    subopt_ax.set_ylabel("Suboptimality gap", fontsize=label_size)

    res_axes[0].set_xlabel("Parameters ($N$)", fontsize=label_size)
    res_axes[1].set_xlabel("Nodes ($n$)", fontsize=label_size)
    res_axes[2].set_xlabel("Spatial dimensions ($d$)", fontsize=label_size)
    [ax.set_xlabel("Mean edge residual", fontsize=label_size) for ax in (subopt_ax, crtloss_ax, stdloss_ax)]
    
    [ax.set_ylabel("Suboptimality gap", fontsize=label_size) for ax in subopt_iso_axes]
    [ax.set_ylabel("SD test loss", fontsize=label_size) for ax in stdloss_iso_axes]
    [ax.set_ylabel("Mean critic test loss", fontsize=label_size) for ax in crtloss_iso_axes]
    [ax.set_xlabel("Mean edge residual", fontsize=label_size) for ax in iso_axes]

    subopt_ax.set_xlim([2e-2, 7])

    res_axes[2].set_xticks([2, 4, 6, 8, 10])

    [ax.set_xticks([1.2, 1.6, 2.0, 2.4], minor=True) for ax in (subopt_iso_axes[1], stdloss_iso_axes[1], crtloss_iso_axes[1])]
    [ax.set_xticks([2.0, 4.0, 6.0], minor=True) for ax in (subopt_iso_axes[-1], stdloss_iso_axes[-1], crtloss_iso_axes[-1])]

    [ax.tick_params(labelsize=tick_label_size) for ax in axes]
    [ax.tick_params(which="minor", labelsize=minor_tick_label_size) for ax in axes]

    res_axes[0].yaxis.set_minor_formatter(mticker.ScalarFormatter())
    
    [ax.xaxis.set_minor_formatter(mticker.ScalarFormatter()) for ax in subopt_iso_axes[:2] + stdloss_iso_axes[:2] + crtloss_iso_axes[:2]]
    [ax.xaxis.set_minor_formatter(mticker.ScalarFormatter()) for ax in (subopt_iso_axes[-1], stdloss_iso_axes[-1], crtloss_iso_axes[-1])]

    [forceAspect(ax, aspect=1.5) for ax in res_axes]
    [forceAspect(ax, aspect=1.75) for ax in iso_axes]

    # legend
    _, decay_fit = res_axes[0].get_lines()
    _, unbounded_fit = res_axes[1].get_lines()
    _, bounded_fit = res_axes[2].get_lines()
    ppo_node, bc_node, ppo_dim10, ppo_dim20, ppo_param, bc_param = subopt_ax.get_lines()
    fig_rel.legend([ppo_param, bc_param, decay_fit, ppo_node, bc_node, unbounded_fit, ppo_dim10, ppo_dim20, bounded_fit], ["PPO parameter scaling", "BC parameter scaling", "Decay", "PPO node scaling", "BC node scaling", "Unbounded growth", "PPO dimension scaling ($n=10$)", "PPO dimension scaling ($n=20$)", "Bounded growth"], loc="upper center", prop={'size': label_size}, bbox_to_anchor=(0.5, 0.01), ncols=3, frameon=False)

    res_axes[0].text(0.95, 0.925, "BC", fontsize=label_size, ha='right', va='top', transform=res_axes[0].transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))
    res_axes[1].text(0.95, 0.075, "BC", fontsize=label_size, ha='right', va='bottom', transform=res_axes[1].transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))
    res_axes[2].text(0.95, 0.075, "$n=20$", fontsize=label_size, ha='right', va='bottom', transform=res_axes[2].transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))

    white_handle = mpatches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none', visible=False)
    fig_iso.legend([ppo_param, bc_param, white_handle, ppo_node, bc_node, unbounded_fit, ppo_dim10, ppo_dim20, white_handle], ["PPO parameter scaling", "BC parameter scaling", " ", "PPO node scaling", "BC node scaling", "Unbounded growth", "PPO dimension scaling ($n=10$)", "PPO dimension scaling ($n=20$)", " "], loc="upper center", prop={'size': label_size}, bbox_to_anchor=(0.5, 0.06), ncols=3, frameon=False)

    # add labelled subplots
    [fig_rel.add_subplot(ax) for ax in rel_axes]
    [fig_iso.add_subplot(ax) for ax in iso_axes]

    # saving
    save_fig(fig_rel, "3_fpc_sol_res_supp_rel", "png")
    save_fig(fig_iso, "3_fpc_sol_res_supp_iso", "png")

    save_fig(fig_rel, "3_fpc_sol_res_supp_rel", "pdf")
    save_fig(fig_iso, "3_fpc_sol_res_supp_iso", "pdf")
    

def make_sol_res_main_plots_nips(client):  # abandoned for nips, no room and too complicated to explain, circle back for thesis
    # setup axes
    fig = plt.figure(figsize=(5.5, 5.0))
    grid = GridSpec(2, 1, hspace=0.25, height_ratios=(1.0, 1.0))
    # deep_grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=grid[0, 0], wspace=0.25)
    subopt_ax = plt.Subplot(fig, grid[0, 0])
    search_ax = plt.Subplot(fig, grid[1, 0], sharex=subopt_ax, sharey=subopt_ax)
    # loss_ax = plt.Subplot(fig, deep_grid[0, 1])

    # raw_grid = GridSpecFromSubplotSpec(1, 3, subplot_spec=grid[0, 0], wspace=0.25)
    # raw_axes = [plt.Subplot(fig, raw_grid[0, col_idx]) for col_idx in range(3)]

    # sparam_ax, sdim_ax, snode_ax = raw_axes
    axes = [subopt_ax, search_ax]   # loss_ax + raw_axes
    
    # [ax.axhline(0, color="black", linewidth=plt.rcParams['xtick.major.width']) for ax in res_axes[1:]]
    # subopt_ax.tick_params("x", which="both", labelbottom=False)

    # gather instance data
    ppo_params_x, params_scale = get_sol_eval(client, "SOL_model_scaling_drl_2", "width", "non_embedding_parameters")
    bc_params_x, _ = get_sol_eval(client, "SOL_model_scaling_il_patch", "width", "non_embedding_parameters")

    bind_exp_name = "EVAL_bind"
    nres, nres_scale = get_bind_eval(client, bind_exp_name, "node", f"drl_node_res.edge.mean")
    d10res, d10res_scale = get_bind_eval(client, bind_exp_name, f"dim10", f"drl_dim_res.edge.mean")
    d20res, d20res_scale = get_bind_eval(client, bind_exp_name, f"dim20", f"drl_dim_res.edge.mean")
    pres, pres_scale = get_bind_eval(client, bind_exp_name, "model", f"drl_model_res.edge.mean")
    bc_nres, _ = get_bind_eval(client, bind_exp_name, "node", f"il_node_res.edge.mean")
    bc_pres, _ = get_bind_eval(client, bind_exp_name, "model", f"il_model_res.edge.mean")

    nres /= 2  # converts XOR to num mismatched edges (base stat is always even)
    d10res /= 2
    d20res /= 2
    pres /= 2
    bc_nres /= 2
    bc_pres /= 2

    ###
    node_cost_y, _ = get_bind_eval(client, bind_exp_name, "node", f"drl_node_cost.mean")
    dim10_cost_y, _ = get_bind_eval(client, bind_exp_name, f"dim10", f"drl_dim_cost.mean")
    dim20_cost_y, _ = get_bind_eval(client, bind_exp_name, f"dim20", f"drl_dim_cost.mean")
    param_cost_y, _ = get_bind_eval(client, bind_exp_name, "model", f"drl_model_cost.mean")
    bc_node_cost_y, _ = get_bind_eval(client, bind_exp_name, "node", f"il_node_cost.mean")
    bc_param_cost_y, _ = get_bind_eval(client, bind_exp_name, "model", f"il_model_cost.mean")

    ncost_opt, _ = get_bind_eval(client, bind_exp_name, "node", f"opt_cost.mean")
    d10cost_opt, _ = get_bind_eval(client, bind_exp_name, f"dim10", f"opt_cost.mean")
    d20cost_opt, _ = get_bind_eval(client, bind_exp_name, f"dim20", f"opt_cost.mean")

    OPTIMAL_N20 = 3.831

    node_subopt = (node_cost_y - ncost_opt)
    dim10_subopt = (dim10_cost_y - d10cost_opt)
    dim20_subopt = (dim20_cost_y - d20cost_opt)
    param_subopt = (param_cost_y - OPTIMAL_N20)
    bc_node_subopt = (bc_node_cost_y - ncost_opt)
    bc_param_subopt = (bc_param_cost_y - OPTIMAL_N20)
    ###

    # node_crt_loss_y, node_crt_loss_scale = get_sol_eval(client, "SOL_node_scaling_drl_2", "nodes", f"eval_critic_loss_avg")
    # bc_node_crt_loss_y, _ = get_sol_eval(client, "SOL_node_scaling_il_patch", "nodes", f"eval_critic_loss_avg")

    # dim10_crt_loss_y, dim10_crt_loss_scale = get_sol_eval(client, "SOL_10n_dim_scaling_drl_2", "dims", f"eval_critic_loss_avg")
    # dim20_crt_loss_y, dim20_crt_loss_scale = get_sol_eval(client, "SOL_20n_dim_scaling_drl_2", "dims", f"eval_critic_loss_avg")

    # param_crt_loss_y, param_crt_loss_scale = get_sol_eval(client, "SOL_model_scaling_drl_2", "width", f"eval_critic_loss_avg")
    # bc_param_crt_loss_y, _ = get_sol_eval(client, "SOL_model_scaling_il_patch", "width", f"eval_critic_loss_avg")
    ###
    d10costs_2opt, _ = get_bind_eval(client, bind_exp_name, f"dim10", f"_2opt_cost.mean")
    d10res_2opt, _ = get_bind_eval(client, bind_exp_name, f"dim10", f"_2opt_res.edge.mean")
    d20costs_2opt, _ = get_bind_eval(client, bind_exp_name, f"dim20", f"_2opt_cost.mean")
    d20res_2opt, _ = get_bind_eval(client, bind_exp_name, f"dim20", f"_2opt_res.edge.mean")

    d10costs_2exc, _ = get_bind_eval(client, bind_exp_name, f"dim10", f"_2swap_cost.mean")
    d10res_2exc, _ = get_bind_eval(client, bind_exp_name, f"dim10", f"_2swap_res.edge.mean")
    d20costs_2exc, _ = get_bind_eval(client, bind_exp_name, f"dim20", f"_2swap_cost.mean")
    d20res_2exc, _ = get_bind_eval(client, bind_exp_name, f"dim20", f"_2swap_res.edge.mean")

    ncosts_2opt, _ = get_bind_eval(client, bind_exp_name, "node", f"_2opt_cost.mean")
    nres_2opt, _ = get_bind_eval(client, bind_exp_name, "node", f"_2opt_res.edge.mean")

    ncosts_2exc, _ = get_bind_eval(client, bind_exp_name, "node", f"_2swap_cost.mean")
    nres_2exc, _ = get_bind_eval(client, bind_exp_name, "node", f"_2swap_res.edge.mean")


    nres_2opt /= 2  # converts XOR to num mismatched edges (base stat is always even)
    nres_2exc /= 2
    d10res_2opt /= 2
    d20res_2opt /= 2
    d10res_2exc /= 2
    d20res_2exc /= 2

    # 2D local search experiments (2opt and 2exchange) both have corrupted, inflated results in EVAL_bind for 10-node version, where 20-node version does not, and I'm not sure why
    # code below substitutes values from earlier evals done on same proxy opt problems, where issue does not show up
    d10costs_2opt[0] = 2.8857
    d10costs_2exc[0] = 2.9238

    node_subopt_2opt = (ncosts_2opt - ncost_opt)
    node_subopt_2exc = (ncosts_2exc - ncost_opt)

    d10_subopt_2opt = (d10costs_2opt - d10cost_opt)
    d10_subopt_2exc = (d10costs_2exc - d10cost_opt)
    d20_subopt_2opt = (d20costs_2opt - d20cost_opt)
    d20_subopt_2exc = (d20costs_2exc - d20cost_opt)
    ###

    node_fit_slc, node_omit_slc = slice(9), None
    dim10_fit_slc, dim10_omit_slc = slice(11), None
    dim20_fit_slc, dim20_omit_slc = slice(9), None #slice(4, 9)
    param_fit_slc, param_omit_slc = slice(9), None
    bc_node_fit_slc, bc_node_omit_slc = slice(6), None
    bc_param_fit_slc, bc_param_omit_slc = slice(5), None

    base_ps = 3
    base_fw = 1.0
    node_artcfg = dict(pm="o-", om="o-", fm=":", ps=base_ps, dw=1, fw=base_fw, dc="#FFB05F", fc="#FFB05F", fe=None)
    dim10_artcfg = dict(pm="o-", om="o-", fm=":", ps=base_ps, dw=1, fw=base_fw, dc="#605FFF", fc="#605FFF", fe=None)
    dim20_artcfg = dict(pm="o-", om="o-", fm=":", ps=base_ps, dw=1, fw=base_fw, dc="#FF5FFE", fc="#FF5FFE", fe=None)
    param_artcfg = dict(pm="o-", om="o-", fm=":", ps=base_ps, dw=1, fw=base_fw, dc="#00CF68", fc="#00CF68", fe=None)
    bc_node_artcfg = dict(pm="s-", om="s-", fm=":", ps=base_ps, dw=1, fw=base_fw, dc="#FF605F", fc="#FF605F", fe=None)
    bc_param_artcfg = dict(pm="s-", om="s-", fm=":", ps=base_ps, dw=1, fw=base_fw, dc="#00CFCF", fc="#00CFCF", fe=None)

    # mod_decay = lambda x: modcfg(x, dict(fm="--", ps=3))
    # mod_bounded = lambda x: modcfg(x, dict(fm="-.", fw=0.5, ps=3))
    # mod_expbounded = lambda x: modcfg(x, dict(fm="-", fw=0.5, ps=3))
    mod_search = lambda x: modcfg(x, dict(pm="8-"))
    
    # plot instances
    powerfit = True
    fit_file_str = "3_joint_residual_nips_fits.txt"
    clear_fits(fit_file_str)

    ## subopt v res
    def plot_instance_with_tail(plt_fn, scale_x, perf_y, fit_slc, omit_slc, use_c0, use_c1, mode, sign, fit_file_str, id_str, artcfg, plot_data=True, powerfit=True, plot_powerfit=True, powerfit_zorder=3, omit_zorder=1, powerfit_full_xbounds=False, near_linear=False, expfit=False, data_zorder=2):
        """
        show only last omit experiment, ignoring omits in-between fit and that tail
        """
        new_scale_x = np.concatenate([scale_x[fit_slc], np.expand_dims(np.asarray(scale_x[-1]), 0)])
        new_perf_y = np.concatenate([perf_y[fit_slc], np.expand_dims(np.asarray(perf_y[-1]), 0)])
        new_omit_slc = slice(len(new_scale_x)-2, len(new_scale_x))
        plot_instance(plt_fn, new_scale_x, new_perf_y, fit_slc, new_omit_slc, use_c0, use_c1, mode, sign, fit_file_str, id_str, artcfg, plot_data=plot_data, powerfit=powerfit, plot_powerfit=plot_powerfit, powerfit_zorder=powerfit_zorder, omit_zorder=omit_zorder, powerfit_full_xbounds=powerfit_full_xbounds, near_linear=near_linear, expfit=expfit, data_zorder=data_zorder)


    plot_instance_with_tail(subopt_ax.loglog, nres, node_subopt, node_fit_slc, None, False, False, "grow", "positive", fit_file_str, "ppo node subopt v res", (node_artcfg), powerfit=powerfit, powerfit_zorder=0)
    plot_instance_with_tail(subopt_ax.loglog, bc_nres, bc_node_subopt, bc_node_fit_slc, None, False, False, "grow", "positive", fit_file_str, "bc node subopt v res", (bc_node_artcfg), powerfit=powerfit, powerfit_zorder=0)
    plot_instance_with_tail(subopt_ax.loglog, d10res, dim10_subopt, dim10_fit_slc, None, False, False, "grow", "positive", fit_file_str, "ppo dim10 subopt v res", (dim10_artcfg), powerfit=powerfit, powerfit_zorder=0)
    plot_instance_with_tail(subopt_ax.loglog, d20res, dim20_subopt, dim20_fit_slc, None, False, False, "grow", "positive", fit_file_str, "ppo dim20 subopt v res", (dim20_artcfg), powerfit=powerfit, powerfit_zorder=0)
    plot_instance_with_tail(subopt_ax.loglog, pres, param_subopt, param_fit_slc, None, False, False, "grow", "positive", fit_file_str, "ppo param subopt v res", (param_artcfg), powerfit=powerfit, data_zorder=3, omit_zorder=2, powerfit_zorder=0)
    plot_instance_with_tail(subopt_ax.loglog, bc_pres, bc_param_subopt, bc_param_fit_slc, None, False, False, "grow", "positive", fit_file_str, "bc param subopt v res", (bc_param_artcfg), powerfit=powerfit, data_zorder=3, omit_zorder=2, powerfit_zorder=0)

    plot_instance(search_ax.loglog, nres_2opt, node_subopt_2opt, slice(len(nres_2opt)), None, False, False, "grow", "positive", fit_file_str, "2opt node subopt v res", mod_search(node_artcfg), powerfit=powerfit, powerfit_zorder=0)
    plot_instance(search_ax.loglog, d10res_2opt, d10_subopt_2opt, slice(len(d10res_2opt)), None, False, False, "grow", "positive", fit_file_str, "2opt dim10 subopt v res", mod_search(dim10_artcfg), powerfit=powerfit, powerfit_zorder=0)
    plot_instance(search_ax.loglog, d20res_2opt, d20_subopt_2opt, slice(len(d20res_2opt)), None, False, False, "grow", "positive", fit_file_str, "2opt dim20 subopt v res", mod_search(dim20_artcfg), powerfit=powerfit, powerfit_zorder=0)



    # plot_instance_with_tail(subopt_ax.loglog, node_subopt_2opt, node_subopt, node_fit_slc, None, False, False, "grow", "positive", fit_file_str, "ppo node subopt v res", (node_artcfg), powerfit=powerfit, powerfit_zorder=0)
    # plot_instance_with_tail(subopt_ax.loglog, node_subopt_2opt, bc_node_subopt, bc_node_fit_slc, None, False, False, "grow", "positive", fit_file_str, "bc node subopt v res", (bc_node_artcfg), powerfit=powerfit, powerfit_zorder=0)
    # plot_instance_with_tail(subopt_ax.loglog, d10_subopt_2opt, dim10_subopt, dim10_fit_slc, None, False, False, "grow", "positive", fit_file_str, "ppo dim10 subopt v res", (dim10_artcfg), powerfit=powerfit, powerfit_zorder=0)
    # plot_instance(subopt_ax.loglog, d20_subopt_2opt, dim20_subopt, dim20_fit_slc, None, False, False, "grow", "positive", fit_file_str, "ppo dim20 subopt v res", (dim20_artcfg), powerfit=powerfit, powerfit_zorder=0)
    # # plot_instance_with_tail(subopt_ax.loglog, pres, param_subopt, param_fit_slc, None, False, False, "grow", "positive", fit_file_str, "ppo param subopt v res", (param_artcfg), powerfit=powerfit, data_zorder=3, omit_zorder=2, powerfit_zorder=0)
    # # plot_instance_with_tail(subopt_ax.loglog, bc_pres, bc_param_subopt, bc_param_fit_slc, None, False, False, "grow", "positive", fit_file_str, "bc param subopt v res", (bc_param_artcfg), powerfit=powerfit, data_zorder=3, omit_zorder=2, powerfit_zorder=0)


    # plot_instance_with_tail(search_ax.loglog, nres_2opt, node_subopt, node_fit_slc, None, False, False, "grow", "positive", fit_file_str, "ppo node subopt v res", (node_artcfg), powerfit=powerfit, powerfit_zorder=0)
    # plot_instance_with_tail(search_ax.loglog, nres_2opt, bc_node_subopt, bc_node_fit_slc, None, False, False, "grow", "positive", fit_file_str, "bc node subopt v res", (bc_node_artcfg), powerfit=powerfit, powerfit_zorder=0)
    # plot_instance_with_tail(search_ax.loglog, d10res_2opt, dim10_subopt, dim10_fit_slc, None, False, False, "grow", "positive", fit_file_str, "ppo dim10 subopt v res", (dim10_artcfg), powerfit=powerfit, powerfit_zorder=0)
    # plot_instance(search_ax.loglog, d20res_2opt, dim20_subopt, dim20_fit_slc, None, False, False, "grow", "positive", fit_file_str, "ppo dim20 subopt v res", (dim20_artcfg), powerfit=powerfit, powerfit_zorder=0)
    # # plot_instance_with_tail(subopt_ax.loglog, pres, param_subopt, param_fit_slc, None, False, False, "grow", "positive", fit_file_str, "ppo param subopt v res", (param_artcfg), powerfit=powerfit, data_zorder=3, omit_zorder=2, powerfit_zorder=0)
    # # plot_instance_with_tail(subopt_ax.loglog, bc_pres, bc_param_subopt, bc_param_fit_slc, None, False, False, "grow", "positive", fit_file_str, "bc param subopt v res", (bc_param_artcfg), powerfit=powerfit, data_zorder=3, omit_zorder=2, powerfit_zorder=0)



    ## critic loss vs res
    # plot_instance_with_tail(loss_ax.loglog, nres, node_crt_loss_y, node_fit_slc, None, False, False, "grow", "positive", fit_file_str, "ppo node crtloss v res", (node_artcfg), powerfit=powerfit, powerfit_zorder=0)
    # plot_instance_with_tail(loss_ax.loglog, bc_nres, bc_node_crt_loss_y, bc_node_fit_slc, None, False, False, "grow", "positive", fit_file_str, "bc node crtloss v res", (bc_node_artcfg), powerfit=powerfit, powerfit_zorder=0)
    # plot_instance_with_tail(loss_ax.loglog, d10res, dim10_crt_loss_y, dim10_fit_slc, None, False, False, "grow", "positive", fit_file_str, "ppo dim10 crtloss v res", (dim10_artcfg), powerfit=powerfit, powerfit_zorder=0)
    # plot_instance_with_tail(loss_ax.loglog, d20res, dim20_crt_loss_y, dim20_fit_slc, None, False, False, "grow", "positive", fit_file_str, "ppo dim20 crtloss v res", (dim20_artcfg), powerfit=powerfit, powerfit_zorder=0)
    # plot_instance_with_tail(loss_ax.loglog, pres, param_crt_loss_y, param_fit_slc, None, False, False, "grow", "positive", fit_file_str, "ppo param crtloss v res", (param_artcfg), powerfit=powerfit, data_zorder=3, omit_zorder=2, powerfit_zorder=0)
    # plot_instance_with_tail(loss_ax.loglog, bc_pres, bc_param_crt_loss_y, bc_param_fit_slc, None, False, False, "grow", "positive", fit_file_str, "bc param crtloss v res", (bc_param_artcfg), powerfit=powerfit, data_zorder=3, omit_zorder=2, powerfit_zorder=0)


    ## raw row
    # plot_instance(sdim_ax.plot, d10res_scale, d10_subopt_2opt, slice(len(d10res_scale)), None, True, False, "decay", "negative", fit_file_str, f"2opt dim10 subopt v res", modcfg(dim10_artcfg, dict(fc="black")), powerfit=powerfit, expfit=True)
    # plot_instance(sdim_ax.plot, d10res_scale, d10_subopt_2exc, slice(len(d10res_scale)), None, True, False, "decay", "negative", fit_file_str, f"2exc dim10 subopt v res", modcfg(dim10_artcfg, dict(fc="black")), powerfit=powerfit, expfit=True)
    # plot_instance(sdim_ax.plot, d20res_scale, d20_subopt_2opt, slice(len(d20res_scale)), None, True, False, "decay", "negative", fit_file_str, f"2opt dim20 subopt v res", modcfg(dim20_artcfg, dict(fc="black")), powerfit=powerfit, expfit=True)
    # plot_instance(sdim_ax.plot, d20res_scale, d20_subopt_2exc, slice(len(d20res_scale)), None, True, False, "decay", "negative", fit_file_str, f"2exc dim20 subopt v res", modcfg(dim20_artcfg, dict(fc="black")), powerfit=powerfit, expfit=True)

    # plot_instance(snode_ax.loglog, nres_2opt, node_subopt_2opt, slice(len(nres_2opt)), None, False, False, "grow", "positive", fit_file_str, "2opt node subopt v res", node_artcfg, powerfit=powerfit, powerfit_full_xbounds=True, near_linear=False)
    # plot_instance(snode_ax.loglog, nres_2exc, node_subopt_2exc, slice(len(nres_2exc)), None, False, False, "grow", "positive", fit_file_str, "2exc node subopt v res", node_artcfg, powerfit=powerfit, powerfit_full_xbounds=True, near_linear=False)



    
    # plot_instance(res_axes[0].loglog, bc_params_x, bc_pres, bc_param_fit_slc, None, False, False, "decay", "positive", fit_file_str, "bc param res", mod_decay(bc_param_artcfg), powerfit=powerfit)
    # plot_instance(res_axes[0].loglog, ppo_params_x, pres, param_fit_slc, param_omit_slc, False, False, "decay", "positive", fit_file_str, "ppo param res", mod_decay(param_artcfg), powerfit=powerfit)
    
    # plot_instance(res_axes[1].plot, nres_scale, bc_nres, bc_node_fit_slc, None, False, True, "grow", "positive", fit_file_str, "bc node res", modcfg(bc_node_artcfg, dict(ps=3)), powerfit=powerfit)
    # plot_instance(res_axes[1].plot, nres_scale, nres, node_fit_slc, node_omit_slc, False, True, "grow", "positive", fit_file_str, "ppo node res", modcfg(node_artcfg, dict(ps=3)), powerfit=powerfit)

    # plot_instance(res_axes[2].plot, d10res_scale, d10res, dim10_fit_slc, dim10_omit_slc, False, True, "grow", "positive", fit_file_str, "ppo dim10 res (power growth)", modcfg(dim10_artcfg, dict(fw=0.5)), powerfit=powerfit, plot_data=False, plot_powerfit=False)
    # plot_instance(res_axes[2].plot, d10res_scale, d10res, dim10_fit_slc, dim10_omit_slc, True, False, "decay", "negative", fit_file_str, "ppo dim10 res (power decay)", mod_bounded(dim10_artcfg), powerfit=powerfit, plot_data=False, plot_powerfit=False)
    # d10_popt = plot_instance(res_axes[2].plot, d10res_scale, d10res, dim10_fit_slc, dim10_omit_slc, True, False, "decay", "negative", fit_file_str, "ppo dim10 res (exp decay)", mod_expbounded(dim10_artcfg), powerfit=powerfit, expfit=True)

    # plot_instance(res_axes[2].plot, d20res_scale, d20res, dim20_fit_slc, dim20_omit_slc, False, True, "grow", "positive", fit_file_str, "ppo dim20 res (power growth)", modcfg(dim20_artcfg, dict(fw=0.5)), powerfit=powerfit, plot_data=False)
    # plot_instance(res_axes[2].plot, d20res_scale, d20res, dim20_fit_slc, dim20_omit_slc, True, False, "decay", "negative", fit_file_str, "ppo dim20 res (power decay)", mod_bounded(dim20_artcfg), powerfit=powerfit, plot_data=False)
    # d20_popt = plot_instance(res_axes[2].plot, d20res_scale, d20res, dim20_fit_slc, dim20_omit_slc, True, False, "decay", "negative", fit_file_str, "ppo dim20 res (exp decay)", mod_expbounded(dim20_artcfg), powerfit=powerfit, expfit=True)


    # labelling axes
    label_size = 6
    tick_label_size = 4
    minor_tick_label_size = 4

    subopt_ax.set_ylabel("Suboptimality gap", fontsize=label_size)
    search_ax.set_ylabel("Suboptimality gap", fontsize=label_size)

    # sparam_ax.set_title("Parameters ($N$)", fontsize=label_size, fontweight="bold")
    # snode_ax.set_title("Nodes ($n$)", fontsize=label_size, fontweight="bold")
    # sdim_ax.set_title("Spatial dimensions ($d$)", fontsize=label_size, fontweight="bold")

    [ax.set_xlabel("Mean edge residual", fontsize=label_size) for ax in axes]

    # res_axes[1].set_ylim(bottom=-0.003)
    # res_axes[2].set_ylim(bottom=-0.012)

    # subopt_ax.set_xlim([0.2, 10])
    # subopt_ax.set_ylim(bottom=5e-4)

    # res_axes[1].set_xticks(list(range(5, 46, 10)))
    # res_axes[2].set_xticks([2, 4, 6, 8, 10, 12])

    [ax.tick_params(labelsize=tick_label_size) for ax in axes]
    [ax.tick_params(which="minor", labelsize=minor_tick_label_size) for ax in axes]

    # res_axes[0].yaxis.set_minor_formatter(mticker.ScalarFormatter())
    
    # [forceAspect(ax, aspect=1.5) for ax in res_axes]

    # legend
    # _, decay_fit = res_axes[0].get_lines()
    # _, unbounded_fit = res_axes[1].get_lines()
    # _, bounded_fit = res_axes[2].get_lines()
    # ppo_node, bc_node, ppo_dim10, ppo_dim20, ppo_param, bc_param = subopt_ax.get_lines()
    # fig.legend([ppo_param, bc_param, decay_fit, ppo_node, bc_node, unbounded_fit, ppo_dim10, ppo_dim20, bounded_fit], ["PPO parameter scaling", "BC parameter scaling", "Decay", "PPO node scaling", "BC node scaling", "Unbounded growth", "PPO dimension scaling ($n=10$)", "PPO dimension scaling ($n=20$)", "Bounded growth"], loc="upper center", prop={'size': label_size}, bbox_to_anchor=(0.5, 0.01), ncols=3, frameon=False)

    # res_axes[0].text(0.95, 0.925, "BC", fontsize=label_size, ha='right', va='top', transform=res_axes[0].transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))
    # res_axes[1].text(0.95, 0.075, "BC", fontsize=label_size, ha='right', va='bottom', transform=res_axes[1].transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))
    # res_axes[2].text(0.95, 0.075, "$n=20$", fontsize=label_size, ha='right', va='bottom', transform=res_axes[2].transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))

    rlnode, _, _, sftnode, _, _, d10, _, _, d20, _, _, rlparam, _, _, sftparam, _, _ = subopt_ax.get_lines()
    subopt_ax.legend([rlparam, sftparam, rlnode, sftnode, d10, d20], ["$N$-RL", "$N$-SFT", "$n$-RL", "$n$-SFT", "$d$-RL ($n=10$)", "$d$-RL ($n=20$)"], loc="upper left", prop={'size': 5}, ncols=1, frameon=True)  #bbox_to_anchor=(0.5, 0.01)

    snode, _, sd10, _, sd20, _, = search_ax.get_lines()
    search_ax.legend([snode, sd10, sd20], ["$n$-2opt", "$d$-2opt ($n=10$)", "$d$-2opt ($n=20$)"], loc="upper left", prop={'size': 5}, ncols=1, frameon=True)  #bbox_to_anchor=(0.5, 0.01)

    # add labelled subplots
    [fig.add_subplot(ax) for ax in axes]

    # saving
    save_fig(fig, "3_joint_residual_nips", "png")
    save_fig(fig, "3_joint_residual_nips", "pdf")



def make_joint_res_iso_plots(client):  # residual == natural performance mentric --> isolation plots
    # setup axes
    fig_iso = plt.figure(figsize=(5.5, 5.0))
    grid_iso = GridSpec(3, 2, wspace=0.3, hspace=0.35)

    rlp_ax = plt.Subplot(fig_iso, grid_iso[0, 0])
    sfp_ax = plt.Subplot(fig_iso, grid_iso[0, 1])
    rln_ax = plt.Subplot(fig_iso, grid_iso[1, 0])
    sfn_ax = plt.Subplot(fig_iso, grid_iso[1, 1])
    d10_ax = plt.Subplot(fig_iso, grid_iso[2, 0])
    d20_ax = plt.Subplot(fig_iso, grid_iso[2, 1])

    axes = [rlp_ax, sfp_ax, rln_ax, sfn_ax, d10_ax, d20_ax]

    [ax.axhline(0, color="black", linewidth=plt.rcParams['xtick.major.width']) for ax in (rln_ax, sfn_ax)]


    # subopt_ax.tick_params("x", which="both", labelbottom=False)

    # gather instance data
    params_x, params_scale = get_sol_eval(client, "SOL_model_scaling_drl_2", "width", "non_embedding_parameters")
    bc_params_x, _ = get_sol_eval(client, "SOL_model_scaling_il_patch", "width", "non_embedding_parameters")

    bind_exp_name = "EVAL_bind"
    nres, nres_scale = get_bind_eval(client, bind_exp_name, "node", f"drl_node_res.edge.mean")
    d10res, d10res_scale = get_bind_eval(client, bind_exp_name, f"dim10", f"drl_dim_res.edge.mean")
    d20res, d20res_scale = get_bind_eval(client, bind_exp_name, f"dim20", f"drl_dim_res.edge.mean")
    pres, pres_scale = get_bind_eval(client, bind_exp_name, "model", f"drl_model_res.edge.mean")
    bc_nres, _ = get_bind_eval(client, bind_exp_name, "node", f"il_node_res.edge.mean")
    bc_pres, _ = get_bind_eval(client, bind_exp_name, "model", f"il_model_res.edge.mean")

    nres /= 2  # converts XOR to num mismatched edges (base stat is always even)
    d10res /= 2
    d20res /= 2
    pres /= 2
    bc_nres /= 2
    bc_pres /= 2

    ###
    node_cost_y, _ = get_bind_eval(client, bind_exp_name, "node", f"drl_node_cost.mean")
    dim10_cost_y, _ = get_bind_eval(client, bind_exp_name, f"dim10", f"drl_dim_cost.mean")
    dim20_cost_y, _ = get_bind_eval(client, bind_exp_name, f"dim20", f"drl_dim_cost.mean")
    param_cost_y, _ = get_bind_eval(client, bind_exp_name, "model", f"drl_model_cost.mean")
    bc_node_cost_y, _ = get_bind_eval(client, bind_exp_name, "node", f"il_node_cost.mean")
    bc_param_cost_y, _ = get_bind_eval(client, bind_exp_name, "model", f"il_model_cost.mean")

    ncost_opt, _ = get_bind_eval(client, bind_exp_name, "node", f"opt_cost.mean")
    # ncost_rand, _ = get_bind_eval(client, bind_exp_name, "node", f"rand_cost.mean")
    d10cost_opt, _ = get_bind_eval(client, bind_exp_name, f"dim10", f"opt_cost.mean")
    # d10cost_rand, _ = get_bind_eval(client, bind_exp_name, f"dim10", f"rand_cost.mean")
    d20cost_opt, _ = get_bind_eval(client, bind_exp_name, f"dim20", f"opt_cost.mean")
    # d20cost_rand, _ = get_bind_eval(client, bind_exp_name, f"dim20", f"rand_cost.mean")

    OPTIMAL_N20 = 3.831
    RAND_N20 = 10.428

    node_subopt = (node_cost_y - ncost_opt)
    dim10_subopt = (dim10_cost_y - d10cost_opt)
    dim20_subopt = (dim20_cost_y - d20cost_opt)
    param_subopt = (param_cost_y - OPTIMAL_N20)
    bc_node_subopt = (bc_node_cost_y - ncost_opt)
    bc_param_subopt = (bc_param_cost_y - OPTIMAL_N20)
    ###

    node_fit_slc, node_omit_slc = slice(9), slice(8, 10)
    dim10_fit_slc, dim10_omit_slc = slice(11), slice(10, 13) # slice(10, 17)
    dim20_fit_slc, dim20_omit_slc = slice(9), slice(8, 13) # slice(8, 17)
    param_fit_slc, param_omit_slc = slice(9), slice(8, 12)
    bc_node_fit_slc, bc_node_omit_slc = slice(6), slice(5, 10)
    bc_param_fit_slc, bc_param_omit_slc = slice(5), slice(4, 12)

    base_ps = 3
    base_fw = 1.0
    node_artcfg = dict(pm="o-", om="o-", fm=":", ps=base_ps, dw=1, fw=base_fw, dc="#FFB05F", fc="black", fe=None)
    dim10_artcfg = dict(pm="o-", om="o-", fm="-.", ps=base_ps, dw=1, fw=base_fw, dc="#605FFF", fc="black", fe=None)
    dim20_artcfg = dict(pm="o-", om="o-", fm="-.", ps=base_ps, dw=1, fw=base_fw, dc="#FF5FFE", fc="black", fe=None)
    param_artcfg = dict(pm="o-", om="o-", fm="--", ps=base_ps, dw=1, fw=base_fw, dc="#00CF68", fc="black", fe=None)
    bc_node_artcfg = dict(pm="s-", om="s-", fm=":", ps=base_ps, dw=1, fw=base_fw, dc="#FF605F", fc="black", fe=None)
    bc_param_artcfg = dict(pm="s-", om="s-", fm="--", ps=base_ps, dw=1, fw=base_fw, dc="#00CFCF", fc="black", fe=None)

    modexp = lambda x: modcfg(x, dict(fm="-", fc="red"))


    # plot instances
    powerfit = True
    tbreak = True
    fit_file_str = "3_joint_res_iso_fits.txt"
    clear_fits(fit_file_str)

    ## residual isolations
    plot_instance(rlp_ax.loglog, params_x, pres, param_fit_slc, param_omit_slc if tbreak else None, False, False, "decay", "positive", fit_file_str, "rl param res", param_artcfg, powerfit=powerfit)
    plot_instance(sfp_ax.loglog, bc_params_x, bc_pres, bc_param_fit_slc, bc_param_omit_slc if tbreak else None, False, False, "decay", "positive", fit_file_str, "sft param res", bc_param_artcfg, powerfit=powerfit)

    plot_instance(rln_ax.plot, nres_scale, nres, node_fit_slc, node_omit_slc if tbreak else None, False, True, "grow", "positive", fit_file_str, "rl node res", node_artcfg, powerfit=powerfit)
    plot_instance(sfn_ax.plot, nres_scale, bc_nres, bc_node_fit_slc, bc_node_omit_slc if tbreak else None, False, True, "grow", "positive", fit_file_str, "sft node res", bc_node_artcfg, powerfit=powerfit)

    plot_instance(d10_ax.plot, d10res_scale, d10res, dim10_fit_slc, dim10_omit_slc if tbreak else None, True, False, "decay", "negative", fit_file_str, "rl dim10 res exp", modexp(dim10_artcfg), powerfit=powerfit, expfit=True, plot_data=False)
    plot_instance(d20_ax.plot, d20res_scale, d20res, dim20_fit_slc, dim20_omit_slc if tbreak else None, True, False, "decay", "negative", fit_file_str, "rl dim20 res exp", modexp(dim20_artcfg), powerfit=powerfit, expfit=True, plot_data=False)

    d10_popt = plot_instance(d10_ax.plot, d10res_scale, d10res, dim10_fit_slc, dim10_omit_slc if tbreak else None, True, False, "decay", "negative", fit_file_str, "rl dim10 res pow", dim10_artcfg, powerfit=powerfit, expfit=False)
    d20_popt = plot_instance(d20_ax.plot, d20res_scale, d20res, dim20_fit_slc, dim20_omit_slc if tbreak else None, True, False, "decay", "negative", fit_file_str, "rl dim20 res pow", dim20_artcfg, powerfit=powerfit, expfit=False)

    # labelling axes
    label_size = 6
    tick_label_size = 4
    minor_tick_label_size = 4

    [ax.set_ylabel("Mean edge residual", fontsize=label_size) for ax in axes]

    [ax.set_xlabel("Parameters ($N$)", fontsize=label_size) for ax in (rlp_ax, sfp_ax)]
    [ax.set_xlabel("Nodes ($n$)", fontsize=label_size) for ax in (rln_ax, sfn_ax)]
    [ax.set_xlabel("Spatial dimensions ($d$)", fontsize=label_size) for ax in (d10_ax, d20_ax)]

    [ax.set_ylim(bottom=0) for ax in (d10_ax, d20_ax)]
    d10_ax.set_ylim(top=2.1)

    [ax.set_xticks([2, 4, 6, 8, 10, 12, 15, 20]) for ax in (d10_ax, d20_ax)]
    [ax.set_xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50]) for ax in (rln_ax, sfn_ax)]

    [ax.tick_params(labelsize=tick_label_size) for ax in axes]
    [ax.tick_params(which="minor", labelsize=minor_tick_label_size) for ax in axes]

    # beta lines
    beta_col = "slategrey"

    blo10_beta = d10_popt[-1] * d10_popt[0][0]
    d10_ax.axhline(blo10_beta, color=beta_col, linewidth=1, zorder=0)
    xmin, _ = d10_ax.get_xlim()
    d10_ax.text(xmin+0.5, blo10_beta-0.025, r"$\beta_{\alpha}$", color=beta_col, fontsize=label_size, ha='left', va='top')
    
    blo20_beta = d20_popt[-1] * d20_popt[0][0]
    d20_ax.axhline(blo20_beta, color=beta_col, linewidth=1, zorder=0)
    xmin, _ = d20_ax.get_xlim()
    d20_ax.text(xmin+0.5, blo20_beta-0.05, r"$\beta_{\alpha}$", color=beta_col, fontsize=label_size, ha='left', va='top')

    # legend
    leg_size = 6
    line, _, fit = rlp_ax.get_lines()
    rlp_ax.legend([line, fit], ["$N$-RL", r"$r \propto N^{-\alpha}$"], loc="upper right", prop={'size': leg_size}, ncols=1, frameon=True)  #bbox_to_anchor=(0.5, 0.01)

    line, _, fit = sfp_ax.get_lines()
    sfp_ax.legend([line, fit], ["$N$-SFT", r"$r \propto N^{-\alpha}$"], loc="upper right", prop={'size': leg_size}, ncols=1, frameon=True)  #bbox_to_anchor=(0.5, 0.01)

    _, line, _, fit = rln_ax.get_lines()
    rln_ax.legend([line, fit], ["$n$-RL", r"$r \propto (n - \gamma)^{\alpha}$"], loc="upper left", prop={'size': leg_size}, ncols=1, frameon=True)  #bbox_to_anchor=(0.5, 0.01)

    _, line, _, fit = sfn_ax.get_lines()
    sfn_ax.legend([line, fit], ["$n$-SFT", r"$r \propto (n - \gamma)^{\alpha}$"], loc="upper left", prop={'size': leg_size}, ncols=1, frameon=True)  #bbox_to_anchor=(0.5, 0.01)

    expfit, line, _, powfit, _ = d10_ax.get_lines()
    d10_ax.legend([line, powfit, expfit], ["$d$-RL ($n=10$)", r"$r - \beta \propto {-d}^{-\alpha}$", r"$r - \beta \propto -\psi^{-d}$"], loc="lower right", prop={'size': leg_size}, ncols=1, frameon=True)  #bbox_to_anchor=(0.5, 0.01)

    expfit, line, _, powfit, _ = d20_ax.get_lines()
    d20_ax.legend([line, powfit, expfit], ["$d$-RL ($n=20$)", r"$r - \beta \propto {-d}^{-\alpha}$", r"$r - \beta \propto -\psi^{-d}$"], loc="lower right", prop={'size': leg_size}, ncols=1, frameon=True)  #bbox_to_anchor=(0.5, 0.01)

    # add labelled subplots
    [fig_iso.add_subplot(ax) for ax in axes]

    # saving
    save_fig(fig_iso, "3_joint_res_iso", "png")
    save_fig(fig_iso, "3_joint_res_iso", "pdf")




def make_sol_res_supp_plots_nips(client):  # TODO base residual scaling figs for reference (maybe add search here too?) and get rid of subopt vs res
    # setup axes
    fig = plt.figure(figsize=(5.5, 2.5))
    grid = GridSpec(2, 1, hspace=0.45, height_ratios=(1, 1.5))
    res_grid = GridSpecFromSubplotSpec(1, 3, subplot_spec=grid[0, 0], wspace=0.25)

    res_axes = [plt.Subplot(fig, res_grid[0, col_idx]) for col_idx in range(3)]
    subopt_ax = plt.Subplot(fig, grid[1, 0])

    axes = res_axes + [subopt_ax]
    
    [ax.axhline(0, color="black", linewidth=plt.rcParams['xtick.major.width']) for ax in res_axes[1:]]

    # subopt_ax.tick_params("x", which="both", labelbottom=False)

    # gather instance data
    ppo_params_x, params_scale = get_sol_eval(client, "SOL_model_scaling_drl_2", "width", "non_embedding_parameters")
    bc_params_x, _ = get_sol_eval(client, "SOL_model_scaling_il_patch", "width", "non_embedding_parameters")

    bind_exp_name = "EVAL_bind"
    nres, nres_scale = get_bind_eval(client, bind_exp_name, "node", f"drl_node_res.edge.mean")
    d10res, d10res_scale = get_bind_eval(client, bind_exp_name, f"dim10", f"drl_dim_res.edge.mean")
    d20res, d20res_scale = get_bind_eval(client, bind_exp_name, f"dim20", f"drl_dim_res.edge.mean")
    pres, pres_scale = get_bind_eval(client, bind_exp_name, "model", f"drl_model_res.edge.mean")
    bc_nres, _ = get_bind_eval(client, bind_exp_name, "node", f"il_node_res.edge.mean")
    bc_pres, _ = get_bind_eval(client, bind_exp_name, "model", f"il_model_res.edge.mean")

    nres /= 2  # converts XOR to num mismatched edges (base stat is always even)
    d10res /= 2
    d20res /= 2
    pres /= 2
    bc_nres /= 2
    bc_pres /= 2

    ###
    node_cost_y, _ = get_bind_eval(client, bind_exp_name, "node", f"drl_node_cost.mean")
    dim10_cost_y, _ = get_bind_eval(client, bind_exp_name, f"dim10", f"drl_dim_cost.mean")
    dim20_cost_y, _ = get_bind_eval(client, bind_exp_name, f"dim20", f"drl_dim_cost.mean")
    param_cost_y, _ = get_bind_eval(client, bind_exp_name, "model", f"drl_model_cost.mean")
    bc_node_cost_y, _ = get_bind_eval(client, bind_exp_name, "node", f"il_node_cost.mean")
    bc_param_cost_y, _ = get_bind_eval(client, bind_exp_name, "model", f"il_model_cost.mean")

    ncost_opt, _ = get_bind_eval(client, bind_exp_name, "node", f"opt_cost.mean")
    d10cost_opt, _ = get_bind_eval(client, bind_exp_name, f"dim10", f"opt_cost.mean")
    d20cost_opt, _ = get_bind_eval(client, bind_exp_name, f"dim20", f"opt_cost.mean")

    OPTIMAL_N20 = 3.831

    node_subopt = (node_cost_y - ncost_opt)
    dim10_subopt = (dim10_cost_y - d10cost_opt)
    dim20_subopt = (dim20_cost_y - d20cost_opt)
    param_subopt = (param_cost_y - OPTIMAL_N20)
    bc_node_subopt = (bc_node_cost_y - ncost_opt)
    bc_param_subopt = (bc_param_cost_y - OPTIMAL_N20)
    ###

    node_fit_slc, node_omit_slc = slice(9), None
    dim10_fit_slc, dim10_omit_slc = slice(11), None
    dim20_fit_slc, dim20_omit_slc = slice(9), None #slice(4, 9)
    param_fit_slc, param_omit_slc = slice(9), None
    bc_node_fit_slc, bc_node_omit_slc = slice(6), None
    bc_param_fit_slc, bc_param_omit_slc = slice(5), None

    base_ps = 2
    node_artcfg = dict(pm="o-", om="o-", fm=":", ps=base_ps, dw=1, fw=0.75, dc="#FFB05F", fc="black", fe=None)
    dim10_artcfg = dict(pm="o-", om="o-", fm=":", ps=base_ps, dw=1, fw=0.75, dc="#605FFF", fc="black", fe=None)
    dim20_artcfg = dict(pm="o-", om="o-", fm=":", ps=base_ps, dw=1, fw=0.75, dc="#FF5FFE", fc="black", fe=None)
    param_artcfg = dict(pm="o-", om="o-", fm=":", ps=base_ps, dw=1, fw=0.75, dc="#00CF68", fc="black", fe=None)
    bc_node_artcfg = dict(pm="o-", om="o-", fm=":", ps=base_ps, dw=1, fw=0.75, dc="#FF605F", fc="black", fe=None)
    bc_param_artcfg = dict(pm="o-", om="o-", fm=":", ps=base_ps, dw=1, fw=0.75, dc="#00CFCF", fc="black", fe=None)

    mod_decay = lambda x: modcfg(x, dict(fm="--", ps=3))
    mod_bounded = lambda x: modcfg(x, dict(fm="-.", fw=0.5, ps=3))
    mod_expbounded = lambda x: modcfg(x, dict(fm="-", fw=0.5, ps=3))
    
    # plot instances
    powerfit = True
    fit_file_str = "3_joint_residual_nips_supp_fits.txt"
    clear_fits(fit_file_str)

    ## res row
    plot_instance(res_axes[0].loglog, bc_params_x, bc_pres, bc_param_fit_slc, None, False, False, "decay", "positive", fit_file_str, "bc param res", mod_decay(bc_param_artcfg), powerfit=powerfit)
    plot_instance(res_axes[0].loglog, ppo_params_x, pres, param_fit_slc, param_omit_slc, False, False, "decay", "positive", fit_file_str, "ppo param res", mod_decay(param_artcfg), powerfit=powerfit)
    
    plot_instance(res_axes[1].plot, nres_scale, bc_nres, bc_node_fit_slc, None, False, True, "grow", "positive", fit_file_str, "bc node res", modcfg(bc_node_artcfg, dict(ps=3)), powerfit=powerfit)
    plot_instance(res_axes[1].plot, nres_scale, nres, node_fit_slc, node_omit_slc, False, True, "grow", "positive", fit_file_str, "ppo node res", modcfg(node_artcfg, dict(ps=3)), powerfit=powerfit)

    plot_instance(res_axes[2].plot, d10res_scale, d10res, dim10_fit_slc, dim10_omit_slc, False, True, "grow", "positive", fit_file_str, "ppo dim10 res (power growth)", modcfg(dim10_artcfg, dict(fw=0.5)), powerfit=powerfit, plot_data=False, plot_powerfit=False)
    plot_instance(res_axes[2].plot, d10res_scale, d10res, dim10_fit_slc, dim10_omit_slc, True, False, "decay", "negative", fit_file_str, "ppo dim10 res (power decay)", mod_bounded(dim10_artcfg), powerfit=powerfit, plot_data=False, plot_powerfit=False)
    d10_popt = plot_instance(res_axes[2].plot, d10res_scale, d10res, dim10_fit_slc, dim10_omit_slc, True, False, "decay", "negative", fit_file_str, "ppo dim10 res (exp decay)", mod_expbounded(dim10_artcfg), powerfit=powerfit, expfit=True)

    plot_instance(res_axes[2].plot, d20res_scale, d20res, dim20_fit_slc, dim20_omit_slc, False, True, "grow", "positive", fit_file_str, "ppo dim20 res (power growth)", modcfg(dim20_artcfg, dict(fw=0.5)), powerfit=powerfit, plot_data=False)
    plot_instance(res_axes[2].plot, d20res_scale, d20res, dim20_fit_slc, dim20_omit_slc, True, False, "decay", "negative", fit_file_str, "ppo dim20 res (power decay)", mod_bounded(dim20_artcfg), powerfit=powerfit, plot_data=False)
    d20_popt = plot_instance(res_axes[2].plot, d20res_scale, d20res, dim20_fit_slc, dim20_omit_slc, True, False, "decay", "negative", fit_file_str, "ppo dim20 res (exp decay)", mod_expbounded(dim20_artcfg), powerfit=powerfit, expfit=True)

    ## relational subopt v res
    plot_instance(subopt_ax.loglog, nres, node_subopt, node_fit_slc, None, False, False, "grow", "positive", fit_file_str, "ppo node subopt v res", (node_artcfg), powerfit=powerfit)
    plot_instance(subopt_ax.loglog, bc_nres, bc_node_subopt, bc_node_fit_slc, None, False, False, "grow", "positive", fit_file_str, "bc node subopt v res", (bc_node_artcfg), powerfit=powerfit)
    plot_instance(subopt_ax.loglog, d10res, dim10_subopt, dim10_fit_slc, None, False, False, "grow", "positive", fit_file_str, "ppo dim10 subopt v res", (dim10_artcfg), powerfit=powerfit)
    plot_instance(subopt_ax.loglog, d20res, dim20_subopt, dim20_fit_slc, None, False, False, "grow", "positive", fit_file_str, "ppo dim20 subopt v res", (dim20_artcfg), powerfit=powerfit)
    plot_instance(subopt_ax.loglog, pres, param_subopt, param_fit_slc, None, False, False, "grow", "positive", fit_file_str, "ppo param subopt v res", (param_artcfg), powerfit=powerfit)
    plot_instance(subopt_ax.loglog, bc_pres, bc_param_subopt, bc_param_fit_slc, None, False, False, "grow", "positive", fit_file_str, "bc param subopt v res", (bc_param_artcfg), powerfit=powerfit)

    # labelling axes
    label_size = 6
    tick_label_size = 4
    minor_tick_label_size = 4

    res_axes[0].set_ylabel("Mean edge residual", fontsize=label_size)
    subopt_ax.set_ylabel("Suboptimality gap", fontsize=label_size)

    res_axes[0].set_xlabel("Parameters ($N$)", fontsize=label_size)
    res_axes[1].set_xlabel("Nodes ($n$)", fontsize=label_size)
    res_axes[2].set_xlabel("Spatial dimensions ($d$)", fontsize=label_size)
    subopt_ax.set_xlabel("Mean edge residual", fontsize=label_size)

    res_axes[1].set_ylim(bottom=-0.003)
    res_axes[2].set_ylim(bottom=-0.012)

    subopt_ax.set_xlim([0.2, 7])
    subopt_ax.set_ylim(bottom=5e-4)

    res_axes[1].set_xticks(list(range(5, 46, 10)))
    res_axes[2].set_xticks([2, 4, 6, 8, 10, 12])

    [ax.tick_params(labelsize=tick_label_size) for ax in axes]
    [ax.tick_params(which="minor", labelsize=minor_tick_label_size) for ax in axes]

    res_axes[0].yaxis.set_minor_formatter(mticker.ScalarFormatter())
    
    # [forceAspect(ax, aspect=1.5) for ax in res_axes]

    # legend
    # _, decay_fit = res_axes[0].get_lines()
    # _, unbounded_fit = res_axes[1].get_lines()
    # _, bounded_fit = res_axes[2].get_lines()
    # ppo_node, bc_node, ppo_dim10, ppo_dim20, ppo_param, bc_param = subopt_ax.get_lines()
    # fig.legend([ppo_param, bc_param, decay_fit, ppo_node, bc_node, unbounded_fit, ppo_dim10, ppo_dim20, bounded_fit], ["PPO parameter scaling", "BC parameter scaling", "Decay", "PPO node scaling", "BC node scaling", "Unbounded growth", "PPO dimension scaling ($n=10$)", "PPO dimension scaling ($n=20$)", "Bounded growth"], loc="upper center", prop={'size': label_size}, bbox_to_anchor=(0.5, 0.01), ncols=3, frameon=False)

    # res_axes[0].text(0.95, 0.925, "BC", fontsize=label_size, ha='right', va='top', transform=res_axes[0].transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))
    # res_axes[1].text(0.95, 0.075, "BC", fontsize=label_size, ha='right', va='bottom', transform=res_axes[1].transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))
    # res_axes[2].text(0.95, 0.075, "$n=20$", fontsize=label_size, ha='right', va='bottom', transform=res_axes[2].transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))

    # add labelled subplots
    [fig.add_subplot(ax) for ax in axes]

    # saving
    save_fig(fig, "3_joint_residual_nips_supp", "png")
    save_fig(fig, "3_joint_residual_nips_supp", "pdf")
    

def make_unlim_lo_res_plots(client, algo_str):
    # setup axes
    fig = plt.figure(figsize=(5.5, 3.1))
    outer_grid = GridSpec(2, 1, hspace=0.0, height_ratios=[2/3, 1/3])
    nd_grid = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer_grid[0, 0], wspace=0.5)
    subopt_res_grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid[1, 0], wspace=0.3)
    
    nd_axes = [plt.Subplot(fig, nd_grid[0, idx]) for idx in range(4)]
    subopt_res_axes = [plt.Subplot(fig, subopt_res_grid[0, idx]) for idx in range(2)]

    node_axes = nd_axes[:2]
    dim_axes = nd_axes[2:]

    axes = nd_axes + subopt_res_axes

    [ax.axhline(0, color="black", linewidth=plt.rcParams['xtick.major.width'], zorder=0) for ax in nd_axes[:2]]

    # gather instance data
    dnodes = 10
    bind_exp_name = "EVAL_bind"

    ncost_opt, _ = get_bind_eval(client, bind_exp_name, "node", f"opt_cost.mean")
    dcost_opt, _ = get_bind_eval(client, bind_exp_name, f"dim{dnodes}", f"opt_cost.mean")

    ncosts, _ = get_bind_eval(client, bind_exp_name, "node", f"_{algo_str}_cost.mean")
    nres, nres_scale = get_bind_eval(client, bind_exp_name, "node", f"_{algo_str}_res.edge.mean")
    dcosts, _ = get_bind_eval(client, bind_exp_name, f"dim{dnodes}", f"_{algo_str}_cost.mean")
    dres, dres_scale = get_bind_eval(client, bind_exp_name, f"dim{dnodes}", f"_{algo_str}_res.edge.mean")

    nres /= 2  # converts XOR to num mismatched edges (base stat is always even)
    dres /= 2

    # 2D local search experiments (2opt and 2exchange) both have corrupted, inflated results in EVAL_bind for 10-node version, where 20-node version does not, and I'm not sure why
    # code below substitutes values from earlier evals done on same proxy opt problems, where issue does not show up
    if dnodes == 10:  
        _2OPT_2D_COST = 2.8857
        _2SWAP_2D_COST = 2.9238
        dcosts[0] = _2OPT_2D_COST if algo_str == "2opt" else _2SWAP_2D_COST

    node_subopt = (ncosts  - ncost_opt)
    dim_subopt = (dcosts - dcost_opt)

    node_fit_slc, node_omit_slc = (slice(3, 10), slice(4)) if algo_str == "2opt" else (slice(1, 10), slice(2))
    dim_fit_slc, dim_omit_slc = slice(11), None

    node_artcfg = dict(pm="o-", om="o-", fm=":", ps=3, dw=1, fw=0.75, dc="#FFB05F", fc="black", fe=None)
    dim_artcfg = dict(pm="o-", om="o-", fm="--", ps=3, dw=1, fw=0.75, dc="#B05FFF", fc="black", fe=None)

    mod_unbounded = lambda x: modcfg(x, dict(fm=":"))
    mod_bounded = lambda x: modcfg(x, dict(fm="-."))
    
    # plot instances
    powerfit = True
    fit_file_str = f"3_fpc_unlim_lo_res_fits_{algo_str}.txt"
    clear_fits(fit_file_str)

    plot_instance(nd_axes[0].plot, nres_scale, nres, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, "node res", node_artcfg, powerfit=powerfit, powerfit_full_xbounds=True, near_linear=True)
    plot_instance(nd_axes[1].plot, nres_scale, node_subopt, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, "node subopt", node_artcfg, powerfit=powerfit and algo_str == "2swap", powerfit_full_xbounds=True, near_linear=algo_str == "2swap")

    plot_instance(nd_axes[2].plot, dres_scale, dres, dim_fit_slc, dim_omit_slc, True, True, "decay", "negative", fit_file_str, f"dim{dnodes} res", mod_bounded(dim_artcfg), powerfit=powerfit)
    plot_instance(nd_axes[3].plot, dres_scale, dim_subopt, dim_fit_slc, dim_omit_slc, True, True, "decay", "negative", fit_file_str, f"dim{dnodes} subopt", mod_bounded(dim_artcfg), powerfit=powerfit)

    plot_instance(subopt_res_axes[0].loglog, nres, node_subopt, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, "node subopt v res", node_artcfg, powerfit=powerfit and algo_str == "2swap", powerfit_full_xbounds=True, near_linear=algo_str == "2swap")
    plot_instance(subopt_res_axes[1].loglog, dres, dim_subopt, dim_fit_slc, dim_omit_slc, True, True, "grow", "positive", fit_file_str, f"dim{dnodes} subopt v res", mod_unbounded(dim_artcfg), powerfit=powerfit)

    # labelling axes
    label_size = 6
    tick_label_size = 4
    minor_tick_label_size = 4

    [ax.set_ylabel("Suboptimality gap", fontsize=label_size) for ax in [nd_axes[1], nd_axes[3]] + subopt_res_axes]
    [ax.set_ylabel("Mean edge residual", fontsize=label_size) for ax in (nd_axes[0], nd_axes[2])]

    [ax.set_xlabel("Nodes ($n$)", fontsize=label_size) for ax in node_axes]
    [ax.set_xlabel("Spatial dimensions ($d$)", fontsize=label_size) for ax in dim_axes]

    [ax.set_xlabel("Mean edge residual", fontsize=label_size) for ax in subopt_res_axes]

    nd_axes[0].set_ylim(bottom=-0.7 if algo_str == "2opt" else -1.5)
    if algo_str == "2swap":
        nd_axes[1].set_ylim(bottom=-0.2)

    [ax.set_xticks(list(range(5, 51, 15))) for ax in node_axes]
    [ax.set_xticks([2, 5, 8, 12]) for ax in dim_axes]
    
    [ax.tick_params(labelsize=tick_label_size) for ax in axes]
    [ax.tick_params(which="minor", labelsize=minor_tick_label_size) for ax in axes]

    if algo_str == "2swap":
        subopt_res_axes[1].xaxis.set_minor_formatter(mticker.ScalarFormatter())
        subopt_res_axes[1].xaxis.set_major_formatter(mticker.ScalarFormatter())

        nd_axes[-1].set_yticks([0.05, 0.10, 0.15])

    [forceAspect(ax) for ax in nd_axes]

    # legend
    white_handle = mpatches.Rectangle((0,0), 1, 1, fill=False, edgecolor='none', visible=False)
    _, node_trend, node_omit, unbounded_fit = nd_axes[0].get_lines()
    dim_trend, bounded_fit = nd_axes[2].get_lines()
    fig.legend([node_trend, unbounded_fit, node_omit, white_handle, dim_trend, bounded_fit], ["Residual vs $n$ trend inputs" if algo_str == "2opt" else "$n$ trend inputs", "Unbounded growth" if algo_str == "2opt" else "Linear growth", "$n$ pre-trend", " ", "$d$ trend inputs", "Bounded growth"], loc="upper center", prop={'size': label_size}, bbox_to_anchor=(0.5, 0.0), ncols=3, frameon=False)

    nd_axes[0].text(0.075, 0.925, r"$\alpha=1.0$", fontsize=label_size, ha='left', va='top', transform=nd_axes[0].transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))

    if algo_str == "2swap":
        nd_axes[1].text(0.075, 0.925, r"$\alpha=1.0$", fontsize=label_size, ha='left', va='top', transform=nd_axes[1].transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))
        subopt_res_axes[0].text(0.05, 0.925, r"$\alpha=1.0$", fontsize=label_size, ha='left', va='top', transform=subopt_res_axes[0].transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))
        subopt_res_axes[1].text(0.05, 0.925, r"$\alpha \approx 1.0$", fontsize=label_size, ha='left', va='top', transform=subopt_res_axes[1].transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))

    # add labelled subplots
    [fig.add_subplot(ax) for ax in axes]

    # saving
    save_str = "3_fpc_unlim_lo_res" if algo_str == "2opt" else "3_fpc_unlim_lo_res_supp"
    save_fig(fig, save_str, FORMAT)
    

def make_lo_searchcap_plots(client, algo_str):
    # setup axes
    fig = plt.figure(figsize=(5.5, 2.5))
    grid = GridSpec(1, 4, wspace=0.6)

    res_ax = plt.Subplot(fig, grid[0, 0])
    subopt_ax = plt.Subplot(fig, grid[0, 1])
    subopt_res_ax = plt.Subplot(fig, grid[0, 2])
    cap_ax = plt.Subplot(fig, grid[0, 3])

    axes = [res_ax, subopt_ax, subopt_res_ax, cap_ax]

    cap_ax.axhline(0, color="black", linewidth=plt.rcParams['xtick.major.width'])
    cap_ax.axhline(100, color="black", linewidth=plt.rcParams['xtick.major.width'])

    # gather instance data
    constrained_exp_name = "EVAL_constrained_local_optima"
    scosts, sres_scale = get_bind_eval(client, constrained_exp_name, f"model_scaling_{algo_str}", f"{algo_str}_local_opt.costs_avg")
    sres, _ = get_bind_eval(client, constrained_exp_name, f"model_scaling_{algo_str}", f"edge.{algo_str}_local_opt.residuals_avg")
    scaps, _ = get_bind_eval(client, constrained_exp_name, f"model_scaling_{algo_str}", "search_cap")
    scap_probs, _ = get_bind_eval(client, constrained_exp_name, f"model_scaling_{algo_str}", f"edge.{algo_str}_local_opt.caps_avg")
   
    sres /= 2

    OPTIMAL_N20 = 3.831
    RAND_N20 = 10.428

    scap_subopt = (scosts - OPTIMAL_N20)

    scap_fit_slc, scap_omit_slc = (slice(len(scaps)), None) if algo_str == "2opt" else (slice(27), None)

    scap_artcfg = dict(pm="o-", om="o-", fm="--", ps=1.5, dw=0.5, fw=0.75, dc="gainsboro", fc="black", fe=None)  # "#00CFCF"

    mod_unbounded = lambda x: modcfg(x, dict(fm=":"))
    mod_bounded = lambda x: modcfg(x, dict(fm="-."))

    # plot instances
    powerfit = False
    fit_file_str = f"3_fpc_searchcap_lo_fits_{algo_str}.txt"
    if powerfit: clear_fits(fit_file_str)

    cap_colors = berlinColormap(scap_probs)

    plot_heterochromia_instance(res_ax, res_ax.plot, scaps, sres, cap_colors, scap_fit_slc, scap_omit_slc, True, True, "decay", "positive", fit_file_str, "scap res", scap_artcfg, powerfit=powerfit)

    plot_heterochromia_instance(subopt_ax, subopt_ax.semilogy, scaps, scap_subopt, cap_colors, scap_fit_slc, scap_omit_slc, True, True, "decay", "positive", fit_file_str, "scap subopt", scap_artcfg, powerfit=powerfit)

    plot_heterochromia_instance(subopt_res_ax, subopt_res_ax.semilogy, sres, scap_subopt, cap_colors, scap_fit_slc, scap_omit_slc, True, True, "grow", "positive", fit_file_str, "scap subopt v res", mod_unbounded(scap_artcfg), powerfit=powerfit)
    
    plot_heterochromia_instance(cap_ax, cap_ax.plot, scaps, 100 * scap_probs, cap_colors, scap_fit_slc, scap_omit_slc, False, False, "NONE", "NONE", fit_file_str, "NONE", scap_artcfg, powerfit=False)

    # labelling axes
    label_size = 6
    tick_label_size = 4
    minor_tick_label_size = 4

    cap_ax.set_ylabel("Max depth %", fontsize=label_size)
    res_ax.set_ylabel("Mean edge residual", fontsize=label_size)
    [ax.set_ylabel("Suboptimality gap", fontsize=label_size) for ax in (subopt_ax, subopt_res_ax)]

    [ax.set_xlabel("Search capacity ($M$)", fontsize=label_size) for ax in (res_ax, subopt_ax, cap_ax)]
    subopt_res_ax.set_xlabel("Mean edge residual", fontsize=label_size)

    res_ax.set_ylim(bottom=0)

    [ax.set_xlim(left=0) for ax in axes]

    [ax.set_xticks(list(range(0, 28, 9))) for ax in (res_ax, subopt_ax, cap_ax)]
    
    [ax.tick_params(labelsize=tick_label_size) for ax in axes]
    [ax.tick_params(which="minor", labelsize=minor_tick_label_size) for ax in axes]

    if algo_str == "2swap":
        [ax.yaxis.set_minor_formatter(mticker.ScalarFormatter()) for ax in (subopt_ax, subopt_res_ax)]
        [ax.yaxis.set_major_formatter(mticker.ScalarFormatter()) for ax in (subopt_ax, subopt_res_ax)]
        [ax.set_yticks([0.7, 0.8, 0.9, 2, 3, 4, 5], ["", "", "", "2", "3", "4", ""], minor=True) for ax in (subopt_ax, subopt_res_ax)]
        [ax.set_yticks([1]) for ax in (subopt_ax, subopt_res_ax)]

    [forceAspect(ax) for ax in axes]

    # colorbar
    cbar_ax = fig.add_axes([axes[-1].get_position().x1 + 0.015, axes[-1].get_position().y0 , 0.01, axes[-1].get_position().y1 - axes[-1].get_position().y0])
    cbar = fig.colorbar(ScalarMappable(cmap=berlinColormap), cax=cbar_ax, location="right", fraction=1, aspect=40)
    cbar_ticks = [0, 0.25, 0.5, 0.75, 1]
    cbar.ax.set_yticks(cbar_ticks, [Text(0, t, f"{int(100 * t)}") for t in cbar_ticks])
    cbar.set_label("Max depth %", rotation=90, fontsize=label_size)
    cbar.ax.tick_params(labelsize=tick_label_size) 

    # add labelled subplots
    [fig.add_subplot(ax) for ax in axes]

    # saving
    save_str = "3_fpc_searchcap_lo" if algo_str == "2opt" else "3_fpc_searchcap_lo_supp"
    save_fig(fig, save_str, FORMAT)
  

def make_node_phase_main_plots(client):
    # setup axes
    fig = plt.figure(figsize=(5.5, 3.1))
    outer_grid = GridSpec(2, 1, hspace=0.15, height_ratios=[0.5, 0.5])
    splash_grid = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_grid[0, 0], wspace=0.4)   
    
    splash_axes = [plt.Subplot(fig, splash_grid[0, idx]) for idx in range(3)]
    res_n_ax = splash_axes[0]
    subopt_n_ax = splash_axes[1]
    subopt_res_ax = splash_axes[2]

    trials = 6
    phase_grid = GridSpecFromSubplotSpec(1, trials, subplot_spec=outer_grid[1, 0], wspace=0.1)
    phase_axes = [plt.Subplot(fig, phase_grid[0, 0])]
    phase_axes += [plt.Subplot(fig, phase_grid[0, col_idx], sharex=phase_axes[0], sharey=phase_axes[0]) for col_idx in range(1, trials)]

    axes = splash_axes + phase_axes

    [ax.axhline(0, color="black", linewidth=plt.rcParams['xtick.major.width']) for ax in splash_axes]

    # gather instance data
    constrained_exp_name = "EVAL_constrained_local_optima"
    bind_exp_name = "EVAL_bind"

    ncost_opt, _ = get_bind_eval(client, bind_exp_name, "node", f"opt_cost.mean")
    ncost_rand, _ = get_bind_eval(client, bind_exp_name, "node", f"rand_cost.mean")
    nres_rand, _ = get_bind_eval(client, bind_exp_name, "node", "rand_res.edge.mean")

    ncosts, _ = get_bind_eval(client, bind_exp_name, "node", "_2opt_cost.mean")
    nres, nres_scale = get_bind_eval(client, bind_exp_name, "node", f"_2opt_res.edge.mean")

    ncosts_constrained, ncap_ref = split_by_cap(client, constrained_exp_name, "plus_node_scaling_2opt", "2opt_local_opt.costs_avg", "search_cap")
    nres_constrained, _ = split_by_cap(client, constrained_exp_name, "plus_node_scaling_2opt", "edge.2opt_local_opt.residuals_avg", "search_cap")
    ncap_probs_constrained, _ = split_by_cap(client, constrained_exp_name, "plus_node_scaling_2opt", "edge.2opt_local_opt.caps_avg", "search_cap")

    nres /= 2  # converts XOR to num mismatched edges (base stat is always even)
    nres_constrained = [arr / 2 for arr in nres_constrained]
    nres_rand /= 2

    node_subopt = ncosts - ncost_opt
    node_span = ncost_rand - ncost_opt
    nsubopts_constrained = [ncc - ncost_opt for ncc in ncosts_constrained]

    node_fit_slc, node_omit_slc = slice(10), None

    base_artcfg = dict(pm="o-", om="o-", fm="--", ps=1.05, dw=1, fw=0.75, dc="#FFB05F", fc="black", fe=None)
    rand_artcfg = modcfg(base_artcfg, dict(dc="black"))
    unlim_artcfg = modcfg(base_artcfg, dict(dc="#FFEE00"))  # #F7F800

    # plot instances
    powerfit = True
    fit_file_str = "3_fpc_node_phase_main_fits.txt"
    clear_fits(fit_file_str)

    ## splash plots
    cap_map = partial_cmap(plt.get_cmap("inferno"), 0.4, 0.9)
    mod_col = lambda x, v: modcfg(x, dict(dc=cap_map((v - 5) / 35)))

    plot_instance(res_n_ax.plot, nres_scale, nres_rand, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, "node rand res", rand_artcfg, powerfit=False) 
    [plot_instance(res_n_ax.plot, nres_scale, nrc, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, f"node {int(cap)}M res", mod_col(base_artcfg, cap), powerfit=False) for nrc, cap in zip(nres_constrained, ncap_ref)]
    plot_instance(res_n_ax.plot, nres_scale, nres, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, "node res", unlim_artcfg, powerfit=False)
    
    plot_instance(subopt_n_ax.plot, nres_scale, node_span, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, "node achievable span", rand_artcfg, powerfit=False)
    [plot_instance(subopt_n_ax.plot, nres_scale, nsc, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, f"node {int(cap)}M subopt", mod_col(base_artcfg, cap), powerfit=False) for nsc, cap in zip(nsubopts_constrained, ncap_ref)]
    plot_instance(subopt_n_ax.plot, nres_scale, node_subopt, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, "node subopt", unlim_artcfg, powerfit=False)

    plot_instance(subopt_res_ax.plot, nres_rand, node_span, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, "node rand subopt v res", rand_artcfg, powerfit=False)
    [plot_instance(subopt_res_ax.plot, nrc, nsc, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, f"node {int(cap)}M subopt v res", mod_col(base_artcfg, cap), powerfit=False) for nsc, nrc, cap in zip(nsubopts_constrained, nres_constrained, ncap_ref)]
    plot_instance(subopt_res_ax.plot, nres, node_subopt, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, "node subopt v res", unlim_artcfg, powerfit=False)

    ## phase plots
    for trial_idx, capv in enumerate(ncap_ref):
        nrc = nres_constrained[trial_idx]
        nsc = nsubopts_constrained[trial_idx]
        cap_probs = ncap_probs_constrained[trial_idx]

        phase_artcfg = modcfg(base_artcfg, dict(pm="-", dc="white", fm=":", ps=3))
        cap_colors = berlinColormap(cap_probs)

        low_thresh = 0.3
        high_thresh = 0.6
        low_buff = 0
        high_buff = 0
        nrc_phased, nsc_phased, cap_col_phased = split_by_triphase([nrc, nsc, cap_colors], cap_probs, low_thresh, high_thresh, low_buff, high_buff)

        ures, pres, fres = nrc_phased 
        usubopt, psubopt, fsubopt = nsc_phased
        ucols, pcols, fcols = cap_col_phased

        plot_heterochromia_instance(phase_axes[trial_idx], phase_axes[trial_idx].loglog, ures, usubopt, ucols, slice(1, len(ures)), slice(1), True, True, "grow", "positive", fit_file_str, f"node unconstrained {int(capv)}M subopt v res", phase_artcfg, powerfit=powerfit and len(ures) > 2)
        plot_heterochromia_instance(phase_axes[trial_idx], phase_axes[trial_idx].loglog, pres, psubopt, pcols, slice(len(pres)), None, True, True, "grow", "positive", fit_file_str, f"node partially-constrained {int(capv)}M subopt v res", phase_artcfg, powerfit=False)
        plot_heterochromia_instance(phase_axes[trial_idx], phase_axes[trial_idx].loglog, fres, fsubopt, fcols, slice(len(fres)), None, True, True, "grow", "positive", fit_file_str, f"node fully-constrained {int(capv)}M subopt v res", phase_artcfg, powerfit=powerfit and len(fres) > 1)


    # labelling axes
    label_size = 6
    tick_label_size = 4
    minor_tick_label_size = 4

    [ax.set_ylabel("Suboptimality gap", fontsize=label_size) for ax in (subopt_n_ax, subopt_res_ax, phase_axes[0])]
    [ax.set_ylabel("Mean edge residual", fontsize=label_size) for ax in (res_n_ax,)]

    [ax.set_xlabel("Nodes ($n$)", fontsize=label_size) for ax in [res_n_ax, subopt_n_ax]]
    [ax.set_xlabel("Mean edge residual", fontsize=label_size) for ax in [subopt_res_ax]]
    phase_axes[3].set_xlabel("Mean edge residual" + 27 * " ", fontsize=label_size)

    [ax.set_xticks(list(range(5, 51, 15))) for ax in splash_axes[:2]]

    [ax.tick_params(labelsize=tick_label_size) for ax in axes]
    [ax.tick_params(which="minor", labelsize=minor_tick_label_size) for ax in axes]

    [ax.tick_params(which="both", labelleft=False) for ax in phase_axes[1:]]

    phase_axes[0].set_yticks([0.001, 0.01, 0.1, 1, 10])

    [forceAspect(ax) for ax in phase_axes]

    # legend
    rand_line = subopt_res_ax.get_lines()[1]
    unlim_line = subopt_res_ax.get_lines()[-1]
    subopt_res_ax.legend([rand_line, unlim_line], ["random", "unconstrained"], loc="upper left", prop={'size': tick_label_size}, ncols=1, frameon=False, framealpha=1)

    power_fit = phase_axes[-2].get_lines()[-1]
    phase_axes[-1].legend([power_fit], ["UPG"], loc="upper left", bbox_to_anchor=(0.0, 0.8), prop={'size': tick_label_size}, ncols=1, frameon=False, framealpha=1)

    [ax.text(0.075, 0.925, f"$M={int(capv)}$", fontsize=5, ha='left', va='top', transform=ax.transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round')) for ax, capv in zip(phase_axes, ncap_ref)]

    # add labelled subplots
    [fig.add_subplot(ax) for ax in axes]

    # colorbars
    splash_cbar_ax = fig.add_axes([splash_axes[-1].get_position().x1 + 0.015, splash_axes[-1].get_position().y0 , 0.01, splash_axes[-1].get_position().y1 - splash_axes[-1].get_position().y0])
    phase_cbar_ax = fig.add_axes([phase_axes[-1].get_position().x1 + 0.015, phase_axes[-1].get_position().y0 , 0.01, phase_axes[-1].get_position().y1 - phase_axes[-1].get_position().y0])

    splash_cbar = fig.colorbar(ScalarMappable(cmap=cap_map), cax=splash_cbar_ax, location="right", fraction=1, aspect=40)
    splash_cbar_ticks = (ncap_ref - 5) / 35
    splash_cbar.ax.set_yticks(splash_cbar_ticks, [Text(0, t, f"{int(v)}") for t, v in zip(splash_cbar_ticks, ncap_ref)])
    splash_cbar.set_label("Search move capacity ($M$)", rotation=90, fontsize=label_size)
    splash_cbar.ax.tick_params(labelsize=tick_label_size) 

    phase_cbar = fig.colorbar(ScalarMappable(cmap=berlinColormap), cax=phase_cbar_ax, location="right", fraction=1, aspect=40)
    phase_cbar_ticks = [0, 0.25, 0.5, 0.75, 1]
    phase_cbar.ax.set_yticks(phase_cbar_ticks, [Text(0, t, f"{int(100 * t)}") for t in phase_cbar_ticks])
    phase_cbar.set_label("Max depth %", rotation=90, fontsize=label_size)
    phase_cbar.ax.tick_params(labelsize=tick_label_size) 

    # saving
    save_fig(fig, "3_fpc_node_phase_main", FORMAT)


def make_node_phase_supp_plots(client):  # 2exchange version
    # setup axes
    fig = plt.figure(figsize=(5.5, 3.1))
    outer_grid = GridSpec(2, 1, hspace=0.15, height_ratios=[0.5, 0.5])
    splash_grid = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_grid[0, 0], wspace=0.4)   
    
    splash_axes = [plt.Subplot(fig, splash_grid[0, idx]) for idx in range(3)]
    res_n_ax = splash_axes[0]
    subopt_n_ax = splash_axes[1]
    subopt_res_ax = splash_axes[2]

    trials = 6
    phase_grid = GridSpecFromSubplotSpec(1, trials, subplot_spec=outer_grid[1, 0], wspace=0.1)
    phase_axes = [plt.Subplot(fig, phase_grid[0, 0])]
    phase_axes += [plt.Subplot(fig, phase_grid[0, col_idx], sharex=phase_axes[0], sharey=phase_axes[0]) for col_idx in range(1, trials)]

    axes = splash_axes + phase_axes

    [ax.axhline(0, color="black", linewidth=plt.rcParams['xtick.major.width']) for ax in splash_axes]

    # gather instance data
    constrained_exp_name = "EVAL_constrained_local_optima"
    bind_exp_name = "EVAL_bind"

    ncost_opt, _ = get_bind_eval(client, bind_exp_name, "node", f"opt_cost.mean")
    ncost_rand, _ = get_bind_eval(client, bind_exp_name, "node", f"rand_cost.mean")
    nres_rand, _ = get_bind_eval(client, bind_exp_name, "node", "rand_res.edge.mean")

    ncosts, _ = get_bind_eval(client, bind_exp_name, "node", "_2swap_cost.mean")
    nres, nres_scale = get_bind_eval(client, bind_exp_name, "node", f"_2swap_res.edge.mean")

    ncosts_constrained, ncap_ref = split_by_cap(client, constrained_exp_name, "plus_node_scaling_2swap", "2swap_local_opt.costs_avg", "search_cap")
    nres_constrained, _ = split_by_cap(client, constrained_exp_name, "plus_node_scaling_2swap", "edge.2swap_local_opt.residuals_avg", "search_cap")
    ncap_probs_constrained, _ = split_by_cap(client, constrained_exp_name, "plus_node_scaling_2swap", "edge.2swap_local_opt.caps_avg", "search_cap")

    nres /= 2  # converts XOR to num mismatched edges (base stat is always even)
    nres_constrained = [arr / 2 for arr in nres_constrained]
    nres_rand /= 2

    node_subopt = ncosts - ncost_opt
    node_span = ncost_rand - ncost_opt
    nsubopts_constrained = [ncc - ncost_opt for ncc in ncosts_constrained]

    node_fit_slc, node_omit_slc = slice(10), None

    base_artcfg = dict(pm="o-", om="o-", fm="--", ps=1.05, dw=1, fw=0.75, dc="#FFB05F", fc="black", fe=None)
    rand_artcfg = modcfg(base_artcfg, dict(dc="black"))
    unlim_artcfg = modcfg(base_artcfg, dict(dc="#FFEE00"))  # #F7F800

    # plot instances
    powerfit = True
    fit_file_str = "3_fpc_node_phase_supp_fits.txt"
    clear_fits(fit_file_str)

    ## splash plots
    cap_map = partial_cmap(plt.get_cmap("inferno"), 0.4, 0.9)
    mod_col = lambda x, v: modcfg(x, dict(dc=cap_map((v - 5) / 35)))

    plot_instance(res_n_ax.plot, nres_scale, nres_rand, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, "node rand res", rand_artcfg, powerfit=False) 
    [plot_instance(res_n_ax.plot, nres_scale, nrc, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, f"node {int(cap)}M res", mod_col(base_artcfg, cap), powerfit=False) for nrc, cap in zip(nres_constrained, ncap_ref)]
    plot_instance(res_n_ax.plot, nres_scale, nres, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, "node res", unlim_artcfg, powerfit=False)
    
    plot_instance(subopt_n_ax.plot, nres_scale, node_span, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, "node achievable span", rand_artcfg, powerfit=False)
    [plot_instance(subopt_n_ax.plot, nres_scale, nsc, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, f"node {int(cap)}M subopt", mod_col(base_artcfg, cap), powerfit=False) for nsc, cap in zip(nsubopts_constrained, ncap_ref)]
    plot_instance(subopt_n_ax.plot, nres_scale, node_subopt, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, "node subopt", unlim_artcfg, powerfit=False)

    plot_instance(subopt_res_ax.plot, nres_rand, node_span, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, "node rand subopt v res", rand_artcfg, powerfit=False)
    [plot_instance(subopt_res_ax.plot, nrc, nsc, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, f"node {int(cap)}M subopt v res", mod_col(base_artcfg, cap), powerfit=False) for nsc, nrc, cap in zip(nsubopts_constrained, nres_constrained, ncap_ref)]
    plot_instance(subopt_res_ax.plot, nres, node_subopt, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, "node subopt v res", unlim_artcfg, powerfit=False)

    ## phase plots
    for trial_idx, capv in enumerate(ncap_ref):
        nrc = nres_constrained[trial_idx]
        nsc = nsubopts_constrained[trial_idx]
        cap_probs = ncap_probs_constrained[trial_idx]

        phase_artcfg = modcfg(base_artcfg, dict(pm="-", dc="white", fm=":", ps=3))
        cap_colors = berlinColormap(cap_probs)

        low_thresh = 0.25
        high_thresh = 0.85
        low_buff = 0
        high_buff = 0
        nrc_phased, nsc_phased, cap_col_phased = split_by_triphase([nrc, nsc, cap_colors], cap_probs, low_thresh, high_thresh, low_buff, high_buff)

        ures, pres, fres = nrc_phased 
        usubopt, psubopt, fsubopt = nsc_phased
        ucols, pcols, fcols = cap_col_phased

        plot_heterochromia_instance(phase_axes[trial_idx], phase_axes[trial_idx].loglog, ures, usubopt, ucols, slice(1, len(ures)), slice(1), True, True, "grow", "positive", fit_file_str, f"node unconstrained {int(capv)}M subopt v res", phase_artcfg, powerfit=powerfit and len(ures) > 2)
        plot_heterochromia_instance(phase_axes[trial_idx], phase_axes[trial_idx].loglog, pres, psubopt, pcols, slice(len(pres)), None, True, True, "grow", "positive", fit_file_str, f"node partially-constrained {int(capv)}M subopt v res", phase_artcfg, powerfit=False)
        plot_heterochromia_instance(phase_axes[trial_idx], phase_axes[trial_idx].loglog, fres, fsubopt, fcols, slice(len(fres)), None, True, True, "grow", "positive", fit_file_str, f"node fully-constrained {int(capv)}M subopt v res", phase_artcfg, powerfit=powerfit and len(fres) > 1)


    # labelling axes
    label_size = 6
    tick_label_size = 4
    minor_tick_label_size = 4

    [ax.set_ylabel("Suboptimality gap", fontsize=label_size) for ax in (subopt_n_ax, subopt_res_ax, phase_axes[0])]
    [ax.set_ylabel("Mean edge residual", fontsize=label_size) for ax in (res_n_ax,)]

    [ax.set_xlabel("Nodes ($n$)", fontsize=label_size) for ax in [res_n_ax, subopt_n_ax]]
    [ax.set_xlabel("Mean edge residual", fontsize=label_size) for ax in [subopt_res_ax]]
    phase_axes[3].set_xlabel("Mean edge residual" + 27 * " ", fontsize=label_size)

    [ax.set_xticks(list(range(5, 51, 15))) for ax in splash_axes[:2]]

    [ax.tick_params(labelsize=tick_label_size) for ax in axes]
    [ax.tick_params(which="minor", labelsize=minor_tick_label_size) for ax in axes]

    [ax.tick_params(which="both", labelleft=False) for ax in phase_axes[1:]]

    phase_axes[0].set_yticks([0.001, 0.01, 0.1, 1, 10])
    phase_axes[0].set_yticks([20], minor=True)

    [forceAspect(ax) for ax in phase_axes]

    # legend
    rand_line = subopt_res_ax.get_lines()[1]
    unlim_line = subopt_res_ax.get_lines()[-1]
    subopt_res_ax.legend([rand_line, unlim_line], ["random", "unconstrained"], loc="upper left", prop={'size': tick_label_size}, ncols=1, frameon=False, framealpha=1)

    power_fit = phase_axes[-2].get_lines()[-1]
    phase_axes[-1].legend([power_fit], ["UPG"], loc="upper left", bbox_to_anchor=(0.0, 0.8), prop={'size': tick_label_size}, ncols=1, frameon=False, framealpha=1)

    [ax.text(0.075, 0.925, f"$M={int(capv)}$", fontsize=5, ha='left', va='top', transform=ax.transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round')) for ax, capv in zip(phase_axes, ncap_ref)]

    # add labelled subplots
    [fig.add_subplot(ax) for ax in axes]

    # colorbars
    splash_cbar_ax = fig.add_axes([splash_axes[-1].get_position().x1 + 0.015, splash_axes[-1].get_position().y0 , 0.01, splash_axes[-1].get_position().y1 - splash_axes[-1].get_position().y0])
    phase_cbar_ax = fig.add_axes([phase_axes[-1].get_position().x1 + 0.015, phase_axes[-1].get_position().y0 , 0.01, phase_axes[-1].get_position().y1 - phase_axes[-1].get_position().y0])

    splash_cbar = fig.colorbar(ScalarMappable(cmap=cap_map), cax=splash_cbar_ax, location="right", fraction=1, aspect=40)
    splash_cbar_ticks = (ncap_ref - 5) / 35
    splash_cbar.ax.set_yticks(splash_cbar_ticks, [Text(0, t, f"{int(v)}") for t, v in zip(splash_cbar_ticks, ncap_ref)])
    splash_cbar.set_label("Search move capacity ($M$)", rotation=90, fontsize=label_size)
    splash_cbar.ax.tick_params(labelsize=tick_label_size) 

    phase_cbar = fig.colorbar(ScalarMappable(cmap=berlinColormap), cax=phase_cbar_ax, location="right", fraction=1, aspect=40)
    phase_cbar_ticks = [0, 0.25, 0.5, 0.75, 1]
    phase_cbar.ax.set_yticks(phase_cbar_ticks, [Text(0, t, f"{int(100 * t)}") for t in phase_cbar_ticks])
    phase_cbar.set_label("Max depth %", rotation=90, fontsize=label_size)
    phase_cbar.ax.tick_params(labelsize=tick_label_size) 

    # saving
    save_fig(fig, "3_fpc_node_phase_supp", FORMAT)


def make_dim_phase_supp_plots(client):  # 2 figures, shotgun results and fully constrained isolations
    # setup axes
    tall_aspect = 0.75

    shotgun_fig = plt.figure(figsize=(5.5, 6.0))
    shot_grid = GridSpec(4, 4, hspace=0.45, wspace=0.4, height_ratios=[1] + 3 * [1 / tall_aspect])
    
    d10_2opt_axes = [plt.Subplot(shotgun_fig, shot_grid[idx, 0]) for idx in range(4)]
    d10_2exc_axes = [plt.Subplot(shotgun_fig, shot_grid[idx, 1], sharex=d10_2opt_axes[idx], sharey=d10_2opt_axes[idx]) for idx in range(4)]
    d20_2opt_axes = [plt.Subplot(shotgun_fig, shot_grid[idx, 2]) for idx in range(4)]
    d20_2exc_axes = [plt.Subplot(shotgun_fig, shot_grid[idx, 3], sharex=d20_2opt_axes[idx], sharey=d20_2opt_axes[idx]) for idx in range(4)]

    shotgun_axes = d10_2opt_axes + d10_2exc_axes + d20_2opt_axes + d20_2exc_axes

    subres_axes = shotgun_axes[0::4]
    res_axes = shotgun_axes[1::4]
    sub_axes = shotgun_axes[2::4]
    cap_axes = shotgun_axes[3::4]

    fc_fig = plt.figure(figsize=(3.67, 3.67))  # fully constrained isolations
    fc_grid = GridSpec(2, 2, wspace=0.45, hspace=0.225)
    
    fc_axes = [plt.Subplot(fc_fig, fc_grid[row_idx, col_idx]) for row_idx, col_idx in zip([0, 0, 1, 1], [0, 1, 0, 1])]

    axes = shotgun_axes + fc_axes

    [ax.axhline(0, color="black", linewidth=plt.rcParams['xtick.major.width']) for ax in shotgun_axes]
    [ax.axhline(100, color="black", linewidth=plt.rcParams['xtick.major.width']) for ax in cap_axes]

    # gather instance data
    CM_LV = 2
    CM_HV = 20
    cap_map = partial_cmap(plt.get_cmap("magma"), 0.2, 0.9)

    fit_file_str = "3_fpc_dim_phase_supp_fits.txt"
    clear_fits(fit_file_str)

    def rowplot(shotgun_axes, fc_ax, cap_map, dnodes, algo_str, powerfit=False):  # shotgun row plus corresponding fc entry
        constrained_exp_name = "EVAL_constrained_local_optima"
        bind_exp_name = "EVAL_bind"

        dcost_opt, _ = get_bind_eval(client, bind_exp_name, f"dim{dnodes}", f"opt_cost.mean")
        dcost_rand, _ = get_bind_eval(client, bind_exp_name, f"dim{dnodes}", f"rand_cost.mean")
        dres_rand, _ = get_bind_eval(client, bind_exp_name, f"dim{dnodes}", "rand_res.edge.mean")

        dcosts, _ = get_bind_eval(client, bind_exp_name, f"dim{dnodes}", f"_{algo_str}_cost.mean")
        dres, dres_scale = get_bind_eval(client, bind_exp_name, f"dim{dnodes}", f"_{algo_str}_res.edge.mean")

        # 2D local search experiments (2opt and 2exchange) both have corrupted, inflated results in EVAL_bind for 10-node version, where 20-node version does not, and I'm not sure why
        # code below substitutes values from earlier evals done on same proxy opt problems, where issue does not show up
        if dnodes == 10:  
            _2OPT_2D_COST = 2.8857
            _2SWAP_2D_COST = 2.9238
            dcosts[0] = _2OPT_2D_COST if algo_str == "2opt" else _2SWAP_2D_COST

        dcosts_constrained, dcap_ref = split_by_cap(client, constrained_exp_name, f"plus_dim{dnodes}_scaling_{algo_str}", f"{algo_str}_local_opt.costs_avg", "search_cap")
        dres_constrained, _ = split_by_cap(client, constrained_exp_name, f"plus_dim{dnodes}_scaling_{algo_str}", f"edge.{algo_str}_local_opt.residuals_avg", "search_cap")
        dcap_probs_constrained, _ = split_by_cap(client, constrained_exp_name, f"plus_dim{dnodes}_scaling_{algo_str}", f"edge.{algo_str}_local_opt.caps_avg", "search_cap")

        dres /= 2  # converts XOR to num mismatched edges (base stat is always even)
        dres_constrained = [arr / 2 for arr in dres_constrained]
        dres_rand /= 2

        dim_subopt = dcosts - dcost_opt
        dim_span = dcost_rand - dcost_opt
        dsubopts_constrained = [dcc - dcost_opt for dcc in dcosts_constrained]

        dim_fit_slc, dim_omit_slc = slice(len(dcosts)), None

        base_artcfg = dict(pm="o-", om="o-", fm=":", ps=1.05, dw=1, fw=0.75, dc="gray", fc="black", fe=None)
        rand_artcfg = modcfg(base_artcfg, dict(dc="black"))
        unlim_artcfg = modcfg(base_artcfg, dict(dc="deeppink"))  # #F7F800

        dim10_artcfg = modcfg(base_artcfg, dict(ps=2, dc="#605FFF"))
        dim20_artcfg = modcfg(base_artcfg, dict(ps=2, dc="#FF5FFE"))

        # plot instances
        ## splash plots
        mod_col = lambda x, v: modcfg(x, dict(dc=cap_map((v - CM_LV) / (CM_HV - CM_LV))))

        subopt_res_ax = shotgun_axes[0]
        res_n_ax = shotgun_axes[1]
        subopt_n_ax = shotgun_axes[2]
        cap_ax = shotgun_axes[3]

        [plot_instance(cap_ax.plot, dres_scale, 100 * cap_prob, dim_fit_slc, dim_omit_slc, False, False, "NONE", "NONE", fit_file_str, "NONE", mod_col(base_artcfg, cap), powerfit=False) for cap_prob, cap in zip(dcap_probs_constrained, dcap_ref)]

        plot_instance(res_n_ax.plot, dres_scale, dres_rand, dim_fit_slc, dim_omit_slc, True, True, "grow", "positive", fit_file_str, "dim rand res", rand_artcfg, powerfit=False) 
        [plot_instance(res_n_ax.plot, dres_scale, nrc, dim_fit_slc, dim_omit_slc, True, True, "grow", "positive", fit_file_str, f"dim {int(cap)}M res", mod_col(base_artcfg, cap), powerfit=False) for nrc, cap in zip(dres_constrained, dcap_ref)]
        plot_instance(res_n_ax.plot, dres_scale, dres, dim_fit_slc, dim_omit_slc, True, True, "grow", "positive", fit_file_str, "dim res", unlim_artcfg, powerfit=False)
        
        plot_instance(subopt_n_ax.plot, dres_scale, dim_span, dim_fit_slc, dim_omit_slc, True, True, "grow", "positive", fit_file_str, "dim achievable span", rand_artcfg, powerfit=False)
        [plot_instance(subopt_n_ax.plot, dres_scale, nsc, dim_fit_slc, dim_omit_slc, True, True, "grow", "positive", fit_file_str, f"dim {int(cap)}M subopt", mod_col(base_artcfg, cap), powerfit=False) for nsc, cap in zip(dsubopts_constrained, dcap_ref)]
        plot_instance(subopt_n_ax.plot, dres_scale, dim_subopt, dim_fit_slc, dim_omit_slc, True, True, "grow", "positive", fit_file_str, "dim subopt", unlim_artcfg, powerfit=False)

        plot_instance(subopt_res_ax.plot, dres_rand, dim_span, dim_fit_slc, dim_omit_slc, True, True, "grow", "positive", fit_file_str, "dim rand subopt v res", rand_artcfg, powerfit=False)
        [plot_instance(subopt_res_ax.plot, nrc, nsc, dim_fit_slc, dim_omit_slc, True, True, "grow", "positive", fit_file_str, f"dim {int(cap)}M subopt v res", mod_col(base_artcfg, cap), powerfit=False) for nsc, nrc, cap in zip(dsubopts_constrained, dres_constrained, dcap_ref)]
        plot_instance(subopt_res_ax.plot, dres, dim_subopt, dim_fit_slc, dim_omit_slc, True, True, "grow", "positive", fit_file_str, "dim subopt v res", unlim_artcfg, powerfit=False)

        ## fully constrained iso plots
        if dnodes == 20 and algo_str == "2opt":
            fc_fit_slc, fc_omit_slc = slice(12), slice(11, 17)
        elif dnodes == 20:
            fc_fit_slc, fc_omit_slc = slice(6), slice(5, 17)
        else:
            fc_fit_slc, fc_omit_slc = slice(len(dres)), None

        plot_instance(fc_ax.loglog, dres, dim_subopt, fc_fit_slc, fc_omit_slc, True, True, "grow", "positive", fit_file_str, f"{algo_str} dim{dnodes} subopt v res", (dim10_artcfg if dnodes == 10 else dim20_artcfg), powerfit=powerfit, near_linear=dnodes == 10 and algo_str == "2swap")

        return dcap_ref


    dcap_ref1 = rowplot(d10_2opt_axes, fc_axes[0], cap_map, 10, "2opt", True)
    dcap_ref2 = rowplot(d20_2opt_axes, fc_axes[2], cap_map, 20, "2opt", True)
    dcap_ref3 = rowplot(d10_2exc_axes, fc_axes[1], cap_map, 10, "2swap", True)
    dcap_ref4 = rowplot(d20_2exc_axes, fc_axes[3], cap_map, 20, "2swap", True)

    all_dcaps = np.unique(np.concatenate([dcap_ref1, dcap_ref2, dcap_ref3, dcap_ref4]))

    # labelling axes
    label_size = 6
    tick_label_size = 4
    minor_tick_label_size = 4

    d10_2opt_axes[0].set_title("2-opt " + r"$\mathbf{n=10}$", fontsize=label_size, fontweight="bold")
    d10_2exc_axes[0].set_title("2-exchange " + r"$\mathbf{n=10}$", fontsize=label_size, fontweight="bold")
    d20_2opt_axes[0].set_title("2-opt " + r"$\mathbf{n=20}$", fontsize=label_size, fontweight="bold")
    d20_2exc_axes[0].set_title("2-exchange " + r"$\mathbf{n=20}$", fontsize=label_size, fontweight="bold")

    [ax.set_ylabel("Suboptimality gap", fontsize=label_size) for ax in subres_axes + sub_axes + fc_axes]
    [ax.set_ylabel("Mean edge residual", fontsize=label_size) for ax in res_axes]
    [ax.set_ylabel("Max depth %", fontsize=label_size) for ax in cap_axes]

    [ax.set_xlabel("Spatial dimensions ($d$)", fontsize=label_size) for ax in shotgun_axes]
    [ax.set_xlabel("Mean edge residual", fontsize=label_size) for ax in subres_axes + fc_axes]

    [ax.tick_params(labelsize=tick_label_size) for ax in axes]
    [ax.tick_params(which="minor", labelsize=minor_tick_label_size) for ax in axes]

    [ax.set_xlim(left=0) for ax in subres_axes]

    fc_axes[0].set_xticks([0.6, 0.8], minor=True)
    fc_axes[1].set_xticks([1.5, 2.0, 2.5], minor=True)

    [ax.xaxis.set_minor_formatter(mticker.ScalarFormatter()) for ax in fc_axes]
    [ax.xaxis.set_major_formatter(mticker.ScalarFormatter()) for ax in fc_axes]

    [forceAspect(ax, aspect=tall_aspect) for ax in shotgun_axes]
    [forceAspect(ax) for ax in subres_axes + fc_axes]

    # legend
    rand_line = subres_axes[0].get_lines()[1]
    unlim_line = subres_axes[0].get_lines()[-1]
    subres_axes[0].legend([rand_line, unlim_line], ["random", "unconstrained"], loc="upper left", prop={'size': tick_label_size}, ncols=1, frameon=False, framealpha=1)

    [ax.text(0.05, 0.95, f"2-opt", fontsize=5, ha='left', va='top', transform=ax.transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round')) for ax in fc_axes[::2]]
    [ax.text(0.05, 0.95, f"2-exchange", fontsize=5, ha='left', va='top', transform=ax.transAxes, bbox=dict(facecolor='none', edgecolor='black', boxstyle='round')) for ax in fc_axes[1::2]]

    d10p, fitline = fc_axes[0].get_lines()
    d20p, d20p_omit, _ = fc_axes[2].get_lines()
    fc_fig.legend([d10p, d20p, fitline, d20p_omit], ["$n=10$ trend inputs", "$n=20$ trend inputs", "Unbounded growth", "$n=20$ trend break"], loc="upper center", prop={'size': label_size}, bbox_to_anchor=(0.5, 0.02), ncols=2, frameon=False)

    # add labelled subplots
    [shotgun_fig.add_subplot(ax) for ax in shotgun_axes]
    [fc_fig.add_subplot(ax) for ax in fc_axes]

    # colorbars
    cap_cbar_ax = shotgun_fig.add_axes([subres_axes[-1].get_position().x1 + 0.025, cap_axes[-1].get_position().y0 , 0.01, subres_axes[-1].get_position().y1 - cap_axes[-1].get_position().y0])

    cap_cbar = shotgun_fig.colorbar(ScalarMappable(cmap=cap_map), cax=cap_cbar_ax, location="right", fraction=1, aspect=40)
    cap_cbar_ticks = (all_dcaps - CM_LV) / (CM_HV - CM_LV)
    cap_cbar.ax.set_yticks(cap_cbar_ticks, [Text(0, t, f"{int(v)}") for t, v in zip(cap_cbar_ticks, all_dcaps)])
    cap_cbar.set_label("Search move capacity ($M$)", rotation=90, fontsize=label_size)
    cap_cbar.ax.tick_params(labelsize=tick_label_size) 

    # saving
    save_fig(shotgun_fig, "3_fpc_dim_phase_supp_shotgun", FORMAT)
    save_fig(fc_fig, "3_fpc_dim_phase_supp_fc", FORMAT)


def make_unexpected_supp_plots(client):
    def plot_algo(algo_str):
        # setup axes
        fig = plt.figure(figsize=(5.0, 5.0))
        grid = GridSpec(3, 3, wspace=0.45, hspace=0.45)

        dim10_axes = [plt.Subplot(fig, grid[0, idx]) for idx in range(3)]
        dim20_axes = [plt.Subplot(fig, grid[1, idx]) for idx in range(3)]
        node_axes = [plt.Subplot(fig, grid[2, idx]) for idx in range(3)]

        rows = (node_axes, dim10_axes, dim20_axes)

        ulo_axes = [axes[0] for axes in rows]
        blo_axes = [axes[1] for axes in rows]
        sm_axes = [axes[2] for axes in rows]

        axes = node_axes + dim10_axes + dim20_axes

        bigk_descents = 100_000
        proxy10_descents = 100
        proxy20_descents = 1000 if algo_str == "2opt" else 100

        node_axes[0].axhline(bigk_descents, color="black", linewidth=plt.rcParams['xtick.major.width'], zorder=0)
        if algo_str == "2opt": node_axes[1].axhline(100, color="black", linewidth=plt.rcParams['xtick.major.width'], zorder=0)

        # gather instance data
        node_sm, node_sm_scale = get_proxynode_data(client, "search_swaps_avg", algo=algo_str)
        node_blo, node_blo_scale = get_bigk_data(client, f"node_scaling_{algo_str}", "blo_hits_avg")
        node_ulo, node_ulo_scale = get_bigk_data(client, f"node_scaling_{algo_str}", "unique_lo_avg")
        #assert all([np.allclose(node_sm_scale, arr) for arr in (node_blo_scale, node_ulo_scale)])
        
        d10_sm, d10_sm_scale = get_proxy10n_data(client, "search_swaps_avg", algo=algo_str)
        d10_blo, d10_blo_scale = get_bigk_data(client, f"dim_scaling_10n_{algo_str}", "blo_hits_avg") 
        d10_ulo, d10_ulo_scale = get_bigk_data(client, f"dim_scaling_10n_{algo_str}", "unique_lo_avg")
        d10_blo_lilk, d10_blo_scale_lilk = get_proxy10n_data(client, "blo_hits_avg", algo=algo_str)
        d10_ulo_lilk, d10_ulo_scale_lilk = get_proxy10n_data(client, "unique_lo_avg", algo=algo_str)
        #assert all([np.allclose(d10_sm_scale, arr) for arr in (d10_blo_scale, d10_ulo_scale, d10_blo_scale_lilk, d10_ulo_scale_lilk)])

        d20_sm, d20_sm_scale = get_proxy20n_data(client, "search_swaps_avg", algo=algo_str)
        d20_blo, d20_blo_scale = get_bigk_data(client, f"dim_scaling_20n_2opt", "blo_hits_avg")
        d20_ulo, d20_ulo_scale = get_bigk_data(client, f"dim_scaling_20n_2opt", "unique_lo_avg")
        d20_blo_lilk, d20_blo_scale_lilk = get_proxy20n_data(client, "blo_hits_avg", algo=algo_str)
        d20_ulo_lilk, d20_ulo_scale_lilk = get_proxy20n_data(client, "unique_lo_avg", algo=algo_str)
        #assert all([np.allclose(d20_sm_scale, arr) for arr in (d20_blo_scale, d20_ulo_scale, d20_blo_scale_lilk, d20_ulo_scale_lilk)])

        node_blo /= bigk_descents * 0.01  # percentage
        d10_blo /= bigk_descents * 0.01
        d20_blo /= bigk_descents * 0.01
        d10_blo_lilk /= proxy10_descents * 0.01
        d20_blo_lilk /= proxy20_descents * 0.01

        node_fit_slc, node_omit_slc = slice(len(node_sm_scale)), None # slice(9), slice(8, 10)
        d10_fit_slc, d10_omit_slc = slice(len(d10_sm_scale)), None # slice(11), None
        d20_fit_slc, d20_omit_slc = slice(len(d20_sm_scale)), None 

        node_artcfg = dict(pm="o-", om="o-", fm=":", ps=3, dw=1, fw=1.0, dc="#FFB05F", fc="black", fe=None)
        d10_artcfg = dict(pm="o-", om="o-", fm="--", ps=3, dw=1, fw=1.0, dc="#605FFF", fc="black", fe=None)
        d20_artcfg = dict(pm="o-", om="o-", fm="--", ps=3, dw=1, fw=1.0, dc="#FF5FFE", fc="black", fe=None)
        lilk_artcfg = dict(pm="^-", om="o-", fm="--", ps=2, dw=0.75, fw=1.0, dc="deepskyblue", fc="black", fe=None)

        # plot instances
        powerfit = True
        fit_file_str = f"3_fpc_unexpected_fits_{algo_str}.txt"
        if powerfit: clear_fits(fit_file_str)

        plot_instance(node_axes[0].semilogy, node_ulo_scale, node_ulo, node_fit_slc, node_omit_slc, True, True, "grow", "positive", fit_file_str, "node ULO", node_artcfg, powerfit=False)
        plot_instance(node_axes[1].semilogy, node_blo_scale, node_blo, node_fit_slc, node_omit_slc, True, False, "decay", "positive", fit_file_str, "node BLO hits", node_artcfg, powerfit=False)
        plot_instance(node_axes[2].plot, node_sm_scale, node_sm, node_fit_slc, node_omit_slc, False, True, "grow", "positive", fit_file_str, "node search moves", node_artcfg, powerfit=powerfit, near_linear=True)

        plot_instance(dim10_axes[0].semilogx, d10_ulo_scale, d10_ulo, d10_fit_slc, d10_omit_slc, True, True, "decay", "negative", fit_file_str, f"dim10 ULO", d10_artcfg, powerfit=False)
        plot_instance(dim10_axes[0].semilogx, d10_ulo_scale_lilk, d10_ulo_lilk, d10_fit_slc, d10_omit_slc, True, True, "decay", "negative", fit_file_str, f"dim10 ULO", lilk_artcfg, powerfit=False)
        plot_instance(dim10_axes[1].loglog, d10_blo_scale, d10_blo, d10_fit_slc, d10_omit_slc, True, True, "decay", "positive", fit_file_str, f"dim10 BLO hits", d10_artcfg, powerfit=False)
        popt_blo10 = plot_instance(dim10_axes[1].loglog, d10_blo_scale_lilk, d10_blo_lilk, d10_fit_slc, d10_omit_slc, True, True, "decay", "positive", fit_file_str, f"dim10 BLO hits", lilk_artcfg, powerfit=powerfit)
        popt_sm10 = plot_instance(dim10_axes[2].loglog, d10_sm_scale, d10_sm, d10_fit_slc, d10_omit_slc, True, True, "decay", "positive", fit_file_str, f"dim10 search moves", d10_artcfg, powerfit=powerfit)

        plot_instance(dim20_axes[0].semilogx, d20_ulo_scale, d20_ulo, d20_fit_slc, d20_omit_slc, True, True, "decay", "negative", fit_file_str, f"dim20 ULO", d20_artcfg, powerfit=False)
        if algo_str == "2opt": plot_instance(dim20_axes[0].semilogx, d20_ulo_scale_lilk, d20_ulo_lilk, d20_fit_slc, d20_omit_slc, True, True, "decay", "negative", fit_file_str, f"dim20 ULO", lilk_artcfg, powerfit=False)
        plot_instance(dim20_axes[1].loglog, d20_blo_scale, d20_blo, d20_fit_slc, d20_omit_slc, True, True, "decay", "positive", fit_file_str, f"dim20 BLO hits", d20_artcfg, powerfit=powerfit and algo_str == "2swap")
        if algo_str == "2opt": popt_blo20 = plot_instance(dim20_axes[1].loglog, d20_blo_scale_lilk, d20_blo_lilk, d20_fit_slc, d20_omit_slc, True, True, "decay", "positive", fit_file_str, f"dim20 BLO hits", lilk_artcfg, powerfit=powerfit)
        popt_sm20 = plot_instance(dim20_axes[2].loglog, d20_sm_scale, d20_sm, d20_fit_slc, d20_omit_slc, True, True, "decay", "positive", fit_file_str, f"dim20 search moves", d20_artcfg, powerfit=powerfit)

        # labelling axes
        label_size = 6
        tick_label_size = 4
        minor_tick_label_size = 4
    
        [ax.set_ylabel("Mean search moves", fontsize=label_size) for ax in sm_axes]
        [ax.set_ylabel("Mean BFLO visitation %", fontsize=label_size) for ax in blo_axes]
        [ax.set_ylabel("Mean local optima count", fontsize=label_size) for ax in ulo_axes]

        [ax.set_xlabel("Nodes ($n$)", fontsize=label_size) for ax in node_axes]
        [ax.set_xlabel("Spatial dimensions ($d$)", fontsize=label_size) for ax in dim10_axes + dim20_axes]

        [ax.set_xticks(list(range(5, 51, 15))) for ax in node_axes]

        [ax.yaxis.set_minor_formatter(mticker.ScalarFormatter()) for ax in dim10_axes[1:] + dim20_axes[1:]]
        [ax.yaxis.set_major_formatter(mticker.ScalarFormatter()) for ax in dim10_axes[1:] + dim20_axes[1:]]

        if algo_str == "2opt":
            node_axes[1].yaxis.set_major_formatter(mticker.ScalarFormatter())

            dim10_axes[1].set_yticks([60, 70, 80], minor=True)
            dim10_axes[2].set_yticks([5.3, 5.4, 5.5, 5.6], minor=True)
            dim20_axes[1].set_yticks([10])
            dim20_axes[1].set_yticks([6, 20, 30, 40], minor=True)
            dim20_axes[2].set_yticks([13.6, 14.0, 14.4, 14.8], minor=True)
            node_axes[1].set_yticks([1, 10, 100])
            
            node_axes[2].set_ylim(bottom=0)
            dim10_axes[0].set_ylim(bottom=0)
            dim20_axes[0].set_ylim(bottom=0)

        elif algo_str == "2swap":
            node_axes[1].set_ylim(top=100)
            node_axes[2].set_ylim(bottom=0)
            dim10_axes[0].set_ylim(bottom=0)
            dim20_axes[0].set_ylim(bottom=0)

            dim10_axes[1].set_yticks([20, 30, 40, 50, 60, 70], minor=True)
            dim10_axes[2].set_yticks([4.4, 4.6, 4.8, 5.0, 5.2], minor=True)
            dim20_axes[1].set_yticks([10])
            dim20_axes[1].set_yticks([6, 20, 30, 40], minor=True)
            dim20_axes[2].set_yticks([10])
            dim20_axes[2].set_yticks([11, 12, 13, 14], minor=True)
            # node_axes[1].set_yticks([1, 10, 100])
            
        [ax.tick_params(labelsize=tick_label_size) for ax in axes]
        [ax.tick_params(axis="y", which="minor", labelsize=minor_tick_label_size) for ax in axes]

        # legend
        d10p, d10p_lilk, decay = dim10_axes[1].get_lines()
        d20p, _ = dim20_axes[-1].get_lines()
        nodep, bound_growth = node_axes[-1].get_lines()
        fig.legend([d10p, d20p, d10p_lilk, nodep, decay, bound_growth], ["Spatial dimension scaling ($n=10$)", "Spatial dimension scaling ($n=20$)", "Small-$k$ / large-batch", "Node scaling", "$y - \\beta \propto (d - \\gamma)^{{-\\alpha}}$", "$y \propto (n - \\gamma)$"], loc="upper center", prop={'size': label_size}, bbox_to_anchor=(0.5, 0.03), ncols=3, frameon=False)

        # betas
        beta_col = "slategrey"

        blo10_beta = popt_blo10[-1] * popt_blo10[0][0]
        dim10_axes[1].axhline(blo10_beta, color=beta_col, linewidth=1, zorder=1)
        xmin, _ = dim10_axes[1].get_xlim()
        dim10_axes[1].text(xmin+0.5, blo10_beta+0.0003, r"$\beta$", color=beta_col, fontsize=label_size, ha='left', va='bottom')
        
        blo20_beta = popt_blo20[-1] * popt_blo20[0][0]
        dim20_axes[1].axhline(blo20_beta, color=beta_col, linewidth=1, zorder=1)
        xmin, _ = dim20_axes[1].get_xlim()
        dim20_axes[1].text(xmin+0.5, blo20_beta+0.0003, r"$\beta$", color=beta_col, fontsize=label_size, ha='left', va='bottom')
        
        sm10_beta = popt_sm10[-1] * popt_sm10[0][0]
        dim10_axes[2].axhline(sm10_beta, color=beta_col, linewidth=1, zorder=1)
        xmin, _ = dim10_axes[1].get_xlim()
        dim10_axes[2].text(xmin+0.5, sm10_beta+0.0003, r"$\beta$", color=beta_col, fontsize=label_size, ha='left', va='bottom')
        
        sm20_beta = popt_sm20[-1] * popt_sm20[0][0]
        dim20_axes[2].axhline(sm20_beta, color=beta_col, linewidth=1, zorder=1)
        xmin, _ = dim20_axes[1].get_xlim()
        dim20_axes[2].text(xmin+0.5, sm20_beta+0.0003, r"$\beta$", color=beta_col, fontsize=label_size, ha='left', va='bottom')
        
        # add labelled subplots
        [fig.add_subplot(ax) for ax in axes]

        # saving
        save_fig(fig, f"3_fpc_unexpected_{algo_str}", "pdf")
        save_fig(fig, f"3_fpc_unexpected_{algo_str}", "png")

    plot_algo("2opt")
    # plot_algo("2swap")




if __name__ == "__main__":
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    client = MlflowClient(tracking_uri)

    if PLOT_SEARCH_PROBLEM_SCALING_NIPS:
        make_search_complexity_plots_nips(client)

    if PLOT_SEARCH_PROBLEM_SCALING_NIPS_2EXC:
        make_search_complexity_plots_nips_2exc(client)

    if PLOT_SOL_RES_MAIN_NIPS:
        make_sol_res_main_plots_nips(client)

    if PLOT_JOINT_RES_ISO:
        make_joint_res_iso_plots(client)
    
    if PLOT_SOL_RES_MAIN:
        make_sol_res_main_plots(client)

    if PLOT_SOL_RES_SUPP:
        make_sol_res_supp_plots(client)

    if PLOT_UNLIM_LO_RES_MAIN:
        make_unlim_lo_res_plots(client, "2opt")

    if PLOT_UNLIM_LO_RES_SUPP:
        make_unlim_lo_res_plots(client, "2swap")

    if PLOT_SEARCHCAP_MAIN:
        make_lo_searchcap_plots(client, "2opt")

    if PLOT_SEARCHCAP_SUPP:
        make_lo_searchcap_plots(client, "2swap")

    if PLOT_NODE_PHASE_MAIN:
        make_node_phase_main_plots(client)

    if PLOT_NODE_PHASE_SUPP:
        make_node_phase_supp_plots(client)

    if PLOT_DIM_PHASE_SUPP:
        make_dim_phase_supp_plots(client)

    if PLOT_UNEXPECTED_SUPP:
        make_unexpected_supp_plots(client)
