
import os.path as osp
import os
from multiprocessing import Process, Queue
import numpy as np
import torch
import argparse

from tsp.utils import seed_rand_gen
from tsp.logger import MLflowLogger as Logger, parse_log_sig
from localsearch.distance import batch_edge_hamming_distance, batch_tei_node_hamming_distance
from localsearch.utils import drain_queue, np_get_costs

parser = argparse.ArgumentParser()
parser.add_argument("scale", default=None, type=str, choices=["node", "dim10", "dim20", "model"])


DEBUG = False  # serially launch just the smallest scale per expirement
LIGHTWEIGHT = None  # use just first K batch slice of each dataset array, or None for full experiment

MLFLOW_EXP = "EVAL_bind"

NODE_RANGE = range(5, 51, 5)
DIM_RANGE = list(range(2, 13)) + [15, 20, 30, 40, 50, 100]
MODEL_WIDTHS = [24, 32, 40, 48, 56, 72, 88, 104, 128, 160, 192, 240]

NSCALE_SIZE = 1_280_000
DSCALE_10N_SIZE = 128_000
DSCALE_20N_SIZE = 64_000
RAND_PER_PROB = 10  # random tours per problem; this must be at least 2; only stat that uses past first 2 problems is for problem-specific-RON

import tsp
root_path = osp.dirname(osp.dirname(tsp.__file__))
BIND_PATH = osp.join(root_path, "bindthem")




def get_random_tours(probs, rand_per_prob):
    b, n = probs.shape[:2]
    rand_tours = torch.stack([torch.randperm(n) for _ in range(rand_per_prob * b)])
    return rand_tours.view(b, rand_per_prob, n).numpy().astype(np.int8)


def get_fitness(probs, tours=None):
    if tours is None:
        return np_get_costs(probs)
    else:
        tours_per_prob = tours.shape[1]
        costs = [np_get_costs(probs, tours[:, idx]) for idx in range(tours_per_prob)]
        return np.stack(costs, axis=1)
    

def get_prob_specific_ron(costs, prob_mean_rand_cost, opt_cost):
    ron_slices = [(costs[:, idx] - opt_cost) / (prob_mean_rand_cost - opt_cost) for idx in range(costs.shape[1])]
    return np.stack(ron_slices, axis=1)


def get_distances(tours_a, tours_b):
    edge_dist = batch_edge_hamming_distance(tours_a, tours_b)
    tei_node_dist = batch_tei_node_hamming_distance(tours_a, tours_b)
    return edge_dist, tei_node_dist


def get_min_distances(exp_tours, ref_tours):  # experiment tours (batch, nodes), and reference tours (batch, sol_per_prob, nodes)
    dists = [get_distances(exp_tours, ref_tours[:, idx]) for idx in range(ref_tours.shape[1])]
    edge_list, tein_list = list(zip(*dists))
    edge_stack, tein_stack = np.stack(edge_list, axis=1), np.stack(tein_list, axis=1)
    
    edge_mins = edge_stack.min(axis=1)
    tein_mins = tein_stack.min(axis=1)

    edge_argmins = np.expand_dims(edge_stack.argmin(axis=1), axis=1)  # take_along_axis() compatible
    tein_argmins = np.expand_dims(tein_stack.argmin(axis=1), axis=1)
    
    return (edge_mins, edge_argmins), (tein_mins, tein_argmins)
    

def get_covariance(fitness, fit_mean, edge_dist, edge_mean, tein_dist, tein_mean):
    fit_delta = fitness - fit_mean
    edge_cov = np.mean(fit_delta * (edge_dist - edge_mean), axis=0)
    tein_cov = np.mean(fit_delta * (tein_dist - tein_mean), axis=0)
    return edge_cov, tein_cov


def get_specific_covariance(fitness, fit_mean, dist, dist_mean):  # for an individual distance measure, rather than both edge and tein
    fit_delta = fitness - fit_mean
    return np.mean(fit_delta * (dist - dist_mean), axis=0)


def get_correlation(edge_cov, tein_cov, fit_std, edge_std, tein_std):
    edge_corr = edge_cov / (fit_std * edge_std)
    tein_corr = tein_cov / (fit_std * tein_std)
    return edge_corr, tein_corr


def get_specific_correlation(cov, fit_std, dist_std):  # for an individual distance measure, rather than both edge and tein
    return cov / (fit_std * dist_std)


def stat(arr):
    return dict(
        mean = arr.mean().item(),
        median = np.median(arr).item(),
        min = arr.min().item(),
        max = arr.max().item(),
        std = arr.std().item(),
        size = arr.size
    )


def dstat(edge_arr, tein_arr):
    return dict(
        edge=stat(edge_arr),
        tein=stat(tein_arr)
    )


def dmet(edge_val, tein_val):
    return dict(
        edge=edge_val,
        tein=tein_val
    )


def eval_node_scale(retv_q, n):
    seed_rand_gen(int.from_bytes(os.urandom(4), "big"))

    dpath = f"node_scale_{n}n_2d.npz" if n != 20 else f"node_scale_{n}n_2d_plus_model_scale.npz"
    dpath = osp.join(BIND_PATH, dpath)
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

    if LIGHTWEIGHT is not None:  # making things wayyy faster
        optimals = optimals[:LIGHTWEIGHT]
    
        _2opt_lo = _2opt_lo[:LIGHTWEIGHT]
        _2opt_starts = _2opt_starts[:LIGHTWEIGHT]
        _2opt_swaps = _2opt_swaps[:LIGHTWEIGHT]

        _2swap_lo = _2swap_lo[:LIGHTWEIGHT]
        _2swap_starts = _2swap_starts[:LIGHTWEIGHT]
        _2swap_swaps = _2swap_swaps[:LIGHTWEIGHT]

        drl_node_sol = drl_node_sol[:LIGHTWEIGHT]
        il_node_sol = il_node_sol[:LIGHTWEIGHT]

    # gen random tours
    rand_sol = get_random_tours(optimals, RAND_PER_PROB)

    # gather fitness (cost aka tour length)
    opt_cost = get_fitness(optimals)
    rand_cost = get_fitness(optimals, rand_sol)
    _2opt_cost = get_fitness(optimals, _2opt_lo)
    _2swap_cost = get_fitness(optimals, _2swap_lo)
    drl_node_cost = get_fitness(optimals, drl_node_sol)
    il_node_cost = get_fitness(optimals, il_node_sol)

    _2opt_start_cost = get_fitness(optimals, _2opt_starts)
    _2swap_start_cost = get_fitness(optimals, _2swap_starts)

    # expectation RON scores
    avg_opt_cost = opt_cost.mean()
    avg_rand_cost = rand_cost[:, 0].mean()  # taking only first slice of tours to not artificially inflate sample size

    avg_2opt_ron = (_2opt_cost[:, 0].mean() - avg_opt_cost) / (avg_rand_cost - avg_opt_cost)  # ditto, and below
    avg_2swap_ron = (_2swap_cost[:, 0].mean() - avg_opt_cost) / (avg_rand_cost - avg_opt_cost) 
    avg_drl_node_ron = (drl_node_cost[:, 0].mean() - avg_opt_cost) / (avg_rand_cost - avg_opt_cost) 
    avg_il_node_ron = (il_node_cost[:, 0].mean() - avg_opt_cost) / (avg_rand_cost - avg_opt_cost) 
    
    # specific per-problem RON scores (random is a noisier estimate based on RAND_PER_PROB; optimal is exact)
    prob_mean_rand_cost = rand_cost.mean(axis=1)  # sol_per_prob axis only

    spec_2opt_ron = get_prob_specific_ron(_2opt_cost, prob_mean_rand_cost, opt_cost)
    spec_2swap_ron = get_prob_specific_ron(_2swap_cost, prob_mean_rand_cost, opt_cost)
    spec_drl_node_ron = get_prob_specific_ron(drl_node_cost, prob_mean_rand_cost, opt_cost)
    spec_il_node_ron = get_prob_specific_ron(il_node_cost, prob_mean_rand_cost, opt_cost)

    # intra-population spread distance of solutions
    rand_spread_edge, rand_spread_tein = get_distances(rand_sol[:, 0], rand_sol[:, 1])
    _2opt_spread_edge, _2opt_spread_tein = get_distances(_2opt_lo[:, 0], _2opt_lo[:, 1])
    _2swap_spread_edge, _2swap_spread_tein = get_distances(_2swap_lo[:, 0], _2swap_lo[:, 1])
    drl_node_spread_edge, drl_node_spread_tein = get_distances(drl_node_sol[:, 0], drl_node_sol[:, 1])
    il_node_spread_edge, il_node_spread_tein = get_distances(il_node_sol[:, 0], il_node_sol[:, 1])

    # global optima residual distances
    opt_sol = np.stack(optimals.shape[0] * [np.arange(optimals.shape[1])], axis=0)

    rand_res_edge, rand_res_tein = get_distances(rand_sol[:, 0], opt_sol)
    _2opt_res_edge, _2opt_res_tein = get_distances(_2opt_lo[:, 0], opt_sol)
    _2swap_res_edge, _2swap_res_tein = get_distances(_2swap_lo[:, 0], opt_sol)
    drl_node_res_edge, drl_node_res_tein = get_distances(drl_node_sol[:, 0], opt_sol)
    il_node_res_edge, il_node_res_tein = get_distances(il_node_sol[:, 0], opt_sol)

    # local search roll distances (between start and delivered local optima)
    _2opt_roll_edge, _2opt_roll_tein = get_distances(_2opt_lo[:, 0], _2opt_starts[:, 0])
    _2swap_roll_edge, _2swap_roll_tein = get_distances(_2swap_lo[:, 0], _2swap_starts[:, 0])

    # remaining inter-population distances of interest (residuals cover distances to optimals, and random isn't interesting)
    _2opt_2swap_dist_edge, _2opt_2swap_dist_tein = get_distances(_2opt_lo[:, 0], _2swap_lo[:, 0])
    drl_2opt_dist_edge, drl_2opt_dist_tein = get_distances(drl_node_sol[:, 0], _2opt_lo[:, 0])
    drl_2swap_dist_edge, drl_2swap_dist_tein = get_distances(drl_node_sol[:, 0], _2swap_lo[:, 0])
    il_2opt_dist_edge, il_2opt_dist_tein = get_distances(il_node_sol[:, 0], _2opt_lo[:, 0])
    il_2swap_dist_edge, il_2swap_dist_tein = get_distances(il_node_sol[:, 0], _2swap_lo[:, 0])
    drl_il_dist_edge, drl_il_dist_tein = get_distances(drl_node_sol[:, 0], il_node_sol[:, 0])

    # global optima residual FDCs
    rand_res_cov_edge, rand_res_cov_tein = get_covariance(rand_cost[:, 0], avg_rand_cost, rand_res_edge, rand_res_edge.mean(), rand_res_tein, rand_res_tein.mean())
    _2opt_res_cov_edge, _2opt_res_cov_tein = get_covariance(_2opt_cost[:, 0], _2opt_cost[:, 0].mean(), _2opt_res_edge, _2opt_res_edge.mean(), _2opt_res_tein, _2opt_res_tein.mean())
    _2swap_res_cov_edge, _2swap_res_cov_tein = get_covariance(_2swap_cost[:, 0], _2swap_cost[:, 0].mean(), _2swap_res_edge, _2swap_res_edge.mean(), _2swap_res_tein, _2swap_res_tein.mean())
    drl_node_res_cov_edge, drl_node_res_cov_tein = get_covariance(drl_node_cost[:, 0], drl_node_cost[:, 0].mean(), drl_node_res_edge, drl_node_res_edge.mean(), drl_node_res_tein, drl_node_res_tein.mean())
    il_node_res_cov_edge, il_node_res_cov_tein = get_covariance(il_node_cost[:, 0], il_node_cost[:, 0].mean(), il_node_res_edge, il_node_res_edge.mean(), il_node_res_tein, il_node_res_tein.mean())

    rand_res_corr_edge, rand_res_corr_tein = get_correlation(rand_res_cov_edge, rand_res_cov_tein, rand_cost[:, 0].std(), rand_res_edge.std(), rand_res_tein.std())
    _2opt_res_corr_edge, _2opt_res_corr_tein = get_correlation(_2opt_res_cov_edge, _2opt_res_cov_tein, _2opt_cost[:, 0].std(), _2opt_res_edge.std(), _2opt_res_tein.std())
    _2swap_res_corr_edge, _2swap_res_corr_tein = get_correlation(_2swap_res_cov_edge, _2swap_res_cov_tein, _2swap_cost[:, 0].std(), _2swap_res_edge.std(), _2swap_res_tein.std())
    drl_node_res_corr_edge, drl_node_res_corr_tein = get_correlation(drl_node_res_cov_edge, drl_node_res_cov_tein, drl_node_cost[:, 0].std(), drl_node_res_edge.std(), drl_node_res_tein.std())
    il_node_res_corr_edge, il_node_res_corr_tein = get_correlation(il_node_res_cov_edge, il_node_res_cov_tein, il_node_cost[:, 0].std(), il_node_res_edge.std(), il_node_res_tein.std())

    # local search roll FDCs, LO experiment and random start reference
    _2opt_roll_lo_cov_edge, _2opt_roll_lo_cov_tein = get_covariance(_2opt_cost[:, 0], _2opt_cost[:, 0].mean(), _2opt_roll_edge, _2opt_roll_edge.mean(), _2opt_roll_tein, _2opt_roll_tein.mean())
    _2swap_roll_lo_cov_edge, _2swap_roll_lo_cov_tein = get_covariance(_2swap_cost[:, 0], _2swap_cost[:, 0].mean(), _2swap_roll_edge, _2swap_roll_edge.mean(), _2swap_roll_tein, _2swap_roll_tein.mean())

    _2opt_roll_lo_corr_edge, _2opt_roll_lo_corr_tein = get_correlation(_2opt_roll_lo_cov_edge, _2opt_roll_lo_cov_tein, _2opt_cost[:, 0].std(), _2opt_roll_edge.std(), _2opt_roll_tein.std())
    _2swap_roll_lo_corr_edge, _2swap_roll_lo_corr_tein = get_correlation(_2swap_roll_lo_cov_edge, _2swap_roll_lo_cov_tein, _2swap_cost[:, 0].std(), _2swap_roll_edge.std(), _2swap_roll_tein.std())
    
    # local search roll FDCs, random start experiment and LO reference
    _2opt_roll_start_cov_edge, _2opt_roll_start_cov_tein = get_covariance(_2opt_start_cost[:, 0], _2opt_start_cost[:, 0].mean(), _2opt_roll_edge, _2opt_roll_edge.mean(), _2opt_roll_tein, _2opt_roll_tein.mean())
    _2swap_roll_start_cov_edge, _2swap_roll_start_cov_tein = get_covariance(_2swap_start_cost[:, 0], _2swap_start_cost[:, 0].mean(), _2swap_roll_edge, _2swap_roll_edge.mean(), _2swap_roll_tein, _2swap_roll_tein.mean())

    _2opt_roll_start_corr_edge, _2opt_roll_start_corr_tein = get_correlation(_2opt_roll_start_cov_edge, _2opt_roll_start_cov_tein, _2opt_start_cost[:, 0].std(), _2opt_roll_edge.std(), _2opt_roll_tein.std())
    _2swap_roll_start_corr_edge, _2swap_roll_start_corr_tein = get_correlation(_2swap_roll_start_cov_edge, _2swap_roll_start_cov_tein, _2swap_start_cost[:, 0].std(), _2swap_roll_edge.std(), _2swap_roll_tein.std())

    # distances of closest-found local optima to ML solutions (searching across all 2opt and all 2swap LO per DRL/IL solution)
    lo_stack = np.concatenate([_2opt_lo, _2swap_lo], axis=1)
    lo_cost_stack = np.concatenate([_2opt_cost, _2swap_cost], axis=1)
    assert lo_stack.shape[1] == 4 and lo_cost_stack.shape[1] == 4
    
    drl_node_minlo_edge, drl_node_minlo_tein = get_min_distances(drl_node_sol[:, 0], lo_stack)
    il_node_minlo_edge, il_node_minlo_tein = get_min_distances(il_node_sol[:, 0], lo_stack)

    drl_node_minlo_edge, drl_node_minlo_edge_idxs = drl_node_minlo_edge
    drl_node_minlo_tein, drl_node_minlo_tein_idxs = drl_node_minlo_tein
    il_node_minlo_edge, il_node_minlo_edge_idxs = il_node_minlo_edge
    il_node_minlo_tein, il_node_minlo_tein_idxs = il_node_minlo_tein
    
    # FDCs with ML solutions as fitness experiment, and closest-found LO as distance reference
    drl_node_minlo_cov_edge, drl_node_minlo_cov_tein = get_covariance(drl_node_cost[:, 0], drl_node_cost[:, 0].mean(), drl_node_minlo_edge, drl_node_minlo_edge.mean(), drl_node_minlo_tein, drl_node_minlo_tein.mean())
    il_node_minlo_cov_edge, il_node_minlo_cov_tein = get_covariance(il_node_cost[:, 0], il_node_cost[:, 0].mean(), il_node_minlo_edge, il_node_minlo_edge.mean(), il_node_minlo_tein, il_node_minlo_tein.mean())

    drl_node_minlo_corr_edge, drl_node_minlo_corr_tein = get_correlation(drl_node_minlo_cov_edge, drl_node_minlo_cov_tein, drl_node_cost[:, 0].std(), drl_node_minlo_edge.std(), drl_node_minlo_tein.std())
    il_node_minlo_corr_edge, il_node_minlo_corr_tein = get_correlation(il_node_minlo_cov_edge, il_node_minlo_cov_tein, il_node_cost[:, 0].std(), il_node_minlo_edge.std(), il_node_minlo_tein.std())

    # FDCs with closest-found LO as fitness experiment (one for each edge and tein closest dist), and ML solutions as reference
    drl_node_minlo_edge_cost = np.take_along_axis(lo_cost_stack, drl_node_minlo_edge_idxs, axis=1).squeeze()
    drl_node_minlo_tein_cost = np.take_along_axis(lo_cost_stack, drl_node_minlo_tein_idxs, axis=1).squeeze()
    il_node_minlo_edge_cost = np.take_along_axis(lo_cost_stack, il_node_minlo_edge_idxs, axis=1).squeeze()
    il_node_minlo_tein_cost = np.take_along_axis(lo_cost_stack, il_node_minlo_tein_idxs, axis=1).squeeze()

    minlo_drl_node_cov_edge = get_specific_covariance(drl_node_minlo_edge_cost, drl_node_minlo_edge_cost.mean(), drl_node_minlo_edge, drl_node_minlo_edge.mean())
    minlo_drl_node_cov_tein = get_specific_covariance(drl_node_minlo_tein_cost, drl_node_minlo_tein_cost.mean(), drl_node_minlo_tein, drl_node_minlo_tein.mean())
    minlo_il_node_cov_edge = get_specific_covariance(il_node_minlo_edge_cost, il_node_minlo_edge_cost.mean(), il_node_minlo_edge, il_node_minlo_edge.mean())
    minlo_il_node_cov_tein = get_specific_covariance(il_node_minlo_tein_cost, il_node_minlo_tein_cost.mean(), il_node_minlo_tein, il_node_minlo_tein.mean())

    minlo_drl_node_corr_edge = get_specific_correlation(minlo_drl_node_cov_edge, drl_node_minlo_edge_cost.std(), drl_node_minlo_edge.std())
    minlo_drl_node_corr_tein = get_specific_correlation(minlo_drl_node_cov_tein, drl_node_minlo_tein_cost.std(), drl_node_minlo_tein.std())
    minlo_il_node_corr_edge = get_specific_correlation(minlo_il_node_cov_edge, il_node_minlo_edge_cost.std(), il_node_minlo_edge.std())
    minlo_il_node_corr_tein = get_specific_correlation(minlo_il_node_cov_tein, il_node_minlo_tein_cost.std(), il_node_minlo_tein.std())

    # correlation between edge and tein distance over random solutions (spread), which should demonstrate strong correlation between distance measures
    inter_dist_spread_cov = get_specific_covariance(rand_spread_edge, rand_spread_edge.mean(), rand_spread_tein, rand_spread_tein.mean())
    inter_dist_spread_corr = get_specific_correlation(inter_dist_spread_cov, rand_spread_edge.std(), rand_spread_tein.std())

    # correlation between local search moves and local optima fitness
    _2opt_swaps_mean = _2opt_swaps[:, 0].mean()
    _2swap_swaps_mean = _2swap_swaps[:, 0].mean()

    _2opt_fit_swap_cov = get_specific_covariance(_2opt_cost[:, 0], _2opt_cost[:, 0].mean(), _2opt_swaps[:, 0], _2opt_swaps_mean)
    _2swap_fit_swap_cov = get_specific_covariance(_2swap_cost[:, 0], _2swap_cost[:, 0].mean(), _2swap_swaps[:, 0], _2swap_swaps_mean)

    _2opt_fit_swap_corr = get_specific_correlation(_2opt_fit_swap_cov, _2opt_cost[:, 0].std(), _2opt_swaps[:, 0].std())
    _2swap_fit_swap_corr = get_specific_correlation(_2swap_fit_swap_cov, _2swap_cost[:, 0].std(), _2swap_swaps[:, 0].std())

    # correlation between local search moves and roll distance
    _2opt_roll_swap_cov_edge, _2opt_roll_swap_cov_tein = get_covariance(_2opt_swaps[:, 0], _2opt_swaps_mean, _2opt_roll_edge, _2opt_roll_edge.mean(), _2opt_roll_tein, _2opt_roll_tein.mean())
    _2swap_roll_swap_cov_edge, _2swap_roll_swap_cov_tein = get_covariance(_2swap_swaps[:, 0], _2swap_swaps_mean, _2swap_roll_edge, _2swap_roll_edge.mean(), _2swap_roll_tein, _2swap_roll_tein.mean())

    _2opt_roll_swap_corr_edge, _2opt_roll_swap_corr_tein = get_correlation(_2opt_roll_swap_cov_edge, _2opt_roll_swap_cov_tein, _2opt_swaps[:, 0].std(), _2opt_roll_edge.std(), _2opt_roll_tein.std())
    _2swap_roll_swap_corr_edge, _2swap_roll_swap_corr_tein = get_correlation(_2swap_roll_swap_cov_edge, _2swap_roll_swap_cov_tein, _2swap_swaps[:, 0].std(), _2swap_roll_edge.std(), _2swap_roll_tein.std())

    # build and ship return message
    msg = dict(
        nodes=n,

        opt_cost=stat(opt_cost),
        rand_cost=stat(rand_cost[:, 0]),
        _2opt_cost=stat(_2opt_cost[:, 0]),
        _2swap_cost=stat(_2swap_cost[:, 0]),
        drl_node_cost=stat(drl_node_cost[:, 0]),
        il_node_cost=stat(il_node_cost[:, 0]),

        avg_2opt_ron=avg_2opt_ron,
        avg_2swap_ron=avg_2swap_ron,
        avg_drl_node_ron=avg_drl_node_ron,
        avg_il_node_ron=avg_il_node_ron,

        spec_2opt_ron=stat(spec_2opt_ron),
        spec_2swap_ron=stat(spec_2swap_ron),
        spec_drl_node_ron=stat(spec_drl_node_ron),
        spec_il_node_ron=stat(spec_il_node_ron),

        rand_spread=dstat(rand_spread_edge, rand_spread_tein),
        _2opt_spread=dstat(_2opt_spread_edge, _2opt_spread_tein),
        _2swap_spread=dstat(_2swap_spread_edge, _2swap_spread_tein),
        drl_node_spread=dstat(drl_node_spread_edge, drl_node_spread_tein),
        il_node_spread=dstat(il_node_spread_edge, il_node_spread_tein),

        rand_res=dstat(rand_res_edge, rand_res_tein),
        _2opt_res=dstat(_2opt_res_edge, _2opt_res_tein),
        _2swap_res=dstat(_2swap_res_edge, _2swap_res_tein),
        drl_node_res=dstat(drl_node_res_edge, drl_node_res_tein),
        il_node_res=dstat(il_node_res_edge, il_node_res_tein),

        _2opt_roll=dstat(_2opt_roll_edge, _2opt_roll_tein),
        _2swap_roll=dstat(_2swap_roll_edge, _2swap_roll_tein),

        _2opt_2swap_dist=dstat(_2opt_2swap_dist_edge, _2opt_2swap_dist_tein),
        drl_2opt_dist=dstat(drl_2opt_dist_edge, drl_2opt_dist_tein),
        drl_2swap_dist=dstat(drl_2swap_dist_edge, drl_2swap_dist_tein),
        il_2opt_dist=dstat(il_2opt_dist_edge, il_2opt_dist_tein),
        il_2swap_dist=dstat(il_2swap_dist_edge, il_2swap_dist_tein),
        drl_il_dist=dstat(drl_il_dist_edge, drl_il_dist_tein),

        rand_res_cov=dmet(rand_res_cov_edge, rand_res_cov_tein),
        _2opt_res_cov=dmet(_2opt_res_cov_edge, _2opt_res_cov_tein),
        _2swap_res_cov=dmet(_2swap_res_cov_edge, _2swap_res_cov_tein),
        drl_node_res_cov=dmet(drl_node_res_cov_edge, drl_node_res_cov_tein),
        il_node_res_cov=dmet(il_node_res_cov_edge, il_node_res_cov_tein),

        rand_res_corr=dmet(rand_res_corr_edge, rand_res_corr_tein),
        _2opt_res_corr=dmet(_2opt_res_corr_edge, _2opt_res_corr_tein),
        _2swap_res_corr=dmet(_2swap_res_corr_edge, _2swap_res_corr_tein),
        drl_node_res_corr=dmet(drl_node_res_corr_edge, drl_node_res_corr_tein),
        il_node_res_corr=dmet(il_node_res_corr_edge, il_node_res_corr_tein),

        _2opt_roll_lo_cov=dmet(_2opt_roll_lo_cov_edge, _2opt_roll_lo_cov_tein),
        _2swap_roll_lo_cov=dmet(_2swap_roll_lo_cov_edge, _2swap_roll_lo_cov_tein),

        _2opt_roll_lo_corr=dmet(_2opt_roll_lo_corr_edge, _2opt_roll_lo_corr_tein),
        _2swap_roll_lo_corr=dmet(_2swap_roll_lo_corr_edge, _2swap_roll_lo_corr_tein),

        _2opt_roll_start_cov=dmet(_2opt_roll_start_cov_edge, _2opt_roll_start_cov_tein),
        _2swap_roll_start_cov=dmet(_2swap_roll_start_cov_edge, _2swap_roll_start_cov_tein),

        _2opt_roll_start_corr=dmet(_2opt_roll_start_corr_edge, _2opt_roll_start_corr_tein),
        _2swap_roll_start_corr=dmet(_2swap_roll_start_corr_edge, _2swap_roll_start_corr_tein),

        drl_node_minlo=dstat(drl_node_minlo_edge, drl_node_minlo_tein),
        il_node_minlo=dstat(il_node_minlo_edge, il_node_minlo_tein),

        drl_node_minlo_cov=dmet(drl_node_minlo_cov_edge, drl_node_minlo_cov_tein),
        il_node_minlo_cov=dmet(il_node_minlo_cov_edge, il_node_minlo_cov_tein),

        drl_node_minlo_corr=dmet(drl_node_minlo_corr_edge, drl_node_minlo_corr_tein),
        il_node_minlo_corr=dmet(il_node_minlo_corr_edge, il_node_minlo_corr_tein),

        minlo_drl_node_cov=dmet(minlo_drl_node_cov_edge, minlo_drl_node_cov_tein),
        minlo_il_node_cov=dmet(minlo_il_node_cov_edge, minlo_il_node_cov_tein),

        minlo_drl_node_corr=dmet(minlo_drl_node_corr_edge, minlo_drl_node_corr_tein),
        minlo_il_node_corr=dmet(minlo_il_node_corr_edge, minlo_il_node_corr_tein),
        
        inter_dist_spread_cov=inter_dist_spread_cov,
        inter_dist_spread_corr=inter_dist_spread_corr,

        _2opt_swaps=stat(_2opt_swaps[:, 0]),
        _2swap_swaps=stat(_2swap_swaps[:, 0]),

        _2opt_fit_swap_cov=_2opt_fit_swap_cov,
        _2swap_fit_swap_cov=_2swap_fit_swap_cov,

        _2opt_fit_swap_corr=_2opt_fit_swap_corr,
        _2swap_fit_swap_corr=_2swap_fit_swap_corr,

        _2opt_roll_swap_cov=dmet(_2opt_roll_swap_cov_edge, _2opt_roll_swap_cov_tein),
        _2swap_roll_swap_cov=dmet(_2swap_roll_swap_cov_edge, _2swap_roll_swap_cov_tein),

        _2opt_roll_swap_corr=dmet(_2opt_roll_swap_corr_edge, _2opt_roll_swap_corr_tein),
        _2swap_roll_swap_corr=dmet(_2swap_roll_swap_corr_edge, _2swap_roll_swap_corr_tein),
    )

    retv_q.put(msg)


def start_node_scale_analyzer():
    retv_q = Queue()

    if DEBUG:
        eval_node_scale(retv_q, NODE_RANGE[0])
        return None, retv_q

    workers = [Process(target=eval_node_scale, args=(retv_q, n)) for n in NODE_RANGE]

    [w.start() for w in workers]

    return workers, retv_q


def eval_model_scale(retv_q, w):
    seed_rand_gen(int.from_bytes(os.urandom(4), "big"))

    dpath = f"node_scale_20n_2d_plus_model_scale.npz"
    dpath = osp.join(BIND_PATH, dpath)
    dset = np.load(dpath, mmap_mode="r")

    optimals = dset["global_optima"]
    
    _2opt_lo = dset["_2opt_local_optima"]

    _2swap_lo = dset["_2swap_local_optima"]

    drl_model_sol = dset[f"drl_model_scaling_{w}w"]
    il_model_sol = dset[f"il_model_scaling_{w}w"]

    if LIGHTWEIGHT is not None:  # making things wayyy faster
        optimals = optimals[:LIGHTWEIGHT]
    
        _2opt_lo = _2opt_lo[:LIGHTWEIGHT]

        _2swap_lo = _2swap_lo[:LIGHTWEIGHT]

        drl_model_sol = drl_model_sol[:LIGHTWEIGHT]
        il_model_sol = il_model_sol[:LIGHTWEIGHT]

    # gen random tours
    rand_sol = get_random_tours(optimals, RAND_PER_PROB)

    # gather fitness (cost aka tour length)
    opt_cost = get_fitness(optimals)
    rand_cost = get_fitness(optimals, rand_sol)
    _2opt_cost = get_fitness(optimals, _2opt_lo)
    _2swap_cost = get_fitness(optimals, _2swap_lo)
    drl_model_cost = get_fitness(optimals, drl_model_sol)
    il_model_cost = get_fitness(optimals, il_model_sol)

    # expectation RON scores
    avg_opt_cost = opt_cost.mean()
    avg_rand_cost = rand_cost[:, 0].mean()  # taking only first slice of tours to not artificially inflate sample size

    avg_drl_model_ron = (drl_model_cost[:, 0].mean() - avg_opt_cost) / (avg_rand_cost - avg_opt_cost)
    avg_il_model_ron = (il_model_cost[:, 0].mean() - avg_opt_cost) / (avg_rand_cost - avg_opt_cost)

    # specific per-problem RON scores (random is a noisier estimate based on RAND_PER_PROB; optimal is exact)
    prob_mean_rand_cost = rand_cost.mean(axis=1)  # sol_per_prob axis only

    spec_drl_model_ron = get_prob_specific_ron(drl_model_cost, prob_mean_rand_cost, opt_cost)
    spec_il_model_ron = get_prob_specific_ron(il_model_cost, prob_mean_rand_cost, opt_cost)

    # intra-population spread distance of solutions
    drl_model_spread_edge, drl_model_spread_tein = get_distances(drl_model_sol[:, 0], drl_model_sol[:, 1])
    il_model_spread_edge, il_model_spread_tein = get_distances(il_model_sol[:, 0], il_model_sol[:, 1])

    # global optima residual distances
    opt_sol = np.stack(optimals.shape[0] * [np.arange(optimals.shape[1])], axis=0)

    drl_model_res_edge, drl_model_res_tein = get_distances(drl_model_sol[:, 0], opt_sol)
    il_model_res_edge, il_model_res_tein = get_distances(il_model_sol[:, 0], opt_sol)

    # remaining inter-population distances of interest
    drl_2opt_dist_edge, drl_2opt_dist_tein = get_distances(drl_model_sol[:, 0], _2opt_lo[:, 0])
    drl_2swap_dist_edge, drl_2swap_dist_tein = get_distances(drl_model_sol[:, 0], _2swap_lo[:, 0])
    il_2opt_dist_edge, il_2opt_dist_tein = get_distances(il_model_sol[:, 0], _2opt_lo[:, 0])
    il_2swap_dist_edge, il_2swap_dist_tein = get_distances(il_model_sol[:, 0], _2swap_lo[:, 0])
    drl_il_dist_edge, drl_il_dist_tein = get_distances(drl_model_sol[:, 0], il_model_sol[:, 0])

    # global optima residual FDCs
    drl_model_res_cov_edge, drl_model_res_cov_tein = get_covariance(drl_model_cost[:, 0], drl_model_cost[:, 0].mean(), drl_model_res_edge, drl_model_res_edge.mean(), drl_model_res_tein, drl_model_res_tein.mean())
    il_model_res_cov_edge, il_model_res_cov_tein = get_covariance(il_model_cost[:, 0], il_model_cost[:, 0].mean(), il_model_res_edge, il_model_res_edge.mean(), il_model_res_tein, il_model_res_tein.mean())

    drl_model_res_corr_edge, drl_model_res_corr_tein = get_correlation(drl_model_res_cov_edge, drl_model_res_cov_tein, drl_model_cost[:, 0].std(), drl_model_res_edge.std(), drl_model_res_tein.std())
    il_model_res_corr_edge, il_model_res_corr_tein = get_correlation(il_model_res_cov_edge, il_model_res_cov_tein, il_model_cost[:, 0].std(), il_model_res_edge.std(), il_model_res_tein.std())

    # distances of closest-found local optima to ML solutions (searching across all 2opt and all 2swap LO per DRL/IL solution)
    lo_stack = np.concatenate([_2opt_lo, _2swap_lo], axis=1)
    lo_cost_stack = np.concatenate([_2opt_cost, _2swap_cost], axis=1)
    assert lo_stack.shape[1] == 4 and lo_cost_stack.shape[1] == 4
    
    drl_model_minlo_edge, drl_model_minlo_tein = get_min_distances(drl_model_sol[:, 0], lo_stack)
    il_model_minlo_edge, il_model_minlo_tein = get_min_distances(il_model_sol[:, 0], lo_stack)

    drl_model_minlo_edge, drl_model_minlo_edge_idxs = drl_model_minlo_edge
    drl_model_minlo_tein, drl_model_minlo_tein_idxs = drl_model_minlo_tein
    il_model_minlo_edge, il_model_minlo_edge_idxs = il_model_minlo_edge
    il_model_minlo_tein, il_model_minlo_tein_idxs = il_model_minlo_tein
    
    # FDCs with ML solutions as fitness experiment, and closest-found LO as distance reference
    drl_model_minlo_cov_edge, drl_model_minlo_cov_tein = get_covariance(drl_model_cost[:, 0], drl_model_cost[:, 0].mean(), drl_model_minlo_edge, drl_model_minlo_edge.mean(), drl_model_minlo_tein, drl_model_minlo_tein.mean())
    il_model_minlo_cov_edge, il_model_minlo_cov_tein = get_covariance(il_model_cost[:, 0], il_model_cost[:, 0].mean(), il_model_minlo_edge, il_model_minlo_edge.mean(), il_model_minlo_tein, il_model_minlo_tein.mean())

    drl_model_minlo_corr_edge, drl_model_minlo_corr_tein = get_correlation(drl_model_minlo_cov_edge, drl_model_minlo_cov_tein, drl_model_cost[:, 0].std(), drl_model_minlo_edge.std(), drl_model_minlo_tein.std())
    il_model_minlo_corr_edge, il_model_minlo_corr_tein = get_correlation(il_model_minlo_cov_edge, il_model_minlo_cov_tein, il_model_cost[:, 0].std(), il_model_minlo_edge.std(), il_model_minlo_tein.std())

    # FDCs with closest-found LO as fitness experiment (one for each edge and tein closest dist), and ML solutions as reference
    drl_model_minlo_edge_cost = np.take_along_axis(lo_cost_stack, drl_model_minlo_edge_idxs, axis=1).squeeze()
    drl_model_minlo_tein_cost = np.take_along_axis(lo_cost_stack, drl_model_minlo_tein_idxs, axis=1).squeeze()
    il_model_minlo_edge_cost = np.take_along_axis(lo_cost_stack, il_model_minlo_edge_idxs, axis=1).squeeze()
    il_model_minlo_tein_cost = np.take_along_axis(lo_cost_stack, il_model_minlo_tein_idxs, axis=1).squeeze()

    minlo_drl_model_cov_edge = get_specific_covariance(drl_model_minlo_edge_cost, drl_model_minlo_edge_cost.mean(), drl_model_minlo_edge, drl_model_minlo_edge.mean())
    minlo_drl_model_cov_tein = get_specific_covariance(drl_model_minlo_tein_cost, drl_model_minlo_tein_cost.mean(), drl_model_minlo_tein, drl_model_minlo_tein.mean())
    minlo_il_model_cov_edge = get_specific_covariance(il_model_minlo_edge_cost, il_model_minlo_edge_cost.mean(), il_model_minlo_edge, il_model_minlo_edge.mean())
    minlo_il_model_cov_tein = get_specific_covariance(il_model_minlo_tein_cost, il_model_minlo_tein_cost.mean(), il_model_minlo_tein, il_model_minlo_tein.mean())

    minlo_drl_model_corr_edge = get_specific_correlation(minlo_drl_model_cov_edge, drl_model_minlo_edge_cost.std(), drl_model_minlo_edge.std())
    minlo_drl_model_corr_tein = get_specific_correlation(minlo_drl_model_cov_tein, drl_model_minlo_tein_cost.std(), drl_model_minlo_tein.std())
    minlo_il_model_corr_edge = get_specific_correlation(minlo_il_model_cov_edge, il_model_minlo_edge_cost.std(), il_model_minlo_edge.std())
    minlo_il_model_corr_tein = get_specific_correlation(minlo_il_model_cov_tein, il_model_minlo_tein_cost.std(), il_model_minlo_tein.std())

    # build and ship return message
    msg = dict(
        width=w,

        drl_model_cost=stat(drl_model_cost[:, 0]),
        il_model_cost=stat(il_model_cost[:, 0]),

        avg_drl_model_ron=avg_drl_model_ron,
        avg_il_model_ron=avg_il_model_ron,

        spec_drl_model_ron=stat(spec_drl_model_ron),
        spec_il_model_ron=stat(spec_il_model_ron),

        drl_model_spread=dstat(drl_model_spread_edge, drl_model_spread_tein),
        il_model_spread=dstat(il_model_spread_edge, il_model_spread_tein),

        drl_model_res=dstat(drl_model_res_edge, drl_model_res_tein),
        il_model_res=dstat(il_model_res_edge, il_model_res_tein),

        drl_2opt_dist=dstat(drl_2opt_dist_edge, drl_2opt_dist_tein),
        drl_2swap_dist=dstat(drl_2swap_dist_edge, drl_2swap_dist_tein),
        il_2opt_dist=dstat(il_2opt_dist_edge, il_2opt_dist_tein),
        il_2swap_dist=dstat(il_2swap_dist_edge, il_2swap_dist_tein),
        drl_il_dist=dstat(drl_il_dist_edge, drl_il_dist_tein),

        drl_model_res_cov=dmet(drl_model_res_cov_edge, drl_model_res_cov_tein),
        il_model_res_cov=dmet(il_model_res_cov_edge, il_model_res_cov_tein),

        drl_model_res_corr=dmet(drl_model_res_corr_edge, drl_model_res_corr_tein),
        il_model_res_corr=dmet(il_model_res_corr_edge, il_model_res_corr_tein),

        drl_model_minlo=dstat(drl_model_minlo_edge, drl_model_minlo_tein),
        il_model_minlo=dstat(il_model_minlo_edge, il_model_minlo_tein),

        drl_model_minlo_cov=dmet(drl_model_minlo_cov_edge, drl_model_minlo_cov_tein),
        il_model_minlo_cov=dmet(il_model_minlo_cov_edge, il_model_minlo_cov_tein),

        drl_model_minlo_corr=dmet(drl_model_minlo_corr_edge, drl_model_minlo_corr_tein),
        il_model_minlo_corr=dmet(il_model_minlo_corr_edge, il_model_minlo_corr_tein),

        minlo_drl_model_cov=dmet(minlo_drl_model_cov_edge, minlo_drl_model_cov_tein),
        minlo_il_model_cov=dmet(minlo_il_model_cov_edge, minlo_il_model_cov_tein),

        minlo_drl_model_corr=dmet(minlo_drl_model_corr_edge, minlo_drl_model_corr_tein),
        minlo_il_model_corr=dmet(minlo_il_model_corr_edge, minlo_il_model_corr_tein),
    )

    retv_q.put(msg)


def start_model_scale_analyzer():
    retv_q = Queue()

    if DEBUG:
        eval_model_scale(retv_q, MODEL_WIDTHS[0])
        return None, retv_q

    workers = [Process(target=eval_model_scale, args=(retv_q, w)) for w in MODEL_WIDTHS]

    [w.start() for w in workers]

    return workers, retv_q
    

def eval_dim_scale(retv_q, d, node_scale):  # basically the same as eval_node_scale() but without IL solution stuff
    seed_rand_gen(int.from_bytes(os.urandom(4), "big"))

    dpath = f"dim_scale_{node_scale}n_{d}d.npz"
    dpath = osp.join(BIND_PATH, dpath)
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

    if LIGHTWEIGHT is not None:
        proxy_optimals = proxy_optimals[:LIGHTWEIGHT]
    
        _2opt_lo = _2opt_lo[:LIGHTWEIGHT]
        _2opt_starts = _2opt_starts[:LIGHTWEIGHT]
        _2opt_swaps = _2opt_swaps[:LIGHTWEIGHT]

        _2swap_lo = _2swap_lo[:LIGHTWEIGHT]
        _2swap_starts = _2swap_starts[:LIGHTWEIGHT]
        _2swap_swaps = _2swap_swaps[:LIGHTWEIGHT]

        drl_dim_sol = drl_dim_sol[:LIGHTWEIGHT]

    # gen random tours
    rand_sol = get_random_tours(proxy_optimals, RAND_PER_PROB)

    # gather fitness (cost aka tour length)
    opt_cost = get_fitness(proxy_optimals)
    rand_cost = get_fitness(proxy_optimals, rand_sol)
    _2opt_cost = get_fitness(proxy_optimals, _2opt_lo)
    _2swap_cost = get_fitness(proxy_optimals, _2swap_lo)
    drl_dim_cost = get_fitness(proxy_optimals, drl_dim_sol)

    _2opt_start_cost = get_fitness(proxy_optimals, _2opt_starts)
    _2swap_start_cost = get_fitness(proxy_optimals, _2swap_starts)

    # expectation RON scores
    avg_opt_cost = opt_cost.mean()
    avg_rand_cost = rand_cost[:, 0].mean()  # taking only first slice of tours to not artificially inflate sample size

    avg_2opt_ron = (_2opt_cost[:, 0].mean() - avg_opt_cost) / (avg_rand_cost - avg_opt_cost)  # ditto, and below
    avg_2swap_ron = (_2swap_cost[:, 0].mean() - avg_opt_cost) / (avg_rand_cost - avg_opt_cost) 
    avg_drl_dim_ron = (drl_dim_cost[:, 0].mean() - avg_opt_cost) / (avg_rand_cost - avg_opt_cost) 
    
    # specific per-problem RON scores (random is a noisier estimate based on RAND_PER_PROB; optimal is exact)
    prob_mean_rand_cost = rand_cost.mean(axis=1)  # sol_per_prob axis only

    spec_2opt_ron = get_prob_specific_ron(_2opt_cost, prob_mean_rand_cost, opt_cost)
    spec_2swap_ron = get_prob_specific_ron(_2swap_cost, prob_mean_rand_cost, opt_cost)
    spec_drl_dim_ron = get_prob_specific_ron(drl_dim_cost, prob_mean_rand_cost, opt_cost)

    # intra-population spread distance of solutions
    rand_spread_edge, rand_spread_tein = get_distances(rand_sol[:, 0], rand_sol[:, 1])
    _2opt_spread_edge, _2opt_spread_tein = get_distances(_2opt_lo[:, 0], _2opt_lo[:, 1])
    _2swap_spread_edge, _2swap_spread_tein = get_distances(_2swap_lo[:, 0], _2swap_lo[:, 1])
    drl_dim_spread_edge, drl_dim_spread_tein = get_distances(drl_dim_sol[:, 0], drl_dim_sol[:, 1])

    # global optima residual distances
    opt_sol = np.stack(proxy_optimals.shape[0] * [np.arange(proxy_optimals.shape[1])], axis=0)

    rand_res_edge, rand_res_tein = get_distances(rand_sol[:, 0], opt_sol)
    _2opt_res_edge, _2opt_res_tein = get_distances(_2opt_lo[:, 0], opt_sol)
    _2swap_res_edge, _2swap_res_tein = get_distances(_2swap_lo[:, 0], opt_sol)
    drl_dim_res_edge, drl_dim_res_tein = get_distances(drl_dim_sol[:, 0], opt_sol)

    # local search roll distances (between start and delivered local optima)
    _2opt_roll_edge, _2opt_roll_tein = get_distances(_2opt_lo[:, 0], _2opt_starts[:, 0])
    _2swap_roll_edge, _2swap_roll_tein = get_distances(_2swap_lo[:, 0], _2swap_starts[:, 0])

    # remaining inter-population distances of interest (residuals cover distances to optimals, and random isn't interesting)
    _2opt_2swap_dist_edge, _2opt_2swap_dist_tein = get_distances(_2opt_lo[:, 0], _2swap_lo[:, 0])
    drl_2opt_dist_edge, drl_2opt_dist_tein = get_distances(drl_dim_sol[:, 0], _2opt_lo[:, 0])
    drl_2swap_dist_edge, drl_2swap_dist_tein = get_distances(drl_dim_sol[:, 0], _2swap_lo[:, 0])

    # global optima residual FDCs
    rand_res_cov_edge, rand_res_cov_tein = get_covariance(rand_cost[:, 0], avg_rand_cost, rand_res_edge, rand_res_edge.mean(), rand_res_tein, rand_res_tein.mean())
    _2opt_res_cov_edge, _2opt_res_cov_tein = get_covariance(_2opt_cost[:, 0], _2opt_cost[:, 0].mean(), _2opt_res_edge, _2opt_res_edge.mean(), _2opt_res_tein, _2opt_res_tein.mean())
    _2swap_res_cov_edge, _2swap_res_cov_tein = get_covariance(_2swap_cost[:, 0], _2swap_cost[:, 0].mean(), _2swap_res_edge, _2swap_res_edge.mean(), _2swap_res_tein, _2swap_res_tein.mean())
    drl_dim_res_cov_edge, drl_dim_res_cov_tein = get_covariance(drl_dim_cost[:, 0], drl_dim_cost[:, 0].mean(), drl_dim_res_edge, drl_dim_res_edge.mean(), drl_dim_res_tein, drl_dim_res_tein.mean())

    rand_res_corr_edge, rand_res_corr_tein = get_correlation(rand_res_cov_edge, rand_res_cov_tein, rand_cost[:, 0].std(), rand_res_edge.std(), rand_res_tein.std())
    _2opt_res_corr_edge, _2opt_res_corr_tein = get_correlation(_2opt_res_cov_edge, _2opt_res_cov_tein, _2opt_cost[:, 0].std(), _2opt_res_edge.std(), _2opt_res_tein.std())
    _2swap_res_corr_edge, _2swap_res_corr_tein = get_correlation(_2swap_res_cov_edge, _2swap_res_cov_tein, _2swap_cost[:, 0].std(), _2swap_res_edge.std(), _2swap_res_tein.std())
    drl_dim_res_corr_edge, drl_dim_res_corr_tein = get_correlation(drl_dim_res_cov_edge, drl_dim_res_cov_tein, drl_dim_cost[:, 0].std(), drl_dim_res_edge.std(), drl_dim_res_tein.std())

    # local search roll FDCs, LO experiment and random start reference
    _2opt_roll_lo_cov_edge, _2opt_roll_lo_cov_tein = get_covariance(_2opt_cost[:, 0], _2opt_cost[:, 0].mean(), _2opt_roll_edge, _2opt_roll_edge.mean(), _2opt_roll_tein, _2opt_roll_tein.mean())
    _2swap_roll_lo_cov_edge, _2swap_roll_lo_cov_tein = get_covariance(_2swap_cost[:, 0], _2swap_cost[:, 0].mean(), _2swap_roll_edge, _2swap_roll_edge.mean(), _2swap_roll_tein, _2swap_roll_tein.mean())

    _2opt_roll_lo_corr_edge, _2opt_roll_lo_corr_tein = get_correlation(_2opt_roll_lo_cov_edge, _2opt_roll_lo_cov_tein, _2opt_cost[:, 0].std(), _2opt_roll_edge.std(), _2opt_roll_tein.std())
    _2swap_roll_lo_corr_edge, _2swap_roll_lo_corr_tein = get_correlation(_2swap_roll_lo_cov_edge, _2swap_roll_lo_cov_tein, _2swap_cost[:, 0].std(), _2swap_roll_edge.std(), _2swap_roll_tein.std())
    
    # local search roll FDCs, random start experiment and LO reference
    _2opt_roll_start_cov_edge, _2opt_roll_start_cov_tein = get_covariance(_2opt_start_cost[:, 0], _2opt_start_cost[:, 0].mean(), _2opt_roll_edge, _2opt_roll_edge.mean(), _2opt_roll_tein, _2opt_roll_tein.mean())
    _2swap_roll_start_cov_edge, _2swap_roll_start_cov_tein = get_covariance(_2swap_start_cost[:, 0], _2swap_start_cost[:, 0].mean(), _2swap_roll_edge, _2swap_roll_edge.mean(), _2swap_roll_tein, _2swap_roll_tein.mean())

    _2opt_roll_start_corr_edge, _2opt_roll_start_corr_tein = get_correlation(_2opt_roll_start_cov_edge, _2opt_roll_start_cov_tein, _2opt_start_cost[:, 0].std(), _2opt_roll_edge.std(), _2opt_roll_tein.std())
    _2swap_roll_start_corr_edge, _2swap_roll_start_corr_tein = get_correlation(_2swap_roll_start_cov_edge, _2swap_roll_start_cov_tein, _2swap_start_cost[:, 0].std(), _2swap_roll_edge.std(), _2swap_roll_tein.std())

    # distances of closest-found local optima to ML solutions (searching across all 2opt and all 2swap LO per DRL/IL solution)
    lo_stack = np.concatenate([_2opt_lo, _2swap_lo], axis=1)
    lo_cost_stack = np.concatenate([_2opt_cost, _2swap_cost], axis=1)
    assert lo_stack.shape[1] == 4 and lo_cost_stack.shape[1] == 4
    
    drl_dim_minlo_edge, drl_dim_minlo_tein = get_min_distances(drl_dim_sol[:, 0], lo_stack)

    drl_dim_minlo_edge, drl_dim_minlo_edge_idxs = drl_dim_minlo_edge
    drl_dim_minlo_tein, drl_dim_minlo_tein_idxs = drl_dim_minlo_tein
    
    # FDCs with ML solutions as fitness experiment, and closest-found LO as distance reference
    drl_dim_minlo_cov_edge, drl_dim_minlo_cov_tein = get_covariance(drl_dim_cost[:, 0], drl_dim_cost[:, 0].mean(), drl_dim_minlo_edge, drl_dim_minlo_edge.mean(), drl_dim_minlo_tein, drl_dim_minlo_tein.mean())

    drl_dim_minlo_corr_edge, drl_dim_minlo_corr_tein = get_correlation(drl_dim_minlo_cov_edge, drl_dim_minlo_cov_tein, drl_dim_cost[:, 0].std(), drl_dim_minlo_edge.std(), drl_dim_minlo_tein.std())

    # FDCs with closest-found LO as fitness experiment (one for each edge and tein closest dist), and ML solutions as reference
    drl_dim_minlo_edge_cost = np.take_along_axis(lo_cost_stack, drl_dim_minlo_edge_idxs, axis=1).squeeze()
    drl_dim_minlo_tein_cost = np.take_along_axis(lo_cost_stack, drl_dim_minlo_tein_idxs, axis=1).squeeze()

    minlo_drl_dim_cov_edge = get_specific_covariance(drl_dim_minlo_edge_cost, drl_dim_minlo_edge_cost.mean(), drl_dim_minlo_edge, drl_dim_minlo_edge.mean())
    minlo_drl_dim_cov_tein = get_specific_covariance(drl_dim_minlo_tein_cost, drl_dim_minlo_tein_cost.mean(), drl_dim_minlo_tein, drl_dim_minlo_tein.mean())

    minlo_drl_dim_corr_edge = get_specific_correlation(minlo_drl_dim_cov_edge, drl_dim_minlo_edge_cost.std(), drl_dim_minlo_edge.std())
    minlo_drl_dim_corr_tein = get_specific_correlation(minlo_drl_dim_cov_tein, drl_dim_minlo_tein_cost.std(), drl_dim_minlo_tein.std())

    # correlation between edge and tein distance over random solutions (spread), which should demonstrate strong correlation between distance measures
    inter_dist_spread_cov = get_specific_covariance(rand_spread_edge, rand_spread_edge.mean(), rand_spread_tein, rand_spread_tein.mean())
    inter_dist_spread_corr = get_specific_correlation(inter_dist_spread_cov, rand_spread_edge.std(), rand_spread_tein.std())

    # correlation between local search moves and local optima fitness
    _2opt_swaps_mean = _2opt_swaps[:, 0].mean()
    _2swap_swaps_mean = _2swap_swaps[:, 0].mean()

    _2opt_fit_swap_cov = get_specific_covariance(_2opt_cost[:, 0], _2opt_cost[:, 0].mean(), _2opt_swaps[:, 0], _2opt_swaps_mean)
    _2swap_fit_swap_cov = get_specific_covariance(_2swap_cost[:, 0], _2swap_cost[:, 0].mean(), _2swap_swaps[:, 0], _2swap_swaps_mean)

    _2opt_fit_swap_corr = get_specific_correlation(_2opt_fit_swap_cov, _2opt_cost[:, 0].std(), _2opt_swaps[:, 0].std())
    _2swap_fit_swap_corr = get_specific_correlation(_2swap_fit_swap_cov, _2swap_cost[:, 0].std(), _2swap_swaps[:, 0].std())

    # correlation between local search moves and roll distance
    _2opt_roll_swap_cov_edge, _2opt_roll_swap_cov_tein = get_covariance(_2opt_swaps[:, 0], _2opt_swaps_mean, _2opt_roll_edge, _2opt_roll_edge.mean(), _2opt_roll_tein, _2opt_roll_tein.mean())
    _2swap_roll_swap_cov_edge, _2swap_roll_swap_cov_tein = get_covariance(_2swap_swaps[:, 0], _2swap_swaps_mean, _2swap_roll_edge, _2swap_roll_edge.mean(), _2swap_roll_tein, _2swap_roll_tein.mean())

    _2opt_roll_swap_corr_edge, _2opt_roll_swap_corr_tein = get_correlation(_2opt_roll_swap_cov_edge, _2opt_roll_swap_cov_tein, _2opt_swaps[:, 0].std(), _2opt_roll_edge.std(), _2opt_roll_tein.std())
    _2swap_roll_swap_corr_edge, _2swap_roll_swap_corr_tein = get_correlation(_2swap_roll_swap_cov_edge, _2swap_roll_swap_cov_tein, _2swap_swaps[:, 0].std(), _2swap_roll_edge.std(), _2swap_roll_tein.std())

    # build and ship return message
    msg = dict(
        dims=d,

        opt_cost=stat(opt_cost),
        rand_cost=stat(rand_cost[:, 0]),
        _2opt_cost=stat(_2opt_cost[:, 0]),
        _2swap_cost=stat(_2swap_cost[:, 0]),
        drl_dim_cost=stat(drl_dim_cost[:, 0]),

        avg_2opt_ron=avg_2opt_ron,
        avg_2swap_ron=avg_2swap_ron,
        avg_drl_dim_ron=avg_drl_dim_ron,

        spec_2opt_ron=stat(spec_2opt_ron),
        spec_2swap_ron=stat(spec_2swap_ron),
        spec_drl_dim_ron=stat(spec_drl_dim_ron),

        rand_spread=dstat(rand_spread_edge, rand_spread_tein),
        _2opt_spread=dstat(_2opt_spread_edge, _2opt_spread_tein),
        _2swap_spread=dstat(_2swap_spread_edge, _2swap_spread_tein),
        drl_dim_spread=dstat(drl_dim_spread_edge, drl_dim_spread_tein),

        rand_res=dstat(rand_res_edge, rand_res_tein),
        _2opt_res=dstat(_2opt_res_edge, _2opt_res_tein),
        _2swap_res=dstat(_2swap_res_edge, _2swap_res_tein),
        drl_dim_res=dstat(drl_dim_res_edge, drl_dim_res_tein),

        _2opt_roll=dstat(_2opt_roll_edge, _2opt_roll_tein),
        _2swap_roll=dstat(_2swap_roll_edge, _2swap_roll_tein),

        _2opt_2swap_dist=dstat(_2opt_2swap_dist_edge, _2opt_2swap_dist_tein),
        drl_2opt_dist=dstat(drl_2opt_dist_edge, drl_2opt_dist_tein),
        drl_2swap_dist=dstat(drl_2swap_dist_edge, drl_2swap_dist_tein),

        rand_res_cov=dmet(rand_res_cov_edge, rand_res_cov_tein),
        _2opt_res_cov=dmet(_2opt_res_cov_edge, _2opt_res_cov_tein),
        _2swap_res_cov=dmet(_2swap_res_cov_edge, _2swap_res_cov_tein),
        drl_dim_res_cov=dmet(drl_dim_res_cov_edge, drl_dim_res_cov_tein),

        rand_res_corr=dmet(rand_res_corr_edge, rand_res_corr_tein),
        _2opt_res_corr=dmet(_2opt_res_corr_edge, _2opt_res_corr_tein),
        _2swap_res_corr=dmet(_2swap_res_corr_edge, _2swap_res_corr_tein),
        drl_dim_res_corr=dmet(drl_dim_res_corr_edge, drl_dim_res_corr_tein),

        _2opt_roll_lo_cov=dmet(_2opt_roll_lo_cov_edge, _2opt_roll_lo_cov_tein),
        _2swap_roll_lo_cov=dmet(_2swap_roll_lo_cov_edge, _2swap_roll_lo_cov_tein),

        _2opt_roll_lo_corr=dmet(_2opt_roll_lo_corr_edge, _2opt_roll_lo_corr_tein),
        _2swap_roll_lo_corr=dmet(_2swap_roll_lo_corr_edge, _2swap_roll_lo_corr_tein),

        _2opt_roll_start_cov=dmet(_2opt_roll_start_cov_edge, _2opt_roll_start_cov_tein),
        _2swap_roll_start_cov=dmet(_2swap_roll_start_cov_edge, _2swap_roll_start_cov_tein),

        _2opt_roll_start_corr=dmet(_2opt_roll_start_corr_edge, _2opt_roll_start_corr_tein),
        _2swap_roll_start_corr=dmet(_2swap_roll_start_corr_edge, _2swap_roll_start_corr_tein),

        drl_dim_minlo=dstat(drl_dim_minlo_edge, drl_dim_minlo_tein),

        drl_dim_minlo_cov=dmet(drl_dim_minlo_cov_edge, drl_dim_minlo_cov_tein),

        drl_dim_minlo_corr=dmet(drl_dim_minlo_corr_edge, drl_dim_minlo_corr_tein),

        minlo_drl_dim_cov=dmet(minlo_drl_dim_cov_edge, minlo_drl_dim_cov_tein),

        minlo_drl_dim_corr=dmet(minlo_drl_dim_corr_edge, minlo_drl_dim_corr_tein),
        
        inter_dist_spread_cov=inter_dist_spread_cov,
        inter_dist_spread_corr=inter_dist_spread_corr,

        _2opt_swaps=stat(_2opt_swaps[:, 0]),
        _2swap_swaps=stat(_2swap_swaps[:, 0]),

        _2opt_fit_swap_cov=_2opt_fit_swap_cov,
        _2swap_fit_swap_cov=_2swap_fit_swap_cov,

        _2opt_fit_swap_corr=_2opt_fit_swap_corr,
        _2swap_fit_swap_corr=_2swap_fit_swap_corr,

        _2opt_roll_swap_cov=dmet(_2opt_roll_swap_cov_edge, _2opt_roll_swap_cov_tein),
        _2swap_roll_swap_cov=dmet(_2swap_roll_swap_cov_edge, _2swap_roll_swap_cov_tein),

        _2opt_roll_swap_corr=dmet(_2opt_roll_swap_corr_edge, _2opt_roll_swap_corr_tein),
        _2swap_roll_swap_corr=dmet(_2swap_roll_swap_corr_edge, _2swap_roll_swap_corr_tein),
    )

    retv_q.put(msg)


def start_dim_scale_analyzer(node_scale):
    retv_q = Queue()

    if DEBUG:
        eval_dim_scale(retv_q, 3, node_scale)
        return None, retv_q

    workers = [Process(target=eval_dim_scale, args=(retv_q, d, node_scale)) for d in DIM_RANGE]

    [w.start() for w in workers]

    return workers, retv_q


def recursive_log(msg, step, prefix=[]):
    if type(msg) is not dict:
        Logger.log(".".join(prefix), msg, step=step)
    else:
        for k, v in msg.items():
            recursive_log(v, step, prefix + [k])


def log_and_join(workers, queue, step_key):
    if workers is None:
        return  # skip the wrap up, just debugging
    
    msgs = drain_queue(queue, len(workers))
    sorted_msgs = sorted(msgs, key = lambda x: x[step_key])

    for msg in sorted_msgs:
        step = msg[step_key]
        recursive_log(msg, step)
                
    [w.join() for w in workers]


def setup_logging(sig):
    if sig is not None:
        resuming, sig_args = parse_log_sig(sig)

        assert not resuming, "resuming unsupported"

        Logger.start(*sig_args)

    else:
        Logger.dummy_init()



if __name__ == "__main__":
    args = parser.parse_args()

    sig = MLFLOW_EXP + f"/{args.scale}"
    setup_logging(sig)

    if args.scale == "node":
        n_wrk, n_q = start_node_scale_analyzer()
        log_and_join(n_wrk, n_q, "nodes")

    elif args.scale == "model":
        m_wrk, m_q = start_model_scale_analyzer()
        log_and_join(m_wrk, m_q, "width")

    elif args.scale == "dim10":
        d10_wrk, d10_q = start_dim_scale_analyzer(10)
        log_and_join(d10_wrk, d10_q, "dims")

    elif args.scale == "dim20":
        d20_wrk, d20_q = start_dim_scale_analyzer(20)
        log_and_join(d20_wrk, d20_q, "dims")
