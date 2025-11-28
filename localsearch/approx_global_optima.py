"""
Generates local optima from several random starting tours
using local search, returning best found optima as approximation
for global optima (which only works well with small problem sizes)

Useful for higher-dimensional TSP where we can't use the Concorde optimal solver

Logs estimated optimal costs and other stats to mlflow

Also outputs TSP problem dataset containing approximate global optima in tour order (dim 1), as float array of shape (num_problems, num_nodes, dimensions)
"""

import pyximport; pyximport.install(language_level=3)
from csearch import batch_two_opt_search, batch_two_swap_search  # eucl_dist, hyper_eucl_dist

from tqdm import tqdm
import os.path as osp
import os
from multiprocessing import Process, Queue
import numpy as np
import torch
from time import time

from tsp.utils import seed_rand_gen, get_coords
from tsp.logger import MLflowLogger as Logger
from localsearch.distance import batch_edge_hamming_distance
from localsearch.utils import setup_logging, drain_queue, stitch_msgs, np_get_costs

from config.approx_global_optima import SUPER_CONFIG

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "cfg", default=None, type=str
)

VERBOSE = False


# def hyper_dist_test():
#     n = 10
#     test_prob_2d = get_coords(1, n, 2).squeeze().double().numpy()
#     test_prob_12d = get_coords(1, n, 12).squeeze().double().numpy()
#     tour = np.arange(n, dtype=np.int32)
    
#     orig = eucl_dist(test_prob_2d, tour, 0, 1)
#     hyper = hyper_eucl_dist(test_prob_2d, tour, 0, 1)
#     assert np.allclose(np.asarray([orig]), np.asarray([hyper]))

#     orig_mod = eucl_dist(test_prob_2d, tour, -1, 0)
#     hyper_mod = hyper_eucl_dist(test_prob_2d, tour, -1, 0)
#     assert np.allclose(np.asarray([orig_mod]), np.asarray([hyper_mod]))

#     hyper_high_d = hyper_eucl_dist(test_prob_12d, tour, 0, 1)
#     assert hyper_high_d > 0


def worker_fn(args, queue, node_scale, dim_scale, proc_id):
    if args.seed is not None:
        base_seed = args.seed
    else:
        base_seed = int.from_bytes(os.urandom(4), "big")

    seed_rand_gen(base_seed + proc_id)
    if VERBOSE: print(f"({proc_id}) Using random seed {base_seed + proc_id}")

    n = node_scale
    d = dim_scale
    k = args.optima_per_problem
    p = args.num_problems

    blo_coords = []
    blo_hits = []
    unique_lo = []
    search_swaps = []

    probs = get_coords(p, n, dimensions=d)

    for prob_idx in range(p):
        prob = probs[prob_idx]

        # generate batch of local optima
        prob_repeated = torch.stack(k * [prob]).numpy().astype(np.double)
        starting_tours = torch.stack([torch.randperm(n) for _ in range(k)]).numpy().astype(np.int32)

        assert args.search_algo in ("2opt", "2swap"), "Only 2opt and 2swap currently supported"
        
        if VERBOSE: before = time()

        if args.search_algo == "2opt":
            optima, swaps = batch_two_opt_search(prob_repeated, starting_tours)
        else:  # 2swap
            optima, swaps = batch_two_swap_search(prob_repeated, starting_tours)

        if VERBOSE:
            duration = time() - before
            print(f"({proc_id}) {duration:.1f}s for {k} probs {n}n")

        # find observable global optima
        costs = np_get_costs(prob_repeated, optima)
        best_local_optima = optima[np.argmin(costs)]
        blo_coord = prob[best_local_optima]

        blo_repeated = np.stack(args.optima_per_problem * [best_local_optima])
        blo_dists = batch_edge_hamming_distance(optima, blo_repeated)

        blo_count = np.isclose(blo_dists, np.zeros_like(blo_dists)).sum()  # counts how many times we reached the same observed global optimum

        unique_optima_count = len(np.unique(costs))  # NOTE very unlikely but two distinct tours COULD have the same tour length; but brute force checking for # of unique tours is non-trivial and expensive

        # gather data
        blo_coords.append(blo_coord)
        blo_hits.append(blo_count)
        unique_lo.append(unique_optima_count)
        search_swaps.append(swaps)

    blo_coords = np.stack(blo_coords)
    blo_hits = np.stack(blo_hits)
    unique_lo = np.stack(unique_lo)
    search_swaps = np.stack(search_swaps)

    # deliver to main process via queue
    msg = dict(
        proc_id=proc_id,
        blo_coords=blo_coords,
        blo_hits=blo_hits,
        unique_lo=unique_lo,
        search_swaps=search_swaps
    )

    queue.put(msg)



if __name__ == "__main__":
    # hyper_dist_test()

    cfg = SUPER_CONFIG[parser.parse_args().cfg]

    if os.getenv("SLURM_ARRAY_TASK_ID") is not None:
        slar_idx = int(os.getenv("SLURM_ARRAY_TASK_ID"))
        cfg.mlflow_logging_signature = cfg.mlflow_logging_signature + f"_{slar_idx}"

        n_idx = slar_idx // len(cfg.dscales)
        d_idx = slar_idx % len(cfg.dscales)

        n_range = [cfg.nscales[n_idx]]
        d_range = [cfg.dscales[d_idx]]
    else:
        n_range = cfg.nscales
        d_range = cfg.dscales

    setup_logging(cfg)
    step_idx = 0

    for nscale in tqdm(n_range, desc="nscale", position=0):

        for dscale in tqdm(d_range, desc="dscale", position=1, leave=False):

            retv_q = Queue()
            procs = [Process(target=worker_fn, args=(cfg, retv_q, nscale, dscale, proc_id)) for proc_id in range(cfg.parallel_jobs)]

            [p.start() for p in procs]
            msgs = drain_queue(retv_q, cfg.parallel_jobs)
            [p.join() for p in procs]

            output = stitch_msgs(msgs)

            log_keys = list(output.keys())
            log_keys.remove("blo_coords")

            for key in log_keys:
                Logger.log_stat(key, output[key], step=step_idx)
                Logger.log(f"size_{key}", len(output[key]), step=step_idx)

            Logger.log("nscale", nscale, step=step_idx)
            Logger.log("dscale", dscale, step=step_idx)

            # save approximate global optima to disk
            if not osp.exists(cfg.output_dataset_path):
                os.makedirs(cfg.output_dataset_path)

            slug = f"{dscale}d_{cfg.search_algo}_{cfg.optima_per_problem}k_{nscale}n_{len(output['blo_coords'])}t.npy"
            out_path = osp.join(cfg.output_dataset_path, slug)

            with open(out_path, "wb") as f:
                np.save(f, output["blo_coords"])

            step_idx += 1
