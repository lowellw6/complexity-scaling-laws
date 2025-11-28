
from tqdm import tqdm
import os.path as osp
import os
from multiprocessing import Process, Queue
import numpy as np
import torch

from tsp.utils import seed_rand_gen, get_costs, get_coords, perm_shuffle, get_num_named_params
from tsp.logger import MLflowLogger as Logger, parse_log_sig
from localsearch.distance import edge_hamming_distance, batch_edge_hamming_distance, tei_node_hamming_distance, batch_tei_node_hamming_distance
from localsearch.utils import setup_logging, drain_queue, stitch_msgs, np_get_costs

from config.eval_constrained_local_optima import cfg


VERBOSE = False
TEST = True

DIST_FN_MAP = {"edge" : batch_edge_hamming_distance, "tei-node" : batch_tei_node_hamming_distance}


def get_slice_size(slc):
    start = slc.start if slc.start is not None else 0
    stop = slc.stop if slc.stop is not None else 0
    step = slc.step if slc.step is not None else 1
    return (stop - start) // step


def get_dataset_names(conf, nscale, dscale, search_cap):
    local_dataset_size = get_slice_size(conf.global_dataset_slice) if conf.global_dataset_slice is not None else conf.global_dataset_size
    search_cap_str = search_cap if search_cap is not None else "inf-"

    if conf.input_dataset_name_style.startswith("pyconcorde"):
        input_dataset_index = conf.input_dataset_name_style.split(":")[-1]
        global_dataset_name = f"sol_{nscale}n_{conf.global_dataset_size}t_{input_dataset_index}.npy"
        local_dataset_suffix = f"{conf.local_optima_per_problem}lo_{nscale}n_{search_cap_str}M_{local_dataset_size}t_{input_dataset_index}.npz"

    elif conf.input_dataset_name_style.startswith("proxy"):
        proxyopt_algo, best_of = conf.input_dataset_name_style.split(":")[1:]
        global_dataset_name = f"{dscale}d_{proxyopt_algo}_{best_of}k_{nscale}n_{conf.global_dataset_size}t.npy"
        local_dataset_suffix = f"{conf.local_optima_per_problem}lo_{dscale}d_{nscale}n_{search_cap_str}M_{local_dataset_size}t.npz"

    else:
        raise ValueError(f"Unrecognized input_dataset_name_style: {conf.input_dataset_name_style}")
    
    return global_dataset_name, local_dataset_suffix


def worker_fn(args, queue, global_path, local_paths, search_cap, batch_idx, proc_id):
    if args.seed is not None:
        base_seed = args.seed
    else:
        base_seed = int.from_bytes(os.urandom(4), "big")

    seed_rand_gen(base_seed + proc_id)
    if VERBOSE: print(f"({proc_id}) Using random seed {base_seed + proc_id}")

    # initialize response message dict
    msg = dict(proc_id=proc_id)
    
    # get batch idxs (expected to be shared for global and local optima)
    local_dataset_size = get_slice_size(args.global_dataset_slice) if args.global_dataset_slice is not None else args.global_dataset_size

    fragment_size = args.batch_size // args.parallel_jobs
    fragment_offset = args.batch_size * batch_idx

    start_idx = fragment_offset + proc_id * fragment_size
    end_idx = min(start_idx + fragment_size, local_dataset_size)

    if start_idx >= local_dataset_size:
        queue.put(msg)  # just proc_id empty message
        return
    
    # load corresponding fragments from global dataset, which are (ordered coordinates) 
    global_dataset_mmap = np.load(global_path, mmap_mode="r")  # keeps on disk
    gdata = np.copy(global_dataset_mmap[start_idx:end_idx]).astype(np.double)  # copy loads slice into memory
    gprobs, gn = gdata.shape[:2]

    # load selected local optima fragments (2opt, 2swap, etc; includes tour integers + swap data)
    ldata = {}  
    for algo, local_path in local_paths.items():
        lmmap = np.load(local_path, mmap_mode="r")
        ldata[algo] = {lkey : np.copy(lmmap[lkey][start_idx:end_idx]).astype(np.int32) for lkey in lmmap.keys()}

    # generate random solution batch as baseline
    random_tours = torch.stack([torch.randperm(gn) for _ in range(args.random_per_problem * gprobs)]).numpy().astype(np.int32)

    # compute costs for global optima, each local optima neighborhood type, and random solutions
    gcosts = np_get_costs(gdata)
    msg["global_optima.costs"] = gcosts

    grpt = np.repeat(gdata, args.local_optima_per_problem, axis=0)  # repeat interleave problems along batch dim according to # of local optima per problem
    for algo, ld in ldata.items():
        lopt = ld["optima"].reshape(-1, gn)  # (batch fragment, # local optima, n) --> (batch fragment * # local optima, n)
        lcosts = np_get_costs(grpt, select_idxs=lopt)
        msg[f"{algo}_local_opt.costs"] = lcosts

    rcosts = np_get_costs(grpt, select_idxs=random_tours)
    msg["random.costs"] = rcosts

    # for each distance function
    for dist_key in args.distance_metrics:
        dist_fn = DIST_FN_MAP[dist_key]

        # compute random residuals w.r.t. global optima
        gtours4rand = np.stack([np.arange(gn) for _ in range(args.random_per_problem * gprobs)], axis=0)
        rand_residuals = dist_fn(gtours4rand, random_tours)
        msg[f"{dist_key}.random.residuals"] = rand_residuals

        # compute random tour distance baseline
        shuf_ridxs_a, shuf_ridxs_b = np.split(np.random.choice(len(random_tours), size=len(random_tours)), 2)
        rand_dists = dist_fn(random_tours[shuf_ridxs_a], random_tours[shuf_ridxs_b])
        msg[f"{dist_key}.random.spreads"] = rand_dists

        # for each local optima neighborhood type dataset
        for algo, ld in ldata.items():
            # compute all local optima residuals
            lopt = ld["optima"].reshape(-1, gn)  # (batch fragment, # local optima, n) --> (batch fragment * # local optima, n)
            gtours4lopt = np.stack([np.arange(gn) for _ in range(args.local_optima_per_problem * gprobs)], axis=0)
            lopt_residuals = dist_fn(gtours4lopt, lopt)
            msg[f"{dist_key}.{algo}_local_opt.residuals"] = lopt_residuals

            # compute all intra-problem local-to-local optima distances (NOTE only supports args.local_optima_per_problem == 2 right now)
            lopt_dists = dist_fn(ld["optima"][:, 0, :], ld["optima"][:, 1, :])
            msg[f"{dist_key}.{algo}_local_opt.spreads"] = lopt_dists

            # compute all start-to-local-optima distances
            starts = ld["starts"].reshape(-1, gn)  # (batch fragment, # local optima, n) --> (batch fragment * # local optima, n)
            start_lopt_dists = dist_fn(starts, lopt)
            msg[f"{dist_key}.{algo}_local_opt.rolls"] = start_lopt_dists

            # extract swap info
            # NOTE updated to 1) fix max_moves + 1 flag for hitting search cap and 2) save count for how many times search hit search cap
            lopt_swaps = np.copy(ld["swaps"])
            capped_mask = lopt_swaps == (search_cap + 1)
            lopt_swaps[capped_mask] = search_cap

            msg[f"{dist_key}.{algo}_local_opt.swaps"] = lopt_swaps.flatten()
            msg[f"{dist_key}.{algo}_local_opt.caps"] = capped_mask.astype(np.int32).flatten()

            # optional tests
            if TEST:
                # validate batch distance with random individual distance computation
                single_dist_fn, dist_max = (edge_hamming_distance, 2 * gn) if dist_key == "edge" else (tei_node_hamming_distance, gn)  # tei-node
                ridx = np.random.choice(len(gtours4lopt))
                single_dist_test = single_dist_fn(gtours4lopt[ridx], lopt[ridx])
                assert lopt_residuals[ridx] == single_dist_test
                assert single_dist_test <= dist_max

                # validate repeat interleave reshaping for flattening out local-optima-per-problem dim
                ridx = np.random.choice(len(gdata))
                for lidx in range(args.local_optima_per_problem):
                    assert np.allclose(gdata[ridx], grpt[2*ridx + lidx])
                    assert np.allclose(ld["optima"][ridx, lidx].astype(np.float32), lopt[2*ridx + lidx].astype(np.float32))

    queue.put(msg)



if __name__ == "__main__":
    setup_logging(cfg)

    local_dataset_size = get_slice_size(cfg.global_dataset_slice) if cfg.global_dataset_slice is not None else cfg.global_dataset_size
    
    n_scale_range = [cfg.nscales] if type(cfg.nscales) is int else cfg.nscales
    d_scale_range = [cfg.dscales] if type(cfg.dscales) is int else cfg.dscales
    scap_range = [cfg.search_caps] if cfg.search_caps is None or type(cfg.search_caps) is int else cfg.search_caps

    log_step_idx = 0
    for nscale in tqdm(n_scale_range, desc="nscale", position=0):

        for dscale in tqdm(d_scale_range, desc="dscale", position=1, leave=False):

            for scap in tqdm(scap_range, desc="search_cap", position=2, leave=False):
                global_dataset_name, local_dataset_suffix = get_dataset_names(cfg, nscale, dscale, scap)

                global_path = osp.join(cfg.global_dataset_path, global_dataset_name)

                local_dataset_names = {prefix : f"{prefix}_{local_dataset_suffix}" for prefix in cfg.local_dataset_prefices}
                local_paths = {prefix : osp.join(cfg.local_dataset_path, ldn) for prefix, ldn in local_dataset_names.items()}

                response_store = []
                num_batches = int(np.ceil(local_dataset_size / cfg.batch_size))
                for batch_idx in tqdm(range(num_batches), desc="batch", position=3, leave=False):
                    retv_q = Queue()
                    procs = [Process(target=worker_fn, args=(cfg, retv_q, global_path, local_paths, scap, batch_idx, proc_id)) for proc_id in range(cfg.parallel_jobs)]

                    [p.start() for p in procs]
                    msgs = drain_queue(retv_q, cfg.parallel_jobs)
                    [p.join() for p in procs]

                    msgs = list(filter(lambda x: len(x.keys()) > 1, msgs))  # remove empty responses which can occur based on batching division logic
                    response_store.append(stitch_msgs(msgs))

                n_stats = stitch_msgs(response_store)

                for key in n_stats.keys():
                    Logger.log_stat(key, n_stats[key], step=log_step_idx)
                    Logger.log(f"size_{key}", len(n_stats[key]), step=log_step_idx)
                    
                Logger.log("nscale", nscale, step=log_step_idx)
                Logger.log("dscale", dscale, step=log_step_idx)
                Logger.log("search_cap", scap, step=log_step_idx)

                log_step_idx += 1
