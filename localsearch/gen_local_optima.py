"""
Generates local optima for a provided input TSP dataset,
storing in a new dataset of tours in the preserved order of the
original dataset

Expects input dataset to be float of shape (num_problems, num_nodes, 2)

Output dataset is int of shape (num_problems, num_optima_per_problem, num_nodes)

Also provides the random starting tours for each optima, and the number of swaps taken to get to the optima

NOTE currently uses int8 to store optima tour idxs, if nscale > 127 this needs to be increased!
"""

import pyximport; pyximport.install(language_level=3)
from csearch import batch_two_opt_search, batch_two_swap_search

from time import time
import os
import os.path as osp
import torch
import numpy as np
from multiprocessing import Process, Queue
from tqdm import tqdm

from tsp.utils import seed_rand_gen

from config.gen_local_optima import args as config_args


VERBOSE = False


def get_slice_size(slc):
    start = slc.start if slc.start is not None else 0
    stop = slc.stop if slc.stop is not None else 0
    step = slc.step if slc.step is not None else 1
    return (stop - start) // step


def get_dataset_name(conf, nscale, dscale, search_cap):
    output_dataset_size = get_slice_size(conf.output_slice) if conf.output_slice is not None else conf.input_dataset_size
    search_cap_str = search_cap if search_cap is not None else "inf-"

    if conf.input_dataset_name_style.startswith("pyconcorde"):
        input_dataset_index = conf.input_dataset_name_style.split(":")[-1]
        dataset_name = f"sol_{nscale}n_{conf.input_dataset_size}t_{input_dataset_index}.npy"
        out_slug = f"{conf.search_algo}_{conf.optima_per_problem}lo_{nscale}n_{search_cap_str}M_{output_dataset_size}t_{input_dataset_index}.npz"

    elif conf.input_dataset_name_style.startswith("proxy"):
        proxyopt_algo, best_of = conf.input_dataset_name_style.split(":")[1:]
        dataset_name = f"{dscale}d_{proxyopt_algo}_{best_of}k_{nscale}n_{conf.input_dataset_size}t.npy"
        out_slug = f"{conf.search_algo}_{conf.optima_per_problem}lo_{dscale}d_{nscale}n_{search_cap_str}M_{output_dataset_size}t.npz"

    else:
        raise ValueError(f"Unrecognized input_dataset_name_style: {conf.input_dataset_name_style}")
    
    return dataset_name, out_slug


def gen_func(args, queue, input_path, search_cap, batch_idx, proc_id):
    if args.seed is not None:
        base_seed = args.seed
    else:
        base_seed = int.from_bytes(os.urandom(4), "big")

    seed_rand_gen(base_seed + proc_id)
    if VERBOSE: print(f"({proc_id}) Using random seed {base_seed + proc_id}")

    # load dataset fragment this process is responsible for
    dataset_mmap = np.load(input_path, mmap_mode="r")  # keeps on disk

    out_size = get_slice_size(args.output_slice) if args.output_slice is not None else args.input_dataset_size
    
    fragment_size = args.batch_size // args.parallel_jobs
    fragment_offset = args.batch_size * batch_idx

    start_idx = fragment_offset + proc_id * fragment_size
    end_idx = min(start_idx + fragment_size, out_size)

    if start_idx >= out_size:
        queue.put(dict(proc_id=proc_id, optima=None, swaps=None))
        return
    
    pdata = np.copy(dataset_mmap[start_idx:end_idx]).astype(np.double)  # copy loads slice into memory

    pprobs, nscale = pdata.shape[:2]
    optima = np.empty((pprobs, 0, nscale), dtype=np.int32)
    swaps = np.empty((pprobs, 0), dtype=np.int32)
    rand_starts = np.empty((pprobs, 0, nscale), dtype=np.int32)
    
    # iterate for requested number of optima
    for _ in range(args.optima_per_problem):
        # create random starting tours
        nscale = pdata.shape[1]
        starting_tours = torch.stack([torch.randperm(nscale) for _ in range(pprobs)]).numpy().astype(np.int32)

        # run local search
        assert args.search_algo in ("2opt", "2swap"), "Only 2opt and 2swap currently supported"
        
        if VERBOSE: before = time()

        if args.search_algo == "2opt":
            b_optima, b_swaps = batch_two_opt_search(pdata, starting_tours, search_cap)
        else:  # 2swap
            b_optima, b_swaps = batch_two_swap_search(pdata, starting_tours, search_cap)

        if VERBOSE:
            duration = time() - before
            print(f"({proc_id}) {duration:.1f}s for {pprobs}t {nscale}n")

        # gather optima along dim 1
        b_optima = np.expand_dims(b_optima, axis=1)
        b_swaps = np.expand_dims(b_swaps, axis=1)
        b_starts = np.expand_dims(starting_tours, axis=1)

        optima = np.concatenate((optima, b_optima), axis=1)
        swaps = np.concatenate((swaps, b_swaps), axis=1)
        rand_starts = np.concatenate((rand_starts, b_starts), axis=1)

    # deliver to main process via queue
    output = dict(
        proc_id=proc_id,
        optima=optima,
        swaps=swaps,
        rand_starts=rand_starts
    )

    queue.put(output)



if __name__ == "__main__":
    conf = config_args

    out_size = get_slice_size(conf.output_slice) if conf.output_slice is not None else conf.input_dataset_size

    assert out_size >= conf.batch_size

    n_scale_range = [conf.nscales] if type(conf.nscales) is int else conf.nscales
    d_scale_range = [conf.dscales] if type(conf.dscales) is int else conf.dscales
    scap_range = [conf.search_caps] if conf.search_caps is None or type(conf.search_caps) is int else conf.search_caps

    for nscale in tqdm(n_scale_range, desc="nscale", position=0):

        for dscale in tqdm(d_scale_range, desc="dscale", position=1, leave=False):

            for scap in tqdm(scap_range, desc="search_cap", position=2, leave=False):
                dataset_name, out_slug = get_dataset_name(conf, nscale, dscale, scap)
                input_path = osp.join(conf.input_dataset_path, dataset_name)

                optima_batches = []
                swaps_batches = []
                rand_starts_batches = []

                num_batches = int(np.ceil(out_size / conf.batch_size))
                for batch_idx in tqdm(range(num_batches), desc="batch", position=3, leave=False):
                    retv_q = Queue()
                    procs = [Process(target=gen_func, args=(conf, retv_q, input_path, scap, batch_idx, proc_id)) for proc_id in range(conf.parallel_jobs)]

                    [p.start() for p in procs]

                    optima_snippets = []
                    msgs_rcv = 0
                    while msgs_rcv < conf.parallel_jobs:
                        optima_snippets.append(retv_q.get())
                        msgs_rcv += 1

                    [p.join() for p in procs]

                    optima_snippets = list(filter(lambda x: x["optima"] is not None, optima_snippets))

                    # sort batches based on provided proc_id tag and gather
                    realigned_opt_snippets = sorted(optima_snippets, key=lambda x: int(x["proc_id"]))

                    optima = np.concatenate([retv["optima"] for retv in realigned_opt_snippets], axis=0)
                    swaps = np.concatenate([retv["swaps"] for retv in realigned_opt_snippets], axis=0)
                    rand_starts = np.concatenate([retv["rand_starts"] for retv in realigned_opt_snippets], axis=0)

                    optima_batches.append(optima)
                    swaps_batches.append(swaps)
                    rand_starts_batches.append(rand_starts)

                optima = np.concatenate(optima_batches, axis=0, dtype=np.int8)  # NOTE if nscale > 127 this needs to be increased!
                swaps = np.concatenate(swaps_batches, axis=0, dtype=np.int16)
                rand_starts = np.concatenate(rand_starts_batches, axis=0, dtype=np.int8)  # NOTE ditto
                
                # save to disk as .npz uncompressed zip
                if not osp.exists(conf.output_dataset_path):
                    os.makedirs(conf.output_dataset_path)

                out_path = osp.join(conf.output_dataset_path, out_slug)

                with open(out_path, "wb") as f:
                    np.savez(f, optima=optima, swaps=swaps, starts=rand_starts)
