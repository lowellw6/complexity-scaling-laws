
from argparse import Namespace

import tsp
import os.path as osp
root_path = osp.dirname(osp.dirname(tsp.__file__))


args = Namespace(
    output_dataset_path = osp.join(root_path, "constrained_search_datasets/dim20_scaling"),
    input_dataset_path = osp.join(root_path, "approx_global_optima_datasets"),
    input_dataset_name_style = "proxy:2opt:1000",  # supports 'pyconcorde:<dataset_idx>' or 'proxy:<search_algo>:<best-of>'; former formatted sol_<nodes>n_<num_problems>t_<dataset_idx>.npy, latter formatted <dims>d_<search_algo>_<best-of>k_<nodes>n_<num_problems>t.npy
    input_dataset_size = 64_000,  # expected constant
    output_slice = slice(32_000),  # slice of input to use for output
    nscales = 20,  # can be single integer or list of problem sizes, in which case generation is repeated for each size
    dscales = list(range(2, 13)) + [15, 20, 30, 40, 50, 100],  # ditto, but for TSP spatial dimensions
    search_caps = [4, 6, 8, 11, 14, 20],  #[5, 7, 9, 12, 15, 20],  # search move maximum, or None for no capacity limit; can be single value or list like node and dim scales
    search_algo = "2swap",  # supports only '2opt' and '2swap' currently
    optima_per_problem = 2,  # number of local optima to generate per input dataset problem
    parallel_jobs = 32,  # number of parallel jobs
    batch_size = 8000, # number of input problems to parallel process before synchronizing and gathering in main process (multiplies by optima_per_problem)
    seed = None,  # seed random tour starts for local search, or use random seed if set to None
)
