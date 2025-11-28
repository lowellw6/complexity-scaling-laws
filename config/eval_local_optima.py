
from argparse import Namespace

import tsp
import os.path as osp
root_path = osp.dirname(osp.dirname(tsp.__file__))

cfg = Namespace(
    mlflow_logging_signature = f"EVAL_local_optima/nscaling_roll_spread_residual_uncorrupted35n",  # format: <MLflow_experiment_group>/<MLflow_run_name> (if not provided, no logging occurs)
    nscales = list(range(5, 51, 5)),  # can be single integer or list of problem sizes, in which case generation is repeated for each size
    dscales = 2,  # ditto, but for TSP spatial dimensions
    
    dataset_sizes = 1_280_000,  # expected constant between global and local optima datasets
    input_dataset_name_style = "pyconcorde:0",  # determines both global and local dataset stubs; supports 'pyconcorde:<dataset_idx>' or 'proxy:<search_algo>:<best-of>'; former formatted sol_<nodes>n_<num_problems>t_<dataset_idx>.npy, latter formatted <dims>d_<search_algo>_<best-of>k_<nodes>n_<num_problems>t.npy

    global_dataset_path = osp.join(root_path, "datasets"),

    local_dataset_path = osp.join(root_path, "local_optima_datasets"),
    local_dataset_prefices = ("2opt", "2swap"),  # one for each neighborhood type; assumes same name formatting as global optima datasets above
    local_optima_per_problem = 2,  # number of local optima generated per global optima dataset problem (expected constant between neighborhood types)

    random_per_problem = 2,  # baseline random solutions per global optima problem to compare stats with

    distance_metrics = ("edge", "tei-node"),  # repeats all distance measurements over each type; supports "edge" and "tei-node"

    parallel_jobs = 32,  # number of parallel jobs
    batch_size = 12_800, # number of input problems to parallel process before synchronizing and gathering in main process (multiplies by optima_per_problem)
    seed = None,  # seed random tour starts for local search, or use random seed if set to None
)
