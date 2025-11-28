
from argparse import Namespace

import tsp
import os.path as osp
root_path = osp.dirname(osp.dirname(tsp.__file__))


# cfg = Namespace(  # model scaling 2opt
#     mlflow_logging_signature = f"EVAL_constrained_local_optima/model_scaling_2opt",  # format: <MLflow_experiment_group>/<MLflow_run_name> (if not provided, no logging occurs)
#     nscales = 20,  # can be single integer or list of problem sizes, in which case generation is repeated for each size
#     dscales = 2,  # ditto, but for TSP spatial dimensions
#     search_caps = list(range(1, 28)),

#     input_dataset_name_style = "pyconcorde:0",  # determines both global and local dataset stubs; supports 'pyconcorde:<dataset_idx>' or 'proxy:<search_algo>:<best-of>'; former formatted sol_<nodes>n_<num_problems>t_<dataset_idx>.npy, latter formatted <dims>d_<search_algo>_<best-of>k_<nodes>n_<num_problems>t.npy

#     global_dataset_path = osp.join(root_path, "datasets"),
#     global_dataset_size = 1_280_000,
#     global_dataset_slice = slice(64_000),  # slice used to generate local optima dataset

#     local_dataset_path = osp.join(root_path, "constrained_search_datasets/model_scaling"),
#     local_dataset_prefices = ("2opt",),  # one for each neighborhood type; assumes same name formatting as global optima datasets above
#     local_optima_per_problem = 2,  # number of local optima generated per global optima dataset problem (expected constant between neighborhood types)

#     random_per_problem = 2,  # baseline random solutions per global optima problem to compare stats with

#     distance_metrics = ("edge", "tei-node"),  # repeats all distance measurements over each type; supports "edge" and "tei-node"

#     parallel_jobs = 32,  # number of parallel jobs
#     batch_size = 12_800, # number of input problems to parallel process before synchronizing and gathering in main process (multiplies by optima_per_problem)
#     seed = None,  # seed random tour starts for local search, or use random seed if set to None
# )


# cfg = Namespace(  # node scaling 2opt
#     mlflow_logging_signature = f"EVAL_constrained_local_optima/plus_node_scaling_2opt",  # format: <MLflow_experiment_group>/<MLflow_run_name> (if not provided, no logging occurs)
#     nscales = list(range(5, 51, 5)),  # can be single integer or list of problem sizes, in which case generation is repeated for each size
#     dscales = 2,  # ditto, but for TSP spatial dimensions
#     search_caps = [5, 10, 15, 20, 25, 40],

#     input_dataset_name_style = "pyconcorde:0",  # determines both global and local dataset stubs; supports 'pyconcorde:<dataset_idx>' or 'proxy:<search_algo>:<best-of>'; former formatted sol_<nodes>n_<num_problems>t_<dataset_idx>.npy, latter formatted <dims>d_<search_algo>_<best-of>k_<nodes>n_<num_problems>t.npy

#     global_dataset_path = osp.join(root_path, "datasets"),
#     global_dataset_size = 1_280_000,
#     global_dataset_slice = slice(64_000),  # slice used to generate local optima dataset

#     local_dataset_path = osp.join(root_path, "constrained_search_datasets/node_scaling"),
#     local_dataset_prefices = ("2opt",),  # one for each neighborhood type; assumes same name formatting as global optima datasets above
#     local_optima_per_problem = 2,  # number of local optima generated per global optima dataset problem (expected constant between neighborhood types)

#     random_per_problem = 2,  # baseline random solutions per global optima problem to compare stats with

#     distance_metrics = ("edge", "tei-node"),  # repeats all distance measurements over each type; supports "edge" and "tei-node"

#     parallel_jobs = 32,  # number of parallel jobs
#     batch_size = 12_800, # number of input problems to parallel process before synchronizing and gathering in main process (multiplies by optima_per_problem)
#     seed = None,  # seed random tour starts for local search, or use random seed if set to None
# )


# cfg = Namespace(  # dim10 scaling 2opt
#     mlflow_logging_signature = f"EVAL_constrained_local_optima/plus_dim10_scaling_2opt",  # format: <MLflow_experiment_group>/<MLflow_run_name> (if not provided, no logging occurs)
#     nscales = 10,  # can be single integer or list of problem sizes, in which case generation is repeated for each size
#     dscales = list(range(2, 13)) + [15, 20, 30, 40, 50, 100],  # ditto, but for TSP spatial dimensions
#     search_caps = [2, 3, 4, 5, 6, 10],

#     input_dataset_name_style = "proxy:2opt:100",  # determines both global and local dataset stubs; supports 'pyconcorde:<dataset_idx>' or 'proxy:<search_algo>:<best-of>'; former formatted sol_<nodes>n_<num_problems>t_<dataset_idx>.npy, latter formatted <dims>d_<search_algo>_<best-of>k_<nodes>n_<num_problems>t.npy

#     global_dataset_path = osp.join(root_path, "approx_global_optima_datasets"),
#     global_dataset_size = 128_000,
#     global_dataset_slice = slice(64_000),  # slice used to generate local optima dataset

#     local_dataset_path = osp.join(root_path, "constrained_search_datasets/dim10_scaling"),
#     local_dataset_prefices = ("2opt",),  # one for each neighborhood type; assumes same name formatting as global optima datasets above
#     local_optima_per_problem = 2,  # number of local optima generated per global optima dataset problem (expected constant between neighborhood types)

#     random_per_problem = 2,  # baseline random solutions per global optima problem to compare stats with

#     distance_metrics = ("edge", "tei-node"),  # repeats all distance measurements over each type; supports "edge" and "tei-node"

#     parallel_jobs = 32,  # number of parallel jobs
#     batch_size = 12_800, # number of input problems to parallel process before synchronizing and gathering in main process (multiplies by optima_per_problem)
#     seed = None,  # seed random tour starts for local search, or use random seed if set to None
# )


# cfg = Namespace(  # dim20 scaling 2opt
#     mlflow_logging_signature = f"EVAL_constrained_local_optima/plus_dim20_scaling_2opt",  # format: <MLflow_experiment_group>/<MLflow_run_name> (if not provided, no logging occurs)
#     nscales = 20,  # can be single integer or list of problem sizes, in which case generation is repeated for each size
#     dscales = list(range(2, 13)) + [15, 20, 30, 40, 50, 100],  # ditto, but for TSP spatial dimensions
#     search_caps = [5, 7, 9, 12, 15, 20],

#     input_dataset_name_style = "proxy:2opt:1000",  # determines both global and local dataset stubs; supports 'pyconcorde:<dataset_idx>' or 'proxy:<search_algo>:<best-of>'; former formatted sol_<nodes>n_<num_problems>t_<dataset_idx>.npy, latter formatted <dims>d_<search_algo>_<best-of>k_<nodes>n_<num_problems>t.npy

#     global_dataset_path = osp.join(root_path, "approx_global_optima_datasets"),
#     global_dataset_size = 64_000,
#     global_dataset_slice = slice(64_000),  # slice used to generate local optima dataset

#     local_dataset_path = osp.join(root_path, "constrained_search_datasets/dim20_scaling"),
#     local_dataset_prefices = ("2opt",),  # one for each neighborhood type; assumes same name formatting as global optima datasets above
#     local_optima_per_problem = 2,  # number of local optima generated per global optima dataset problem (expected constant between neighborhood types)

#     random_per_problem = 2,  # baseline random solutions per global optima problem to compare stats with

#     distance_metrics = ("edge", "tei-node"),  # repeats all distance measurements over each type; supports "edge" and "tei-node"

#     parallel_jobs = 32,  # number of parallel jobs
#     batch_size = 12_800, # number of input problems to parallel process before synchronizing and gathering in main process (multiplies by optima_per_problem)
#     seed = None,  # seed random tour starts for local search, or use random seed if set to None
# )



# cfg = Namespace(   # model scaling 2swap
#     mlflow_logging_signature = f"EVAL_constrained_local_optima/model_scaling_2swap",  # format: <MLflow_experiment_group>/<MLflow_run_name> (if not provided, no logging occurs)
#     nscales = 20,  # can be single integer or list of problem sizes, in which case generation is repeated for each size
#     dscales = 2,  # ditto, but for TSP spatial dimensions
#     search_caps = list(range(1, 36)),

#     input_dataset_name_style = "pyconcorde:0",  # determines both global and local dataset stubs; supports 'pyconcorde:<dataset_idx>' or 'proxy:<search_algo>:<best-of>'; former formatted sol_<nodes>n_<num_problems>t_<dataset_idx>.npy, latter formatted <dims>d_<search_algo>_<best-of>k_<nodes>n_<num_problems>t.npy

#     global_dataset_path = osp.join(root_path, "datasets"),
#     global_dataset_size = 1_280_000,
#     global_dataset_slice = slice(32_000),  # slice used to generate local optima dataset

#     local_dataset_path = osp.join(root_path, "constrained_search_datasets/model_scaling"),
#     local_dataset_prefices = ("2swap",),  # one for each neighborhood type; assumes same name formatting as global optima datasets above
#     local_optima_per_problem = 2,  # number of local optima generated per global optima dataset problem (expected constant between neighborhood types)

#     random_per_problem = 2,  # baseline random solutions per global optima problem to compare stats with

#     distance_metrics = ("edge", "tei-node"),  # repeats all distance measurements over each type; supports "edge" and "tei-node"

#     parallel_jobs = 32,  # number of parallel jobs
#     batch_size = 8000, # number of input problems to parallel process before synchronizing and gathering in main process (multiplies by optima_per_problem)
#     seed = None,  # seed random tour starts for local search, or use random seed if set to None
# )


# cfg = Namespace(  # node scaling 2swap
#     mlflow_logging_signature = f"EVAL_constrained_local_optima/plus_node_scaling_2swap",  # format: <MLflow_experiment_group>/<MLflow_run_name> (if not provided, no logging occurs)
#     nscales = list(range(5, 51, 5)),  # can be single integer or list of problem sizes, in which case generation is repeated for each size
#     dscales = 2,  # ditto, but for TSP spatial dimensions
#     search_caps = [5, 10, 15, 20, 25, 40],

#     input_dataset_name_style = "pyconcorde:0",  # determines both global and local dataset stubs; supports 'pyconcorde:<dataset_idx>' or 'proxy:<search_algo>:<best-of>'; former formatted sol_<nodes>n_<num_problems>t_<dataset_idx>.npy, latter formatted <dims>d_<search_algo>_<best-of>k_<nodes>n_<num_problems>t.npy

#     global_dataset_path = osp.join(root_path, "datasets"),
#     global_dataset_size = 1_280_000,
#     global_dataset_slice = slice(32_000),  # slice used to generate local optima dataset

#     local_dataset_path = osp.join(root_path, "constrained_search_datasets/node_scaling"),
#     local_dataset_prefices = ("2swap",),  # one for each neighborhood type; assumes same name formatting as global optima datasets above
#     local_optima_per_problem = 2,  # number of local optima generated per global optima dataset problem (expected constant between neighborhood types)

#     random_per_problem = 2,  # baseline random solutions per global optima problem to compare stats with

#     distance_metrics = ("edge", "tei-node"),  # repeats all distance measurements over each type; supports "edge" and "tei-node"

#     parallel_jobs = 32,  # number of parallel jobs
#     batch_size = 8000, # number of input problems to parallel process before synchronizing and gathering in main process (multiplies by optima_per_problem)
#     seed = None,  # seed random tour starts for local search, or use random seed if set to None
# )


# cfg = Namespace(  # dim10 scaling 2swap
#     mlflow_logging_signature = f"EVAL_constrained_local_optima/plus_dim10_scaling_2swap",  # format: <MLflow_experiment_group>/<MLflow_run_name> (if not provided, no logging occurs)
#     nscales = 10,  # can be single integer or list of problem sizes, in which case generation is repeated for each size
#     dscales = list(range(2, 13)) + [15, 20, 30, 40, 50, 100],  # ditto, but for TSP spatial dimensions
#     search_caps = [2, 3, 4, 5, 6, 10],

#     input_dataset_name_style = "proxy:2opt:100",  # determines both global and local dataset stubs; supports 'pyconcorde:<dataset_idx>' or 'proxy:<search_algo>:<best-of>'; former formatted sol_<nodes>n_<num_problems>t_<dataset_idx>.npy, latter formatted <dims>d_<search_algo>_<best-of>k_<nodes>n_<num_problems>t.npy

#     global_dataset_path = osp.join(root_path, "approx_global_optima_datasets"),
#     global_dataset_size = 128_000,
#     global_dataset_slice = slice(32_000),  # slice used to generate local optima dataset

#     local_dataset_path = osp.join(root_path, "constrained_search_datasets/dim10_scaling"),
#     local_dataset_prefices = ("2swap",),  # one for each neighborhood type; assumes same name formatting as global optima datasets above
#     local_optima_per_problem = 2,  # number of local optima generated per global optima dataset problem (expected constant between neighborhood types)

#     random_per_problem = 2,  # baseline random solutions per global optima problem to compare stats with

#     distance_metrics = ("edge", "tei-node"),  # repeats all distance measurements over each type; supports "edge" and "tei-node"

#     parallel_jobs = 32,  # number of parallel jobs
#     batch_size = 8000, # number of input problems to parallel process before synchronizing and gathering in main process (multiplies by optima_per_problem)
#     seed = None,  # seed random tour starts for local search, or use random seed if set to None
# )


cfg = Namespace(  # dim20 scaling 2swap
    mlflow_logging_signature = f"EVAL_constrained_local_optima/plus_dim20_scaling_2swap",  # format: <MLflow_experiment_group>/<MLflow_run_name> (if not provided, no logging occurs)
    nscales = 20,  # can be single integer or list of problem sizes, in which case generation is repeated for each size
    dscales = list(range(2, 13)) + [15, 20, 30, 40, 50, 100],  # ditto, but for TSP spatial dimensions
    search_caps = [4, 6, 8, 11, 14, 20],

    input_dataset_name_style = "proxy:2opt:1000",  # determines both global and local dataset stubs; supports 'pyconcorde:<dataset_idx>' or 'proxy:<search_algo>:<best-of>'; former formatted sol_<nodes>n_<num_problems>t_<dataset_idx>.npy, latter formatted <dims>d_<search_algo>_<best-of>k_<nodes>n_<num_problems>t.npy

    global_dataset_path = osp.join(root_path, "approx_global_optima_datasets"),
    global_dataset_size = 64_000,
    global_dataset_slice = slice(32_000),  # slice used to generate local optima dataset

    local_dataset_path = osp.join(root_path, "constrained_search_datasets/dim20_scaling"),
    local_dataset_prefices = ("2swap",),  # one for each neighborhood type; assumes same name formatting as global optima datasets above
    local_optima_per_problem = 2,  # number of local optima generated per global optima dataset problem (expected constant between neighborhood types)

    random_per_problem = 2,  # baseline random solutions per global optima problem to compare stats with

    distance_metrics = ("edge", "tei-node"),  # repeats all distance measurements over each type; supports "edge" and "tei-node"

    parallel_jobs = 32,  # number of parallel jobs
    batch_size = 8000, # number of input problems to parallel process before synchronizing and gathering in main process (multiplies by optima_per_problem)
    seed = None,  # seed random tour starts for local search, or use random seed if set to None
)
