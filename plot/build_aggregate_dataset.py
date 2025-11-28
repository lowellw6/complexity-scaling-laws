"""
One dataset to rule them all
"""

import os.path as osp
import os
import numpy as np
from tqdm import tqdm

import tsp
root_path = osp.dirname(osp.dirname(tsp.__file__))

OPTIMAL_PATH = osp.join(root_path, "datasets")
PROXYOPT_PATH = osp.join(root_path, "approx_global_optima_datasets")
ML_SOL_PATH = osp.join(root_path, "ml_sol_datasets")
NSCALE_LO_PATH = osp.join(root_path, "local_optima_datasets")
DSCALE_10N_LO_PATH = osp.join(root_path, "dscaling_local_optima_datasets/100k")
DSCALE_20N_LO_PATH = osp.join(root_path, "dscaling_local_optima_datasets/1000k")

OUT_PATH = osp.join(root_path, "bindthem")

NODE_RANGE = range(5, 51, 5)
DIM_RANGE = list(range(2, 13)) + [15, 20, 30, 40, 50, 100]
MODEL_WIDTHS = [24, 32, 40, 48, 56, 72, 88, 104, 128, 160, 192, 240]



def load_mmap(dataset_dir, dataset_stub, mmap_mode="r"):
    dpath = osp.join(dataset_dir, dataset_stub)
    assert osp.exists(dpath), f"can't find dataset: {dpath}"

    dataset = np.load(dpath, mmap_mode=mmap_mode)

    return dataset


def save_bind(output_dir, output_stub, **bind_kwargs):
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    opath = osp.join(output_dir, output_stub)

    with open(opath, "wb") as f:
        np.savez(f, **bind_kwargs)


def nscale_optimal_builder():
    print("Binding node and model scaling experiments...")
    for n in tqdm(NODE_RANGE, desc="nodes"):
        optimals = load_mmap(OPTIMAL_PATH, f"sol_{n}n_1280000t_0.npy")

        _2opt_lo = load_mmap(NSCALE_LO_PATH, f"2opt_2lo_{n}n_1280000t_0.npz")
        _2swap_lo = load_mmap(NSCALE_LO_PATH, f"2swap_2lo_{n}n_1280000t_0.npz")
        
        drl_node_sol = load_mmap(ML_SOL_PATH, f"drl_node_scaling_184w_{n}n_2d_2sol_1280000t.npy")
        il_node_sol = load_mmap(ML_SOL_PATH, f"il_node_scaling_184w_{n}n_2d_2sol_1280000t.npy")

        bind_dict = dict(
            global_optima=optimals,

            _2opt_local_optima=_2opt_lo["optima"],
            _2opt_starts=_2opt_lo["starts"],
            _2opt_swaps=_2opt_lo["swaps"],

            _2swap_local_optima=_2swap_lo["optima"],
            _2swap_starts=_2swap_lo["starts"],
            _2swap_swaps=_2swap_lo["swaps"],

            drl_node_scaling=drl_node_sol,

            il_node_scaling=il_node_sol
        )

        if n == 20:
            drl_model_sols = {f"drl_model_scaling_{w}w" : load_mmap(ML_SOL_PATH, f"drl_model_scaling_{w}w_20n_2d_2sol_1280000t.npy") for w in MODEL_WIDTHS}
            il_model_sols = {f"il_model_scaling_{w}w" : load_mmap(ML_SOL_PATH, f"il_model_scaling_{w}w_20n_2d_2sol_1280000t.npy") for w in MODEL_WIDTHS}

            bind_dict = {**bind_dict, **drl_model_sols, **il_model_sols}

            output_stub = f"node_scale_{n}n_2d_plus_model_scale.npz"

        else:
            output_stub = f"node_scale_{n}n_2d.npz"

        save_bind(OUT_PATH, output_stub, **bind_dict)


def dscale_10n_proxy_builder():
    print("Binding 10n dimension scaling experiments...")
    for d in tqdm(DIM_RANGE, desc="dimensions"):
        proxy_optimals = load_mmap(PROXYOPT_PATH, f"{d}d_2opt_100k_10n_128000t.npy")

        _2opt_lo = load_mmap(DSCALE_10N_LO_PATH, f"2opt_2lo_{d}d_10n_128000t.npz")
        _2swap_lo = load_mmap(DSCALE_10N_LO_PATH, f"2swap_2lo_{d}d_10n_128000t.npz")

        drl_dim_sol = load_mmap(ML_SOL_PATH, f"drl_dim_scaling_184w_10n_{d}d_2sol_128000t.npy")
        
        bind_dict = dict(
            proxy_global_optima=proxy_optimals,

            _2opt_local_optima=_2opt_lo["optima"],
            _2opt_starts=_2opt_lo["starts"],
            _2opt_swaps=_2opt_lo["swaps"],

            _2swap_local_optima=_2swap_lo["optima"],
            _2swap_starts=_2swap_lo["starts"],
            _2swap_swaps=_2swap_lo["swaps"],
        )

        if d == 2:  # edge case 2d solution batch came from node scaling experiment
            bind_dict["drl_node_scaling"] = drl_dim_sol
        else:
            bind_dict["drl_dim_scaling"] = drl_dim_sol

        output_stub = f"dim_scale_10n_{d}d.npz"

        save_bind(OUT_PATH, output_stub, **bind_dict)


def dscale_20n_proxy_builder():
    print("Binding 20n dimension scaling experiments...")
    for d in tqdm(DIM_RANGE, desc="dimensions"):
        proxy_optimals = load_mmap(PROXYOPT_PATH, f"{d}d_2opt_1000k_20n_64000t.npy")

        _2opt_lo = load_mmap(DSCALE_20N_LO_PATH, f"2opt_2lo_{d}d_20n_64000t.npz")
        _2swap_lo = load_mmap(DSCALE_20N_LO_PATH, f"2swap_2lo_{d}d_20n_64000t.npz")

        drl_dim_sol = load_mmap(ML_SOL_PATH, f"drl_dim_scaling_184w_20n_{d}d_2sol_64000t.npy")
        
        bind_dict = dict(
            proxy_global_optima=proxy_optimals,

            _2opt_local_optima=_2opt_lo["optima"],
            _2opt_starts=_2opt_lo["starts"],
            _2opt_swaps=_2opt_lo["swaps"],

            _2swap_local_optima=_2swap_lo["optima"],
            _2swap_starts=_2swap_lo["starts"],
            _2swap_swaps=_2swap_lo["swaps"],
        )

        if d == 2:  # edge case 2d solution batch came from node scaling experiment
            bind_dict["drl_node_scaling"] = drl_dim_sol
        else:
            bind_dict["drl_dim_scaling"] = drl_dim_sol

        output_stub = f"dim_scale_20n_{d}d.npz"

        save_bind(OUT_PATH, output_stub, **bind_dict)


if __name__ == "__main__":

    nscale_optimal_builder()
        
    dscale_10n_proxy_builder()
        
    dscale_20n_proxy_builder()

        




# for each nscale in 2d
    # fetch concorde GO .npy (0th index) in /datasets
    # fetch drl_node_scaling .npy in /ml_sol_datasets
    # fetch il_node_scaling .npy in /ml_sol_datasets
    # iff n = 20
        # fetch drl_model_scaling .npy in /ml_sol_datasets
        # fetch il_model_Scaling .npy in /ml_sol_datasets
    # fetch 2opt LO .npz in /local_optima_datasets
    # fetch 2swap LO .npz in /local_optima_datasets

# for each dscale at 10n
    # fetch proxy GO .npy (2opt_100k_10n_128000t) in /approx_global_optima_datasets
    # fetch drl_dim_scaling (10n) .npy in /ml_sol_datasets
    # fetch 2opt LO .npz in /dscaling_local_optima_datasets/100k
    # fetch 2swap LO .npz in /dscaling_local_optima_datasets/100k

# for each dscale at 20n
    # fetch proxy GO .npy (2opt_1000k_20n_64000t) in /approx_global_optima_datasets
    # fetch drl_dim_scaling (20n) .npy in /ml_sol_datasets
    # fetch 2opt LO .npz in /dscaling_local_optima_datasets/1000k
    # fetch 2swap LO .npz in /dscaling_local_optima_datasets/1000k