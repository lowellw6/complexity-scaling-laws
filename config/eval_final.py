"""
Evalute final checkpoint for model, node, and dimension scaling experiments
Compute scaling evals are handled separately

1) Replace <EXPERIMENT_KEY_HERE> in eval/final.sh with the SUPER_CONFIG key below (e.g. 'drl_model_scaling')
2) Replace <CHECKPOINT_MLFLOW_RUN_ID> values for the training run corresponding to each scale

contains:
models
scale parameters for each model
input dataset for evals
output solution dataset location

model tuple format: (modified_scale, run_id)
modified_scale can be (width, nodes, dims) and is assumed based on that missing in the config namespace
"""

from argparse import Namespace
import os
import os.path as osp

import tsp
root_path = osp.dirname(osp.dirname(tsp.__file__))
dataset_path = osp.join(root_path, "datasets")
tour_path = osp.join(root_path, "ml_sol_datasets")


SUPER_CONFIG = dict(

    drl_model_scaling = Namespace(
        mlflow_logging_signature = f"eval_model_scaling_drl/run_{os.getenv('SLURM_ARRAY_TASK_ID')}",  # format: <MLflow_experiment_group>/<MLflow_run_name> (if not provided, no logging occurs)
        nodes = 20,
        dims = 2,
        check_itr = 250_000,
        model_batch_size = 6_400,
        sol_per_problem = 2,
        dataset_dir = dataset_path,
        dataset_stub = "sol_20n_1280000t_0.npy",  # no special character '$' since all models are evaluated on same dataset
        output_dir = tour_path,
        output_prefix = "drl_model_scaling",
        models = [
            (240, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (192, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (160, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (128, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (104, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (88, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (72, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (56, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (48, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (40, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (32, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (24, "<CHECKPOINT_MLFLOW_RUN_ID>"),
        ],

    ),

    drl_node_scaling = Namespace(
        mlflow_logging_signature = f"eval_node_scaling_drl/run_{os.getenv('SLURM_ARRAY_TASK_ID')}",  # format: <MLflow_experiment_group>/<MLflow_run_name> (if not provided, no logging occurs)
        width = 184,
        dims = 2,
        check_itr = 250_000,
        model_batch_size = 6_400,
        sol_per_problem = 2,
        dataset_dir = dataset_path,
        dataset_stub = "sol_$n_1280000t_0.npy",
        output_dir = tour_path,
        output_prefix = "drl_node_scaling",
        models = [
            (50, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (45, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (40, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (35, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (30, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (25, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (20, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (15, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (10, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (5, "<CHECKPOINT_MLFLOW_RUN_ID>"),
        ],

    ),

    drl_10n_dim_scaling = Namespace(
        mlflow_logging_signature = f"eval_10n_dim_scaling_drl/run_{os.getenv('SLURM_ARRAY_TASK_ID')}",  # format: <MLflow_experiment_group>/<MLflow_run_name> (if not provided, no logging occurs)
        width = 184,
        nodes = 10,
        check_itr = 250_000,
        model_batch_size = 6_400,
        sol_per_problem = 2,
        dataset_dir = dataset_path,
        dataset_stub = "$d_2opt_100k_10n_128000t.npy",
        output_dir = tour_path,
        output_prefix = "drl_10n_dim_scaling",
        models = [
            (2, "<CHECKPOINT_MLFLOW_RUN_ID>"),  # from node scaling run array, to compare accuracy with proxy optimal dataset
            (3, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (4, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (5, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (6, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (7, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (8, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (9, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (10, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (11, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (12, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (15, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (20, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (30, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (40, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (50, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (100, "<CHECKPOINT_MLFLOW_RUN_ID>"),
        ],

    ),

    drl_20n_dim_scaling = Namespace(
        mlflow_logging_signature = f"eval_20n_dim_scaling_drl/run_{os.getenv('SLURM_ARRAY_TASK_ID')}",  # format: <MLflow_experiment_group>/<MLflow_run_name> (if not provided, no logging occurs)
        width = 184,
        nodes = 20,
        check_itr = 250_000,
        model_batch_size = 6_400,
        sol_per_problem = 2,
        dataset_dir = dataset_path,
        dataset_stub = "$d_2opt_1000k_20n_64000t.npy",
        output_dir = tour_path,
        output_prefix = "drl_20n_dim_scaling",
        models = [
            (2, "<CHECKPOINT_MLFLOW_RUN_ID>"),  # from node scaling run array, to compare accuracy with proxy optimal dataset
            (3, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (4, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (5, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (6, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (7, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (8, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (9, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (10, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (11, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (12, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (15, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (20, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (30, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (40, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (50, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (100, "<CHECKPOINT_MLFLOW_RUN_ID>"),
        ],

    ),

    sft_model_scaling = Namespace(
        mlflow_logging_signature = f"eval_model_scaling_sft/run_{os.getenv('SLURM_ARRAY_TASK_ID')}",  # format: <MLflow_experiment_group>/<MLflow_run_name> (if not provided, no logging occurs)
        nodes = 20,
        dims = 2,
        check_itr = 73_143,
        model_batch_size = 6_400,
        sol_per_problem = 2,
        dataset_dir = dataset_path,
        dataset_stub = "sol_20n_1280000t_0.npy",  # no special character '$' since all models are evaluated on same dataset
        output_dir = tour_path,
        output_prefix = "sft_model_scaling",
        models = [
            (240, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (192, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (160, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (128, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (104, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (88, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (72, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (56, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (48, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (40, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (32, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (24, "<CHECKPOINT_MLFLOW_RUN_ID>"),
        ],

    ),

    sft_node_scaling = Namespace(
        mlflow_logging_signature = f"eval_node_scaling_sft/run_{os.getenv('SLURM_ARRAY_TASK_ID')}",  # format: <MLflow_experiment_group>/<MLflow_run_name> (if not provided, no logging occurs)
        width = 184,
        dims = 2,
        check_itr = 73_143,
        model_batch_size = 6_400,
        sol_per_problem = 2,
        dataset_dir = dataset_path,
        dataset_stub = "sol_$n_1280000t_0.npy",
        output_dir = tour_path,
        output_prefix = "sft_node_scaling",
        models = [
            (50, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (45, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (40, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (35, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (30, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (25, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (20, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (15, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (10, "<CHECKPOINT_MLFLOW_RUN_ID>"),
            (5, "<CHECKPOINT_MLFLOW_RUN_ID>"),
        ],

    ),

)