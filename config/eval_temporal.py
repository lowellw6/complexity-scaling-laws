"""
Evalute mlflow runs over time using checkpoints saved at regular intervals
Used for compute scaling evaluations

See config/eval_final.py for docs
"""


from argparse import Namespace
import os
import os.path as osp

import tsp
root_path = osp.dirname(osp.dirname(tsp.__file__))
dataset_path = osp.join(root_path, "datasets")


SUPER_CONFIG = dict(
    
    drl_compute_scaling = Namespace(
        mlflow_logging_signature = f"eval_compute_scaling_drl/run_{os.getenv('SLURM_ARRAY_TASK_ID')}",  # format: <MLflow_experiment_group>/<MLflow_run_name> (if not provided, no logging occurs)
        loss_type = "ppo",
        nodes = 20,
        model_batch_size = 6_400,
        dataset_dir = dataset_path,
        dataset_stub = "sol_20n_1280000t_0.npy", 
        dataset_slice = slice(12_800),  # use slice batch of datset for evals (speeds things up if dataset is big)
        checkpoints = list(range(1_000, 250_000 + 1, 1_000)),
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
        ]

    ),

    sft_compute_scaling = Namespace(
        mlflow_logging_signature = f"eval_compute_scaling_sft/run_{os.getenv('SLURM_ARRAY_TASK_ID')}",  # format: <MLflow_experiment_group>/<MLflow_run_name> (if not provided, no logging occurs)
        loss_type = "supervised",
        nodes = 20,
        model_batch_size = 6_400,
        dataset_dir = dataset_path,
        dataset_stub = "sol_20n_1280000t_0.npy",
        dataset_slice = slice(12_800),  # use slice batch of datset for evals (speeds things up if dataset is big)
        checkpoints = list(range(500, 73_000 + 1, 500)) + [73_143],
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
    )
)