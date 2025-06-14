#!/bin/bash

#SBATCH -J node_scaling
#SBATCH -p v100_normal_q
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH -t 72:00:00
#SBATCH --array=0-9

export MLFLOW_TRACKING_URI="file:///path/to/metric/tracking"
export MLFLOW_ARTIFACT_LOCATION="file:///path/to/artifact/storage"


cd ..
python launch/train_model_supervised.py node_scaling_sft/run_${SLURM_ARRAY_TASK_ID} --slurm_array_config config/supervised_node_scaling_slarc.json \
    --epochs 1 \
    --perm_shuffle \
    --lr 9.37e-4 --cos_lr_schedule 0.3 73.143 0.0 \
    --minibatch_epochs 1 \
    --minibatches 1 \
    --batch_size 175 \
    --critic_coeff 0.52 \
    --grad_norm_clip 0.24 \
    --model_dim 184 \
    --n_enc 3 \
    --n_dec 2 \
    --n_crt 2 \
    --check_period 1000 \
    --eval_period 1000 \
    --eval_samples 100000 \
    --eval_batch_size 128 \
    --device 0