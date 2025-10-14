#!/bin/bash

#SBATCH -J dimension_scaling
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH -t 144:00:00
#SBATCH --array=0-10

export MLFLOW_TRACKING_URI="file:///path/to/metric/tracking"
export MLFLOW_ARTIFACT_LOCATION="file:///path/to/artifact/storage"


cd ..
python launch/train_model_agent.py resume/<SLURM_ARRAY_TAG_HERE> --slurm_array_config config/dimension_scaling_slarc.json \
    --nodes 20 \
    --itr 62.5 \
    --algo ppo \
    --minibatch_epochs 1 \
    --minibatches 4 \
    --ratio_clip 0.17 \
    --batch_size 700 \
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

