#!/bin/bash

#SBATCH -J SFT_model_scaling
#SBATCH -p v100_normal_q
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=21
#SBATCH --gres=gpu:1
#SBATCH -t 144:00:00
#SBATCH --array=0-11

export MLFLOW_TRACKING_URI="file:///path/to/metric/tracking"
export MLFLOW_ARTIFACT_LOCATION="file:///path/to/artifact/storage"


cd ..
python launch/train_model_supervised.py model_scaling_sft/run_${SLURM_ARRAY_TASK_ID} --slurm_array_config config/supervised_model_scaling_slarc.json \
    --epochs 1 \
    --perm_shuffle \
    --lr 9.37e-4 --cos_lr_schedule 0.3 73.143 0.0 \
    --minibatch_epochs 1 \
    --minibatches 1 \
    --batch_size 175 \
    --critic_coeff 0.52 \
    --grad_norm_clip 0.24 \
    --n_enc 3 \
    --n_dec 2 \
    --n_crt 2 \
    --check_period 500 \
    --eval_period 500 \
    --eval_samples 100000 \
    --eval_batch_size 128 \
    --device 0