#!/bin/bash

#SBATCH -J eval_final
#SBATCH -p v100_normal_q
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH --array=0-11

export MLFLOW_TRACKING_URI="file:///path/to/metric/tracking"
export MLFLOW_ARTIFACT_LOCATION="file:///path/to/artifact/storage"

cd ..
python launch/eval_final.py <EXPERIMENT_KEY_HERE>
