#!/bin/bash

#SBATCH -J eval_temporal
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH -t 1:30:00

#export MLFLOW_TRACKING_URI="file:///path/to/metric/tracking"

cd ..
python launch/eval_temporal.py $1
