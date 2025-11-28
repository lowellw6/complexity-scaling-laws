#!/bin/bash

#SBATCH -J approx_global_optima
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=28
#SBATCH --gres=gpu:0
#SBATCH -t 144:00:00
#SBATCH --array=0-16

cd ../localsearch
python approx_global_optima.py $1
