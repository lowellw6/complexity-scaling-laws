#!/bin/bash

#SBATCH -J supervised_dataset_gen
#SBATCH -N 8 --ntasks-per-node=1 --cpus-per-task=25
#SBATCH --gres=gpu:0
#SBATCH -t 144:00:00


cd ..

TEMP_DIR_NAME="temp_solver_files"
if [ ! -d "$TEMP_DIR_NAME" ]; then
    mkdir -p "$TEMP_DIR_NAME"
fi

cd $(TEMP_DIR_NAME)  # throwaway directory to host unmanageable amount of PyConcorde output files

python ../launch/gen_supervised_dataset.py  # generate optimal tour data with pyconcorde

rm *.res *.sol *.pul *.sav  # cleanup junk from pyconcorde

python ../launch/merge_supervised_dataset.py  # merge into chunky, fixed sized datasets
