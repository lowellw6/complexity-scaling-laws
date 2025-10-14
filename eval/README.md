___
___
### Run scripts from this directory
---
---
<br/>

## Model, node, and dimension scaling

### Downloading checkpoints
To evaluate the paper checkpoints...  

Make the directory ```/eval/checkpoint_downloads``` and download checkpoints from [TODO]().

### Your checkpoints
To evaluate final checkpoints produced via train/ scripts...  

Specify the run ids in [config/eval_final.py](../config/eval_final.py) for the checkpoints that correspond to each scale. Replace each <CHECKPOINT_MLFLOW_RUN_ID> in the ```models``` list for the experiments specified in step 1 below.

### Runing evaluation

1. Modify [launch/eval_final.py](../launch/eval_final.py) hardcodes if needed. Some relevant defaults are ```CUDA_IDX=0``` and ```SAVE_SOL=True```. The latter defaults to saving model tours used in evaluation at ```/ml_sol_datasets```, which can be disabled if you just want the MLflow eval stats.

2. Fill in your ```export MLFLOW_TRACKING_URI``` in [final.sh](final.sh). Or omit this line to log to ```/mlruns``` in this directory.

3. Submit sbatch evaluation:  
*(See the readme in [train](../train) for advice on running without Slurm.)*
    | Experiment | Command |
    | ------------- | ------------- |
    | RL Model Scaling  | ```sbatch --array=0-11 final.sh drl_model_scaling```  |
    | RL Node Scaling  | ```sbatch --array=0-9 final.sh drl_node_scaling```  |
    | RL Dimension Scaling<br/>(10-node)  | ```sbatch --array=0-16 final.sh drl_10n_dim_scaling```  |
    | RL Dimension Scaling<br/>(20-node)  | ```sbatch --array=0-16 final.sh drl_20n_dim_scaling```  |
    | SFT Model Scaling  | ```sbatch --array=0-11 final.sh sft_model_scaling```  |
    | SFT Node Scaling  | ```sbatch --array=0-9 final.sh sft_node_scaling```  |



## Compute scaling
*Checkpoints from the original paper are not provided because this would require cloud storage for several thousand checkpoints.*

### Your checkpoints  
To evaluate periodic checkpoints produced via train/ scripts...  

Specify the run ids in [config/eval_temporal.py](../config/eval_temporal.py) for the checkpoints that correspond to each scale. Replace each <CHECKPOINT_MLFLOW_RUN_ID> in the ```models``` list for the experiments specified in step 1 below.

### Runing evaluation

1. Modify the ```CUDA_IDX=0``` hardcode in [launch/eval_temporal.py](../launch/eval_temporal.py) if needed.

2. Fill in your ```export MLFLOW_TRACKING_URI``` in [temporal.sh](temporal.sh). Or omit this line to log to ```/mlruns``` in this directory.

3. Submit sbatch evaluation:  
*(See the readme in [train](../train) for advice on running without Slurm.)*
    | Experiment | Command |
    | ------------- | ------------- |
    | RL Compute Scaling  | ```sbatch --array=0-11 temporal.sh drl_compute_scaling```  |
    | SFT Compute Scaling  | ```sbatch --array=0-11 temporal.sh sft_compute_scaling```  |

