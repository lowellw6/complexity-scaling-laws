___
___
### Run scripts from this directory
---
---
<br/>

## Model, node, and dimension scaling
### Your checkpoints
To evaluate final checkpoints produced via train/ scripts...  
1. Specify experiment in [final.sh](final.sh) using the following keys:  
    | Experiment | Key |
    | ------------- | ------------- |
    | RL Model Scaling  | drl_model_scaling  |
    | RL Node Scaling  | drl_node_scaling  |
    | RL Dimension Scaling<br/>(10-node)  | drl_10n_dim_scaling  |
    | RL Dimension Scaling<br/>(20-node)  | drl_20n_dim_scaling  |
    | SFT Model Scaling  | sft_model_scaling |
    | SFT Node Scaling  | sft_node_scaling  |
2. Specify the run ids in [config/eval_final.py](../config/eval_final.py) for the checkpoints that correspond to each scale. Replace each <CHECKPOINT_MLFLOW_RUN_ID> in the ```models``` list for the experiments specified in step 1. When I get time I'll make this automatic by extending the Slurm array tag resume feature used during training.
3. Modify [launch/eval_final.py](../launch/eval_final.py) hardcodes if needed. Some relevant defaults are ```CUDA_IDX=0``` and ```SAVE_SOL=True```. The latter defaults to saving model tours used in evaluation at ```/ml_sol_datasets```, which can be disabled if you just want the MLflow eval stats.
4. Run ```final.sh``` after modifying ```--array=``` to match the ```models``` list length. See the readme in [train](../train) for advice on running without Slurm.

### Downloaded checkpoints
**WIP** evals using downloads of original paper checkpoints.

## Compute scaling
**WIP**
