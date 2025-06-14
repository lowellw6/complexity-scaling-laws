___
___
### Run scripts from this directory
---
---

## Launching job arrays
```launch_*.sh``` bash scripts initiate model training.

[Slurm](https://slurm.schedmd.com/documentation.html) was used to launch jobs in parallel via Slurm Array Jobs. If also using Slurm, you'll need to update the #SBATCH specifications at the top of each bash script. We provide ours as example compute usage. ```--array``` should *not* be modified unless you want to train a subset of model/problem scales.

If not using Slurm, you can copy everything after the #SBATCH commands (starting with exports) to your own workflow and spoof Slurm array ID environment variables like so:
```
export SLURM_ARRAY_TASK_ID=<IDX>
``` 
replacing \<IDX> with integers between [0, len(job_array)-1]. Each ID indexes the configuration provided via ```--slurm_array_config```.

## Resuming job arrays
```resume_*.sh``` scripts support resuming job arrays if, for example, your compute resource limits run time per job. 

For each job, resuming loads the last model checkpoint, optimizer state, and learning rate scheduler state for a seamless continuation. MLflow runs pick up where they left off using the same evaluation data, which persists as an MLflow artifact.

A unique ID is generated for each job array for convenience, avoiding the need to manually copy each MLflow run_id. The shared array ID is logged as an MLflow tag called ```slar_tag``` (Slurm array tag), and also printed to stdout in the format:
```
Exported run_id to <<< 64c8ed30c34115450b646d584_0.id >>>
```
where in the above example ```64c8ed30c34115450b646d584``` is the shared array ID and ```0``` is the specific SLURM_ARRAY_TASK_ID. This example would store the corresponding Mlflow run_id at ```/run_id_map/64c8ed30c34115450b646d584_0.id```.

To use a ```resume_*.sh``` script, in this example, you would replace ```resume/<SLURM_ARRAY_TAG_HERE>``` with ```resume/64c8ed30c34115450b646d584```.

## PyTorch device
We use the 0th CUDA index by default for each job. For a different index, modify:
```
--device 0
```
at the end of any launch or resume script. (Or remove it for CPU training.)

## Model and compute scaling
**RL:** run ```launch_model_scaling.sh``` to launch then ```resume_model_scaling.sh``` to resume. MLflow exports and the Slurm array tag placeholder (when resuming) must be modified. Each cycle specifies 250K PPO minibatch updates, so we required 3 resumes to reach 1M updates. To run all training from launch, specify ```--itr 250``` in ```launch_model_scaling.sh``` for 250K iterations of 4 PPO minibatch updates. (This may require weeks of training depending on your hardware acceleration.)

**SFT:** Make a directory called ```/datasets``` in the repo root and download all 10 of the 20-node Concorde datasets with format ```sol_20n_1280000t_[0-9].npy``` (2GB) to this directory. Run ```launch_model_scaling.sh``` to launch after modifying MLflow exports. No resuming was used since one epoch of training requires less than 10% the gradient updates compared to RL.

## Node scaling
**RL:** run ```launch_node_scaling.sh``` to launch then ```resume_node_scaling.sh``` to resume. MLflow exports and the Slurm array tag placeholder (when resuming) must be modified. Either resume 3 times or modify the launch script as described for the RL model and compute scaling.

**SFT:** Make a directory called ```/datasets``` in the repo root and download all 100 of the Concorde datasets with format ```sol_[5-50]n_1280000t_[0-9].npy``` (**27GB**) to this directory. Skip the 20-node datasets if you already downloaded them for model and compute scaling above. Run ```launch_node_scaling.sh``` to launch after modifying MLflow exports. No resuming was used since one epoch of training requires less than 10% the gradient updates compared to RL.

## Spatial dimension scaling
**RL:** run ```launch_dimension_scaling.sh``` to launch then ```resume_dimension_scaling.sh``` to resume. MLflow exports and the Slurm array tag placeholder (when resuming) must be modified. Either resume 3 times or modify the launch script as described for the RL model and compute scaling. 

By default this reproduces the 20-node experiment up through 12 dimensions. To reproduce the 10-node experiment (or try another node scale) modify ```--nodes 20``` when launching and resuming. To reproduce scales beyond 12 dimensions that failed to converge, append these scales in ```config/dimension_scaling_slarc.json``` and extend the ```--array``` range when launching/resuming (```--array=0-16``` if including dimensions [15, 20, 30, 40, 50, 100]).
