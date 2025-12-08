<h1 style="margin-bottom:0;">Complexity Scaling Laws for Neural Models <br> using Combinatorial Optimization</h1>
<p style="margin-top:2px;">
     Lowell Weissman, Michael Krumdick, A. Lynn Abbott
</p>

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-00b3b3.svg)](https://neurips.cc/virtual/2025/poster/119119
) [![arXiv](https://img.shields.io/badge/arXiv-2506.12932-b31b1b.svg)](https://arxiv.org/abs/2506.12932) 

[![eval](https://img.shields.io/badge/Data-Evaluation-blue)](https://doi.org/10.7294/29374511)
[![train](https://img.shields.io/badge/Data-Training-blue)](https://doi.org/10.7294/29374535)
[![models](https://img.shields.io/badge/Data-Models-blue)](https://doi.org/10.7294/30359275)

Training and evaluation datasets are TSP problem numpy arrays with shape: ```(batch, nodes, spatial_dimensions)```  
They are provided in optimal tour order along ```nodes```

2D TSP datasets along with "sft" prefix models are restricted to academic research use following [Concorde's liscense](https://www.math.uwaterloo.ca/tsp/concorde.html). Higher-dimensional approximately optimal datasets (generated with local search) and "drl" prefix models do not share this restriction. 

## Install
### Poetry
Install poetry. You can do this with or without a virtual environment.
```
pip install poetry
```
Install this package and dependencies. Poetry will create a virtual environment if one is not already activated.
```
poetry install
```
Activate poetry's virtual environment (skip if you made your own).
```
eval $(poetry env activate)
```
_For older poetry versions try_ ```poetry shell```

Test core modules.
```
python test/run.py
```

## Core

### Scaling law fits
[```/fits```](./fits) &rarr; full-precision regression fits for scaling laws and other observed trends  
(we summarize a subset of these in Appendix A of the paper)

### Training models
[```/train```](./train) &rarr; bash scripts and a README with examples of launching and resuming training runs

### Evaluating models
[```/eval```](./eval) &rarr; bash scripts and a README with examples of evaluating trained models

### Matplotlib plotting
[```/plot```](./plot) &rarr; matplotlib visualization scripts for the main-paper experiments

### Dataset generation
[```/datagen```](./datagen) &rarr; generate datasets of optimal and approximately optimal TSP tours

### Local search experiments
[```/localsearch```](./localsearch/) &rarr; generate locally optimal TSP tours and run various fitness/distance analyses


## Organization
- [tsp](./tsp) - core model, algorithm, and data modules
    - [train.py](./tsp/train.py) - top-level ```TspTrainer```
    - [agent.py](./tsp/agent.py) - ```TspAgent``` class shared for RL and SFT
    - [model](./tsp/model) - PyTorch modules
        - [base.py](./tsp/model/base.py) - top-level model and decoder modules
        - [submodules.py](./tsp/model/submodules.py) - encoder, critic, and positional encoding submodules
    - [algo.py](./tsp/algo.py) - RL algorithms (REINFORCE, A2C, PPO) and negative log-likelihood SFT
    - [eval.py](./tsp/eval.py) - batch eval of cost and loss
    - [select.py](./tsp/select.py) - next-node selection method (sample, greedy, teacher-forced)
    - [datagen.py](./tsp/datagen.py) - TSP data generation and dataset loading
    - [bridge.py](./tsp/bridge.py) - wraps TSPSolver from PyConcorde submodule
    - [utils.py](./tsp/utils.py) - assorted utilities
    - [logger.py](./tsp/.py) - ```MLflowLogger``` class wrapping MLflow's client API
- [launch](./launch) - launch experiments
    - [train_model_agent.py](./launch/train_model_agent.py) - RL training
    - [train_model_supervised.py](./launch/train_model_supervised.py) - SFT training
    - [hpo_model_agent.py](./launch/hpo_model_agent.py) - RL HPO experiment using [BOHB](https://arxiv.org/pdf/1807.01774) implemented with [Optuna](https://optuna.org/)
    - [eval_final.py](./launch/eval_final.py) - model, TSP node, and TSP dimension scaling evaluation (last checkpoint)
    - [eval_temporal.py](./launch/eval_temporal.py) - compute scaling evaluation (periodic checkpoints throughout training)
    - [gen_supervised_dataset.py](./launch/gen_supervised_dataset.py) - generate optimal TSP tours with PyConcorde (or brute force search)
    - [merge_supervised_dataset.py](./launch/merge_supervised_dataset.py) - merge dataset chunks output by parallel workers of [gen_supervised_dataset.py](./launch/gen_supervised_dataset.py)
- [config](./config) - configure experiments
- [solvers](./solvers) - PyConcorde submodule
- [mlf_utils](./mlf_utils) - custom MLflow utilities
    - [slim_mlf_exp.py](./mlf_utils/slim_mlf_exp.py) - subsample full MLflow experiment and make a lightweight copy  
    (training experiments log a lot of data which leads to problems plotting metrics with MLflow)
    - [shared.py](./mlf_utils/shared.py) - replacements for several MLflow utilities which break when metric logs get large
- [hpo](./hpo) - utilities for hyperparameter optimization performed via [launch/hpo_model_agent.py](./launch/hpo_model_agent.py)
- [draw](./draw) - TSP tour visualization code
- [test](./test) - unit tests for core modules in ```/tsp```
- [fits](./fits) - scaling law regression fits
- [train](./train) - training scripts to reproduce main-paper experiments
- [eval](./eval) - evaluation scripts to reproduce main-paper experiments
- [plot](./plot) - matplotlib visualization of MLflow metrics
- [datagen](./datagen) - generate optimal and approximately optimal TSP data
- [localsearch](./localsearch) - local optima generation and eval



## PyConcorde Install [Optional]

Install the [PyConcorde](https://github.com/jvkersch/pyconcorde) submodule to generate optimal TSP tours:
```
git submodule update --init --recursive
```
Ensure there's a C compiler (e.g. gcc) and make on your base environment PATH for building Concorde:
```
sudo apt update
sudo apt install gcc make
```
Overwrite setup.py with our custom version to fix an install bug:  
(urllib, used to download QSOPT, doesn't support HTTP 308 redirect, so we swap urllib with requests)
```
cd solvers
cp custom_pyconcorde_setup.py pyconcorde/setup.py
```
Install pyconcorde in editable mode ("-e" with pip):  
(Otherwise the wrapped Concorde binaries may not correctly install, even if PyConcorde does)
```
cd pyconcorde
pip install -r requirements.txt
pip install -e .
```
If the pyconcorde install fails due to "ModuleNotFoundError: No module named 'requests'" then run setup.py directly:
```
python setup.py install
```

## MLflow Logging
Logging is handled via [MLflow](https://mlflow.org/docs/latest/index.html).

Two environment variables affect where logs are stored:

1. **MLFLOW_TRACKING_URI** : base URI for all MLflow metrics and parameters

2. **MLFLOW_ARTIFACT_LOCATION** : base URI for all MLflow artifacts (including model checkpoints)

Export these (or e.g. add to .bashrc) before running desired experiments. If MLFLOW_TRACKING_URI is not set, /mlruns created in the root directory of this repo is used as a default. If MLFLOW_ARTIFACT_LOCATION is not set, the MLFLOW_TRACKING_URI root is used as a default.

See the [docs for examples of acceptable local and remote URIs](https://mlflow.org/docs/latest/python_api/mlflow.html?highlight=set_experiment#mlflow.set_tracking_uri).


## Liscense
This repository forks [MITRE tsp](https://github.com/mitre/tsp), the open source version of the seedling effort for this work. We carry forward the Apache-2.0 license.

The [PyConcorde](https://github.com/jvkersch/pyconcorde) submodule is provided under a BSD-3-Clause license, and relies on Concorde and QSOpt which are provided under different liscenses.

Under [Concorde's liscense](https://www.math.uwaterloo.ca/tsp/concorde.html), our optimal tour datasets for 2D TSP are restricted to academic research use.


## Citation
```
@inproceedings{weissman2025complexity,
  title     = {Complexity Scaling Laws for Neural Models using Combinatorial Optimization},
  author    = {Weissman, Lowell and Krumdick, Michael and Abbott, A Lynn},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025},
  url       = {https://neurips.cc/virtual/2025/poster/119119}
}
```
