___
___
### Run scripts from this directory
---
---
<br/>

## Optimal dataset generation

The following steps were used to generate the [2D evaluation](https://doi.org/10.7294/29374511) and [SFT training](https://doi.org/10.7294/29374535) datasets. 

1. [Install PyConcorde](https://github.com/lowellw6/complexity-scaling-laws/tree/main?tab=readme-ov-file#pyconcorde-install-optional)
(otherwise this defaults to brute force search and may take until the heat death of the universe)

2. Modify configuration in [config/gen_supervised_dataset.py](../config/gen_supervised_dataset.py) 

3. Run ```sbatch optimal.sh```


## Approximately optimal dataset generation (using local search)

The following steps were used to generate the [higher-dimensional evaluation](https://doi.org/10.7294/29374511) datasets via mass local search (selecting the best-found local optimum).

1. Modify/create a configuration in [config/approx_global_optima.py](../config/approx_global_optima.py)

2. Run ```sbatch approx_optimal.sh <CONFIG_NAME>``` replacing ```<CONFIG_NAME>``` with the name in the configuration file

By default this outputs datasets to ```/approx_global_optima_datasets```