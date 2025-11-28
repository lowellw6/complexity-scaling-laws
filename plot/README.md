

## Matplotlib visualization

Matplotlib scripts used for the main paper plots.  
**Disclaimer:** this code is (substantially) less organized than [train](../train/) and [eval](../eval) code.



Several plots rely on aggregated model and local search evals via [build_aggregate_dataset.py](./build_aggregate_dataset.py) &rarr; [eval_aggregate_dataset.py](./eval_aggregate_dataset.py).

| Plot | Function |
| ------------- | ------------- |
| Figure 1  | [```1_subopt_joint.py:make_plots```](./1_subopt_joint.py#L273)  |
| Figure 2  | [```2_cover.py:make_cover_plots```](./2_cover.py#L195)  |
| Figure 3  | [```2_problem_fitness.py:make_joint_main_plots_nips```](./2_problem_fitness.py#L497)  |
| Figure 4  | [```2_problem_fitness.py:make_drl_main_plots_nips```](./2_problem_fitness.py#L284)  |
| Figure 5  | [```3_fpc.py:make_search_complexity_plots_nips```](./3_fpc.py#L205)  |
