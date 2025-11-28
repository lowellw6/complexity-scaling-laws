## Local search

Generate locally optimal TSP tours and run interesting fitness/distance landscape analysis.

### Glossary
| Term | Description |
| ------------- | ------------- |
| cost  | TSP tour length |
| fitness | negative cost |
| distance | how dissimilar are two tour permutations, e.g. [edge](./distance.py#L34) or [node](./distance.py#L9) |
| landscape | fitness/distance surface characterizing a single TSP problem |
| neighborhood | local search adjacency definition, distinct from distance, e.g. [2-opt](./csearch.pyx#L44) or [2-exchange](./csearch.pyx#122) |
| swap | one local search move, hill climbing the landscape to an adjacent tour in the neighborhood with higher fitness |
| residual | (informal) distance between a local optimum and the global optimum |
| spread | (informal) distance between a two distinct local optima |
| roll | (informal) distance between a randomly intialized tour and its local optimum (basin of attraction) |


### Local optima generation
[gen_local_optima.py](./gen_local_optima.py) runs Cythonized local search.  
- [Configureable](../config/gen_local_optima.py), supports [2-opt](./csearch.pyx#L44) and [2-exchange (aka 2-swap)](./csearch.pyx#122) neighborhood definitions  
- Input: TSP problem dataset  
- Output: locally optimal tour indices dataset (relative to the input)

### Local optima evaluation
[eval_local_optima.py](./eval_local_optima.py) batch evaluates gen_local_optima.py outputs for stats about local optima and fitness/distance analysis.  
- [Configureable](../config/eval_local_optima.py), supports [edge Hamming distance](./distance.py#L34) and [TSP-equivalence-invariant (TEI) Hamming distance](./distance.py#L9)
- Input: TSP problem dataset in optimal order (likely used to generate local optima dataset) and the local optima dataset  
- Output: MLflow run logging stats on cost, # of swaps, residual, spread, and roll

### Constrained local optima
[eval_constrained_local_optima.py](./eval_constrained_local_optima.py) extends local optima evaluation for when [```search_caps```](../config/gen_local_optima.py#L17) is not None. Setting search_caps enforces a limit to the number of search swaps (as done in Figure 5 bottom).

### Approximately optimal dataset generation
Global optimality estimated with mass local search via [approx_global_optima.py](./approx_global_optima.py). See [/datagen](../datagen) for instructions.