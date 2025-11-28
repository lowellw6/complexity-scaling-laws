
from argparse import Namespace


args = Namespace(
    problem_size = list(range(5, 51, 5)),  # can be single integer or list of problem sizes, in which case generation is repeated for each size
    num_tours = 1600,
    batch_size = 128,
    parallel_jobs = 200,  # number of parallel jobs
    output_len = 1_280_000,  # size of each dataset after merging via merge_supervised_dataset.py
    name = "sol",  # leave empty for no prefix description in dataset name
    seed = None,  # seed random generation, or use random seed if set to None
)
