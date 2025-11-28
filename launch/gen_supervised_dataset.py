"""
Generates and writes to disk a datset of optimal tours
using TspOracleAgent. Coordinates are saved in optimal
selection order (with arbitrary first choice) so that
no other data needs to be saved.

Note this may be too slow unless an external solver is
used (e.g. PyConcorde).
"""

import time
import os
import os.path as osp
import torch
import numpy as np
from multiprocessing import Process

import tsp
from tsp.agent import TspOracleAgent
from tsp.utils import get_coords, seed_rand_gen, random_hex_str

from config.gen_supervised_dataset import args as config_args



def gen_func(args, proc_id):
    agent = TspOracleAgent()

    if args.seed is not None:
        base_seed = args.seed
    else:
        base_seed = int.from_bytes(os.urandom(4), "big")

    seed_rand_gen(base_seed + proc_id)
    print(f"({proc_id}) Using random seed {base_seed + proc_id}")

    problem_sizes = [args.problem_size] if type(args.problem_size) is int else args.problem_size

    for psize in problem_sizes:
        print(
            f"({proc_id}) Generating and saving solutions for {args.num_tours} tours of size {psize}..."
        )
        before = time.time()

        problems = get_coords(args.num_tours, psize)
        batches = torch.split(problems, args.batch_size, dim=0)

        solutions = []
        for batch in batches:
            b_solutions, _ = agent.solve(batch)
            solutions.append(b_solutions)

        solutions = torch.cat(solutions, dim=0)

        print(f"({proc_id}) ...done, {time.time() - before:.2f}s")

        # write array of solutions to disk
        save_dir = osp.join(osp.dirname(osp.dirname(tsp.__file__)), "datasets")
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

        file_name = f"{psize}n_{args.num_tours}t"

        if args.name:
            file_name = args.name + "_" + file_name

        file_name += f"_{random_hex_str(30)}.npy"

        save_path = osp.join(save_dir, file_name)

        with open(save_path, "wb") as f:
            np.save(f, solutions.numpy())

        print(f"({proc_id}) Saved dataset @ {save_path}")


        # slurm_id = os.getenv("SLURM_ARRAY_TASK_ID")
        # if slurm_id is not None:
        #     file_name += f"_{int(slurm_id)}b"


if __name__ == "__main__":
    procs = [Process(target=gen_func, args=(config_args, proc_id)) for proc_id in range(config_args.parallel_jobs)]

    [p.start() for p in procs]
    [p.join() for p in procs]
