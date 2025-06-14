
import argparse
import pickle

from tsp.datagen import TspDataset


parser = argparse.ArgumentParser()

parser.add_argument("samples", type=int)
parser.add_argument(
    "--nodes", nargs="+", default=None, type=int
)  # can be space-separated list of problem sizes
parser.add_argument(
    "--node_range", nargs=2, default=None, type=int
)  # use this or --nodes but not both



def interpret_problem_sizes(args):
    if args.nodes is not None and args.node_range is not None:
        raise ValueError("May specify custom '--nodes' or '--node_range' but not both")
    elif args.node_range is not None:
        start, end = args.node_range
        return range(start, end + 1)  # inclusive end bound
    elif args.nodes is not None:
        return args.nodes
    else:
        raise ValueError("Must specify either '--nodes' or '--node_range'")



if __name__ == "__main__":
    args = parser.parse_args()

    nodes = interpret_problem_sizes(args)
    dataset = TspDataset(size=nodes, num_samples=args.samples)

    with open(f"{nodes[0] if len(nodes) == 1 else nodes}n_{args.samples}t.pkl", "wb") as f:
        pickle.dump(dataset, f)
