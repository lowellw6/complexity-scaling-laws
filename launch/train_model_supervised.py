import argparse
import os
import os.path as osp
import json
import torch
from math import ceil

import tsp
from tsp.datagen import TspDataset, OracleDataset
from tsp.model.base import TspModel, TspAcModel
from tsp.agent import TspAgent
from tsp.algo import TspAcMinibatchSupervised
from tsp.train import TspTrainer
from tsp.logger import MLflowLogger as Logger, parse_log_sig
from tsp.utils import get_num_named_params, seed_rand_gen

from train_model_agent import load_slar_config, load_sd_component_if_given, get_seed, create_eval_datasets, setup_logging, get_lr, setup_lr_scheduler, fetch_fork_state_dicts


default_dataset_path = osp.join(osp.dirname(osp.dirname(tsp.__file__)), "datasets")
DIMENSIONS = 2  # hardcoded for now since no support for 3+ dimensions in supervised context

parser = argparse.ArgumentParser()

parser.add_argument(
    "mlflow_logging_signature", nargs="?", default=None, type=str, help="format: <MLflow_experiment_group>/<MLflow_run_name>"
)  # (if not provided, no logging occurs)
parser.add_argument(
    "--dataset_path", default=default_dataset_path, type=str, help="path to TSP solutions (ordered node tensors); if not provided, uses scale-fpc/datasets; MUST be constant size problems (no padding)"
)
parser.add_argument(
    "--dataset_idxs", nargs="+", default=None, type=int, help="suffix indices of datasets to train on for this run (one epoch becomes one epoch over each dataset with index provided)"
)
parser.add_argument(
    "--eval_nodes", nargs="+", default=None, type=int, help="specify eval dataset node size, otherwise default to shape of training dataset"
)  # can be space-separated list of problem sizes
parser.add_argument("--batch_size", default=512, type=int)
parser.add_argument("--eval_samples", default=10000, type=int)
parser.add_argument("--eval_batch_size", default=1000, type=int)
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--check_period", default=100, type=int)
parser.add_argument("--eval_period", default=1000, type=int)
parser.add_argument("--minibatch_epochs", default=4, type=int)
parser.add_argument("--minibatches", default=4, type=int)
parser.add_argument("--perm_shuffle", action="store_true", default=False, help="shuffle sequence dim of data/labels before training")
parser.add_argument("--grad_norm_clip", default=1.0, type=float)
parser.add_argument("--critic_coeff", default=1.0, type=float)
parser.add_argument("--model_dim", default=128, type=int)
parser.add_argument("--n_enc", default=6, type=int)
parser.add_argument("--n_dec", default=6, type=int)
parser.add_argument("--n_crt", default=6, type=int)
parser.add_argument("--lr", default=None, type=float, help="learning rate; if using scheduler, this is the max learning rate after warmup; has no effect if loading an optimizer when resuming or forking")
parser.add_argument("--exp_lr_schedule", default=None, nargs='*', type=float, help="warm up exponential learning rate scheduler, can be 3 or 4 args, see docs in setup_lr_scheduler(); --lr is used as max learning rate; mutually exclusive with --cos_lr_decay; only valid when not loading existing scheduler during resume or fork")
parser.add_argument("--cos_lr_schedule", default=None, nargs='*', type=float, help="warm up cosine learning rate scheduler, can be 3 or 4 args, see docs in setup_lr_scheduler(); --lr is used as max learning rate; mutually exclusive with --exp_lr_decay; only valid when not loading existing scheduler during resume or fork")
parser.add_argument("--no_opt_load", action="store_true", help="skips loading optimizer from checkpoint if True, even when optimizer state dict is present; only has an effect during resuming or forking")
parser.add_argument("--no_sch_load", action="store_true", help="skips loading scheduler from checkpoint if True, even when scheduler state dict is present; only has an effect during resuming or forking")
parser.add_argument("--device", default=None, type=int)
parser.add_argument("--fork_from_run", default=None, type=str, help="MLflow run id to load model and optimizer states from; if --fork_from_checkpoint isn't provided, this loads the last checkpoint")
parser.add_argument("--fork_from_checkpoint", default=None, type=int, help="checkpoint iteration to load; must be used with --fork_from_run")
parser.add_argument("--slurm_array_config", default=None, type=str, help="path to task configuration dict defining slurm job array; NOTE overrides args specified")
parser.add_argument("--seed", default=None, type=int, help="random gen seed; if omitted, a random seed is built using os.urandom() and the slurm task id, if applicable")
parser.add_argument("--slurm_constant_seeding", action="store_true", help="shares --seed value among all runs in a slurm array; default is to add rank to seed; only valid when --seed and --slurm_array_config are specified")
parser.add_argument("--debug", action="store_true", help="debug run; right now this just reroutes the persistent run_id map to avoid clutter")



def load_slar_config_supervised(args):
    slarc, slar_id, slar_nodes, slar_dimensions, slar_model_dim, slar_lr_sch = load_slar_config(args)

    task_slarc = slarc[slar_id]

    # parse out IL-specific slurm array overrides here
    slar_dataset_prefix = task_slarc["dataset_prefix"] if "dataset_prefix" in task_slarc else None
    if slar_dataset_prefix is not None:
        args.dataset_prefix = slar_dataset_prefix

    slar_dataset_idxs = task_slarc["dataset_idxs"] if "dataset_idxs" in task_slarc else None
    if slar_dataset_idxs is not None:
        args.dataset_idxs = slar_dataset_idxs

    return slarc, slar_id, slar_nodes, slar_model_dim, slar_lr_sch, slar_dataset_prefix, slar_dataset_idxs



if __name__ == "__main__":

    args = parser.parse_args()

    seed = get_seed(args)
    seed_rand_gen(seed)
    print(f"Seed <<< {seed} >>>")

    if args.debug:
        Logger.debug_mode_on()

    # load Slurm array config and task id, if applicable
    if args.slurm_array_config is not None:
        slarc, slar_id, slar_nodes, slar_model_dim, slar_lr_sch, slar_dataset_prefix, slar_dataset_idxs = load_slar_config_supervised(args)
    else:
        slarc = None

    # initialize MLflowLogger
    if args.mlflow_logging_signature is not None:
        resuming, last_logged_itr = setup_logging(args, slarc, slar_id, seed, ignore_param_keys=["mlflow_logging_signature", "seed", "lr", "exp_lr_schedule", "cos_lr_schedule", "dataset_idxs"])
    else:
        Logger.dummy_init()
        resuming = False

    # assert valid experiment loading args, if applicable (resuming and forking are exclusive options)
    if resuming:
        assert args.fork_from_run is None and args.fork_from_checkpoint is None, "Resuming and forking a run are exclusive options and can't be done jointly"
    elif args.fork_from_run is None:
        assert args.fork_from_checkpoint is None, "--fork_from_checkpoint must be used with --fork_from_run"

    # create TSP datasets
    dataset_idxs = slar_dataset_idxs if slarc is not None and slar_dataset_idxs is not None else args.dataset_idxs
    Logger.log_hyperparam(f"dataset_idxs_at_itr_{last_logged_itr if resuming else 0}", dataset_idxs)
    
    dataset = OracleDataset(
        osp.join(args.dataset_path, slar_dataset_prefix),
        seq_shuffle=args.perm_shuffle,
        merge_idxs=dataset_idxs,
        ext="npy"
    )

    if resuming:
        eval_dataset = Logger.load_pickle_artifact("eval_problems", artifact_dir="datasets")
    elif args.fork_from_run is not None:
        eval_dataset = Logger.load_pickle_artifact("eval_problems", artifact_dir="datasets", run_id=args.fork_from_run)
        Logger.save_pickle_artifact(eval_dataset, "eval_problems", artifact_dir="datasets")  # need to copy over to forked run if resuming going forward
    else:
        eval_nodes = [slar_nodes] if slarc is not None and slar_nodes is not None else ([args.eval_nodes] if args.eval_nodes is not None else [dataset.num_nodes])
        eval_dataset = create_eval_datasets(eval_nodes, DIMENSIONS, args)
        Logger.save_pickle_artifact(eval_dataset, "eval_problems", artifact_dir="datasets")

    # initialize model and agent
    model = TspAcModel(
        dim_model=slar_model_dim if slarc is not None and slar_model_dim is not None else args.model_dim,
        num_enc_layers=args.n_enc,
        num_dec_layers=args.n_dec,
        num_crt_layers=args.n_crt,
        tsp_dimensions=DIMENSIONS
    )
    agent = TspAgent(model)

    if args.device is not None:
        agent.to(args.device)

    # log number of non-embedding parameters (for scaling experiment book keeping)
    non_embed_params = agent.named_parameters()  # TSP agents have no vocab or positional embeddings gathered here
    num_non_embed = get_num_named_params(non_embed_params)
    Logger.log_hyperparam(f"non_embedding_parameters", num_non_embed)

    # initialize optimizer
    max_lr = get_lr(args, resuming, slar_lr_sch if slarc is not None else None)
    optimizer = torch.optim.Adam(agent.parameters(), lr=max_lr)

    # initialize scheduler, if applicable
    scheduler = setup_lr_scheduler(args, optimizer, resuming, max_lr, slar_lr_sch if slarc is not None else None)

    # load last checkpoint, if resuming an existing run
    if resuming:
        model_state_dict, optimizer_state_dict, scheduler_state_dict = Logger.load_checkpoint(last_logged_itr, no_mlflow=True, run_id=Logger.active_run_id(), device=torch.device(args.device if args.device is not None else "cpu"))
        
        agent.load_state_dict(model_state_dict)
        optimizer = load_sd_component_if_given("optimizer", optimizer, optimizer_state_dict, args.no_opt_load)
        scheduler = load_sd_component_if_given("scheduler", scheduler, scheduler_state_dict, args.no_sch_load)
    
    # load specified checkpoint, if forking an existing run
    elif args.fork_from_run is not None:
        model_state_dict, optimizer_state_dict, scheduler_state_dict = fetch_fork_state_dicts(args)
        
        agent.load_state_dict(model_state_dict)
        optimizer = load_sd_component_if_given("optimizer", optimizer, optimizer_state_dict, args.no_opt_load)
        scheduler = load_sd_component_if_given("scheduler", scheduler, scheduler_state_dict, args.no_sch_load)

    # initialize algorithm
    algo = TspAcMinibatchSupervised(
        optimizer,
        scheduler=scheduler, 
        epochs=args.minibatch_epochs, 
        minibatches=args.minibatches, 
        grad_norm_clip=args.grad_norm_clip, 
        critic_coeff=args.critic_coeff
    )

    # build runner and start training
    runner = TspTrainer(dataset, agent, algo, eval_datasets=eval_dataset)

    runner.start(
        epochs=args.epochs,
        batch_size=args.batch_size,
        check_period=args.check_period,
        eval_period=args.eval_period,
        eval_batch_size=args.eval_batch_size,
        init_logger_step= 1 if not resuming else last_logged_itr + 1,
        logger_step_multiplier= args.minibatch_epochs * args.minibatches  # aligns logging step with number of gradient updates
    )
