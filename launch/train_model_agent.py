import argparse
import os
import os.path as osp
import json
import torch
from math import ceil
import hashlib

import tsp
from tsp.datagen import TspLiveDatagen, TspDataset
from tsp.model.base import TspModel, TspAcModel, TspGreedyBaselineModel
from tsp.agent import TspAgent
from tsp.algo import TspReinforce, TspA2C, TspPPO, TspPPOGreedyRollout
from tsp.train import TspTrainer, TspTrainerRolloutCritic
from tsp.logger import MLflowLogger as Logger, parse_log_sig
from tsp.utils import get_num_named_params, seed_rand_gen

parser = argparse.ArgumentParser()

parser.add_argument(
    "mlflow_logging_signature", nargs="?", default=None, type=str, help="format: <MLflow_experiment_group>/<MLflow_run_name>"
)  # (if not provided, no logging occurs)
parser.add_argument(
    "--nodes", nargs="+", default=None, type=int
)  # can be space-separated list of problem sizes
parser.add_argument(
    "--node_range", nargs=2, default=None, type=int
)  # use this or --nodes but not both
parser.add_argument("--dimensions", default=2, type=int, help="TSP dimensions (trailing feature length of problem points); must be >=2")
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--eval_samples", default=10000, type=int)
parser.add_argument("--eval_batch_size", default=1000, type=int)
parser.add_argument("--itr", default=250, type=float, help="in thousands")
parser.add_argument("--check_period", default=100, type=int)
parser.add_argument("--eval_period", default=1000, type=int)
parser.add_argument("--grad_norm_clip", default=1.0, type=float)
parser.add_argument("--critic_coeff", default=1.0, type=float)
parser.add_argument("--model_dim", default=128, type=int)
parser.add_argument("--n_enc", default=6, type=int)
parser.add_argument("--n_dec", default=6, type=int)
parser.add_argument("--n_crt", default=6, type=int)
parser.add_argument("--lr", default=None, type=float, help="learning rate; if using scheduler, this is the max learning rate after warmup; has no effect if loading an optimizer when resuming or forking")
parser.add_argument("--exp_lr_schedule", default=None, nargs='*', type=float, help="warm up exponential learning rate scheduler, can be 3 or 4 args, see docs in setup_lr_scheduler(); --lr is used as max learning rate; mutually exclusive with --cos_lr_decay; only valid when not loading existing scheduler during resume or fork")
parser.add_argument("--cos_lr_schedule", default=None, nargs='*', type=float, help="warm up cosine learning rate scheduler, can be 3 or 4 args, see docs in setup_lr_scheduler(); --lr is used as max learning rate; mutually exclusive with --exp_lr_decay; only valid when not loading existing scheduler during resume or fork")
parser.add_argument("--algo", default="ppo", choices=["a2c", "ppo", "ppo_rollout"], type=str, help="train with compositional A2C or PPO; or PPO with greedy rollouts for critic values; defaults to PPO")
parser.add_argument("--minibatch_epochs", default=1, type=int, help="PPO only; number of minibatch epochs aka how many times to use data before getting new samples")
parser.add_argument("--minibatches", default=1, type=int, help="PPO only; how many minibatches (gradient updates) to take per minibatch_epoch")
parser.add_argument("--ratio_clip", default=0.1, type=float, help="PPO only; clamp threshold during creation of surrogate objectives in loss (smaller means more stable but more constrained gradient updates)")
parser.add_argument("--only_state_values", action="store_true", help="PPO only (and not rollout compatible); ablates compositional value learning in PPO loss, only using state values to inform both actor and critic updates")
parser.add_argument("--no_opt_load", action="store_true", help="skips loading optimizer from checkpoint if True, even when optimizer state dict is present; only has an effect during resuming or forking")
parser.add_argument("--no_sch_load", action="store_true", help="skips loading scheduler from checkpoint if True, even when scheduler state dict is present; only has an effect during resuming or forking")
parser.add_argument("--device", default=None, type=int)
parser.add_argument("--fork_from_run", default=None, type=str, help="MLflow run id to load model and optimizer states from; if --fork_from_checkpoint isn't provided, this loads the last checkpoint")
parser.add_argument("--fork_from_checkpoint", default=None, type=int, help="checkpoint iteration to load; must be used with --fork_from_run")
parser.add_argument("--slurm_array_config", default=None, type=str, help="path to task configuration dict defining slurm job array; NOTE overrides args specified")
parser.add_argument("--seed", default=None, type=int, help="random gen seed; if omitted, a random seed is built using os.urandom() and the slurm task id, if applicable")
parser.add_argument("--slurm_constant_seeding", action="store_true", help="shares --seed value among all runs in a slurm array; default is to add rank to seed; only valid when --seed and --slurm_array_config are specified")
parser.add_argument("--debug", action="store_true", help="debug run; right now this just reroutes the persistent run_id map to avoid clutter")



def interpret_problem_sizes(args):
    if args.nodes is not None and args.node_range is not None:
        raise ValueError("May specify custom '--nodes' or '--node_range' but not both")
    elif args.node_range is not None:
        start, end = args.node_range
        return range(start, end + 1)  # inclusive end bound
    elif args.nodes is not None:
        return args.nodes
    else:
        return (20,)  # default
    

def load_sd_component_if_given(name, host_obj, state_dict, no_load_flag):
    """
    For PyTorch optimizer or scheduler as host_obj
    """
    if not no_load_flag and state_dict is not None:
        host_obj.load_state_dict(state_dict)
        print(f"Loading checkpoint *with {name}* state dict\n")
    else:
        print(f"WARNING Loading checkpoint *without {name}* state dict")
        print(f"{name.capitalize()} state will be re-initialized\n")

    return host_obj


def get_seed(args):
    assert args.seed is not None or not args.slurm_constant_seeding, "Must specify --seed to use --slurm_constant_seeding!"
    assert args.slurm_array_config is not None or not args.slurm_constant_seeding, "Can only use --slurm_constant_seeding with slurm array job! Specify --slurm_array_config"

    base_seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "big")
    addend = int(os.getenv("SLURM_ARRAY_TASK_ID")) if args.slurm_array_config is not None and not args.slurm_constant_seeding else 0

    return base_seed + addend


def create_eval_datasets(nodes, dimensions, args):
    eval_datasets = [("all", TspDataset(size=nodes, num_samples=args.eval_samples, dimensions=dimensions))]

    if len(nodes) > 1:
        # also eval on data strictly containing lower and upper problem size bounds
        low, high = min(nodes), max(nodes)
        eval_datasets += [
            (f"{low}n", TspDataset(size=low, num_samples=args.eval_samples, dimensions=dimensions))
        ]
        eval_datasets += [
            (f"{high}n", TspDataset(size=high, num_samples=args.eval_samples, dimensions=dimensions))
        ]

    return eval_datasets


def get_slar_tag(exp_name, slar_size, rank):
    """
    Generates a 25-length hash id which will be shared among all parallel Slurm array
    jobs, even though its generated independently by each job, via some fancy logic

    With this common identifier and the Slurm array job's rank (SLURM_ARRAY_TASK_ID),
    the logger can save and look up the run_id, avoiding tedious manual copying

    slar_size needs to be part of the hash to avoid ambiguous cases where two array
    jobs in the same experiment have different sizes, so the first "open" counter
    index will be at different hashes for the ranks present in one but not the other

    exp_name: MLflow experiment name (not unique)
    slar_size: total number of jobs for the Slurm array this job is part of
    rank: Slurm rank of this job

    returns: unique Slurm array job id tag (shared with parallel jobs)
    """
    counter = 0
    slar_tag = hashlib.sha1(f"{exp_name}_{slar_size}_{counter}".encode("utf-8")).hexdigest()[:25]
    
    while not Logger.valid_export_name(slar_tag, rank):
        counter += 1
        slar_tag = hashlib.sha1(f"{exp_name}_{slar_size}_{counter}".encode("utf-8")).hexdigest()[:25]

    return slar_tag


def load_slar_config(args):
    with open(args.slurm_array_config, 'r') as f:
        slarc = json.load(f)
    
    slar_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))

    assert len(slarc) > slar_id, "Bad Slurm Array ID, index out of bounds"
    task_slarc = slarc[slar_id]

    # parse out specific slurm array overrides here
    slar_nodes = task_slarc["nodes"] if "nodes" in task_slarc else None
    slar_dimensions = task_slarc["dimensions"] if "dimensions" in task_slarc else None

    slar_model_dim = task_slarc["model_dim"] if "model_dim" in task_slarc else None
    
    slar_max_lr = task_slarc["lr"] if "lr" in task_slarc else None
    slar_exp_lr_sch = task_slarc["exp_lr_schedule"] if "exp_lr_schedule" in task_slarc else None
    slar_cos_lr_sch = task_slarc["cos_lr_schedule"] if "cos_lr_schedule" in task_slarc else None
    assert slar_max_lr is not None or (slar_exp_lr_sch is None and slar_cos_lr_sch is None), "Can't slarc override lr_schedule without specifying (max) 'lr'!"
    slar_lr_sch = dict(max_lr=slar_max_lr, exp=slar_exp_lr_sch, cos=slar_cos_lr_sch) if slar_max_lr is not None or slar_exp_lr_sch is not None or slar_cos_lr_sch is not None else None

    if slar_nodes is not None:
        args.nodes = slar_nodes  # for mlflow logging

    if slar_dimensions is not None:
        args.dimensions = slar_dimensions  # for mlflow logging

    if slar_model_dim is not None:
        args.model_dim = slar_model_dim  # for mlflow logging

    return slarc, slar_id, slar_nodes, slar_dimensions, slar_model_dim, slar_lr_sch


def setup_logging(args, slarc, slar_id, seed, ignore_param_keys=()):
    resuming, sig_args = parse_log_sig(args.mlflow_logging_signature)

    if resuming:
        if slarc is not None:  # override resume id for job arrays
            slar_tag = sig_args[1]
            slar_resume_id = Logger.import_run_id(slar_tag, slar_id)
            print(f"Imported run_id from <<< {slar_tag}_{slar_id}.id >>>\n")

            sig_args = (sig_args[0],  slar_resume_id)

        last_logged_itr = Logger.resume(*sig_args, itr_key="iteration")

        # dont't log args when resuming, other than seed which (typically) changes every run
        seed_itr_str = f"seed_{last_logged_itr / 1000}K" if last_logged_itr % 1000 == 0 else f"seed_{last_logged_itr}" 
        Logger.log_hyperparam(seed_itr_str, seed)

    else:
        Logger.start(*sig_args)
        Logger.log_hyperparam_dict(vars(args), ignore_keys=ignore_param_keys)
        Logger.log_hyperparam("seed_0", seed)

        if slarc is not None:
            exp_name = sig_args[1]
            slar_size = len(slarc)
            slar_tag = get_slar_tag(exp_name, slar_size, slar_id)
            Logger.export_run_id(slar_tag, slar_id)
            print(f"Exported run_id to <<< {slar_tag}_{slar_id}.id >>>\n")
        
            Logger.log_tag("slar_tag", slar_tag)
            Logger.log_tag("slar_rank", slar_id)  # NOTE critical to keep constant --> can't swap order of slurm array config with import/export logic

        last_logged_itr = None

    return resuming, last_logged_itr


def get_lr(args, resuming, slar_lr_sch=None):
    """
    Extracts learning rate hyperparameter based on priorities
    listed in docs of setup_lr_sheduler() below
    """
    if not args.no_sch_load and resuming:
        assert args.lr is None, "Don't specify lr when loading existing scheduler"
        max_lr = Logger.get_hyperparam("lr")
    
    elif slar_lr_sch is not None:  # use slarc values if provided
        max_lr = slar_lr_sch["max_lr"]

    else:  # default to input kwargs if nothing else
        max_lr = args.lr

    assert max_lr is not None, "No (max) learning rate 'lr' specified!"

    if not resuming:  # logged here to account for possible slarc override
        Logger.log_hyperparam("lr", max_lr)

    return max_lr


def setup_lr_scheduler(args, optimizer, resuming, max_lr, slar_lr_sch=None):
    """    
    Initialize learning rate scheduler

    Suports three options:
        - No scheduling
        - Exponential scheduling
        - Cosine scheduling

    Expenential and Cosine scheduling optionally support:
        A) Linear warmup
        B1) Terminal constant floor after reaching end of primary decay
        OR
        B2) Linear 'cooldown' to 0 beginning after reaching end of primary decay 

    When resuming loads existing logged parameter values (unless --no_sch_load is asserted)

    Otherwise, uses Slurm array config settings (per-job in the array) to configure

    Otherwise, defaults to input kwargs --exp_lr_schedule or --cos_lr_schedule

    NOTE assumes schedule param format based on length:
        3 (no cooldown)   --> [warmup_itrs_in_thousands, decay_finish_itr_in_thousands, lr_floor]
        4 (with cooldown) --> [warmup_itrs_in_thousands, primary_decay_finish_itr_in_thousands, primary_decay_final_lr, cooldown_decay_finish_itr_in_thousands]
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, lineno=163)  # SequentialLR internally uses deprecated PyTorch code, this quiets an annoying warning

    if not args.no_sch_load and resuming:
        assert args.exp_lr_schedule is None and args.cos_lr_schedule is None, "Don't specify lr schedule when loading existing scheduler"
        exp_lr_schedule = Logger.get_hyperparam("exp_lr_schedule")  # None if not present
        cos_lr_schedule = Logger.get_hyperparam("cos_lr_schedule")  # ditto
    
    elif slar_lr_sch is not None:  # use slarc values if provided
        exp_lr_schedule = slar_lr_sch["exp"]
        cos_lr_schedule = slar_lr_sch["cos"]

    else:  # default to input kwargs if nothing else
        exp_lr_schedule = args.exp_lr_schedule
        cos_lr_schedule = args.cos_lr_schedule

    if not resuming:  # logged here to account for possible slarc override
        Logger.log_hyperparam("exp_lr_schedule", exp_lr_schedule)
        Logger.log_hyperparam("cos_lr_schedule", cos_lr_schedule)

    assert exp_lr_schedule is None or cos_lr_schedule is None, "Exponential and cosine learning rate scheduling are mutually exclusive!"

    if exp_lr_schedule is None and cos_lr_schedule is None:
        return None # no LR scheduling --> initial LR set externally in optimizer
    
    lr_sch_params = exp_lr_schedule if exp_lr_schedule is not None else cos_lr_schedule

    if len(lr_sch_params) == 3:  # no cooldown
        warmup_itrs, decay_finish_itr, lr_floor = lr_sch_params
        cooldown_finish_itr = None
    else:  # with cooldown
        warmup_itrs, decay_finish_itr, lr_floor, cooldown_finish_itr = lr_sch_params
        cooldown_finish_itr = 1000 * cooldown_finish_itr
    
    warmup_itrs, decay_finish_itr = 1000 * warmup_itrs, 1000 * decay_finish_itr  # itrs expressed in thousands
    assert decay_finish_itr > warmup_itrs

    num_decay_itrs = decay_finish_itr - warmup_itrs

    schedulers = []

    if warmup_itrs > 0:
        schedulers.append(torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, total_iters=warmup_itrs - 1))  # -1 as we count the last update at max lr as part of the warmup, pytorch doesn't

    if cos_lr_schedule is not None:
        schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_decay_itrs, eta_min=lr_floor))

    elif exp_lr_schedule is not None:
        assert lr_floor >= 1e-8, f"Exponential learning rate schedule floor '{lr_floor}' is below epsilon; very small values must be avoided to not decay too fast"
        gamma = (lr_floor / max_lr) ** (1 / num_decay_itrs)
        schedulers.append(torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma))

    floor_factor = lr_floor / max_lr
    if cooldown_finish_itr is not None:
        cooldown_itrs = cooldown_finish_itr - decay_finish_itr
        schedulers.append(torch.optim.lr_scheduler.LinearLR(optimizer, floor_factor, 0.0, total_iters=cooldown_itrs))
    else:
        schedulers.append(torch.optim.lr_scheduler.ConstantLR(optimizer, floor_factor, total_iters=int(1e12)))

    milestones = [warmup_itrs - 1, decay_finish_itr - 1] if warmup_itrs > 0 else [decay_finish_itr - 1]  # -1 as LR scheduler starts at 0, we start at 1
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers, milestones=milestones)


def fetch_fork_state_dicts(args):
    print(f"Forking from run id <<< {args.fork_from_run} >>>")
    
    if args.fork_from_checkpoint is not None:
        model_state_dict, optimizer_state_dict, scheduler_state_dict = Logger.load_checkpoint(args.fork_from_checkpoint, no_mlflow=True, run_id=args.fork_from_run, device=torch.device(args.device if args.device is not None else "cpu"))
        Logger.log_hyperparam(f"last_iteration_before_fork", args.fork_from_checkpoint)  # already logged from args logging, but just to be consistent
        print(f"Forking from iteration ==> {args.fork_from_checkpoint}\n")
    
    else:
        model_state_dict, optimizer_state_dict, scheduler_state_dict, forked_last_itr = Logger.load_latest_checkpoint("iteration", run_id=args.fork_from_run, device=torch.device(args.device if args.device is not None else "cpu"))
        Logger.log_hyperparam(f"last_iteration_before_fork", forked_last_itr)
        print(f"Forking from iteration ==> {forked_last_itr}\n")

    return model_state_dict, optimizer_state_dict, scheduler_state_dict


def setup_algorithm(args, optimizer, scheduler):
    if args.algo == "ppo":
        AlgoCls = TspPPO
        extra_algo_kwargs = dict(
            epochs=args.minibatch_epochs,
            minibatches=args.minibatches,
            ratio_clip=args.ratio_clip,
            critic_coeff=args.critic_coeff,
            only_state_values=args.only_state_values
        )
    elif args.algo == "ppo_rollout":
        assert not args.only_state_values, "--only_state_value not applicable to ppo_rollout; we already use state values out of necessity"

        AlgoCls = TspPPOGreedyRollout
        extra_algo_kwargs = dict(
            epochs=args.minibatch_epochs,
            minibatches=args.minibatches,
            ratio_clip=args.ratio_clip
        )
    elif args.algo == "a2c":
        assert args.minibatch_epochs == 1 and args.minibatches == 1, "Minibatching not supported by A2C, leave 'minibatch_epochs' and 'minibatches' arguments as defaults (1)"
        assert not args.only_state_values, "Pure state-value learning ablation only implemented for PPO, not A2C"

        AlgoCls = TspA2C
        extra_algo_kwargs = dict(critic_coeff=args.critic_coeff)
    else:
        raise Exception(f"Unrecognized training algorithm '{args.algo}'")

    return AlgoCls(
        optimizer, grad_norm_clip=args.grad_norm_clip, scheduler=scheduler, **extra_algo_kwargs
    )



if __name__ == "__main__":

    args = parser.parse_args()

    seed = get_seed(args)
    seed_rand_gen(seed)
    print(f"Seed <<< {seed} >>>")

    if args.debug:
        Logger.debug_mode_on()

    # load Slurm array config and task id, if applicable
    if args.slurm_array_config is not None:
        slarc, slar_id, slar_nodes, slar_dimensions, slar_model_dim, slar_lr_sch = load_slar_config(args)
    else:
        slarc = None
        slar_id = None

    # initialize MLflowLogger
    if args.mlflow_logging_signature is not None:
        resuming, last_logged_itr = setup_logging(args, slarc, slar_id, seed, ignore_param_keys=["mlflow_logging_signature", "seed", "lr", "exp_lr_schedule", "cos_lr_schedule"])
    else:
        Logger.dummy_init()
        resuming = False

    # assert valid experiment loading args, if applicable (resuming and forking are exclusive options)
    if resuming:
        assert args.fork_from_run is None and args.fork_from_checkpoint is None, "Resuming and forking a run are exclusive options and can't be done jointly"
    elif args.fork_from_run is None:
        assert args.fork_from_checkpoint is None, "--fork_from_checkpoint must be used with --fork_from_run"

    # create TSP datasets    
    nodes = [slar_nodes] if slarc is not None and slar_nodes is not None else interpret_problem_sizes(args)
    
    dimensions = slar_dimensions if slarc is not None and slar_dimensions is not None else args.dimensions
    assert dimensions >= 2, "Dimensions of TSP problems must be >=2"

    total_samples = ceil(1e3 * args.itr * args.batch_size)

    dataset = TspLiveDatagen(size=nodes, epoch_size=total_samples, dimensions=dimensions)

    if resuming:
        eval_datasets = Logger.load_pickle_artifact("eval_problems", artifact_dir="datasets")
    elif args.fork_from_run is not None:
        eval_datasets = Logger.load_pickle_artifact("eval_problems", artifact_dir="datasets", run_id=args.fork_from_run)
        Logger.save_pickle_artifact(eval_datasets, "eval_problems", artifact_dir="datasets")  # need to copy over to forked run if resuming going forward
    else:
        eval_datasets = create_eval_datasets(nodes, dimensions, args)
        Logger.save_pickle_artifact(eval_datasets, "eval_problems", artifact_dir="datasets")

    # initialize model and agent
    if args.algo == "ppo_rollout":
        model = TspGreedyBaselineModel(  # no critic layers
            dim_model=slar_model_dim if slarc is not None and slar_model_dim is not None else args.model_dim,
            num_enc_layers=args.n_enc,
            num_dec_layers=args.n_dec,
            tsp_dimensions=dimensions 
        )
    else:
        model = TspAcModel(
            dim_model=slar_model_dim if slarc is not None and slar_model_dim is not None else args.model_dim,
            num_enc_layers=args.n_enc,
            num_dec_layers=args.n_dec,
            num_crt_layers=args.n_crt,
            tsp_dimensions=dimensions
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
    algo = setup_algorithm(args, optimizer, scheduler)

    # build runner and start training
    TrainerCls = TspTrainerRolloutCritic if args.algo == "ppo_rollout" else TspTrainer
    runner = TrainerCls(dataset, agent, algo, eval_datasets=eval_datasets)

    runner.start(
        epochs=1,
        batch_size=args.batch_size,
        check_period=args.check_period,
        eval_period=args.eval_period,
        eval_batch_size=args.eval_batch_size,
        init_logger_step= 1 if not resuming else last_logged_itr + 1,
        logger_step_multiplier= args.minibatch_epochs * args.minibatches  # aligns logging step with number of gradient updates
    )
