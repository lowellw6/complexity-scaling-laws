"""
HPO for TSP DRL agent using BOHB implemented with Optuna
BOHB = Multivariate TPE (BO) + Hyperband (HB)

Note the total number of trials (across all parallel jobs,
so cfg.n_trials multiplied by the Slurm array size)
should be substantially greater than the startup trials
consumed by the TPESampler model construction, which by
default is 10. But for TPE to adapt its search space to each 
Hyperband bracket (SuccessiveHalvingPruner), you need to
multiply by the number of brackets. So if HyperbandPruner
uses 4 SucessiveHalvingPruners, then 4 * 10 = 40 trials
are used just for startup.
"""

import os
import os.path as osp
import json
import torch
from math import ceil
import hashlib
from copy import deepcopy
import pickle
import time

import optuna
from optuna.trial import TrialState
from optuna.storages import JournalStorage, JournalFileStorage, JournalRedisStorage
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner, MedianPruner

import tsp
from tsp.datagen import TspLiveDatagen, TspDataset
from tsp.model.base import TspModel, TspAcModel
from tsp.agent import TspAgent
from tsp.algo import TspReinforce, TspA2C, TspPPO
from tsp.train import HpoTspTrainer
from tsp.logger import MLflowLogger as Logger, parse_log_sig
from tsp.utils import get_num_named_params, seed_rand_gen

from train_model_agent import interpret_problem_sizes, load_sd_component_if_given, get_seed, create_eval_datasets, load_slar_config, setup_logging, get_lr, setup_lr_scheduler, fetch_fork_state_dicts, setup_algorithm

from fpc_config.hpo_model_agent import cfg


VAL_DATASET = None


def sprint(msg, slar_id):
    print(f"({slar_id}) {msg}" if slar_id is not None else msg)


def is_main_job(slar_id):
    return slar_id == 0 or slar_id is None
    

def objective(trial):
    """
    TODO
    - restart-safe logger inside objective (in addition to one hpo run in main for overall stats) (this might be hard)
    """
    # sample hyperparameters upfront
    ppo_num_minibatches = trial.suggest_int("ppo_num_minibatches", *cfg.ppo_num_minibatches_range)
    ppo_ratio_clip = trial.suggest_float("ppo_ratio_clip", *cfg.ppo_ratio_clip_range, step=0.01)
    critic_coeff = trial.suggest_float("critic_coeff", *cfg.critic_coeff_range, step=0.01)

    max_lr = trial.suggest_float("max_lr", *cfg.max_lr_range, log=True)
    decay_finish_update = trial.suggest_int("decay_finish_update", *cfg.decay_finish_update_range)
    grad_norm_clip = trial.suggest_float("grad_norm_clip", *cfg.grad_norm_clip_range, log=True)

    model_dim = trial.suggest_int("model_dim", *cfg.model_dim_range, step=8)  # 8 heads in arch, must be divisble by this for pytorch multi-head attn
    num_enc_layers = trial.suggest_int("num_enc_layers", *cfg.enc_layers_range)
    num_dec_layers = trial.suggest_int("num_dec_layers", *cfg.dec_layers_range)
    num_crt_layers = trial.suggest_int("num_crt_layers", *cfg.crt_layers_range)

    print("Launching Trial with hyperparameters:")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # compute indirect hyperparameters
    batch_size = ppo_num_minibatches * cfg.ppo_minibatch_size

    # create TSP datasets
    nodes = interpret_problem_sizes(cfg)
    
    total_samples = ceil(1e3 * cfg.itr * cfg.ppo_minibatch_size)  # itr expressed in gradient updates

    dataset = TspLiveDatagen(size=nodes, epoch_size=total_samples)
    val_dataset = VAL_DATASET  # no copy, just reference pass, should be read only within trials

    # initialize model and agent
    model = TspAcModel(
        dim_model=model_dim,
        num_enc_layers=num_enc_layers,
        num_dec_layers=num_dec_layers,
        num_crt_layers=num_crt_layers,
    )
    agent = TspAgent(model)

    if cfg.use_gpu:
        agent.to(0)

    # TODO log number of non-embedding parameters (for scaling experiment book keeping)
    # non_embed_params = agent.named_parameters()  # TSP agents have no vocab or positional embeddings gathered here
    # num_non_embed = get_num_named_params(non_embed_params)
    # Logger.log_hyperparam(f"non_embedding_parameters", num_non_embed)

    # initialize optimizer
    optimizer = torch.optim.Adam(agent.parameters(), lr=max_lr)

    # initialize scheduler, if applicable
    warmup_itrs = round(1000 * cfg.warmup_updates / ppo_num_minibatches) / 1000
    decay_finish_itr = round(1000 * decay_finish_update / ppo_num_minibatches) / 1000

    cfg.cos_lr_schedule = (warmup_itrs, decay_finish_itr, cfg.lr_floor)
    scheduler = setup_lr_scheduler(cfg, optimizer, False, max_lr, None)

    # initialize algorithm
    algo = TspPPO(
        optimizer,
        grad_norm_clip=grad_norm_clip,
        scheduler=scheduler,
        critic_coeff=critic_coeff,
        epochs=1,
        minibatches=ppo_num_minibatches,
        ratio_clip=ppo_ratio_clip
    )

    # build runner and start training
    runner = HpoTspTrainer(
        trial, 
        dataset, 
        val_dataset, 
        agent, 
        algo, 
        trial_step_multiplier=ppo_num_minibatches
    )

    last_avg_val_cost = runner.start(
        epochs=1,
        batch_size=batch_size,
        check_period=cfg.check_period,
        eval_period= round(cfg.val_period / ppo_num_minibatches),  # Optuna validation period w.r.t. # of gradient updates (with rounding)
        eval_batch_size=cfg.eval_batch_size,
        init_logger_step=1,
        logger_step_multiplier=ppo_num_minibatches  # aligns logging step with number of gradient updates
    )

    return last_avg_val_cost



if __name__ == "__main__":
    # slar_id = int value if running in slurm array, otherwise None
    slar_id = os.getenv("SLURM_ARRAY_TASK_ID")
    if slar_id is not None:
        slar_id = int(slar_id)

    seed = get_seed(cfg)
    seed_rand_gen(seed)
    sprint(f"Seed <<< {seed} >>>", slar_id)

    # some race condition seems to exist for JournalStorage on creation, resulting in a byte decoding bug
    # this stagger appears to fix the problem, at least most of the time
    time.sleep(2 * slar_id)

    if cfg.debug:
        Logger.debug_mode_on()

    # initialize MLflowLogger
    if cfg.mlflow_logging_signature is not None:
        setup_logging(cfg, None, None, seed)
    else:
        Logger.dummy_init()

    # load global validation dataset
    VAL_DATASET = TspDataset(filename=cfg.val_dataset_path)

    # initialize/load storage
    study_path = osp.join(cfg.local_storage_path, cfg.study_name)
    if not osp.exists(study_path):
        os.makedirs(study_path, exist_ok=True)  # race condition to get here so need exist_ok=True

    storage = JournalStorage(JournalRedisStorage(cfg.redis_storage_url))
    #log_path = osp.join(study_path, "journal.log")
    #storage = JournalStorage(JournalFileStorage(log_path)) #, lock_obj=optuna.storages.JournalFileOpenLock(log_path)))

    # initialize/load sampler
    sampler_save_path = osp.join(study_path, f"sampler_{slar_id}.pkl" if slar_id is not None else "sampler.pkl")
    if osp.exists(sampler_save_path):
        sampler = pickle.load(open(sampler_save_path, "rb"))
    else:
        sampler = TPESampler(multivariate=True)

    # initialize/load pruner
    pruner_save_path = osp.join(study_path, f"pruner_{slar_id}.pkl" if slar_id is not None else "pruner.pkl")
    if osp.exists(pruner_save_path):
        pruner = pickle.load(open(pruner_save_path, "rb"))
    else:
        pruner = HyperbandPruner(min_resource=int(cfg.min_resource), max_resource=int(1000 * cfg.itr), reduction_factor=cfg.reduction_factor)
        ### pruner = MedianPruner()  # TEST
    
    # create/load Optuna study and run HPO
    study = optuna.create_study(study_name=cfg.study_name, load_if_exists=True, storage=storage, sampler=sampler, pruner=pruner, direction="minimize")

    resuming = osp.exists(sampler_save_path) and osp.exists(pruner_save_path)

    if not resuming and is_main_job(slar_id):
        study.enqueue_trial(params=cfg.warm_start_trial_params)
    
    study.optimize(objective, n_trials=cfg.n_trials, timeout=cfg.timeout, gc_after_trial=True)

    # save sampler and pruner states
    with open(sampler_save_path, "wb") as f:
        pickle.dump(study.sampler, f)

    with open(pruner_save_path, "wb") as f:
        pickle.dump(study.pruner, f)

    # display current study stats
    if is_main_job(slar_id):
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))
        # print("  Number of startup trials (TPE model construction): ", study.pruner._n_brackets * study.sampler._n_startup_trials)

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
