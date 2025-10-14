"""
Revised specifically for offline evaluation for compute scaling loss stats (with model scaling experiments)
"""

import torch
from tqdm import tqdm
import mlflow
import os
import os.path as osp
from time import sleep
import argparse
import numpy as np

import tsp
from tsp.agent import TspAgent
from tsp.eval import batched_eval_with_loss_and_solutions

from tsp.utils import seed_rand_gen, get_num_named_params, perm_shuffle
from tsp.logger import MLflowLogger as Logger, parse_log_sig
from tsp.select import sample_select
from tsp.model.base import TspAcModel

from config.eval_temporal import SUPER_CONFIG

root_path = osp.dirname(osp.dirname(tsp.__file__))
checkpoint_download_path = osp.join(root_path, "eval", "checkpoint_downloads")

parser = argparse.ArgumentParser()
parser.add_argument("cfg", default=None, type=str)



CUDA_IDX = 0
SELECT_FN = sample_select
SHUFFLE_BATCH_SIZE = 3_200


def stagger_for_mlf():
    sleep(2 * int(os.getenv('SLURM_ARRAY_TASK_ID')))


def get_cfg():
    return SUPER_CONFIG[parser.parse_args().cfg]


def mlf_init(cfg):
    if cfg.mlflow_logging_signature is not None:
        resuming, sig_args = parse_log_sig(cfg.mlflow_logging_signature)

        assert not resuming, "resuming unsupported for this script"

        Logger.start(*sig_args)
        Logger.log_hyperparam_dict(vars(cfg), ignore_keys=["mlflow_logging_signature", "checkpoints", "models"])
        
        seed_rand_gen(1234)

    else:
        Logger.dummy_init()


def fetch_scale_and_run_id(cfg):    
    slar_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    width, run_id = cfg.models[slar_id]

    nodes = cfg.nodes

    return (width, nodes), run_id


def create_agent(width):
    model = TspAcModel(dim_model=width, num_enc_layers=3, num_dec_layers=2, num_crt_layers=2, tsp_dimensions=2)
    agent = TspAgent(model)
    agent.set_select(SELECT_FN)

    if CUDA_IDX is not None:
        agent.to(CUDA_IDX)

    num_non_embed = get_num_named_params(agent.named_parameters())
    Logger.log("non_embedding_parameters", num_non_embed, step=0)

    return agent


def load_checkpoint(agent, run_id, check_itr):
    model_state_dict, _, _ = Logger.load_checkpoint(check_itr, run_id=run_id, device=torch.device(CUDA_IDX if CUDA_IDX is not None else "cpu"), no_mlflow=True)

    agent.load_state_dict(model_state_dict)
    agent.eval_mode()


def load_dataset(cfg):

    dpath = osp.join(cfg.dataset_dir, cfg.dataset_stub)
    assert osp.exists(dpath), f"can't find dataset: {dpath}"

    dataset = np.load(dpath, mmap_mode="r")

    return np.copy(dataset[cfg.dataset_slice])


def get_shuffle(dataset, nodes):
    lbls = torch.stack(len(dataset) * [torch.arange(nodes)], dim=0).numpy()
    perms = torch.stack([torch.randperm(nodes) for _ in range(len(dataset))], dim=0).numpy()
    shuf_data, shuf_relbls = perm_shuffle(dataset, lbls, perms, batch_size=SHUFFLE_BATCH_SIZE)  # takes a while

    shuf_data_pyt = torch.from_numpy(shuf_data)
    shuf_relbls_pyt = torch.from_numpy(shuf_relbls)

    return shuf_data_pyt, shuf_relbls_pyt


def eval(loss_type, agent, shuf_data_pyt, shuf_relbls_pyt, model_batch_size, check_itr):
    _, costs, actor_losses, critic_losses = batched_eval_with_loss_and_solutions(
        agent, 
        shuf_data_pyt, 
        batch_size=model_batch_size, 
        sol_per_problem=1, 
        algo=loss_type, 
        sup_labels=shuf_relbls_pyt
    )

    assert all([arr.shape[1] == 1 for arr in [costs, actor_losses, critic_losses]])  # 1 sol per problem

    Logger.log("total_tours", len(costs), step=check_itr)
    Logger.log_stat("eval_cost", costs[:, 0].double(), step=check_itr)  # casting to double precision before reduction
    Logger.log_stat("eval_actor_loss", actor_losses[:, 0].double(), step=check_itr)
    Logger.log_stat("eval_critic_loss", critic_losses[:, 0].double(), step=check_itr)

    # print(f"act loss shape {actor_losses.shape}")
    # print(f"crt loss shape {critic_losses.shape}")
    total_losses = actor_losses[:, 0].mean(dim=-1) + 0.52 * critic_losses[:, 0].mean(dim=-1)  # mean over seq dim before adding (critic loss has n+1 dimensions there from pure state-value estimate)
    Logger.log_stat("eval_total_loss", total_losses.double(), step=check_itr)



if __name__ == "__main__":
    stagger_for_mlf()

    cfg = get_cfg()

    mlf_init(cfg)

    scale, run_id = fetch_scale_and_run_id(cfg)
    width, nodes = scale

    Logger.log_hyperparam("model_run_id", run_id)
    Logger.log_hyperparam("model_dim", width)

    agent = create_agent(width)

    dataset = load_dataset(cfg)

    shuf_data_pyt, shuf_relbls_pyt = get_shuffle(dataset, nodes)

    print(f"Evaluating performance over {len(cfg.checkpoints)} checkpoints on {len(dataset)} tours of size {nodes}...")
    for check_itr in tqdm(cfg.checkpoints, desc="Checkpoints"):
        load_checkpoint(agent, run_id, check_itr)
        eval(cfg.loss_type, agent, shuf_data_pyt, shuf_relbls_pyt, cfg.model_batch_size, check_itr)