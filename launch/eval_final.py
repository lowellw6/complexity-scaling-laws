"""
Evaluates deep model performance on a single TSP scale using pre-existing supervised datasets, logging to MLflow
Saves deep model solutions to dataset solutions as a new dataset

Intended to be used with Slurm array jobs for parallelization (one job per model and scale)

WARNING for node scales larger than 127, you'll need to upgrade from storing as np.int8
"""

import torch
import os.path as osp
import os
from time import sleep
import numpy as np
from tqdm import tqdm
import argparse

import tsp
from tsp.agent import TspAgent
from tsp.eval import batched_eval_with_loss_and_solutions, get_costs
from tsp.utils import seed_rand_gen, get_coords, perm_shuffle, correct_shuffled_tours, get_num_named_params
from tsp.logger import MLflowLogger as Logger, parse_log_sig
from tsp.select import sample_select
from tsp.model.base import TspAcModel

from config.eval_final import SUPER_CONFIG

root_path = osp.dirname(osp.dirname(tsp.__file__))
checkpoint_download_path = osp.join(root_path, "eval", "checkpoint_downloads")

parser = argparse.ArgumentParser()
parser.add_argument("cfg", default=None, type=str)

SAVE_SOL = True  # set to false to just log to MLflow and avoid saving the solution datasets

CUDA_IDX = 0
SELECT_FN = sample_select
SHUFFLE_BATCH_SIZE = 3_200

SHUFFLE_CORRUPTION_CHECK = True  # validates whether post-shuffle-corrected tours match costs on original data raw tour output costs on shuffled data 



def stagger_for_mlf():
    sleep(2 * int(os.getenv('SLURM_ARRAY_TASK_ID')))


def get_cfg():
    return SUPER_CONFIG[parser.parse_args().cfg]


def mlf_init(cfg):
    if cfg.mlflow_logging_signature is not None:
        resuming, sig_args = parse_log_sig(cfg.mlflow_logging_signature)

        assert not resuming, "resuming unsupported for this script"

        Logger.start(*sig_args)
        Logger.log_hyperparam_dict(vars(cfg), ignore_keys=["mlflow_logging_signature", "models"])
        
        seed_rand_gen(1234)

    else:
        Logger.dummy_init()


def fetch_scale_and_run_id_and_dataset_stub(cfg):
    width = getattr(cfg, "width", None)
    nodes = getattr(cfg, "nodes", None)
    dims = getattr(cfg, "dims", None)

    assert sum([width is None, nodes is None, dims is None]) == 1

    slar_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))

    if width is None:
        width, run_id = cfg.models[slar_id]
        dataset_stub = cfg.dataset_stub.replace("$", str(width))
        Logger.log_hyperparam("width", width)
    elif nodes is None:
        nodes, run_id = cfg.models[slar_id]
        dataset_stub = cfg.dataset_stub.replace("$", str(nodes))
        Logger.log_hyperparam("nodes", nodes)
    elif dims is None:
        dims, run_id = cfg.models[slar_id]
        dataset_stub = cfg.dataset_stub.replace("$", str(dims))
        Logger.log_hyperparam("dims", dims)

    return (width, nodes, dims), run_id, dataset_stub


def create_agent(width, dims):
    model = TspAcModel(dim_model=width, num_enc_layers=3, num_dec_layers=2, num_crt_layers=2, tsp_dimensions=dims)
    agent = TspAgent(model)
    agent.set_select(SELECT_FN)

    if CUDA_IDX is not None:
        agent.to(CUDA_IDX)

    num_non_embed = get_num_named_params(agent.named_parameters())
    Logger.log("non_embedding_parameters", num_non_embed, step=0)

    return agent


def load_checkpoint_artifact(agent, run_id, check_itr):
    model_state_dict, _, _ = Logger.load_checkpoint(check_itr, run_id=run_id, device=torch.device(CUDA_IDX if CUDA_IDX is not None else "cpu"), no_mlflow=True)

    agent.load_state_dict(model_state_dict)
    agent.eval_mode()

    Logger.log_hyperparam("model_run_id", run_id)
    Logger.log_hyperparam("model_checkpoint_itr", check_itr)


def load_checkpoint_download(agent, download_path):
    model_state_dict = torch.load(download_path, map_location=torch.device(CUDA_IDX if CUDA_IDX is not None else "cpu"))

    agent.load_state_dict(model_state_dict)
    agent.eval_mode()

    Logger.log_hyperparam("checkpoint_path", download_path)


def load_dataset(dataset_dir, dataset_stub):
    dpath = osp.join(dataset_dir, dataset_stub)
    assert osp.exists(dpath), f"can't find dataset: {dpath}"

    with open(dpath, "rb") as f:
        dataset = np.load(f)
    # dataset = np.load(dpath, mmap_mode="r")

    return dataset


def validate_post_shuffle_correction(dataset, corrected_sel, costs, batch, sol_per_prob):
    ref_costs = costs.flatten()
    ref_dataset = torch.repeat_interleave(torch.from_numpy(dataset), sol_per_prob, dim=0)

    dims = ref_dataset.shape[-1]
    ref_corrected_sel = torch.stack(dims * [torch.from_numpy(corrected_sel).view(batch * sol_per_prob, nodes)], dim=-1)

    sorted_tours = torch.gather(ref_dataset, 1, ref_corrected_sel)
    assert torch.allclose(get_costs(sorted_tours), ref_costs), "Solution order corrupted in post-shuffle permutation correction!"


def eval_and_get_solutions(cfg, loss_type, agent, dataset, nodes):
    lbls = torch.stack(len(dataset) * [torch.arange(nodes)], dim=0).numpy()
    perms = torch.stack([torch.randperm(nodes) for _ in range(len(dataset))], dim=0).numpy()
    shuf_data, shuf_relbls = perm_shuffle(dataset, lbls, perms, batch_size=SHUFFLE_BATCH_SIZE)  # takes a while

    shuf_data_pyt = torch.from_numpy(shuf_data)
    shuf_relbls_pyt = torch.from_numpy(shuf_relbls)

    selections, costs, actor_losses, critic_losses = batched_eval_with_loss_and_solutions(agent, shuf_data_pyt, cfg.model_batch_size, cfg.sol_per_problem, algo=loss_type, sup_labels=shuf_relbls_pyt)

    # NOTE we only log stats for first solution of every problem, if sol_per_problem > 1, to avoid inflating apparent sample size of dataset
    Logger.log("total_tours", len(costs), step=0)
    Logger.log_stat("eval_cost", costs[:, 0].double(), step=0)  # casting to double precision before reduction
    Logger.log_stat("eval_actor_loss", actor_losses[:, 0].double(), step=0)
    Logger.log_stat("eval_critic_loss", critic_losses[:, 0].double(), step=0)

    print(f"act loss shape {actor_losses.shape}")
    print(f"crt loss shape {critic_losses.shape}")
    total_losses = actor_losses[:, 0].mean(dim=-1) + 0.52 * critic_losses[:, 0].mean(dim=-1)  # mean over seq dim before adding (critic loss has n+1 dimensions there from pure state-value estimate)
    Logger.log_stat("eval_total_loss", total_losses.double(), step=0)

    # relabel/permute selections to correspond with original unshuffled data (broadcasted across all sol_per_problem 1st index)
    batch, sol_per_prob = selections.shape[:2]
    flat_sel = selections.view(batch * sol_per_prob, nodes).numpy()
    data_rpt = np.repeat(dataset, sol_per_prob, axis=0)
    shuf_data_rpt = np.repeat(shuf_data, sol_per_prob, axis=0)

    corrected_sel = correct_shuffled_tours(data_rpt, shuf_data_rpt, flat_sel, batch_size=SHUFFLE_BATCH_SIZE)
    corrected_sel = corrected_sel.reshape((batch, sol_per_prob, nodes))

    if SHUFFLE_CORRUPTION_CHECK:
        validate_post_shuffle_correction(dataset, corrected_sel, costs, batch, sol_per_prob)

    return corrected_sel


def save_ml_solutions(cfg, selections):
    if not osp.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    output_stub = cfg.output_prefix + f"_{width}w_{nodes}n_{dims}d_{cfg.sol_per_problem}sol_{len(selections)}t.npy"
    opath = osp.join(cfg.output_dir, output_stub)

    with open(opath, "wb") as f:
        np.save(f, selections.astype(np.int8))  # WARNING max node range 127


def get_loss_type(cfg):
    if cfg.output_prefix.startswith("drl"):
        return "ppo"
    if cfg.output_prefix.startswith("sft"):
        return "supervised"
    raise Exception(f"Couldn't parse loss type from output prefix '{cfg.output_prefix}'")


def get_suffix(output_prefix, scale):
    width, nodes, dims = scale
    if "model" in output_prefix:
        return f"{width}w"
    if "node" in output_prefix:
        return f"{nodes}n"
    if "dim" in output_prefix:
        return f"{dims}d"



if __name__ == "__main__":
    print("WARNING for node scales larger than 127, you'll need to update np.int8 in line 179")

    stagger_for_mlf()

    cfg = get_cfg()

    mlf_init(cfg)

    scale, run_id, dataset_stub = fetch_scale_and_run_id_and_dataset_stub(cfg)
    width, nodes, dims = scale

    agent = create_agent(width, dims)

    download_path = osp.join(checkpoint_download_path, f"{cfg.output_prefix}_{get_suffix(cfg.output_prefix, scale)}.pth")
    if run_id.startswith("<") and osp.exists(download_path):
        print(f"Loading downloaded checkpoint --> {download_path}")
        load_checkpoint_download(agent, download_path)
        
    elif not run_id.startswith("<"):
        print(f"Loading checkpoint with MLflow run id @ iteration --> {run_id} @ {cfg.check_itr}")
        load_checkpoint_artifact(agent, run_id, cfg.check_itr)

    else:
        raise Exception("Either an MLflow run id or downloaded checkpoint path must be provided")

    dataset = load_dataset(cfg.dataset_dir, dataset_stub)
    assert dataset.shape[1:] == (nodes, dims)

    loss_type = get_loss_type(cfg)

    selections = eval_and_get_solutions(cfg, loss_type, agent, dataset, nodes)

    if SAVE_SOL:
        save_ml_solutions(cfg, selections)

    print("DONE")
