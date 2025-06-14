from numpy.lib import select
import torch
from torch.distributions.categorical import Categorical
import numpy as np
import pkg_resources
import random
import os
import binascii
import time

from tsp.datagen import SENTINEL


def get_coords(batch_size, problem_size, dimensions=2):
    """Get single-tensor problems dataset with shape (N, S, d)."""
    from tsp.datagen import TspDataset

    dataset = TspDataset(size=problem_size, num_samples=batch_size, dimensions=dimensions)
    return torch.stack(dataset[:], dim=0)


def get_costs(problems, select_idxs=None):
    """
    Adapted from code provided with the paper "Attention, Learn to Solve Routing Problems!"
    https://github.com/wouterkool/attention-learn-to-route/blob/master/problems/tsp/problem_tsp.py

    Modified to allow not passing selection indices, in which case we assume the problems stack
    contains solutions where nodes are already sorted in order of intended travel.

    Also modified to support variable problem sizes, where shorter problems are padded by sentinels
    defined in datagen.py.
    """
    if select_idxs is not None:
        # (Removed check for tour validity as this is non-trivial with variable problem sizes)

        pad_mask = generate_idxs_pad_mask(select_idxs)

        # Gather solution in order of tour, swapping sentinels with first index to not break gather
        select_idxs = reset_pads(select_idxs, pad_mask, val=0)
        s = problems.gather(1, select_idxs.unsqueeze(-1).expand_as(problems))

        # Reset padding locations to sentinel values
        s = reset_pads(s, pad_mask, val=SENTINEL)
    else:
        pad_mask = generate_padding_mask(problems)
        s = problems

    # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
    # Modified to handle padded problem tensors with varying sequence lengths
    edge_costs = (s[:, 1:] - s[:, :-1]).norm(p=2, dim=2)
    edge_costs = reset_pads(edge_costs, pad_mask[:, 1:])

    seq_len = s.shape[1]
    last_idxs = seq_len - pad_mask.sum(dim=1) - 1
    last_nodes = batch_select_gather(s, last_idxs).squeeze(1)
    circ_edge_costs = (s[:, 0] - last_nodes).norm(p=2, dim=1)

    return edge_costs.sum(dim=1) + circ_edge_costs


def all_to(tensors, device="cpu"):
    """Move all tensors to device at once."""
    on_device = [tensor.to(device) for tensor in tensors]
    return on_device


def batch_select_gather(problems, selection_idxs):
    """
    Extracts coordinates from problems at selection_idxs.
    Expects problems to have shape (N, S, d)
    and selection_idxs to have shape (N,).
    """
    sel_idx_g = torch.stack([selection_idxs] * problems.shape[-1], dim=-1).unsqueeze(1)
    return torch.gather(problems, dim=1, index=sel_idx_g)  # (N, 1, d)


def batch_dist_gather(distributions, batch_select_idxs):
    """
    Extracts batched distribution probabilities (or
    log probs) at the indices specified by
    batch_select_idxs. Expects distributions to have
    shape (N, S, S) and batch_select_idxs to have
    shape (N, S).
    """
    select_idxs_g = batch_select_idxs.unsqueeze(dim=-1)
    return torch.gather(distributions, dim=-1, index=select_idxs_g).squeeze(-1)


def pad_safe_dist_gather(distributions, batch_select_idxs, reset_val):
    """
    Adaptation of 'batch_dist_gather' which doesn't break
    when 'batch_select_idxs' is padded with sentinels.
    """
    pad_mask = generate_idxs_pad_mask(batch_select_idxs)
    safe_select_idxs = reset_pads(batch_select_idxs, pad_mask, val=0)

    sel_dist = batch_dist_gather(distributions, safe_select_idxs)
    return reset_pads(sel_dist, pad_mask, val=reset_val)


def get_grad_norm(parameters):
    """Get the gradient (vector) L2 norm from parameters iterable."""
    grad_params = filter(lambda p: p.grad is not None, parameters)
    flat_grads = [p.grad.flatten() for p in grad_params]
    grads_vec = torch.cat(flat_grads, dim=0)
    norm = torch.norm(grads_vec, p=2)
    return norm.item()


def float_equality(x, y, eps=1e-4):
    return (x < y + eps) and (x > y - eps)


def get_entropy(log_probs):
    """
    Returns entropy for each decision.
    Assumes log_probs has shape (N, S, S).
    Returns entropy matrix of shape (N, S),
    computing over the last 'S' dim.
    """
    assert log_probs.shape[1] == log_probs.shape[2]
    dist = Categorical(probs=torch.exp(log_probs))
    return dist.entropy()


def is_installed(module_name):
    installs = [pkg.key for pkg in pkg_resources.working_set]
    return module_name in installs


def generate_square_subsequent_mask(sz):
    """
    Taken directly from PyTorch docs:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def generate_padding_mask(problems):
    """
    Generates boolean mask where 'problems'
    matches the sentinel used in datagen.
    Assumes 'problems' is shape (N, S, d)
    or (S, N, d), returning a mask matching
    the shape of the first two dimensions.
    """
    return (problems == SENTINEL)[:, :, 0]


def generate_idxs_pad_mask(idxs):
    """
    Like 'generate_padding_mask' for
    for selection indices. Assumes
    'idxs' is shape (N, S) and has
    an integer datatype.

    WARNING: sentinel value MUST
    be interpretable as an integer
    for this to work.
    """
    return idxs == int(SENTINEL)


def reset_pads(tensor, pad_mask, val=0.0):
    """
    Sets each padded location defined
    by 'pad_mask' in the the input 'tensor'.
    'tensor' should have the same leading
    dimensions as 'pad_mask'. Additional
    dimensions are assumed to be feature
    dimensions which zeroing is broadcasted
    over. Defaults to setting at 0.0.
    """
    tensor = tensor.clone()
    tensor[pad_mask] = val
    return tensor


def seed_rand_gen(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def perm_shuffle(problems, labels, perms, batch_size=10_000):
    """
    Resort 'problems' and 'labels' according to
    'perms'. i.e. if non-batched,
    problem[label] == problem[perm][relabel]

    Or in words, the relabelled indices are the
    indices in the perms where you find the values
    in the original labels.

    Expects leading dimensions (N, S).

    Processes in batches of 'batch_size' if greater
    to avoid out of memory errors during matrix expansion.
    """
    def shuffle_op(problems, labels, perms):
        b, n, d = problems.shape

        comp_perm_mat = np.stack([perms] * n, axis=1)
        comp_lbl_mat = np.stack([labels] * n, axis=2)
        batch_extract_idxs, _, relabels = (comp_perm_mat == comp_lbl_mat).nonzero()
        relabels = relabels.reshape((b, n))

        shuf_probs = problems[batch_extract_idxs, perms.flatten()].reshape((b, n, d))

        return shuf_probs, relabels
    
    if len(problems) < batch_size:
        return shuffle_op(problems, labels, perms)
    else:
        start_time = time.time()
        print(f"Shuffling permutations of {len(problems)} problems, this could take a while...")
        
        batch_idxs = [slice(start_idx, start_idx + batch_size) for start_idx in range(0, len(problems) - batch_size + 1, batch_size)]
        shuffled_batches = [shuffle_op(problems[bidx], labels[bidx], perms[bidx]) for bidx in batch_idxs]

        print(f"...done! {time.time() - start_time} s")
        shuf_prob_batches, shuf_relbl_batches = zip(*shuffled_batches)
        return np.concatenate(shuf_prob_batches, axis=0), np.concatenate(shuf_relbl_batches, axis=0)
    

def correct_shuffled_tours(dataset, shuf_data, sel_for_shuf, batch_size=10_000):
    """
    Assumes sel_for_shuf are tour idx selections for shuf_data problems
    And that shuf_data is a shuffled version of dataset
    This returns the corresponding selections in dataset

    If non-batched,
    dataset[return_value] == shuf_data[sel_for_shuf]

    Expects leading dimensions (N, S).

    Processes in batches of 'batch_size' if greater
    to avoid out of memory errors during matrix expansion.
    """
    def resort_op(dataset, shuf_data, sel_for_shuf):
        b, n, d = dataset.shape

        repeat_sel_for_shuf = np.stack([sel_for_shuf] * d, axis=-1).astype(np.int64)
        shuf_sorted = np.take_along_axis(shuf_data, repeat_sel_for_shuf, axis=1)

        comp_data_mat = np.stack([dataset] * n, axis=1)
        comp_sort_mat = np.stack([shuf_sorted] * n, axis=2)

        _, _, corrected_selections = np.isclose(comp_data_mat, comp_sort_mat).all(axis=-1).nonzero()
        corrected_selections = corrected_selections.reshape((b, n))

        return corrected_selections
    
    if len(dataset) < batch_size:
        return resort_op(dataset, shuf_data, sel_for_shuf)
    else:
        start_time = time.time()
        print(f"Correcting permutations of {len(dataset)} problems, this could take a while...")
        
        batch_idxs = [slice(start_idx, start_idx + batch_size) for start_idx in range(0, len(dataset) - batch_size + 1, batch_size)]
        corrected_sel_batches = [resort_op(dataset[bidx], shuf_data[bidx], sel_for_shuf[bidx]) for bidx in batch_idxs]
        print(f"...done! {time.time() - start_time} s")

        return np.concatenate(corrected_sel_batches, axis=0)


def filter_params(named_params, blacklist):
    """
    Prunes keyword matches from a torch named_parameters() generator.
    blacklist can be a single string, or iterator of strings
    """
    if type(blacklist) is str:
        blacklist = (blacklist,)

    for key in blacklist:
        named_params = filter(lambda x: key not in x[0], named_params)

    return named_params


def get_num_named_params(named_params):
    """
    Returns total (flattened) number of named parameters
    """
    flat_params = [p[1].flatten() for p in named_params]
    return len(torch.cat(flat_params, dim=0))


def random_hex_str(num_hex):
    """
    Returns (unseedable) random hex string of length num_hex (units of 2, rounds down on odd)
    """
    return binascii.b2a_hex(os.urandom(num_hex // 2)).decode("ascii")
