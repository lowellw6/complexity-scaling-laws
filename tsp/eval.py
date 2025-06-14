import torch
import torch.nn.functional as F
from itertools import permutations
import multiprocessing as mp
import copy
import math
from tqdm import tqdm

from tsp.utils import get_coords, get_costs, batch_select_gather, seed_rand_gen, pad_safe_dist_gather


def batched_eval(agent, problems, batch_size=None):
    """
    Generate solutions and return corresponding costs.
    Allows for batched forward pass over problems, which is useful for large
    evaluation routines which need to happen on GPU.
    Omit 'batch_size' param to generate all tour costs in one batch.
    """
    if batch_size is None:
        batch_size = len(problems)

    batches = torch.split(problems, batch_size, dim=0)

    costs = torch.empty(0)
    for batch in batches:
        b_solutions, _ = agent.solve(batch)
        b_costs = get_costs(b_solutions)
        costs = torch.cat((costs, b_costs), dim=0)

    return costs


def batched_eval_repeat(agent, problems, repeats, batch_size=None):
    """
    Like 'batched_eval' but repeated 'repeats' times for each problem.
    Output costs have shape (repeats, num_samples, problem_size, 2).
    """
    assert type(repeats) is int and repeats >= 1
    num_problems = problems.shape[0]

    problems = torch.cat([problems] * repeats, dim=0)
    costs = batched_eval(agent, problems, batch_size)
    split_costs = torch.split(costs, num_problems, dim=0)
    repeat_costs = torch.stack(split_costs, dim=0)

    return repeat_costs


def evaluate_agent(agent, problem_size, num_samples, batch_size=None, best_of=1):
    """
    Evaluate agent on generated problems with fixed number of nodes.
    Omit 'batch_size' param to run all 'num_samples' evalutations in one batch.

    Setting 'best_of' > 1 repeats each problem that many times and reports
    the lowest costs across these repetitions. This is only useful
    if the agent selection strategy is non-deterministic.
    """
    problems = get_coords(num_samples, problem_size)

    if best_of > 1:
        repeat_costs = batched_eval_repeat(
            agent, problems, best_of, batch_size=batch_size
        )
        costs, _ = torch.min(repeat_costs, dim=0)
    else:
        costs = batched_eval(agent, problems, batch_size)

    min_cost = costs.min().item()
    max_cost = costs.max().item()
    avg_cost = costs.mean().item()
    std_cost = costs.std().item()

    return min_cost, max_cost, avg_cost, std_cost


def parallel_eval_agent(agent, problem_size, num_samples, batch_size, num_workers):
    """
    TODO
    """
    assert num_workers >= 1

    def worker_routine(barrier, queue, seed, itr, agent, problem_size, batch_size):
        seed_rand_gen(seed)
        for _ in range(itr):
            problems = get_coords(batch_size, problem_size)
            solutions, _ = agent.solve(problems)
            costs = get_costs(solutions)
            queue.put(costs)
        barrier.wait()  # not waiting for main process here breaks queue, but may be a better way to do this

    queue = mp.Queue()
    barriers = [mp.Barrier(2) for _ in range(num_workers)]
    itr = math.ceil(num_samples / (batch_size * num_workers))

    worker_kwargs = [
        dict(
            barrier=barriers[idx],
            queue=queue,
            seed=idx + 1,
            itr=itr,
            agent=copy.deepcopy(agent),  # expensive if model is large
            problem_size=problem_size,
            batch_size=batch_size,
        )
        for idx in range(num_workers)
    ]

    workers = [
        mp.Process(target=worker_routine, kwargs=wk_kwargs)
        for wk_kwargs in worker_kwargs
    ]

    for p in workers:
        p.start()

    costs = torch.empty(0)
    for _ in tqdm(range(num_workers * itr)):
        sample = queue.get()
        costs = torch.cat((costs, sample), dim=0)

    for idx, p in enumerate(workers):
        barriers[idx].wait()
        p.join()

    min_cost = costs.min().item()
    max_cost = costs.max().item()
    avg_cost = costs.mean().item()
    std_cost = costs.std().item()

    return min_cost, max_cost, avg_cost, std_cost


def oracle(problem):
    """
    Returns exact solution with corresponding
    tour indices and cost. Note this is O(n!)
    with problem size!

    Only accepts a single problem since
    this is SLOW and there is no numpy or
    torch support for batched permutations.
    Use sparingly.
    """
    assert len(problem.shape) == 2  # S, 2
    assert isinstance(problem, torch.Tensor)

    problem_size = len(problem)
    if problem_size > 10:
        print(
            "WARNING: attempting exact solve of TSP using brute-force method "
            f"with large problem size = {problem_size}"
        )
        print("This becomes exceptionally slow after around 10 nodes")

    perms = torch.tensor(list(permutations(torch.arange(problem_size).tolist())))

    problem_stack = torch.stack([problem] * len(perms), dim=0)
    problem_perm_list = [
        batch_select_gather(problem_stack, perms[:, step_idx])
        for step_idx in range(problem_size)
    ]
    problem_perms = torch.cat(problem_perm_list, dim=1)

    costs = get_costs(problem_perms)
    min_cost, min_idx = torch.min(costs, dim=0)

    best_tour = problem_perms[min_idx]
    best_perm = perms[min_idx]

    return best_tour, best_perm, min_cost


@torch.no_grad()
def ppo_actor_loss_offline(selections, values):
    """
    Compositional PPO actor loss for offline eval
    (Just the advantages w/o minibatching as clip ratio = 1 for first update --> so no need for log_probs)
    No reduction occurs (neither batch nor seq dim) so this can be used in batched eval
    """
    tour_costs = get_costs(selections)
    tour_costs_repeated = tour_costs.unsqueeze(-1).repeat(1, values.shape[1])

    # decision-level advantages
    # (removing last index based on value predictions where all selections have been made)
    advantages = (tour_costs_repeated - values)[:, :-1]
    
    return advantages.detach()  # non-negative since using costs, not rewards


@torch.no_grad()
def nll_actor_loss_offline(log_probs, labels):
    """
    Supervised NLL loss version of offline actor loss
    """
    flat_log_probs = log_probs.reshape(-1, log_probs.shape[-1])
    actor_loss = F.nll_loss(flat_log_probs, labels.flatten(), reduction="none")
    return actor_loss.reshape(*labels.shape)


@torch.no_grad()
def critic_loss_offline(selections, values):
    """
    Compositional critic MSE loss for offline eval
    No reduction occurs (neither batch nor seq dim) so this can be used in batched eval
    """
    tour_costs = get_costs(selections)
    tour_costs_repeated = tour_costs.unsqueeze(-1).repeat(1, values.shape[1])
    return F.mse_loss(values, tour_costs_repeated, reduction="none").detach()


@torch.no_grad()
def ppo_actor_critic_loss(selections, log_probs, values, labels=None):
    """
    Combined function of ppo_actor_loss_offline and critic_loss_offline
    (saves an extra call to get_costs)

    labels does nothing and just allows matching API with nll_actor_critic_loss()
    """
    tour_costs = get_costs(selections)
    tour_costs_repeated = tour_costs.unsqueeze(-1).repeat(1, values.shape[1])

    # decision-level advantages
    # (removing last index based on value predictions where all selections have been made)
    advantages = (tour_costs_repeated - values)[:, :-1]
    
    actor_loss = advantages.detach()  # non-negative since using costs, not rewards
    critic_loss = F.mse_loss(values, tour_costs_repeated, reduction="none").detach()

    return actor_loss, critic_loss, tour_costs


@torch.no_grad()
def nll_actor_critic_loss(selections, log_probs, values, labels):
    """
    Convenience to match API of ppo_actor_critic_loss()
    """
    tour_costs = get_costs(selections)
    tour_costs_repeated = tour_costs.unsqueeze(-1).repeat(1, values.shape[1])
    
    actor_loss = nll_actor_loss_offline(log_probs, labels)
    critic_loss = F.mse_loss(values, tour_costs_repeated, reduction="none").detach()

    return actor_loss, critic_loss, tour_costs


def batched_eval_with_loss(agent, problems, batch_size=None):
    """
    batched_eval that also returns actor and critic loss evals
    """
    raise Exception("DEPRECATED see batched_eval_with_loss_and_solutions() which correctly handles actor loss for both DRL and SFT")

    if batch_size is None:
        batch_size = len(problems)

    batches = torch.split(problems, batch_size, dim=0)

    costs = torch.empty(0)
    actor_losses = torch.empty(0)
    critic_losses = torch.empty(0)
    
    for batch in batches:
        with torch.no_grad():
            b_solutions, _, _, b_values = agent(batch)
        
        b_actor_losses, b_critic_losses, b_costs = actor_critic_loss(b_solutions, b_values)
        
        costs = torch.cat((costs, b_costs), dim=0)
        actor_losses = torch.cat((actor_losses, b_actor_losses), dim=0)
        critic_losses = torch.cat((critic_losses, b_critic_losses), dim=0)

    return costs, actor_losses, critic_losses


def batched_eval_with_loss_and_solutions(agent, problems, batch_size=None, sol_per_problem=1, algo="ppo", sup_labels=None):
    """
    batched_eval_with_loss that also returns solution selection indexes
    Also adds option for >1 solution per problem
    All tensors returned have shape (len(problems), sol_per_problem, *feature_dims)

    algo in (ppo, supervised) which toggles the actor loss accordingly
    """
    assert algo in ("ppo", "supervised")
    if algo == "supervised":
        assert sup_labels is not None

    if batch_size is None:
        batch_size = len(problems)

    batches = torch.split(problems, batch_size, dim=0)
    label_batches = torch.split(sup_labels, batch_size, dim=0) if algo == "supervised" else len(batches) * [None]

    selections = torch.empty(0)
    costs = torch.empty(0)
    actor_losses = torch.empty(0)
    critic_losses = torch.empty(0)
    
    for batch, labels in zip(batches, label_batches):

        l_selections = []
        l_costs = []
        l_actor_losses = []
        l_critic_losses = []

        for _ in range(sol_per_problem):
            if algo == "ppo":
                with torch.no_grad():
                    s_solutions, s_selections, s_log_probs, s_values = agent(batch)
                s_actor_losses, s_critic_losses, s_costs = ppo_actor_critic_loss(s_solutions, s_log_probs, s_values)
            else:  # algo == "supervised"
                with torch.no_grad():
                    s_solutions, _, s_log_probs, s_values = agent.use(batch, labels)
                s_actor_losses, s_critic_losses, _ = nll_actor_critic_loss(s_solutions, s_log_probs, s_values, labels)

                with torch.no_grad():
                    s_solutions, s_selections, _, _ = agent(batch)  # on-policy costs --> teacher-forced solutions from agent.use() are optimal tours
                s_costs = get_costs(s_solutions)

            l_selections.append(s_selections)
            l_costs.append(s_costs)
            l_actor_losses.append(s_actor_losses)
            l_critic_losses.append(s_critic_losses)

        b_selections = torch.stack(l_selections, dim=1)
        b_costs = torch.stack(l_costs, dim=1)
        b_actor_losses = torch.stack(l_actor_losses, dim=1)
        b_critic_losses = torch.stack(l_critic_losses, dim=1)
        
        selections = torch.cat((selections, b_selections), dim=0)
        costs = torch.cat((costs, b_costs), dim=0)
        actor_losses = torch.cat((actor_losses, b_actor_losses), dim=0)
        critic_losses = torch.cat((critic_losses, b_critic_losses), dim=0)

    return selections, costs, actor_losses, critic_losses
