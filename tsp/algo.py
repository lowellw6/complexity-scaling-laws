import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np

from tsp.utils import get_costs, get_entropy, pad_safe_dist_gather, perm_shuffle
from tsp.logger import MLflowLogger as Logger


def iterate_mb_idxs(data_length, minibatch_size, shuffle=False):
    """
    Yields minibatches of indexes, to use as a for-loop iterator, with
    option to shuffle.

    Taken directly from rlpyt: https://github.com/astooke/rlpyt/blob/f04f23db1eb7b5915d88401fca67869968a07a37/rlpyt/utils/misc.py#L6
    """
    if shuffle:
        indexes = np.arange(data_length)
        np.random.shuffle(indexes)
    for start_idx in range(0, data_length - minibatch_size + 1, minibatch_size):
        batch = slice(start_idx, start_idx + minibatch_size)
        if shuffle:
            batch = indexes[batch]
        yield batch



class TspAlgoBase:
    """
    Base API for TSP RL algorithms.
    Expects a torch optimizer to be provided.
    """

    def __init__(self, optimizer, grad_norm_clip="inf", scheduler=None):
        self.optimizer = optimizer
        self.grad_norm_clip = grad_norm_clip
        self.scheduler = scheduler

    def optimize_agent(self, agent, problems):
        """
        TODO
        """
        agent_outputs = agent(problems)
        loss = self.loss(*agent_outputs)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm_(agent.parameters(), self.grad_norm_clip)
        self.optimizer.step()

        Logger.log("loss", loss.item())
        Logger.log("grad_norm", grad_norm.item())

        if self.scheduler is not None:
            Logger.log("learning_rate", self.scheduler.get_last_lr()[0])
            self.scheduler.step()


class TspReinforce(TspAlgoBase):
    """
    REINFORCE for TSP.

    Undiscounted in-between decisions,
    so each problem "episode" is treated
    as a factorized bandit problem.
    """

    def loss(self, selections, select_idxs, log_probs):
        tour_costs = get_costs(selections)

        sel_log_probs = pad_safe_dist_gather(log_probs, select_idxs, reset_val=0.0)
        tour_log_probs = torch.sum(sel_log_probs, dim=1)

        Logger.log_stat("train_cost", tour_costs)
        Logger.log_stat("log_pi", tour_log_probs.detach())
        Logger.log_stat("entropy", get_entropy(log_probs))

        return torch.mean(
            tour_costs * tour_log_probs
        )  # non-negative since using costs, not rewards


class TspA2C(TspReinforce):
    """
    Advantage Actor-Critic for TSP.

    A value is provided for each next-node
    decision based on all previous selection
    (the state in this context).

    The critic is trained to predict the
    future outcome of the current policy
    after rolling out the remaining selections.

    This is a factorized bandit problem,
    so no dicounting occurs. The value head
    learns the average tour cost generated
    from a given partial set of selections.
    """

    def __init__(self, optimizer, grad_norm_clip="inf", scheduler=None, critic_coeff=1.0):
        super().__init__(optimizer, grad_norm_clip=grad_norm_clip, scheduler=scheduler)
        self.critic_coeff = critic_coeff

    def loss(self, selections, select_idxs, log_probs, values):
        tour_costs = get_costs(selections)

        sel_log_probs = pad_safe_dist_gather(log_probs, select_idxs, reset_val=0.0)
        tour_log_probs = torch.sum(sel_log_probs, dim=1)  # only for logging here

        Logger.log_stat("train_cost", tour_costs)
        Logger.log_stat("log_pi", tour_log_probs.detach())
        Logger.log_stat("entropy", get_entropy(log_probs))

        tour_costs_repeated = tour_costs.unsqueeze(-1).repeat(1, values.shape[1])

        # decision-level advantages
        # (removing last index based on value predictions where all selections have been made)
        advantages = (tour_costs_repeated - values)[:, :-1]

        Logger.log_stat("value", values.detach())
        Logger.log_stat("advantage", advantages.detach())

        actor_loss = torch.mean(
            advantages.detach() * sel_log_probs
        )  # non-negative since using costs, not rewards
        critic_loss = F.mse_loss(values, tour_costs_repeated)

        Logger.log("actor_loss", actor_loss.item())
        Logger.log("critic_loss", critic_loss.item())

        return actor_loss + self.critic_coeff * critic_loss


class TspPPO(TspA2C):
    """
    Proximal Policy Optimization for TSP

    Compositional w.r.t. values and policy distributions
    for node-wise decisions like TspA2C

    Only distinctions are 1) the PPO loss
    and 2) support for epochs of minibatch updates
    """
    def __init__(
            self, 
            optimizer, 
            epochs=4,
            minibatches=4,
            ratio_clip=1.0,
            grad_norm_clip="inf",
            scheduler=None, 
            critic_coeff=1.0,
            only_state_values=False
        ):
        super().__init__(optimizer, grad_norm_clip=grad_norm_clip, scheduler=scheduler, critic_coeff=critic_coeff)
        
        self.epochs = epochs
        self.minibatches = minibatches
        self.ratio_clip = ratio_clip

        self.only_state_values = only_state_values  # if True, ablates compositional value usage in actor and critic loss, relying only on initial full-state value

    def optimize_agent(self, agent, problems):
        # gather pre-determined selection sequences for PPO minibatch updates (can't re-roll-out each time when comparing with old_log_probs in PPO objective)
        # and gather starting policy distributions for PPO importance sampling ratio
        with torch.no_grad():
            _, old_select_idxs, old_log_probs, old_values = agent(problems)

        # determine minibatch sample size
        batch_size = len(problems)
        mb_size = batch_size // self.minibatches

        for epch_num in range(self.epochs):

            for mb_num, mb_idxs in enumerate(iterate_mb_idxs(batch_size, mb_size, shuffle=True)):
                Logger.set_offset(epch_num * self.minibatches + mb_num - self.epochs * self.minibatches + 1)  # log in steps between previous and current iteration's step

                # extract minibatch of inputs for agent and loss
                problems_mb = problems[mb_idxs]
                select_idxs_mb = old_select_idxs[mb_idxs]
                old_log_probs_mb = old_log_probs[mb_idxs]
                old_values_mb = old_values[mb_idxs]

                # compute model outputs with gradient tracking
                selections, re_select_idxs, log_probs, values = agent.use(problems_mb, select_idxs_mb)

                # double check model took the same subdecision rollouts as old_select_idxs (very important)
                assert torch.allclose(select_idxs_mb.float(), re_select_idxs.float())

                # compute PPO clip loss
                loss = self.loss(selections, select_idxs_mb, log_probs, old_log_probs_mb, values, old_values_mb)

                # optimization step
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = clip_grad_norm_(agent.parameters(), self.grad_norm_clip)
                self.optimizer.step()

                # logging
                Logger.log("loss", loss.item())
                Logger.log("grad_norm", grad_norm.item())

        Logger.set_offset(0)

        if self.scheduler is not None:
            Logger.log("learning_rate", self.scheduler.get_last_lr()[0])
            self.scheduler.step()

    def loss(self, *args):
        if self.only_state_values:
            return self._state_values_loss(*args)
        else:
            return self._compositional_values_loss(*args)
        
    def _compositional_values_loss(self, selections, select_idxs, log_probs, old_log_probs, values, old_values):
        tour_costs = get_costs(selections)

        sel_log_probs = pad_safe_dist_gather(log_probs, select_idxs, reset_val=0.0)
        old_sel_log_probs = pad_safe_dist_gather(old_log_probs, select_idxs, reset_val=0.0)

        tour_log_probs = torch.sum(sel_log_probs, dim=1)  # only for logging here
        
        Logger.log_stat("train_cost", tour_costs)
        Logger.log_stat("log_pi", tour_log_probs.detach())
        Logger.log_stat("entropy", get_entropy(log_probs))
        Logger.log_stat("pi_drift", torch.exp(sel_log_probs.detach()) - torch.exp(old_sel_log_probs))
        Logger.log_stat("abs_pi_drift", torch.abs(torch.exp(sel_log_probs.detach()) - torch.exp(old_sel_log_probs)))

        tour_costs_repeated = tour_costs.unsqueeze(-1).repeat(1, old_values.shape[1])

        # decision-level advantages
        # (removing last index based on value predictions where all selections have been made)
        advantages = (tour_costs_repeated - old_values)[:, :-1]

        Logger.log_stat("value", values.detach())
        Logger.log_stat("advantage", advantages)
        Logger.log_stat("value_drift", values.detach() - old_values)
        Logger.log_stat("abs_value_drift", torch.abs(values.detach() - old_values))

        # compute PPO clip ratio and surrogate objective (old_log_probs should already have no tracked gradients, but detaching just in case)
        importance_ratio = torch.exp(sel_log_probs - old_sel_log_probs.detach())
        clipped_ratio = torch.clamp(importance_ratio, 1. - self.ratio_clip, 1. + self.ratio_clip)

        surr_1 = importance_ratio * advantages.detach()  # advantages should already not have tracked gradients, but detaching just to be safe
        surr_2 = clipped_ratio * advantages.detach()
        surrogate_objective = -torch.max(surr_1, surr_2)  # -max(-x, -y) == min(x, y) --> surrogates actually have sign flipped since we use costs, not rewards

        # compute PPO clip loss over non-pad (i.e. gradient-tracked actor and critic) decisions
        actor_loss = -torch.mean(surrogate_objective)
        critic_loss = F.mse_loss(values, tour_costs_repeated)

        Logger.log("actor_loss", actor_loss.item())
        Logger.log("critic_loss", critic_loss.item())

        return actor_loss + self.critic_coeff * critic_loss
    
    def _state_values_loss(self, selections, select_idxs, log_probs, old_log_probs, values, old_values):
        tour_costs = get_costs(selections)

        sel_log_probs = pad_safe_dist_gather(log_probs, select_idxs, reset_val=0.0)
        old_sel_log_probs = pad_safe_dist_gather(old_log_probs, select_idxs, reset_val=0.0)

        tour_log_probs = torch.sum(sel_log_probs, dim=1)  # only for logging here
        
        Logger.log_stat("train_cost", tour_costs)
        Logger.log_stat("log_pi", tour_log_probs.detach())
        Logger.log_stat("entropy", get_entropy(log_probs))
        Logger.log_stat("pi_drift", torch.exp(sel_log_probs.detach()) - torch.exp(old_sel_log_probs))
        Logger.log_stat("abs_pi_drift", torch.abs(torch.exp(sel_log_probs.detach()) - torch.exp(old_sel_log_probs)))

        tour_costs_repeated = tour_costs.unsqueeze(-1).repeat(1, old_values.shape[1])

        state_values = values[:, 0]  # 0th index corresponds to first value estimate only informed by state and start token
        old_state_values = old_values[:, 0]
        old_state_values_repeated = old_state_values.unsqueeze(-1).repeat(1, old_values.shape[1])

        # state-level advantages copied over each subdecision index
        advantages = (tour_costs_repeated - old_state_values_repeated)[:, :-1]

        Logger.log_stat("value", state_values.detach())
        Logger.log_stat("advantage", advantages[:, 0])
        Logger.log_stat("value_drift", state_values.detach() - old_state_values)
        Logger.log_stat("abs_value_drift", torch.abs(state_values.detach() - old_state_values))

        # compute PPO clip ratio and surrogate objective (old_log_probs should already have no tracked gradients, but detaching just in case)
        importance_ratio = torch.exp(sel_log_probs - old_sel_log_probs.detach())
        clipped_ratio = torch.clamp(importance_ratio, 1. - self.ratio_clip, 1. + self.ratio_clip)

        surr_1 = importance_ratio * advantages.detach()  # advantages should already not have tracked gradients, but detaching just to be safe
        surr_2 = clipped_ratio * advantages.detach()
        surrogate_objective = -torch.max(surr_1, surr_2)  # -max(-x, -y) == min(x, y) --> surrogates actually have sign flipped since we use costs, not rewards

        # compute PPO clip loss over non-pad (i.e. gradient-tracked actor and critic) decisions
        actor_loss = -torch.mean(surrogate_objective)
        critic_loss = F.mse_loss(state_values, tour_costs)

        Logger.log("actor_loss", actor_loss.item())
        Logger.log("critic_loss", critic_loss.item())

        return actor_loss + self.critic_coeff * critic_loss
    

class TspPPOGreedyRollout(TspPPO):
    """
    Modifies TspPPO to use greedy rollouts from a critic policy as baseline value estimations.
    
    Similar to baseline algorithm used in Attention, Learn to Solve Routing Problems! (ICLR 2019)
    https://arxiv.org/pdf/1803.08475

    Notable differences include... (both are handled in TspTrainerRolloutCritic class)
        1) Not using paired t-test significance as a condition to update baseline policy state dict,
           rather simply updating if the held-out evaluation performs better to any degree
        2) Not regenerating the held-out evaluation dataset when updating the baseline policy,
           so one should use a larger dataset size (e.g. 100K) to help avoid overfitting
    """
    def __init__(
        self, 
        optimizer, 
        epochs=4,
        minibatches=4,
        ratio_clip=1.0,
        grad_norm_clip="inf",
        scheduler=None
        ):
        super().__init__(
            optimizer, 
            epochs=epochs, 
            minibatches=minibatches, 
            ratio_clip=ratio_clip, 
            grad_norm_clip=grad_norm_clip, 
            scheduler=scheduler, 
            critic_coeff=None,  # N/A here
            only_state_values=False  # ditto
        )

    def optimize_agent(self, agent, problems):
        agent.model.spur_critic()  # activates critic rollouts only for first forward pass where old log probs and values are produced
        super().optimize_agent(agent, problems)

    def loss(self, selections, select_idxs, log_probs, old_log_probs, crt_sel, old_crt_sel):
        """
        Trailing value terms are replaced with model's critic selections (rollouts)
        Note in the inherited optimize_agent() function these are still called values

        crt_sel is empty as we avoid redundant critic rollouts for efficiency, only generating
        old_crt_sel via the spur_critic() trigger
        """
        assert crt_sel.numel() == 0

        tour_costs = get_costs(selections)

        sel_log_probs = pad_safe_dist_gather(log_probs, select_idxs, reset_val=0.0)
        old_sel_log_probs = pad_safe_dist_gather(old_log_probs, select_idxs, reset_val=0.0)

        tour_log_probs = torch.sum(sel_log_probs, dim=1)  # NOTE not just for logging here, used in actor loss
        old_tour_log_probs = torch.sum(old_sel_log_probs, dim=1)
        
        Logger.log_stat("train_cost", tour_costs)
        Logger.log_stat("log_pi", tour_log_probs.detach())
        Logger.log_stat("entropy", get_entropy(log_probs))
        Logger.log_stat("pi_drift", torch.exp(sel_log_probs.detach()) - torch.exp(old_sel_log_probs))
        Logger.log_stat("abs_pi_drift", torch.abs(torch.exp(sel_log_probs.detach()) - torch.exp(old_sel_log_probs)))

        state_values = get_costs(old_crt_sel)

        # state-level advantages
        advantages = tour_costs - state_values

        Logger.log_stat("value", state_values.detach())
        Logger.log_stat("advantage", advantages.detach())

        # compute PPO clip ratio and surrogate objective (old_log_probs should already have no tracked gradients, but detaching just in case)
        importance_ratio = torch.exp(tour_log_probs - old_tour_log_probs.detach())
        clipped_ratio = torch.clamp(importance_ratio, 1. - self.ratio_clip, 1. + self.ratio_clip)

        surr_1 = importance_ratio * advantages.detach()  # advantages should already not have tracked gradients, but detaching just to be safe
        surr_2 = clipped_ratio * advantages.detach()
        surrogate_objective = -torch.max(surr_1, surr_2)  # -max(-x, -y) == min(x, y) --> surrogates actually have sign flipped since we use costs, not rewards

        # compute PPO clip loss over non-pad (i.e. gradient-tracked actor) decisions
        actor_loss = -torch.mean(surrogate_objective)

        Logger.log("actor_loss", actor_loss.item())

        return actor_loss


class TspSupervisedBase(TspAlgoBase):
    """
    Base API for TSP SL algorithms.
    """

    def optimize_agent(self, agent, problems):
        """
        TODO
        """
        data, labels = problems
        agent_outputs = agent.use(data, labels)
        loss = self.loss(*agent_outputs)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm_(agent.parameters(), self.grad_norm_clip)
        self.optimizer.step()

        Logger.log("loss", loss.item())
        Logger.log("grad_norm", grad_norm.item())

        if self.scheduler is not None:
            Logger.log("learning_rate", self.scheduler.get_last_lr()[0])
            self.scheduler.step()


class TspActorCriticSupervised(TspSupervisedBase):
    """
    Actor-Critic for supervised TSP.

    Critic is unchanged from RL and tries to predict
    the undiscounted return of the actor policy.
    Actor is now trained through a NLL loss on
    the ground truth labels for each decision.
    """

    def __init__(self, optimizer, grad_norm_clip="inf", scheduler=None, critic_coeff=1.0):
        super().__init__(optimizer, grad_norm_clip=grad_norm_clip, scheduler=scheduler)
        self.critic_coeff = critic_coeff

    def loss(self, selections, labels, log_probs, values):
        tour_costs = get_costs(selections)

        tour_costs_repeated = tour_costs.unsqueeze(-1).repeat(1, values.shape[1])

        Logger.log_stat("train_cost", tour_costs)
        Logger.log_stat("value", values.detach())

        flat_log_probs = log_probs.reshape(-1, log_probs.shape[-1])
        actor_loss = F.nll_loss(flat_log_probs, labels.flatten())

        critic_loss = F.mse_loss(values, tour_costs_repeated)

        Logger.log("actor_loss", actor_loss.item())
        Logger.log("critic_loss", critic_loss.item())

        return actor_loss + self.critic_coeff * critic_loss


class TspAcMinibatchSupervised(TspActorCriticSupervised):
    """
    Performs several epochs of SFT.
    """

    def __init__(self, optimizer, epochs=4, minibatches=4, **kwargs):
        super().__init__(optimizer, **kwargs)
        self.epochs = epochs
        self.minibatches = minibatches

    def optimize_agent(self, agent, problems):
        data, labels = problems

        batch_size = len(data)
        mb_size = batch_size // self.minibatches

        for epch_num in range(self.epochs):

            for mb_num, mb_idxs in enumerate(iterate_mb_idxs(batch_size, mb_size, shuffle=True)):
                Logger.set_offset(epch_num * self.minibatches + mb_num - self.epochs * self.minibatches + 1)  # log in steps between previous and current iteration's step

                data_mb = data[mb_idxs]
                labels_mb = labels[mb_idxs]

                agent_outputs = agent.use(data_mb, labels_mb)
                loss = self.loss(*agent_outputs)

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = clip_grad_norm_(agent.parameters(), self.grad_norm_clip)
                self.optimizer.step()

                Logger.log("loss", loss.item())
                Logger.log("grad_norm", grad_norm.item())

        Logger.set_offset(0)

        if self.scheduler is not None:
            Logger.log("learning_rate", self.scheduler.get_last_lr()[0])
            self.scheduler.step()
