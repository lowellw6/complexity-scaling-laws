import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
import optuna

from tsp.logger import MLflowLogger as Logger
from tsp.datagen import TspDataset
from tsp.eval import batched_eval
from tsp.agent import TspAgent


class TspTrainer:
    """
    Root object for training TSP agents via RL or SFT
    """

    def __init__(self, dataset, agent, algo, eval_datasets=None):
        """
        Provide TSP data, agent, and algorithm.
        """
        self.dataset = dataset
        self.agent = agent
        self.algo = algo

        self.eval_data = []
        for name, eval_dataset in eval_datasets:
            if isinstance(eval_dataset, TspDataset):
                self.eval_data.append((name, torch.stack(eval_dataset[:], dim=0)))
            elif isinstance(eval_dataset, torch.Tensor):
                self.eval_data.append((name, eval_dataset))
            elif eval_dataset is not None:
                raise ValueError(
                    f"Unrecognized eval_dataset type '{type(eval_dataset)}'"
                )

    def _offline_eval_routine(self, iteration, eval_period, eval_batch_size):
        """
        Run offline evaluation using all provided eval datasets.
        """
        if eval_period is not None and iteration % eval_period == 0:
            for name, problems in self.eval_data:
                eval_costs = batched_eval(
                    self.agent, problems, batch_size=eval_batch_size
                )
                Logger.log_stat(f"eval_cost_{name}", eval_costs)

    def start(
        self,
        epochs,
        batch_size,
        num_workers=0,
        eval_period=None,
        eval_batch_size=None,
        check_period=None,
        init_logger_step=1,
        logger_step_multiplier=1
    ):
        """
        Start training agent using algo and TSP dataset.

        epochs: number of times to cycle over the training datset
        batch_size: number of TSP tours to train with for one algo optimization step
        num_workers: number of subprocesses to spawn for dataloading during training (0 for main process only)
        eval_period: period between offline agent evaluation (None for no offline eval)
        eval_batch_size: number of TSP tours to use during offline agent evaluation (None defaults to training batch_size)
        check_period: checkpoint period in units of algo optimization steps (None default skips checkpoint logging)
        init_logger_step: what Logger.step to begin on (useful when resuming runs)
        logger_step_multiplier: each trainer iteration jumps this many steps, allowing signals logging more frequently to populate the steps in between
        """
        dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        print(
            f"Training for {epochs} epochs of {len(dataloader)} algorithm iterations with {batch_size} batch-size"
        )

        if eval_period is not None:
            assert (
                self.eval_data is not None
            ), "Eval period set but no eval dataset provided during init"
            eval_batch_size = (
                eval_batch_size if eval_batch_size is not None else batch_size
            )
            print(
                f"Evaluating offline every {eval_period} iterations with {eval_batch_size} batch-size"
            )

        iteration = init_logger_step
        start_time = time()
        for epoch_idx in range(epochs):

            for batch_idx, problem_batch in tqdm(enumerate(dataloader)):
                # update logger index (must happen before logging calls inside train routine)
                Logger.step = iteration * logger_step_multiplier

                # train routine
                self.agent.train_mode()
                self.algo.optimize_agent(self.agent, problem_batch)

                # (offline) eval routine
                self.agent.eval_mode()
                self._offline_eval_routine(iteration, eval_period, eval_batch_size)

                # log writing, checkpoint saving, etc.
                Logger.log("iteration", iteration)
                Logger.log("time", time() - start_time)

                if check_period is not None and iteration % check_period == 0:
                    Logger.save_checkpoint(
                        iteration, 
                        self.agent.state_dict(), 
                        self.algo.optimizer.state_dict(), 
                        self.algo.scheduler.state_dict() if self.algo.scheduler is not None else None
                    )

                iteration += 1

        # always save a checkpoint at the end of training, unless the final iteration already did or checkpoint logging is turned off
        if check_period is not None and (iteration - 1) % check_period != 0:
            Logger.save_checkpoint(
                iteration - 1, 
                self.agent.state_dict(), 
                self.algo.optimizer.state_dict(),
                self.algo.scheduler.state_dict() if self.algo.scheduler is not None else None
            )



class HpoTspTrainer(TspTrainer):
    """
    Optuna-enabled child of TspTrainer for hyperparameter optimization (HPO)
    """
    def __init__(self, trial, train_dataset, val_dataset, agent, algo, trial_step_multiplier=1):
        """
        trial Optuna object added
        val_dataset (validation dataset) takes place of eval_dataset and is now required
        trial_step_multiplier is the trial iteration multiplier (e.g. # of gradient updates per algorithm iteration, like logger_step_multiplier in TspTrainer.start())
        """
        self.trial = trial
        self.dataset = train_dataset
        self.agent = agent
        self.algo = algo
        self.trial_step_multiplier = trial_step_multiplier

        if isinstance(val_dataset, TspDataset):
            self.eval_data = torch.stack(val_dataset[:], dim=0)
        elif isinstance(val_dataset, torch.Tensor):
            self.eval_data = val_dataset
        else:
            raise ValueError(f"Unrecognized val_dataset type '{type(val_dataset)}'")
        
        # per-trial mlflow logging currently unsupported (each Logger needs it's own process which may not be the case here)
        if not Logger.setup:
            Logger.dummy_init()

    def _offline_eval_routine(self, iteration, eval_period, eval_batch_size):
        """
        Report mean validation cost to Optuna trial object and prune this trial if specified
        """
        assert eval_period is not None, "Must have positive eval_period for validation tracking in Optuna"

        if iteration % eval_period == 0:
            val_costs = batched_eval(
                self.agent, self.eval_data, batch_size=eval_batch_size
            )

            avg_val_cost = val_costs.mean().item()
            trial_step = iteration * self.trial_step_multiplier
            
            self.trial.report(avg_val_cost, trial_step)

            self.last_avg_val_cost = avg_val_cost

            # prune based on last reported value
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
    def start(self, *args, **kwargs):
        """Return last mean validation cost after training trial completes"""
        super().start(*args, **kwargs)
        return getattr(self, "last_avg_val_cost", None)


class TspTrainerRolloutCritic(TspTrainer):
    """
    Extends TspTrainer for use with rollout critic algorithms,
    overwriting the rollout critic policy with the current actor
    policy when its score on the held-out eval data improves.
    """

    def __init__(self, dataset, agent, algo, eval_datasets=None):
        super().__init__(dataset, agent, algo, eval_datasets)
        assert len(self.eval_data) == 1, "Using multiple eval datasets makes the rollout critic update condition ambiguous and is not supported"

        self.critic_cost = None

    def _offline_eval_routine(self, iteration, eval_period, eval_batch_size):
        if self.critic_cost is None:  # runs once first iteration
            name, problems = self.eval_data[0]
            baseline_agent = TspAgent(self.agent.model.critic)

            baseline_eval_costs = batched_eval(baseline_agent, problems, batch_size=eval_batch_size)
            self.critic_cost = baseline_eval_costs.mean().item()
    
        if eval_period is not None and iteration % eval_period == 0:  # runs every evaluation period
            name, problems = self.eval_data[0]

            eval_costs = batched_eval(self.agent, problems, batch_size=eval_batch_size)
            Logger.log_stat(f"eval_cost_{name}", eval_costs)

            actor_cost = eval_costs.mean().item()
            if actor_cost < self.critic_cost:
                self.agent.model.sync_baseline()
                
                self.critic_cost = actor_cost
                Logger.log("baseline_sync_itrs", iteration)
