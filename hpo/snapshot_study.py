
import os.path as osp
import pickle
import json

import optuna
from optuna.trial import TrialState
from optuna.storages import JournalStorage, JournalFileStorage, JournalRedisStorage
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner, MedianPruner



STUDY_NAME = "grad_clip_BOHB_1000-min_100000-max_3-reduction_50n_drl_hpo"
STUDY_PATH = "/home/lowell6/scale-fpc/hpo/storage/grad_clip_BOHB_1000-min_100000-max_3-reduction_50n_drl_hpo"
REDIS_STORAGE_URL = "redis://default:H6upctK2hrMleacjdzNeXHpEb57EvOVn@redis-19726.c326.us-east-1-3.ec2.cloud.redislabs.com:19726"  # set as None if local journal.log already exists and we can skip fetching from Redis cloud


if __name__ == "__main__":
    
    local_storage_path = osp.join(STUDY_PATH, "journal.log")

    if REDIS_STORAGE_URL is not None:
        assert not osp.exists(local_storage_path), "Stopping to not overwrite existing journal.log!"
        
        cloud_storage = JournalStorage(JournalRedisStorage(REDIS_STORAGE_URL))
        redis_logs = cloud_storage._backend.read_logs(0)  # start from first log and read until end

        local_storage = JournalStorage(JournalFileStorage(local_storage_path))
        local_storage._backend.append_logs(redis_logs)

    else:
        assert osp.exists(local_storage_path), "journal.log must already exist at STUDY_PATH if not pulling from Redis cloud!"
        local_storage = JournalStorage(JournalFileStorage(local_storage_path))

    sampler = TPESampler(multivariate=True)
    pruner = HyperbandPruner()

    study = optuna.load_study(study_name=STUDY_NAME, storage=local_storage, sampler=sampler, pruner=pruner)

    with open(osp.join(STUDY_PATH, "study.pkl"), "wb") as f:
        pickle.dump(study, f)

    # save high-level results json alongside frozen study
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    results_dict = dict(
        total_trials=len(study.trials),
        pruned_trials=len(pruned_trials),
        complete_trials=len(complete_trials)
    )

    if len(pruned_trials) + len(complete_trials) > 0:
        best_trial = study.best_trial
        results_dict["best_trial_value"] = best_trial.value
        results_dict["best_trial_number"] = best_trial.number
        results_dict["best_trial_params"] = best_trial.params

    with open(osp.join(STUDY_PATH, "results.json"), "w") as f:
        json.dump(results_dict, f, indent=4)

    # save full rankings json as well
    trial_outcomes = [(t.value, t.number, t.params) for t in complete_trials + pruned_trials]
    ranked_outcomes = sorted(trial_outcomes, key=lambda x: x[0])

    with open(osp.join(STUDY_PATH, "trial_rankings.json"), "w") as f:
        json.dump(ranked_outcomes, f, indent=4)
