"""
Makes a lightweight copy of an entire MLflow experiment by downsampling the metrics

Params and tags are copied over, artifacts are NOT copied

Useful where MLflow struggles to load and display very large metric logs (timing out instead)
"""

import argparse
from mlflow import MlflowClient
from tqdm import tqdm
import os
from multiprocessing import Process

from mlf_utils.shared import get_run_paths, get_run_info, get_metas, to_param_seq, get_metric_keys


parser = argparse.ArgumentParser()

parser.add_argument("input_experiment_name", type=str, help="original (existing) experiment to input for downsample")

parser.add_argument("sample_period", type=int, help="downsampling period, according to MLflow steps")
# if a metric is already logged with a frequency <= 'sample_period' it will be copied unchanged
# NOTE this implicitly assumes all signal logging periods align with the 'sample_period' step, e.g. 100 period means logging at step 100, 200, 300...



def slim_run(tracking_uri, input_run_id, input_run_path, output_exp_id, tqdm_position=0):
    client = MlflowClient(tracking_uri)

    # create new run to hold downsample
    input_start_time = get_run_info(input_run_path)["start_time"]
    input_run_tags = get_metas(input_run_path, "tags")
    input_run_name = input_run_tags["mlflow.runName"]

    output_run = client.create_run(output_exp_id, input_start_time, input_run_tags, input_run_name)

    # log original run id as a tag
    client.set_tag(output_run.info.run_id, "full_data_run_id", input_run_id)

    # copy over input run parameters
    input_run_params = get_metas(input_run_path, "params")
    client.log_batch(output_run.info.run_id, params=to_param_seq(input_run_params))

    # for each input run metric key
    for mkey in tqdm(get_metric_keys(input_run_path), desc=input_run_name, position=tqdm_position):

        # load metric data
        metric_data = client.get_metric_history(input_run_id, mkey)  # NOTE this mlflow load works without blocking for a very long time

        # downsample metric data
        downsampled_data = list(filter(lambda m: m.step % args.sample_period == 0, metric_data))

        # save downsampled metric data to output experiment
        client.log_batch(output_run.info.run_id, metrics=downsampled_data)



if __name__ == "__main__":
    args = parser.parse_args()

    # using MLflowClient, which provides CRUD interface for managing several runs/experiments at once
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    client = MlflowClient(tracking_uri)

    # get input experiment
    input_exp = client.get_experiment_by_name(args.input_experiment_name)
    assert input_exp is not None, f"Input experiment '{args.input_experiment_name}' does not exist"

    # create lightweight output experiment
    output_experiment_name = f"SLIM_{args.sample_period}x_{args.input_experiment_name}"
    assert client.get_experiment_by_name(output_experiment_name) is None, f"Output experiment '{output_experiment_name}' already exists"
    output_exp_id = client.create_experiment(output_experiment_name)

    # launch a subprocess dedicated to each run in the input experiment
    run_procs = []
    for run_idx, (input_run_id, input_run_path) in enumerate(get_run_paths(tracking_uri, input_exp.experiment_id)):
        
        rp = Process(target=slim_run, args=(tracking_uri, input_run_id, input_run_path, output_exp_id, run_idx))
        rp.start()

        run_procs.append(rp)

    # join subprocesses
    [rp.join() for rp in run_procs]
