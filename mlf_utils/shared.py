"""
Custom MLflow utilities, mainly related to side-stepping where
MLflow struggles to load and display very large metric logs (timing out instead)

Much of this will only work with a local /mlruns path
"""


import mlflow
import os
import os.path as osp
from urllib.parse import urlparse
import yaml



def get_run_paths(tracking_uri, exp_id):
    """
    Non-blocking function to get run paths for a mlflow.entities.Experiment
    mlflow.search_runs() blocks trying to statically load large quantities of metric data

    Returns tuples (run_id, full_run_path)
    """
    exp_path = osp.join(urlparse(tracking_uri).path, exp_id)
    run_ids = filter(lambda x: osp.isdir(osp.join(exp_path, x)), os.listdir(exp_path))
    return [(rid, osp.join(exp_path, rid)) for rid in run_ids]


def get_run_info(run_path):
    """Get meta.yaml info directly from run_path, no mlflow involved"""
    meta_path = osp.join(run_path, "meta.yaml")
    
    with open(meta_path, 'r') as f:
        meta_content = yaml.safe_load(f)

    return meta_content


def get_metas(run_path, meta_name):
    """Get tags or params directly from run_path, no mlflow involved"""
    metas_path = osp.join(run_path, meta_name)
    metas_dict = dict()
    
    for mname, mpath in [(meta_name, osp.join(metas_path, meta_name)) for meta_name in os.listdir(metas_path)]:
        with open(mpath, 'r') as f:
            metas_dict[mname] = f.read()

    return metas_dict


def to_param_seq(param_dict):
    return [mlflow.entities.Param(k, v) for k, v in param_dict.items()]


def get_metric_keys(run_path):
    """Get metric keys directly from run_path, no mlflow involved"""
    metric_path = osp.join(run_path, "metrics")
    return os.listdir(metric_path)


def find_run(tracking_uri, run_id):
    """Get experiment id and corresponding experiment path and run path from run id (or None if not found), no mlflow involved"""
    mlf_root = urlparse(tracking_uri).path
    exp_ids = filter(lambda x: osp.isdir(osp.join(mlf_root, x)) and x != "models", os.listdir(mlf_root))
    
    for exp_id in exp_ids:
        exp_path = osp.join(urlparse(tracking_uri).path, exp_id)
        run_ids = filter(lambda x: osp.isdir(osp.join(exp_path, x)), os.listdir(exp_path))

        if run_id in run_ids:
            run_path = osp.join(exp_path, run_id)
            return exp_id, exp_path, run_path
        
    return None
