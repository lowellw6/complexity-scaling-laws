import mlflow
import numpy as np
import torch
import os
import os.path as osp
from urllib.parse import urlparse
import pickle
import ast

import tsp
from tsp.utils import random_hex_str

from mlf_utils.shared import find_run


def parse_log_sig(mlflow_logging_signature):
    """
    Parses a string one of two formats...
        For starting a new run: <MLflow_experiment_group>/<MLflow_run_name>
        For resuming an existing run: resume/<run_id>

    Returns the corresponding pieces along with the env variables of tracking uri, and artifact location when starting a new run,
    preceeded by a True signal for a resume, and a False signal for 'not resuming'

    NOTE this means 'resume' is reserved as a special logging signature keyword, and cannot be used as an experiment name
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    # logs inside /mlruns in repo root directory if MLFLOW_TRACKING_URI environment variable doesn't exist
    if tracking_uri is None:
        tracking_uri = osp.join("file://" + osp.dirname(osp.dirname(tsp.__file__)), "mlruns")

    try:
        if mlflow_logging_signature.startswith("resume"):
            run_id = mlflow_logging_signature.split("/")[1]

            return True, (tracking_uri, run_id)

        else:
            experiment_group, run_name = mlflow_logging_signature.split("/")
            artifact_location = os.getenv("MLFLOW_ARTIFACT_LOCATION")  # mlflow defaults to location within tracking_uri if not provided

            return False, (tracking_uri, experiment_group, run_name, artifact_location) 

    except:
        print("Invalid mlflow_logging_signature")
        print("Format for starting a new run: <MLflow_experiment_group>/<MLflow_run_name>")
        print("Format for resuming an existing run: resume/<MLflow_run_id>")
        quit()


def module_to_locals_dict(module):
    """
    Converts a python module to a dictionary of its local (non-private) variables,
    which can be useful for hyperparameter logging. But this is a heavy-weight approach,
    e.g. it will grab all non-builtin imports too.
    """
    public_locals = [var for var in dir(module) if not var.startswith("__")]
    return { key : getattr(module, key) for key in public_locals }



class MLflowLogger:
    """
    A global logger class that can be called anywhere.
    Wraps MLflow's API.

    Logger must be started or resumed, but not both, before use
    """

    prefix: str = None
    step: int = 0
    offset: int = 0
    setup: bool = False
    dummy: bool = False  # dummy initialization --> skipping most logging routines
    debug: bool = False  # debug mode toggled on --> no functionality change, reroutes run_id_map to DEBUG_run_id_map
    run_id_map_dir : str = "run_id_map"
    debug_run_id_map_dir : str = "DEBUG_run_id_map"

    @classmethod
    def start(cls, mlf_tracking_uri, mlf_experiment_group, mlf_run_name, mlf_artifact_location=None):
        """
        Start a new MLflow run (and possibly a new experiment)

        If mlf_artifact_location is None, the default location is used within mlf_tracking_uri
        """
        if cls.setup:
            raise Exception("Global Logger already initialized!")
                
        try:
            mlflow.set_tracking_uri(mlf_tracking_uri)

            existing_mlf_exp = mlflow.get_experiment_by_name(mlf_experiment_group)

            # explicitly check if experiment exists so we can set artifact location when creating a new one
            if existing_mlf_exp is None:
                mlflow.create_experiment(mlf_experiment_group, artifact_location=mlf_artifact_location)

            elif mlf_artifact_location is not None and mlf_artifact_location != existing_mlf_exp.artifact_location:
                print(f"WARNING: '{mlf_experiment_group}' experiment exists @ tracking URI {mlf_tracking_uri}")
                print("Using existing artifact location rather than the one newly provided")
                
            mlflow.set_experiment(mlf_experiment_group)                    

            run = mlflow.start_run(run_name=mlf_run_name)

        except Exception as e:
            print("MLflow initialization error:", e)
            quit()

        print("\nMLFLOW TRACKING LAUNCHED")
        print("Experiment, run name, experiment id, run id, tracking URI, artifact location:\n")
        print(mlf_experiment_group)
        print(mlf_run_name)
        print(run.info.experiment_id)
        print(run.info.run_id)
        print(mlf_tracking_uri)
        print(run.info.artifact_uri, '\n')

        cls.setup = True

    @classmethod
    def resume(cls, mlf_tracking_uri, mlf_run_id, itr_key=None):
        """
        Resume an existing MLflow run with the given run id
        
        Searching for run id from run and experiment names is not implemented
        since neither field is necessarily unique

        If itr_key is provided, this returns the latest iteration index
        e.g. to inform model checkpoint loading. itr_key's value is used 
        to search for this in the run metrics as MLflow doesn't track 
        this itself because it supports inconsistent logging intervals
        between metrics
        """
        if cls.setup:
            raise Exception("Global Logger already initialized!")
        
        try:
            mlflow.set_tracking_uri(mlf_tracking_uri)
                  
            run = mlflow.start_run(run_id=mlf_run_id)

            exp = mlflow.get_experiment(run.info.experiment_id)

        except Exception as e:
            print("MLflow initialization error:", e)
            quit()

        if itr_key is not None:
            try:
                itr_val = int(run.data.metrics[itr_key])
                print(f"Resuming @ iteration {itr_val}", '\n')
            except:
                print(f"Failed to find iteration key '{itr_key}' in resumed run")
                quit()

        print("\nMLFLOW TRACKING RESUMED")

        if itr_key is not None:
            print(f"Last logged iteration ==> {itr_val}")

        print("Experiment, run name, experiment id, run id, tracking URI, artifact location:\n")
        print(exp.name)
        print(run.info.run_name)
        print(run.info.experiment_id)
        print(run.info.run_id)
        print(mlf_tracking_uri)
        print(run.info.artifact_uri, '\n')

        cls.setup = True

        if itr_key is not None:
            return itr_val

    @classmethod
    def dummy_init(cls):
        """
        Allows calling Logger without actually logging.
        Useful for testing routines.
        """
        if cls.setup:
            raise Exception("Global Logger already initialized!")

        cls.setup = True
        cls.dummy = True

    @classmethod
    def debug_mode_on(cls):
        """Turn on debug mode"""
        cls.debug = True

    @classmethod
    def debug_mode_off(cls):
        """Turn off debug mode"""
        cls.debug = False

    @classmethod
    def active_run_id(cls):
        """
        Getter method for active mlflow run id
        """
        if not cls.setup or cls.dummy:
            raise Exception("Global Logger not initialized!")
        
        return mlflow.active_run().info.run_id

    @classmethod
    def scope(cls, prefix):
        """Set scope of subsequent logging with prefix"""
        cls.prefix = str(prefix) if prefix else None

    @classmethod
    def set_offset(cls, offset):
        """
        Set step offset, which adjusts logging step to cls.step + offset.
        Useful for aligning signals with different log rates.
        """
        cls.offset = offset

    @classmethod
    def log(cls, key, value, step=None):
        if not cls.dummy:
            pkey = cls.prefix + "_" + key if cls.prefix else key
            step = step if step is not None else cls.step + cls.offset
            mlflow.log_metric(pkey, value, step)

    @classmethod
    def log_stat(cls, key, values, step=None, log_sample_size=True):
        """Logs mean, median, min, max, and std for numpy and torch tensors"""
        if values is not None:
            mean = values.mean().item()
            median = (
                np.median(values).item()
                if isinstance(values, np.ndarray)
                else values.median().item()
            )
            minv = values.min().item()
            maxv = values.max().item()
            std = values.std().item()
            size = (
                values.size
                if isinstance(values, np.ndarray)
                else values.numel()
            )
        else:
            mean = None
            median = None
            minv = None
            maxv = None
            std = None
            size = None

        cls.log(key + "_avg", mean, step)
        cls.log(key + "_med", median, step)
        cls.log(key + "_min", minv, step)
        cls.log(key + "_max", maxv, step)
        cls.log(key + "_std", std, step)

        if log_sample_size:
            cls.log(key + "_siz", size, step)

    @classmethod
    def log_hyperparam(cls, key, value):
        if not cls.dummy:
            mlflow.log_param(key, value)

    @classmethod
    def log_hyperparam_dict(cls, hp_dict, ignore_keys=()):
        hp_dict = dict(hp_dict)  # shallow copy
        
        for k in ignore_keys:
            hp_dict.pop(k)

        for k, v in hp_dict.items():
            cls.log_hyperparam(k, v)

    @classmethod
    def get_hyperparam(cls, key):
        if not cls.dummy:
            params = mlflow.active_run().data.params
            assert key in params, f"'{key}' not found in run params!"

            try:
                return ast.literal_eval(params[key])
            except:
                return params[key]  # likely a string

    @classmethod
    def log_tag(cls, key, value):
        if not cls.dummy:
            mlflow.set_tag(key, value)

    @classmethod
    def save_checkpoint(cls, iteration, model_state_dict, optimizer_state_dict=None, scheduler_state_dict=None, name=None):
        """
        Save model, optimizer, and learning rate scheduler state dict.
        If optimizer_state_dict is provided, a parent dictionary with the keys 'model' and 'optimizer' is created.
        If scheduler_state_dict is also present, a 'scheduler' key/value is added on (only supported if optimizer is present though)
        Otherwise, the model state dict is saved directly with no wrapping dictionary.
        """
        if not cls.dummy:
            chk_name = f"{name}_{iteration}.pt" if name is not None else f"{iteration}.pt"
            chk_full = f"checkpoints/{chk_name}"

            if optimizer_state_dict is not None:
                save_dict = dict(
                    model=model_state_dict,
                    optimizer=optimizer_state_dict
                )

                if scheduler_state_dict is not None:
                    save_dict.update(dict(scheduler=scheduler_state_dict))

            else:
                assert scheduler_state_dict is None, "Saving scheduler state without optimizer state is not currently supported"
                save_dict = model_state_dict

            mlflow.pytorch.log_state_dict(save_dict, chk_full)

    @classmethod
    def load_checkpoint(cls, iteration, name=None, run_id=None, device=None, no_mlflow=False):
        """
        Load model, optimizer, and learning rate scheduler state dict saved with save_checkpoint().
        If no optimizer state dict was provided when saving the checkpoint, this returns None for the second return value.
        Likewise with the scheduler state dict for the third return value.

        If no run_id is provided, this assumes you're loading from the active MLflow run's artifact path.

        Setting no_mlflow=True side-steps (very) long loading times for big experiments, but only works with local tracking URIs
        MLflow unnecessarily tries to load the metrics alongside other run data -> poor design
        """
        if cls.dummy and run_id is None:
            raise Exception("Must specify a run_id to load a model checkpoint in Logger dummy mode")
        
        chk_name = f"{name}_{iteration}.pt" if name is not None else f"{iteration}.pt"
        chk_full = f"checkpoints/{chk_name}"

        if no_mlflow:  # FAST but only works with local tracking and specified run_id
            assert run_id is not None, "Must specify run_id to use no_mlflow mode with load_checkpoint()"
            
            artifact_location = os.getenv("MLFLOW_ARTIFACT_LOCATION")
            if artifact_location is not None:
                run_path = osp.join(urlparse(artifact_location).path, run_id)
            else:
                tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
                _, _, run_path = find_run(tracking_uri, run_id)

            state_dict_path = osp.join(run_path, "artifacts", chk_full, "state_dict.pth")

            load_dict = torch.load(state_dict_path, map_location=device if device is not None else torch.device("cpu"))
            
        else:  # SLOW for large experiments
            state_dict_uri = f"runs:/{run_id}/{chk_full}" if run_id is not None else f"runs:/{mlflow.active_run().info.run_id}/{chk_full}"
            
            load_dict = mlflow.pytorch.load_state_dict(state_dict_uri, map_location=device if device is not None else torch.device("cpu"))

        if "optimizer" in load_dict:
            model_state_dict = load_dict["model"]
            optimizer_state_dict = load_dict["optimizer"]
            scheduler_state_dict = load_dict["scheduler"] if "scheduler" in load_dict else None
        else:
            model_state_dict = load_dict
            optimizer_state_dict = None
            scheduler_state_dict = None

        return model_state_dict, optimizer_state_dict, scheduler_state_dict
    
    @classmethod
    def load_latest_checkpoint(cls, itr_key, name=None, run_id=None, device=None):
        """
        Load *latest* model and optimizer state dict saved with save_checkpoint()
        based on the last logged iteration metric (found with itr_key).

        Returns last_logged_itr along with state dicts.

        Otherwise identical to load_checkpoint().
        """
        run_data = mlflow.get_run(run_id).data if run_id is not None else mlflow.active_run().data
        last_logged_itr = int(run_data.metrics[itr_key])

        model_state_dict, optimizer_state_dict, scheduler_state_dict = cls.load_checkpoint(last_logged_itr, name, run_id, device)

        return model_state_dict, optimizer_state_dict, scheduler_state_dict, last_logged_itr
    
    @classmethod
    def get_last_log(cls, key):
        """
        Returns last value logged for metric 'key' for current active run, or None if 'key' isn't found.
        """
        if not cls.setup:
            raise Exception("'get_last_log' called before setup and only supported for active runs")

        curr_run = mlflow.active_run()
        return curr_run.data.metrics[key] if key in curr_run.data.metrics else None

    @classmethod
    def save_pickle_artifact(cls, obj, name, artifact_dir=None):
        """
        Log an arbitrary python object 'obj' as a pickle file '<name>.pickle'

        artifact_dir is the directory with artifact_uri to save, if None
        defaults to root in artifact_dir

        Saves pickle to temporary random hex folder before logging with mlflow
        to avoid problems with parallel job loggers writing to the same file name
        """
        temp_dir = f"/tmp/{random_hex_str(30)}"
        os.mkdir(temp_dir)

        temp_file_path = osp.join(temp_dir, f"{name}.pickle")
        with open(temp_file_path, "wb") as f:
            pickle.dump(obj, f)

        mlflow.log_artifact(temp_file_path, artifact_dir)

        os.remove(temp_file_path)
        os.rmdir(temp_dir)

    @classmethod
    def load_pickle_artifact(cls, name, artifact_dir=None, run_id=None):
        """
        WARNING: this likely only works for local mlflow servers

        Loads an arbitrary python object from a pickle file
        saved with save_pickle_artifact()

        artifact_dir is the directory with artifact_uri to save, if None
        defaults to root in artifact_dir

        If run_id is left as None, this pulls from the currently active run
        """
        if run_id is None:
            run_id = mlflow.active_run().info.run_id

        artifact_location = os.getenv("MLFLOW_ARTIFACT_LOCATION")
        if artifact_location is not None:
            run_path = osp.join(urlparse(artifact_location).path, run_id)
        else:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            _, _, run_path = find_run(tracking_uri, run_id)

        artifact_dir_path = osp.join(run_path, "artifacts", artifact_dir) if artifact_dir is not None else osp.join(run_path, "artifacts")
        artifact_path = osp.join(artifact_dir_path, f"{name}.pickle")

        with open(artifact_path, "rb") as f:
            artifact = pickle.load(f)

        return artifact

    @classmethod
    def export_run_id(cls, unique_name, rank:int=None):
        """
        Save mlflow run_id with a unique name for an easy lookup

        IDs export to a file map directory in project home

        Especially useful with batch jobs (like Slurm) to assign
        a unique name to the full batch of runs, in which case the rank
        parameter should also be used
        """
        assert type(rank) is int

        map_dir = cls.run_id_map_dir if not cls.debug else cls.debug_run_id_map_dir

        export_dir = osp.join(osp.dirname(osp.dirname(tsp.__file__)), map_dir)
        if not osp.exists(export_dir):
            os.mkdir(export_dir)

        id_file_name = f"{unique_name}_{rank}.id" if rank is not None else f"{unique_name}.id"
        with open(osp.join(export_dir, id_file_name), "x") as f:
            f.write(mlflow.active_run().info.run_id)

    @classmethod
    def import_run_id(cls, unique_name, rank:int=None):
        """
        Import run_id through lookup saved via export_run_id()
        """
        map_dir = cls.run_id_map_dir if not cls.debug else cls.debug_run_id_map_dir
        import_dir = osp.join(osp.dirname(osp.dirname(tsp.__file__)), map_dir)
        id_file_name = f"{unique_name}_{rank}.id" if rank is not None else f"{unique_name}.id"
        
        import_path = osp.join(import_dir, id_file_name)
        assert osp.exists(import_path), f"MLflow run_id file {id_file_name} does not exist!"

        with open(import_path, "r") as f:
            run_id = f.readline()

        return run_id
    
    @classmethod
    def valid_export_name(cls, name, rank:int=None):
        """
        True if name/rank provided is not already being used in
        the run_id file map export, False otherwise
        """
        map_dir = cls.run_id_map_dir if not cls.debug else cls.debug_run_id_map_dir
        export_dir = osp.join(osp.dirname(osp.dirname(tsp.__file__)), map_dir)
        if not osp.exists(export_dir):
            return True
        
        existing_id_file_names = os.listdir(export_dir)

        if rank is None and f"{name}.id" in existing_id_file_names:
            return False
        
        if rank is not None and f"{name}_{rank}.id" in existing_id_file_names:
            return False
        
        return True
