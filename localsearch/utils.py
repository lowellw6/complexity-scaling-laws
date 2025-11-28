
import numpy as np
import torch

from tsp.utils import get_costs
from tsp.logger import MLflowLogger as Logger, parse_log_sig



def setup_logging(cfg):
    if cfg.mlflow_logging_signature is not None:
        resuming, sig_args = parse_log_sig(cfg.mlflow_logging_signature)

        assert not resuming, "resuming unsupported"

        Logger.start(*sig_args)
        Logger.log_hyperparam_dict(vars(cfg), ignore_keys=["mlflow_logging_signature"])

    else:
        Logger.dummy_init()


def drain_queue(queue, num_workers):
    msgs = []
    msgs_rcv = 0
    while msgs_rcv < num_workers:
        msgs.append(queue.get())
        msgs_rcv += 1
    return msgs


def np_get_costs(problems, select_idxs=None):
    return get_costs(torch.from_numpy(problems), select_idxs=torch.from_numpy(select_idxs).to(torch.int64) if select_idxs is not None else None).numpy()


def stitch_msgs(msgs):
    """Inputs list of dictionaries containing np arrays, outputs one dictionary with each array concatenated along axis 0 in proc_id order"""
    ref_keys = msgs[0].keys()

    if "proc_id" in ref_keys:
        ref_keys = list(ref_keys)
        ref_keys.remove("proc_id")
        
        msgs = sorted(msgs, key=lambda x: x["proc_id"])

    stitched_msg = {}
    for key in ref_keys:
        kmsgs = [m[key] for m in msgs]
        # assert np.all([len(km.shape) == 1 for km in kmsgs])
        assert np.allclose([len(km) for km in kmsgs], len(kmsgs[0]))
        
        stitched_msg[key] = np.concatenate(kmsgs, axis=0)

    return stitched_msg
