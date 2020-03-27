import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import sys
from distutils.dir_util import copy_tree
import aws_utils
import pickle
from copy import deepcopy


class RNGSeed:
    def __init__(self, seed, deterministic=True):
        self.seed = seed
        self.deterministic = deterministic
        self.set_random_seeds()

    def set_random_seeds(self):
        seed = self.seed
        random.seed(seed)
        np.random.seed(seed)
        cudnn.enabled = True

        if self.deterministic:
            cudnn.benchmark = False
            cudnn.deterministic = True
        else:
            cudnn.benchmark = True

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def get_save_states(self):
        rng_states = {
            "random_state": random.getstate(),
            "np_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "torch_cuda_random_state": torch.cuda.get_rng_state_all(),
        }
        return rng_states

    def load_states(self, rng_states):
        random.setstate(rng_states["random_state"])
        np.random.set_state(rng_states["np_random_state"])
        torch.set_rng_state(rng_states["torch_random_state"])
        torch.cuda.set_rng_state_all(rng_states["torch_cuda_random_state"])


def save(
    folder,
    epochs,
    rng_seed,
    model,
    optimizer,
    history=None,
    s3_bucket=None,
):

    checkpoint = {
        "epochs": epochs,
        "rng_seed": rng_seed.get_save_states(),
        "optimizer": optimizer.state_dict(),
        "model": model.state_dict()
        "arch_params": model._modules['module']._arch_parameters
    }

    ckpt = os.path.join(folder, "model.ckpt")
    torch.save(checkpoint, ckpt)

    if history is not None:
        history_file = os.path.join(folder, "history.pkl")
        with open(history_file, "wb") as f:
            pickle.dump(history, f)

    log = os.path.join(folder, "log.txt")

    if s3_bucket is not None:
        aws_utils.upload_to_s3(ckpt, s3_bucket, ckpt)
        aws_utils.upload_to_s3(log, s3_bucket, log)
        if history is not None:
            aws_utils.upload_to_s3(history_file, s3_bucket, history_file)

def load(folder, rng_seed, model, optimizer, s3_bucket=None):
    # Try to download log and ckpt from s3 first to see if a ckpt exists.
    ckpt = os.path.join(folder, "model.ckpt")
    history_file = os.path.join(folder, "history.pkl")
    history = None

    if s3_bucket is not None:
        aws_utils.download_from_s3(ckpt, s3_bucket, ckpt)
        try:
            aws_utils.download_from_s3(history_file, s3_bucket, history_file)
        except:
            logging.info("history.pkl not in s3 bucket")

    if os.path.exists(history_file):
        with open(history_file, "rb") as f:
            history = pickle.load(f)

    checkpoint = torch.load(ckpt)

    epochs = checkpoint["epochs"]
    rng_seed.load_states(checkpoint["rng_seed"])
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for i, p in enumerate(checkpoint['arch_params']):
        model._modules['module']._arch_parameters[i] = p

    logging.info("Resumed model trained for %d epochs" % epochs)

    return epochs, history

