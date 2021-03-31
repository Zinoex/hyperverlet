import random
import torch
import numpy as np
import json
import pickle
import os


def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_pickle(path):
    with open(path, 'rb') as f:
        results = pickle.load(f)
    return results


def save_pickle(save_path, dict):
    path_dir = os.path.join(*save_path.split("/")[:-1])
    os.makedirs(path_dir, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(dict, f)


def seed_randomness(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def send_to_device(extra_args, device, non_blocking=False):
    return {
        k: v.to(device, non_blocking=non_blocking) for k, v in extra_args.items()
    }


def torch_to_numpy(extra_args):
    return {
        k: v.numpy() for k, v in extra_args.items()
    }