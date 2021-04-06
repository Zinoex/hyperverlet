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
    path_dir = os.path.dirname(save_path)
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


def format_save_path(config):
    save_path = config['save_path']
    dataset_args = config["dataset_args"]
    model_args = config["model_args"]

    return save_path.format(dataset=dataset_args["dataset"], solver=model_args['solver'],
                            train_duration=dataset_args["train_duration"], train_trajectory_length=dataset_args["train_trajectory_length"],
                            coarsening_factor=dataset_args["coarsening_factor"], test_duration=dataset_args["test_duration"],
                            test_trajectory_length=dataset_args["test_trajectory_length"])