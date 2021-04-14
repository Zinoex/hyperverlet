import random
from datetime import datetime

import torch
import numpy as np
import json
import pickle
import os


def load_config(path):
    with open(path, 'r') as f:
        base_config = json.load(f)

    if 'dataset_args_path' in base_config:
        with open(base_config['dataset_args_path'], 'r') as f:
            base_config['dataset_args'] = json.load(f)

    if 'nn_path' in base_config['model_args']:
        with open(base_config['model_args']["nn_path"], 'r') as f:
            base_config["model_args"]['nn_args'] = json.load(f)

    if 'plotting_args_path' in base_config:
        with open(base_config['plotting_args_path'], 'r') as f:
            base_config['plotting'] = json.load(f)

    if 'train_args_path' in base_config:
        with open(base_config['train_args_path'], 'r') as f:
            base_config['train_args'] = json.load(f)

    return base_config


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


def format_path(config, path, **kwargs):
    dataset_args = config["dataset_args"]
    model_args = config["model_args"]

    return path.format(
        dataset=dataset_args["dataset"],
        solver=model_args['solver'],
        train_duration=dataset_args["train_duration"],
        train_trajectory_length=dataset_args["train_trajectory_length"],
        train_sequence_length=dataset_args["train_sequence_length"],
        train_num_configurations=dataset_args["train_num_configurations"],
        coarsening_factor=dataset_args["coarsening_factor"],
        test_duration=dataset_args["test_duration"],
        test_trajectory_length=dataset_args["test_trajectory_length"],
        test_num_configurations=dataset_args["test_num_configurations"],
        datetime=datetime.now(),
        **kwargs
   )