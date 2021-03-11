import random
import torch
import numpy as np
import json


def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)


def seed_randomness(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def send_to_device(extra_args, device, non_blocking=False):
    return {
        k: v.to(device, non_blocking=non_blocking) for k, v in extra_args.items()
    }
