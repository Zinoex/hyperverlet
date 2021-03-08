import random
import torch
import numpy as np


def seed_randomness(seed=1000):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
