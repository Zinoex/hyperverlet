import torch
from torch.utils.data import DataLoader

from hyperverlet.factories.dataset_factory import construct_dataset


class Executor:
    def __init__(self, config_args, debug):
        if debug:
            num_samples = 100
            limit = 10
            num_workers = 0
            pin_memory = False
        else:
            num_samples = 9995
            limit = None
            num_workers = 8
            pin_memory = True

        batch_size = config_args["batch_size"]

        self.data = construct_dataset(config_args)
        self.data_loader = DataLoader(self.data, num_workers=num_workers, pin_memory=pin_memory, batch_size=batch_size)
        self.n_data = len(self.data_loader)

    def train(self):
        raise NotImplementedError

    @torch.no_grad()
    def test(self):
        raise NotImplementedError

    @torch.no_grad()
    def rollout(self):
        raise NotImplementedError
