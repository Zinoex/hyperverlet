import torch


class Coarsening:
    def __init__(self, coarsening_factor, trajectory_length=None):
        self.coarsening_factor = coarsening_factor

        if trajectory_length is not None:
            assert (trajectory_length - 1) % coarsening_factor == 0

    def __call__(self, q, p, t):
        q, p, t = self.coarse(q), self.coarse(p), self.coarse(t)
        return q, p, t

    def coarse(self, x):
        return torch.cat([x[:1], x[1::self.coarsening_factor]], dim=0)
