import torch


class Coarsening:
    def __init__(self, coarsening_factor, trajectory_length=None):
        self.coarsening_factor = coarsening_factor
        self.trajectory_length = trajectory_length

        if trajectory_length is not None:
            assert (trajectory_length - 1) % coarsening_factor == 0

    def __call__(self, q, p, t):
        q, p, t = self.coarse(q), self.coarse(p), self.coarse(t)
        return q, p, t

    def coarse(self, x):
        return x[::self.coarsening_factor]

    @property
    def new_trajectory_length(self):
        return (self.trajectory_length - 1) // self.coarsening_factor + 1

    def compute_new_trajectory_length(self, trajectory_length):
        return (trajectory_length - 1) // self.coarsening_factor + 1
