import torch
from math import pi

from torch import nn
from torch.utils.data import Dataset

from hyperverlet.transforms import Coarsening


class ExperimentDataset(Dataset):
    def __init__(self, base_solver, duration, num_samples, num_configurations, coarsening_factor, sequence_length=None):
        self.base_solver = base_solver
        self.duration = duration
        self.num_samples = num_samples
        self.num_configurations = num_configurations
        self.coarsening = Coarsening(coarsening_factor, num_samples)
        self.sequence_length = sequence_length or self.coarsening.new_trajectory_length
        self.configuration_length = self.coarsening.new_trajectory_length - self.sequence_length

        self.trajectory = torch.linspace(0, duration, num_samples)
        # Første parameter base_solver bør være eksperiment, men den kender vi jo ikke i base_solveren?
        self.q, self.p = self.base_solver.trajectory(self.experiment, self.q0, self.p0, self.mass, self.trajectory, **self.extra_args)

    def __len__(self):
        return self.configuration_length * self.num_configurations

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        config_idx, time_idx = idx // self.configuration_length, idx % self.configuration_length

        return {
            'q': self.q[time_idx:time_idx + self.sequence_length, config_idx],
            'p': self.p[time_idx:time_idx + self.sequence_length, config_idx],
            'mass': self.mass[config_idx]
        }


class Pendulum(nn.Module):
    def __init__(self, g=9.807):
        super().__init__()

        self.g = g

    def forward(self, q, p, m, t, length, **kwargs):
        dq = p / (m * length ** 2)
        dp = -m * self.g * length * torch.sin(q)
        return dq, dp


class PendulumDataset(ExperimentDataset):
    def __init__(self, base_solver, duration, num_samples, num_configurations, coarsening_factor, sequence_length=None,
                 length_mean=1.0, length_std=0.5, mass_mean=0.9, mass_std=0.1, g=9.807):

        self.experiment = Pendulum(g)
        self.q0 = (torch.rand(num_configurations, 1) * 2 - 1) * pi
        self.p0 = torch.randn(num_configurations, 1) * 0.1
        self.mass = torch.randn(num_configurations, 1) * mass_std + mass_mean
        self.extra_args = {
            'length': torch.randn(num_configurations, 1) * length_std + length_mean
        }

        super().__init__(base_solver, duration, num_samples, num_configurations, coarsening_factor, sequence_length)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        config_idx, time_idx = idx // self.configuration_length, idx % self.configuration_length

        return {
            'q': self.q[time_idx:time_idx + self.sequence_length, config_idx],
            'p': self.p[time_idx:time_idx + self.sequence_length, config_idx],
            'mass': self.mass[config_idx],
            'extra_args': self.extra_args
        }


class LenardJones(nn.Module):
    def __init__(self, eps=1, sigma=1):
        super().__init__()

        self.eps = eps
        self.sigma = sigma

    def forward(self, q, p, m, t):
        # q = q.requires_grad_(True)
        # q.requires_grad = True

        disp = self.displacement(q)
        r = self.distance(disp)
        # acceleration = force / mass
        dq = p / m
        dp = self.force(r, disp).sum(axis=1)

        return dq, dp

    def force(self, r, disp):
        prefix = 48 * self.eps
        sigma12 = self.sigma ** 12
        sigma6 = self.sigma ** 6

        r = r.fill_diagonal_(1)

        r13 = r ** 13
        r7 = r ** 7

        inner = sigma12 / r13 - 0.5 * sigma6 / r7

        direction = disp / r.unsqueeze(-1)

        return -prefix * inner.fill_diagonal_(0).unsqueeze(-1) * direction

        # energy = self.potential(r).sum()
        # scalar_force = torch.autograd.grad(energy, r, retain_graph=True)[0].fill_diagonal_(0)
        # return -scalar_force.unsqueeze(2) * (disp / r.fill_diagonal_(1).unsqueeze(2))

    def potential(self, r):
        return (4 * self.eps * ((self.sigma / r) ** 12 - (self.sigma / r) ** 6)).fill_diagonal_(0)

    def displacement(self, q):
        a = torch.unsqueeze(q, 1)
        b = torch.unsqueeze(q, 0)
        return a - b

    def distance(self, disp):
        return disp.norm(dim=2)


class LennardJonesDataset(ExperimentDataset):
    def __init__(self, base_solver, duration, num_samples, num_configurations, coarsening_factor, sequence_length=None, eps=1, sigma=1, num_particles=10):
        self.experiment = LenardJones(eps, sigma)
        self.q0 = torch.rand((num_configurations, num_particles, 3)) * 5
        p0 = (torch.rand_like(self.q0) * 2 - 1) * torch.tensor([0.03]).sqrt()
        self.p0 = p0 - p0.mean(axis=1)
        self.mass = torch.ones((num_configurations, num_particles, 1))
        self.extra_args = {}

        super().__init__(base_solver, duration, num_samples, num_configurations, coarsening_factor, sequence_length)

