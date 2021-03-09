import abc

import torch
from math import pi

from torch import nn
from torch.utils.data import Dataset

from hyperverlet.transforms import Coarsening


class ExperimentDataset(Dataset, abc.ABC):
    def __init__(self, base_solver, duration, num_samples, num_configurations, coarsening_factor):
        self.base_solver = base_solver
        self.duration = duration
        self.num_samples = num_samples
        self.num_configurations = num_configurations
        self.coarsening = Coarsening(coarsening_factor, num_samples)

        self.experiment, self.q0, self.p0, self.mass, self.trajectory = self.initial_conditions()
        self.q, self.p = self.base_solver.trajectory(self.experiment, self.q0, self.p0, self.mass, self.trajectory)

    def __len__(self):
        return self.coarsening.new_trajectory_length * self.num_configurations

    @abc.abstractmethod
    def initial_conditions(self):
        raise NotImplementedError()


class Pendulum(nn.Module):
    def __init__(self, l, m=0.9, g=9.807):
        super().__init__()

        self.l = l
        self.g = g

        self.register_buffer('q0', torch.tensor([pi / 2]))
        self.register_buffer('p0', torch.tensor([0.0]))
        self.register_buffer('mass', torch.tensor([m]))

    def forward(self, q, p, m, t):
        dq = p / (m * self.l ** 2)
        dp = -m * self.g * self.l * torch.sin(q)
        return dq, dp


class PendulumDataset(ExperimentDataset):

    def __init__(self, base_solver, duration, num_samples, num_configurations, coarsening_factor, l, g=9.807):
        self.l = l
        self.g = g

    def initial_conditions(self):
        return self.experiment, self.q0, self.p0, self.mass, self.trajectory


class LenardJones(nn.Module):
    def __init__(self, eps=1, sigma=1):
        super().__init__()

        self.eps = eps
        self.sigma = sigma

        self.num_particles = 10
        self.register_buffer('q0', torch.rand((self.num_particles, 3)) * 5)
        p0 = (torch.rand_like(self.q0) * 2 - 1) * torch.tensor([0.03]).sqrt()
        self.register_buffer('p0', p0 - p0.mean(axis=0))
        self.register_buffer('mass', torch.ones((self.num_particles, 1)))

    def forward(self, q, p, m, t):
        # q = q.requires_grad_(True)
        # q.requires_grad = True

        displacement = self.displacement(q)
        r = self.distance(displacement)
        # acceleration = force / mass
        dq = p / m
        dp = self.force(r, displacement).sum(axis=1)

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
