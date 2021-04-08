import torch
from torch.utils.data import Dataset

import numpy as np

from hyperverlet.distributions import sample_parameterized_truncated_normal
from hyperverlet.experiments import Pendulum, SpringMass, ThreeBodySpringMass, LennardJones
from hyperverlet.timer import timer
from hyperverlet.transforms import Coarsening


class ExperimentDataset(Dataset):
    def __init__(self, base_solver, duration, num_samples, num_configurations, coarsening_factor, sequence_length=None):
        self.base_solver = base_solver
        self.duration = duration
        self.num_samples = num_samples
        self.num_configurations = num_configurations
        self.coarsening = Coarsening(coarsening_factor, num_samples)
        self.sequence_length = sequence_length

        self.trajectory = torch.linspace(0, duration, num_samples)
        self.q, self.p = timer(lambda: self.base_solver.trajectory(self.experiment, self.q0, self.p0, self.mass, self.trajectory, **self.extra_args), 'data generation')

        self.q, self.p, self.trajectory = self.coarsening(self.q, self.p, self.trajectory)

        if self.sequence_length is not None:
            assert self.coarsening.new_trajectory_length >= self.sequence_length, 'Trajectory length too short for coarsening'

    def __len__(self):
        if self.sequence_length is None:
            return self.num_configurations
        else:
            assert self.coarsening.new_trajectory_length >= self.sequence_length, 'Trajectory length too short for coarsening'
            configuration_length = self.coarsening.new_trajectory_length - self.sequence_length
            return configuration_length * self.num_configurations

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        q = self.q
        p = self.p
        trajectory = self.trajectory

        if self.sequence_length is None:
            config_idx = idx
        else:
            configuration_length = self.coarsening.new_trajectory_length - self.sequence_length

            config_idx, time_idx = idx // configuration_length, idx % configuration_length

            q = q[time_idx:time_idx + self.sequence_length + 1]
            p = p[time_idx:time_idx + self.sequence_length + 1]
            trajectory = trajectory[time_idx:time_idx + self.sequence_length + 1]

        return {
            'q': q[:, config_idx],
            'p': p[:, config_idx],
            'mass': self.mass[config_idx],
            'trajectory': trajectory,
            'extra_args': self.config_extra_args(config_idx)
        }

    def config_extra_args(self, config_idx):
        return self.extra_args


class PendulumDataset(ExperimentDataset):
    def __init__(self, base_solver, duration, num_samples, num_configurations, coarsening_factor, sequence_length=None,
                 length_mean=1.0, length_std=0.5, mass_mean=0.9, mass_std=0.1, g=9.807):

        self.experiment = Pendulum()
        self.q0 = (torch.rand(num_configurations, 1) * 2 - 1) * (np.pi / 2)
        self.p0 = torch.randn(num_configurations, 1) * 0.1
        self.mass = torch.randn(num_configurations, 1) * mass_std + mass_mean
        self.extra_args = {
            'length': torch.randn(num_configurations, 1) * length_std + length_mean,
            'g': torch.full((num_configurations, 1), g)
        }

        super().__init__(base_solver, duration, num_samples, num_configurations, coarsening_factor, sequence_length)

    def config_extra_args(self, config_idx):
        return {
            'length': self.extra_args['length'][config_idx],
            'g': self.extra_args['g'][config_idx]
        }


class SpringMassDataset(ExperimentDataset):
    def __init__(self, base_solver, duration, num_samples, num_configurations, coarsening_factor, sequence_length=None, mass_mean=0.9, mass_std=0.1):

        self.experiment = SpringMass()
        length = sample_parameterized_truncated_normal((num_configurations, 1), 0.8, 0.35, 0.1, 1.5)
        self.q0 = torch.rand(num_configurations, 1) * length * 2
        self.p0 = torch.randn(num_configurations, 1) * 0.1
        self.mass = torch.randn(num_configurations, 1) * mass_std + mass_mean
        self.extra_args = {
            'length': length,
            'k': sample_parameterized_truncated_normal((num_configurations, 1), 0.8, 0.35, 0.1, 1.5)
        }

        super().__init__(base_solver, duration, num_samples, num_configurations, coarsening_factor, sequence_length)

    def config_extra_args(self, config_idx):
        return {
            'length': self.extra_args['length'][config_idx],
            'k': self.extra_args['k'][config_idx]
        }


class ThreeBodySpringMassDataset(ExperimentDataset):
    def __init__(self, base_solver, duration, num_samples, num_configurations, coarsening_factor, sequence_length=None, mass_mean=0.9, mass_std=0.1):

        num_particles = 3
        num_springs = num_particles * (num_particles - 1) // 2
        num_euclid = 2

        self.experiment = ThreeBodySpringMass()
        length = sample_parameterized_truncated_normal((num_configurations, num_springs), 0.8, 0.1, 0.1, 1.5)
        self.q0 = (torch.randn(num_configurations, num_particles, num_euclid) * 2 - 1) * length.max(dim=1, keepdim=True)[0].unsqueeze(2)

        # Don't judge, just accept
        first_index = [i for i in range(0, num_particles) for j in range(i + 1, num_particles)]
        second_index = [j for i in range(0, num_particles) for j in range(i + 1, num_particles)]

        length_matrix = torch.zeros((num_configurations, num_particles, num_particles))
        length_matrix[:, first_index, second_index] = length
        length_matrix = length_matrix + length_matrix.transpose(1, 2)

        k_matrix = torch.zeros((num_configurations, num_particles, num_particles))
        k_matrix[:, first_index, second_index] = sample_parameterized_truncated_normal((num_configurations, num_springs), 0.8, 0.35, 0.1, 1.5)
        k_matrix = k_matrix + k_matrix.transpose(1, 2)

        p0 = torch.randn(num_configurations, num_particles, num_euclid) * 0.1
        self.p0 = p0 - torch.mean(p0, dim=1, keepdim=True)

        self.mass = torch.randn(num_configurations, num_particles, 1) * mass_std + mass_mean
        self.extra_args = {
            'length': length_matrix,
            # The spring constant
            'k': k_matrix
        }

        super().__init__(base_solver, duration, num_samples, num_configurations, coarsening_factor, sequence_length)

    def config_extra_args(self, config_idx):
        return {
            'length': self.extra_args['length'][config_idx],
            'k': self.extra_args['k'][config_idx]
        }


class LennardJonesDataset(ExperimentDataset):
    def __init__(self, base_solver, duration, num_samples, num_configurations, coarsening_factor, sequence_length=None, eps=1, sigma=1, num_particles=10):
        self.experiment = LennardJones(eps, sigma)
        self.q0 = torch.rand((num_configurations, num_particles, 3)) * 5
        p0 = (torch.rand_like(self.q0) * 2 - 1) * torch.tensor([0.03]).sqrt()
        self.p0 = p0 - p0.mean(axis=1)
        self.mass = torch.ones((num_configurations, num_particles, 1))
        self.extra_args = {}

        super().__init__(base_solver, duration, num_samples, num_configurations, coarsening_factor, sequence_length)