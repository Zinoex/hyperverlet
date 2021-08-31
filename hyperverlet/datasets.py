import os

from statistics import NormalDist

import torch
from torch import distributions
from torch.utils.data import Dataset

import numpy as np

from hyperverlet.experiments import Pendulum, SpringMass
from hyperverlet.utils.timer import timer
from hyperverlet.transforms import Coarsening
from hyperverlet.utils.misc import load_pickle, save_pickle


class ExperimentDataset(Dataset):
    def __init__(self, base_solver, duration, duration_stddev, num_samples, num_configurations, coarsening_factor, cache_path, sequence_length=None):
        self.base_solver = base_solver
        self.duration = duration
        self.num_samples = num_samples
        self.num_configurations = num_configurations
        self.coarsening = Coarsening(coarsening_factor, num_samples)
        self.sequence_length = sequence_length

        samples = distributions.Normal(duration, duration * duration_stddev).sample((num_configurations,))
        self.trajectory = torch.stack([torch.linspace(0, sample, num_samples) for sample in samples], dim=1)

        self.load_data(cache_path)

        self.q, self.p, self.trajectory = self.coarsening(self.q, self.p, self.trajectory)
        self.dt = self.trajectory[1:] - self.trajectory[:-1]

        if self.sequence_length is not None:
            assert self.coarsening.new_trajectory_length >= self.sequence_length, 'Trajectory length too short for coarsening'

    def load_data(self, cache_path):
        if os.path.exists(cache_path):
            data = load_pickle(cache_path)
            self.q = data['q']
            self.p = data['p']
            self.trajectory = data['trajectory']
            self.mass = data['mass']
            self.extra_args = data['extra_args']
        else:
            self.q, self.p = timer(lambda: self.base_solver.trajectory(self.experiment, self.q0, self.p0, self.mass, self.trajectory, **self.extra_args), 'data generation')

            save_pickle(cache_path, {
                'q': self.q,
                'p': self.p,
                'trajectory': self.trajectory,
                'mass': self.mass,
                'extra_args': self.extra_args
            })

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
        dt = self.dt

        if self.sequence_length is None:
            config_idx = idx
        else:
            configuration_length = self.coarsening.new_trajectory_length - self.sequence_length

            config_idx, time_idx = idx // configuration_length, idx % configuration_length

            q = q[time_idx:time_idx + self.sequence_length + 1]
            p = p[time_idx:time_idx + self.sequence_length + 1]
            trajectory = trajectory[time_idx:time_idx + self.sequence_length + 1]
            dt = dt[time_idx:time_idx + self.sequence_length]

        return {
            'q': q[:, config_idx],
            'p': p[:, config_idx],
            'mass': self.mass[config_idx],
            'trajectory': trajectory[:, config_idx],
            'dt': dt[:, config_idx],
            'extra_args': self.config_extra_args(config_idx)
        }

    def config_extra_args(self, config_idx):
        return {k: v[config_idx] for k, v in self.extra_args.items()}


class PendulumDataset(ExperimentDataset):
    def __init__(self, base_solver, duration, duration_stddev, num_samples, num_configurations, coarsening_factor, cache_path, sequence_length=None,
                 length_min=0.5, length_max=2.0, mass_min=0.9, mass_max=1.1, g=9.807):

        self.experiment = Pendulum()
        self.q0 = (torch.rand(num_configurations, 1) * 2 - 1) * (np.pi / 2)
        self.p0 = torch.randn(num_configurations, 1) * 0.1

        self.mass = distributions.Uniform(mass_min, mass_max).sample((num_configurations, 1))
        self.extra_args = {
            'length': distributions.Uniform(length_min, length_max).sample((num_configurations, 1)),
            'g': torch.full((num_configurations, 1), g)
        }

        assert torch.all(self.extra_args['length'] > 0).item()
        assert torch.all(self.extra_args['g'] > 0).item()
        assert torch.all(self.mass > 0).item()

        super().__init__(base_solver, duration, duration_stddev, num_samples, num_configurations, coarsening_factor, cache_path, sequence_length)


class SpringMassDataset(ExperimentDataset):
    def __init__(self, base_solver, duration, duration_stddev, num_samples, num_configurations, coarsening_factor, cache_path, sequence_length=None, length_min=0.5, length_max=2.0, k_min=0.8, k_max=1.2, mass_min=0.9, mass_max=1.1):

        self.experiment = SpringMass()

        self.mass = distributions.Uniform(mass_min, mass_max).sample((num_configurations, 1))
        self.extra_args = {
            'length': distributions.Uniform(length_min, length_max).sample((num_configurations, 1)),
             'k': distributions.Uniform(k_min, k_max).sample((num_configurations, 1)),
        }

        self.q0 = torch.rand(num_configurations, 1) * self.extra_args['length'] * 2
        self.p0 = torch.randn(num_configurations, 1) * 0.1

        assert torch.all(self.extra_args['length'] > 0).item()
        assert torch.all(self.extra_args['k'] > 0).item()
        assert torch.all(self.mass > 0).item()

        super().__init__(base_solver, duration, duration_stddev, num_samples, num_configurations, coarsening_factor, cache_path, sequence_length)
