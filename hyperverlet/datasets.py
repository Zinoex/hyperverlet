import os

# import h5py
import torch
from torch.utils.data import Dataset

import numpy as np

from hyperverlet.distributions import sample_parameterized_truncated_normal
from hyperverlet.experiments import Pendulum, SpringMass, ThreeBodySpringMass, LennardJones
from hyperverlet.timer import timer
from hyperverlet.transforms import Coarsening
from hyperverlet.utils.misc import load_pickle, save_pickle


class ExperimentDataset(Dataset):
    def __init__(self, base_solver, duration, num_samples, num_configurations, coarsening_factor, cache_path, sequence_length=None):
        self.base_solver = base_solver
        self.duration = duration
        self.num_samples = num_samples
        self.num_configurations = num_configurations
        self.coarsening = Coarsening(coarsening_factor, num_samples)
        self.sequence_length = sequence_length

        self.trajectory = torch.stack([torch.linspace(0, duration + math.normal(0, 0.1 * duration), num_samples) for _ in range(num_configurations)], dim=1)

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
            'trajectory': trajectory[config_idx],
            'dt': dt,
            'extra_args': self.config_extra_args(config_idx)
        }

    def config_extra_args(self, config_idx):
        return {k: v[config_idx] for k, v in self.extra_args.items()}


class PendulumDataset(ExperimentDataset):
    def __init__(self, base_solver, duration, num_samples, num_configurations, coarsening_factor, cache_path, sequence_length=None,
                 length_mean=1.0, length_std=0.5, mass_mean=0.9, mass_std=0.1, g=9.807):

        self.experiment = Pendulum()
        self.q0 = (torch.rand(num_configurations, 1) * 2 - 1) * (np.pi / 2)
        self.p0 = torch.randn(num_configurations, 1) * 0.1
        self.mass = torch.randn(num_configurations, 1) * mass_std + mass_mean
        self.extra_args = {
            'length': torch.randn(num_configurations, 1) * length_std + length_mean,
            'g': torch.full((num_configurations, 1), g)
        }

        super().__init__(base_solver, duration, num_samples, num_configurations, coarsening_factor, cache_path, sequence_length)


class SpringMassDataset(ExperimentDataset):
    def __init__(self, base_solver, duration, num_samples, num_configurations, coarsening_factor, cache_path, sequence_length=None, mass_mean=0.9, mass_std=0.1):

        self.experiment = SpringMass()
        length = sample_parameterized_truncated_normal((num_configurations, 1), 0.8, 0.35, 0.1, 1.5)
        self.q0 = torch.rand(num_configurations, 1) * length * 2
        self.p0 = torch.randn(num_configurations, 1) * 0.1
        self.mass = torch.randn(num_configurations, 1) * mass_std + mass_mean
        self.extra_args = {
            'length': length,
            'k': sample_parameterized_truncated_normal((num_configurations, 1), 0.8, 0.35, 0.1, 1.5)
        }

        super().__init__(base_solver, duration, num_samples, num_configurations, coarsening_factor, cache_path, sequence_length)


class ThreeBodySpringMassDataset(ExperimentDataset):
    def __init__(self, base_solver, duration, num_samples, num_configurations, coarsening_factor, cache_path, sequence_length=None, mass_mean=0.9, mass_std=0.1):

        num_particles = 3
        num_springs = num_particles * (num_particles - 1) // 2
        num_euclid = 2

        self.experiment = ThreeBodySpringMass()
        length = sample_parameterized_truncated_normal((num_configurations, num_springs), 5, 2, 0.1, 10)
        self.q0 = (torch.randn(num_configurations, num_particles, num_euclid) * 2 - 1) * length.max(dim=1, keepdim=True)[0].unsqueeze(2)

        # Don't judge, just accept
        first_index = [i for i in range(0, num_particles) for j in range(i + 1, num_particles)]
        second_index = [j for i in range(0, num_particles) for j in range(i + 1, num_particles)]

        length_matrix = torch.zeros((num_configurations, num_particles, num_particles))
        length_matrix[:, first_index, second_index] = length
        length_matrix = length_matrix + length_matrix.transpose(1, 2)

        k_matrix = torch.zeros((num_configurations, num_particles, num_particles))
        k_matrix[:, first_index, second_index] = sample_parameterized_truncated_normal((num_configurations, num_springs), 1.2, 0.5, 0.1, 2.5)
        k_matrix = k_matrix + k_matrix.transpose(1, 2)

        p0 = torch.randn(num_configurations, num_particles, num_euclid) * 0.1
        self.p0 = p0 - torch.mean(p0, dim=1, keepdim=True)

        self.mass = torch.randn(num_configurations, num_particles, 1) * mass_std + mass_mean
        self.extra_args = {
            'length': length_matrix,
            # The spring constant
            'k': k_matrix
        }

        super().__init__(base_solver, duration, num_samples, num_configurations, coarsening_factor, cache_path, sequence_length)


class LennardJonesDataset(ExperimentDataset):
    def __init__(self, base_solver, duration, num_samples, num_configurations, coarsening_factor, cache_path, sequence_length=None, eps=1, sigma=1, num_particles=10):
        self.experiment = LennardJones()

        self.spatial_dims = 3
        boundary_conditions = ['f' for _ in range(num_configurations)]
        bbox = self.generate_bbox(num_configurations, self.spatial_dims)
        bbox_size = bbox[:, :, 1] - bbox[:, :, 0]

        self.q0 = (torch.rand((num_configurations, num_particles, self.spatial_dims)) - 0.5) * bbox_size.unsqueeze(1)

        p0 = (torch.rand_like(self.q0) * 2 - 1) * torch.tensor([0.03]).sqrt()
        self.p0 = p0 - p0.mean(axis=1, keepdim=True)

        self.mass = torch.ones((num_configurations, num_particles, 1))
        self.extra_args = {
            'eps': torch.full((num_configurations, num_particles, num_particles, 1), eps),
            'sigma': torch.full((num_configurations, num_particles, num_particles, 1), sigma),
            'boundary_conditions': boundary_conditions,
            'bbox': bbox
        }

        super().__init__(base_solver, duration, num_samples, num_configurations, coarsening_factor, cache_path, sequence_length)

    @staticmethod
    def generate_bbox(num_configurations, spatial_dims, side_length=5):

        bbox = torch.ones(num_configurations, spatial_dims, 2)
        bbox[:, :, 0] = -1

        return bbox * (side_length / 2)


# class KobAndersenDataset(Dataset):
#     def __init__(self,
#                  data_dir: str,
#                  coarsening_factor,
#                  sequence_length=None):
#         self.data_dir = data_dir
#         self.filenames = os.listdir(data_dir)
#         self.files = []
#
#         for filename in self.filenames:
#             assert os.path.splitext(filename)[1] == 'h5'
#
#         self.coarsening = Coarsening(coarsening_factor)
#         self.sequence_length = sequence_length
#
#         self.set_files()
#
#         self.dataset_lengths = [file['timestep'].shape[0] for file in self.files]
#
#     def __del__(self):
#         for file in self.files:
#             if file is not None:
#                 file.close()
#
#     def set_files(self):
#         self.files = [h5py.File(os.path.join(self.data_dir, filename), "r") for filename in self.filenames]
#
#         for file in self.files:
#             # important to compute momentum
#             assert 'velocity' in file
#
#     def __len__(self):
#         if self.sequence_length is None:
#             return len(self.filenames)
#         else:
#             configuration_lengths = [self.coarsening.compute_new_trajectory_length(length) - self.sequence_length for length in self.dataset_lengths]
#             return sum(configuration_lengths)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         if self.sequence_length is None:
#             config_idx = idx
#             file = self.files[config_idx]
#
#             q = file['positions']
#             p = file['velocity']  # All masses of Kob-Andersen is 1
#             trajectory = file['timestep']
#         else:
#             for config_idx, length in enumerate(self.dataset_lengths):
#                 if idx < length:
#                     break
#                 idx -= length
#
#             time_idx = idx
#             file = self.files[config_idx]
#
#             q = file['positions'][time_idx:time_idx + self.sequence_length + 1]
#             p = file['velocity'][time_idx:time_idx + self.sequence_length + 1]  # All masses of Kob-Andersen is 1
#             trajectory = file['timestep'][time_idx:time_idx + self.sequence_length + 1]
#
#         q, p, trajectory = self.coarsening(q, p, trajectory)
#
#         # (type a, type b, epsilon, sigma)
#         # pair_coeff 1 1 1 1
#         # pair_coeff 1 2 1.5 0.80
#         # pair_coeff 2 2 0.5 0.88
#
#         particle_type = file['particle_type']
#         cross_particle_type =
#
#         return {
#             'q': q,
#             'p': p,
#             'mass': torch.ones(q.shape[0], 1),
#             'trajectory': trajectory,
#             'extra_args': {
#                 'eps': torch.full((num_particles, num_particles, 1), eps),
#                 'sigma': torch.full((num_configurations, num_particles, num_particles, 1), sigma),
#                 'boundary_conditions': self.boundary_conditions,
#                 'bbox': self.bbox
#             }
#         }
#
#     def worker_init(self, worker_id):
#         self.set_files()

