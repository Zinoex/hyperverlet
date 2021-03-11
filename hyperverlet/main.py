import torch
import matplotlib.pyplot as plt

from hyperverlet.experiments import Pendulum, LenardJones, PendulumDataset
from hyperverlet.solvers import HyperEulerSolver, EulerSolver, VelocityVerletSolver, HyperVelocityVerletSolver, \
    StormerVerletSolver, ThirdOrderRuthSolver, FourthOrderRuthSolver
from hyperverlet.models import PendulumModel, LennardJonesMLP
from hyperverlet.test import test
from hyperverlet.timer import timer
from hyperverlet.train import train
from hyperverlet.transforms import Coarsening
from hyperverlet.utils import seed_randomness
from plotting.plotting import plot_3d_pos, plot_phasespace, pendulum_plot, plot_2d_pos

seed_randomness()
device = torch.device('cpu')

# Dataset creation
traj_len = 100 * 6000 + 1
duration = 100 * 0.6
num_config = 10
coarsening_factor = 2000
base_solver = ThirdOrderRuthSolver().to(device)

train_dataset = PendulumDataset(base_solver, duration, traj_len, num_config, coarsening_factor, sequence_length=1)
test_dataset = PendulumDataset(base_solver, duration, traj_len, 1, coarsening_factor, sequence_length=None)

solver = HyperVelocityVerletSolver(PendulumModel()).to(device)
# solver = VelocityVerletSolver()

if solver.trainable:
    train(solver, train_dataset, device)


if __name__ == '__main__':
    q, q_base, p, p_base, mass, trajectory, extra_args = test(solver, test_dataset, device)

    # plt.quiver(q[:-1, :, 0], q[:-1, :, 1], q[1:, :, 0] - q[:-1, :, 0], q[1:, :, 1] - q[:-1, :, 1], scale_units='xy', angles='xy', scale=1)
    # color = np.stack([np.full((trajectory.size(0),), 'r'), np.full((trajectory.size(0),), 'g'), np.full((trajectory.size(0),), 'b')], axis=1)

    pendulum_plot(trajectory, mass, test_dataset.experiment.g, extra_args['length'], q, p, plot_every=1)
