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

#Dataset creation
traj_len = 100 * 6000 + 1
duration = 100 * 0.6
trajectory = torch.linspace(0, duration, traj_len, device=device)
num_config = 10
coarsening_factor = 2000
base_solver = ThirdOrderRuthSolver().to(device)
experiment = PendulumDataset(base_solver, duration, traj_len, num_config, coarsening_factor).to(device)

#experiment = LenardJones().to(device)

q_base, p_base = timer(lambda: solver.trajectory(experiment.experiment, experiment.q0, experiment.p0, experiment.mass, trajectory), 'data generation')

solver = HyperVelocityVerletSolver(PendulumModel()).to(device)
# solver = VelocityVerletSolver()

if solver.trainable:
    train(solver, experiment, q_base, p_base, trajectory)


if __name__ == '__main__':
    q, p = test(solver, experiment, trajectory, q_base, p_base)

    # plt.quiver(q[:-1, :, 0], q[:-1, :, 1], q[1:, :, 0] - q[:-1, :, 0], q[1:, :, 1] - q[:-1, :, 1], scale_units='xy', angles='xy', scale=1)
    # color = np.stack([np.full((trajectory.size(0),), 'r'), np.full((trajectory.size(0),), 'g'), np.full((trajectory.size(0),), 'b')], axis=1)

    pendulum_plot(trajectory, experiment.mass, experiment.g, experiment.l, q, p, plot_every=1)