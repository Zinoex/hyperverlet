import torch
from torch import nn
import matplotlib.pyplot as plt

from hyperverlet.experiments import Pendulum, LenardJones
from hyperverlet.solvers import HyperEulerSolver, EulerSolver, VelocityVerletSolver, HyperVelocityVerletSolver, \
    StormerVerletSolver, ThirdOrderRuthSolver, FourthOrderRuthSolver
from hyperverlet.models import PendulumMLP, LennardJonesMLP
from hyperverlet.timer import timer
from hyperverlet.train import train
from hyperverlet.transforms import Coarsening
from hyperverlet.utils import seed_randomness
from plotting.plotting import plot_3d_pos, plot_phasespace, pendulum_plot

seed_randomness()

device = torch.device('cpu')
experiment = Pendulum(l=1, m=0.9).to(device)
#experiment = LenardJones().to(device)
solver = ThirdOrderRuthSolver().to(device)

traj_len = 10 * 6000 + 1
duration = 10 * 0.6
trajectory = torch.linspace(0, duration, traj_len).to(device)

q_base, p_base = timer(lambda: solver.trajectory(experiment, experiment.q0, experiment.p0, experiment.mass, trajectory), 'data generation')

coarsening = Coarsening(coarsening_factor=10, trajectory_length=traj_len)
q_base, p_base, trajectory = coarsening(q_base, p_base, trajectory)


solver = HyperVelocityVerletSolver(PendulumMLP()).to(device)
#solver = VelocityVerletSolver()
criterion = nn.MSELoss()

if solver.trainable:
    train(solver, experiment, q_base, p_base, trajectory)


if __name__ == '__main__':
    with torch.no_grad():
        q, p = timer(lambda: solver.trajectory(experiment, experiment.q0, experiment.p0, experiment.mass, trajectory), 'solving')

        q_loss, p_loss = criterion(q, q_base), criterion(p, p_base)
        print(f'final loss: {q_loss.item(), p_loss.item()}')

        # plt.quiver(q[:-1, :, 0], q[:-1, :, 1], q[1:, :, 0] - q[:-1, :, 0], q[1:, :, 1] - q[:-1, :, 1], scale_units='xy', angles='xy', scale=1)
        # color = np.stack([np.full((trajectory.size(0),), 'r'), np.full((trajectory.size(0),), 'g'), np.full((trajectory.size(0),), 'b')], axis=1)

        plot_every = max(1, 500 // coarsening.coarsening_factor)
        pendulum_plot(trajectory, experiment.mass, experiment.g, experiment.l, q, p, plot_every=10)
        #plot_phasespace(q, p)
