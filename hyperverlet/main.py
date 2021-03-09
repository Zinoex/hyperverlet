import torch
from torch import optim, nn
import matplotlib.pyplot as plt
from tqdm import trange
import random
import time
import numpy as np

from hyperverlet.experiments import Pendulum, LenardJones
from hyperverlet.solvers import HyperEulerSolver, EulerSolver, VelocityVerletSolver, HyperVelocityVerletSolver, \
    StormerVerletSolver, ThirdOrderRuthSolver, FourthOrderRuthSolver
from hyperverlet.models import PendulumMLP, LennardJonesMLP
from hyperverlet.train import train
from hyperverlet.utils import seed_randomness


seed_randomness()

device = torch.device('cpu')
# experiment = Pendulum(l=0.5).to(device)
experiment = LenardJones().to(device)
solver = ThirdOrderRuthSolver().to(device)

traj_len = 10 * 6000 + 1
duration = 10 * 0.6
trajectory = torch.linspace(0, duration, traj_len).to(device)
coarsening_factor = 500
assert (traj_len - 1) % coarsening_factor == 0

t1 = time.time()
q_base, p_base = solver.trajectory(experiment, experiment.q0, experiment.p0, experiment.mass, trajectory)
t2 = time.time()
print(f'data generation took: {t2 - t1}s')

q_base, p_base = torch.cat([q_base[:1], q_base[1::coarsening_factor]], dim=0), torch.cat([p_base[:1], p_base[1::coarsening_factor]], dim=0)
trajectory = torch.linspace(0, duration, int((traj_len - 1) / coarsening_factor) + 1).to(device)

solver = HyperVelocityVerletSolver(LennardJonesMLP()).to(device)
# solver = VelocityVerletSolver()
criterion = nn.MSELoss()

if solver.trainable:
    train(solver, experiment, q_base, p_base, trajectory)


if __name__ == '__main__':
    with torch.no_grad():
        t1 = time.time()
        q, p = solver.trajectory(experiment, experiment.q0, experiment.p0, experiment.mass, trajectory)
        t2 = time.time()
        print(f'solving took: {t2 - t1}s')

        q_loss, p_loss = criterion(q, q_base), criterion(p, p_base)
        print(f'final loss: {q_loss.item(), p_loss}')

        # plt.quiver(q[:-1, :, 0], q[:-1, :, 1], q[1:, :, 0] - q[:-1, :, 0], q[1:, :, 1] - q[:-1, :, 1], scale_units='xy', angles='xy', scale=1)
        # color = np.stack([np.full((trajectory.size(0),), 'r'), np.full((trajectory.size(0),), 'g'), np.full((trajectory.size(0),), 'b')], axis=1)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        plot_every = max(1, 500 // coarsening_factor)

        ax.scatter(q[::plot_every, :, 0].flatten(), q[::plot_every, :, 1].flatten(), q[::plot_every, :, 2].flatten(), marker='x')
        # plt.scatter(q, p, marker='x')
        # plt.xlabel('q')
        # plt.ylabel('p')
        # plt.title('Phase space trajectory')
        plt.show()
