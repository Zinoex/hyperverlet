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
optimizer = optim.AdamW(solver.parameters(), lr=0.5 * 1e-2)
criterion = nn.MSELoss()

# for iteration in trange(1000):
#     optimizer.zero_grad(set_to_none=True)
#
#     batch_size = 4
#     if iteration < -1:
#         start = 0
#     else:
#         start = random.randint(0, trajectory.size(0) - batch_size)
#
#     end = start + batch_size
#
#     q, p = solver.trajectory(experiment, q_base[start], p_base[start], experiment.mass, trajectory[start:end])
#     loss = criterion(q, q_base[start:end]) + criterion(p, p_base[start:end])
#     # loss = solver.loss(experiment, q_base[start:end], p_base[start:end], experiment.mass, trajectory[start:end])
#     loss.backward()
#     optimizer.step()
#
#     print(f'loss: {loss.item()}')

solver = VelocityVerletSolver()

if __name__ == '__main__':
    with torch.no_grad():
        t1 = time.time()
        q, p = solver.trajectory(experiment, experiment.q0, experiment.p0, experiment.mass, trajectory)
        t2 = time.time()
        print(f'solving took: {t2 - t1}s')

        loss = criterion(q, q_base) + criterion(p, p_base)
        print(f'final loss: {loss.item()}')

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
