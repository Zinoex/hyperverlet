import torch
from torch import optim, nn
import matplotlib.pyplot as plt
from tqdm import trange
import random

from hyperverlet.experiments import Pendulum
from hyperverlet.solvers import HyperEulerSolver, EulerSolver, VelocityVerletSolver, HyperVelocityVerletSolver
from hyperverlet.models import PendulumMLP

device = torch.device('cpu')
pendulum = Pendulum(l=0.5).to(device)
solver = VelocityVerletSolver().to(device)

traj_len = 1001
duration = 1
trajectory = torch.linspace(0, duration, traj_len).to(device)
mass = torch.tensor([0.9]).to(device)
q0 = torch.tensor([1.5707963268]).to(device)
p0 = torch.tensor([0.0]).to(device)
coarsing_factor = 10

q_base, p_base = solver.trajectory(pendulum, q0, p0, mass, trajectory)
q_base, p_base = torch.cat([q_base[:1], q_base[1::coarsing_factor]], dim=0), torch.cat([p_base[:1], p_base[1::coarsing_factor]], dim=0)
trajectory = torch.linspace(0, duration, int((traj_len - 1) / coarsing_factor) + 1).to(device)

solver = HyperEulerSolver(PendulumMLP()).to(device)
optimizer = optim.AdamW(solver.parameters(), lr=0.5 * 1e-2)
criterion = nn.MSELoss()

# for iteration in trange(10000):
#     optimizer.zero_grad(set_to_none=True)
#
#     batch_size = 16
#     if iteration < 100:
#         start = 0
#     else:
#         start = random.randint(0, trajectory.size(0) - batch_size)
#
#     end = start + batch_size
#
#     q, p = solver.trajectory(pendulum, q_base[start], p_base[start], mass, trajectory[start:end])
#     loss = criterion(q, q_base[start:end]) + criterion(p, p_base[start:end])
#     loss.backward()
#     optimizer.step()
#
#     print(f'loss: {loss.item()}')

solver = VelocityVerletSolver()

if __name__ == '__main__':
    with torch.no_grad():
        q, p = solver.trajectory(pendulum, q0, p0, mass, trajectory)
        loss = criterion(q, q_base) + criterion(p, p_base)
        print(f'final loss: {loss.item()}')
        # plt.quiver(q[1:], p[1:], q[:-1], p[:-1])
        plt.plot(q, p, '->')
        plt.show()
