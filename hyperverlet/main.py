import torch
from torch import optim, nn
import matplotlib.pyplot as plt
from tqdm import trange
from math import pi
import random
import time

from hyperverlet.experiments import Pendulum
from hyperverlet.solvers import HyperEulerSolver, EulerSolver, VelocityVerletSolver, HyperVelocityVerletSolver, \
    StormerVerletSolver, ThirdOrderRuthSolver, FourthOrderRuthSolver
from hyperverlet.models import PendulumMLP

device = torch.device('cpu')
pendulum = Pendulum(l=0.5).to(device)
solver = ThirdOrderRuthSolver().to(device)

traj_len = 100001
duration = 1000
trajectory = torch.linspace(0, duration, traj_len).to(device)
mass = torch.tensor([0.9]).to(device)
q0 = torch.tensor([pi / 2]).to(device)
p0 = torch.tensor([0.0]).to(device)
coarsing_factor = 20
assert (traj_len - 1) % coarsing_factor == 0

t1 = time.time()
q_base, p_base = solver.trajectory(pendulum, q0, p0, mass, trajectory)
t2 = time.time()
print(f'data generation took: {t2 - t1}s')

q_base, p_base = torch.cat([q_base[:1], q_base[1::coarsing_factor]], dim=0), torch.cat([p_base[:1], p_base[1::coarsing_factor]], dim=0)
trajectory = torch.linspace(0, duration, int((traj_len - 1) / coarsing_factor) + 1).to(device)

solver = HyperVelocityVerletSolver(PendulumMLP()).to(device)
optimizer = optim.AdamW(solver.parameters(), lr=0.5 * 1e-2)
criterion = nn.MSELoss()

for iteration in trange(1000):
    optimizer.zero_grad(set_to_none=True)

    batch_size = 16
    if iteration < 100:
        start = 0
    else:
        start = random.randint(0, trajectory.size(0) - batch_size)

    end = start + batch_size

    # q, p = solver.trajectory(pendulum, q_base[start], p_base[start], mass, trajectory[start:end])
    # loss = criterion(q, q_base[start:end]) + criterion(p, p_base[start:end])
    loss = solver.loss(pendulum, q_base[start:end], p_base[start:end], mass, trajectory[start:end])
    loss.backward()
    optimizer.step()

    print(f'loss: {loss.item()}')

# solver = VelocityVerletSolver()

if __name__ == '__main__':
    with torch.no_grad():
        t1 = time.time()
        q, p = solver.trajectory(pendulum, q0, p0, mass, trajectory)
        t2 = time.time()
        print(f'solving took: {t2 - t1}s')

        loss = criterion(q, q_base) + criterion(p, p_base)
        print(f'final loss: {loss.item()}')

        # plt.quiver(q[:-1], p[:-1], q[1:] - q[:-1], p[1:] - p[:-1], scale_units='xy', angles='xy', scale=1)
        plt.scatter(q, p, marker='x')
        plt.xlabel('q')
        plt.ylabel('p')
        plt.title('Phase space trajectory')
        plt.show()
