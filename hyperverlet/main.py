import torch
from torch import optim, nn
import matplotlib.pyplot as plt
from tqdm import trange
from math import pi
import random

from hyperverlet.experiments import Pendulum, LenardJones
from hyperverlet.solvers import HyperEulerSolver, EulerSolver, VelocityVerletSolver, HyperVelocityVerletSolver
from hyperverlet.models import PendulumMLP

if __name__ == '__main__':
    device = torch.device('cpu')

    #Systems
    pendulum = Pendulum(l=0.5).to(device)
    lj = LenardJones().to(device)

    #Solver
    solver = VelocityVerletSolver().to(device)

    #Data setup
    traj_len = 1001
    duration = 1
    trajectory = torch.linspace(0, duration, traj_len).to(device)
    num_particles = 5
    euclidean_dim = 3
    mass = torch.tensor([0.9]).to(device)

    # Lennard Jones dataset
    q0 = torch.rand(num_particles, euclidean_dim)
    p0 = torch.rand(num_particles, euclidean_dim)

    #Pendulum dataset
    #q0 = torch.tensor([pi / 2]).to(device)
    #p0 = torch.tensor([0.0]).to(device)

    coarsing_factor = 10

    q_base, p_base = solver.trajectory(lj, q0, p0, mass, trajectory)
    q_base, p_base = torch.cat([q_base[:1], q_base[1::coarsing_factor]], dim=0), torch.cat(
        [p_base[:1], p_base[1::coarsing_factor]], dim=0)
    trajectory = torch.linspace(0, duration, int((traj_len - 1) / coarsing_factor) + 1).to(device)

    solver = HyperEulerSolver(PendulumMLP()).to(device)
    optimizer = optim.AdamW(solver.parameters(), lr=0.5 * 1e-2)
    criterion = nn.MSELoss()

    with torch.no_grad():
        q, p = solver.trajectory(lj, q0, p0, mass, trajectory)
        loss = criterion(q, q_base) + criterion(p, p_base)
        print(f'final loss: {loss.item()}')
        plt.quiver(q[:-1], p[:-1], q[1:] - q[:-1], p[1:] - p[:-1], scale_units='xy', angles='xy', scale=1)
        plt.xlabel('q')
        plt.ylabel('p')
        plt.title('Phase space trajectory')
        plt.show()
