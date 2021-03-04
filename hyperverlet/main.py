import torch
import matplotlib.pyplot as plt

from hyperverlet.experiment import Pendulum
from hyperverlet.solvers import EulerSolver, VelocityVerletSolver

pendulum = Pendulum(l=0.5)
solver = EulerSolver()

trajectory = torch.linspace(0, 10, 100)
mass = torch.tensor([0.9])
q0 = torch.tensor([1.5707963268])
p0 = torch.tensor([0.0])

if __name__ == '__main__':
    q, p = solver.trajectory(pendulum, q0, p0, mass, trajectory)
    print(q, p)
    plt.plot(q, p, '->')
    plt.show()
