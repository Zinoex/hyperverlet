import torch
from torch import nn
from math import pi


class Pendulum(nn.Module):
    def __init__(self, l, g=9.807):
        super().__init__()

        self.l = l
        self.g = g

        self.q0 = torch.tensor([pi / 2])
        self.p0 = torch.tensor([0.0])
        self.mass = torch.tensor([0.9])

    def forward(self, q, p, m, t):
        dq = p / (m * self.l ** 2)
        dp = -m * self.g * self.l * torch.sin(q)
        return dq, dp


class LenardJones(nn.Module):
    def __init__(self, eps=1, sigma=1):
        super().__init__()

        self.eps = eps
        self.sigma = sigma

        self.num_particles = 10
        self.q0 = torch.randn((self.num_particles, 3))
        self.p0 = torch.randn((self.num_particles, 3))
        self.mass = torch.ones((self.num_particles, 1))

    def forward(self, q, p, m, t):
        # q = q.requires_grad_(True)
        # q.requires_grad = True

        displacement = self.displacement(q)
        r = self.distance(displacement)
        # acceleration = force / mass
        dq = p / m
        dp = self.force(r, displacement).sum(axis=1)

        return dq, dp

    def force(self, r, disp):
        prefix = 48 * self.eps
        sigma12 = self.sigma ** 12
        sigma6 = self.sigma ** 6

        r13 = r ** 13
        r7 = r ** 7

        inner = sigma12 / r13 - 0.5 * sigma6 / r7

        direction = disp / r.fill_diagonal_(1).unsqueeze(-1)

        return -prefix * inner.fill_diagonal_(0).unsqueeze(-1) * direction

        # energy = self.potential(r).sum()
        # scalar_force = torch.autograd.grad(energy, r, retain_graph=True)[0].fill_diagonal_(0)
        # return -scalar_force.unsqueeze(2) * (disp / r.fill_diagonal_(1).unsqueeze(2))

    def potential(self, r):
        return (4 * self.eps * ((self.sigma / r) ** 12 - (self.sigma / r) ** 6)).fill_diagonal_(0)

    def displacement(self, q):
        a = torch.unsqueeze(q, 1)
        b = torch.unsqueeze(q, 0)
        return a - b

    def distance(self, disp):
        return disp.norm(dim=2)
