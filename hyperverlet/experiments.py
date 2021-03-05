import torch
from torch import nn


class Pendulum(nn.Module):
    def __init__(self, l, g=9.807):
        super().__init__()

        self.l = l
        self.g = g

    def forward(self, q, p, m, t):
        dq = p / (m * self.l ** 2)
        dp = -m * self.g * self.l * torch.sin(q)
        return dq, dp


class LenardJones(nn.Module):
    def __init__(self, eps=4, sigma=1):
        super().__init__()

        self.eps = eps
        self.sigma = sigma

    def forward(self, q, p, m, t):
        displacement = self.displacement(q)
        r = self.distance(displacement)
        # acceleration = force / mass
        dq = p / m
        dp = self.force(r, displacement)
        return dq, dp

    def force(self, r, disp):
        return torch.neg(torch.autograd.grad(self.potential, r) * (disp / r.unsqueeze(2)))

    def potential(self, r):
        return 4 * self.eps * ((self.sigma / r) ** 12 - (self.sigma / r) ** 6)

    def displacement(self, q):
        a = torch.unsqueeze(q, 1)
        b = torch.unsqueeze(q, 0)
        return a - b

    def distance(self, disp):
        return disp.norm(dim=2)
