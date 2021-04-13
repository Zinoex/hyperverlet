import abc

import torch
from torch import nn

from hyperverlet.energy.energy import PendulumEnergy, SpringMassEnergy, ThreeBodySpringMassEnergy


class Experiment(nn.Module, abc.ABC):
    def forward(self, q, p, m, t, **kwargs):
        return self.dq(p, m, t, **kwargs), self.dp(q, m, t, **kwargs)

    def dq(self, p, m, t, length, **kwargs):
        return p / m

    @abc.abstractmethod
    def dp(self, q, m, t, length, g, **kwargs):
        raise NotImplementedError()

    def shift(self, q, **kwargs):
        return q


class Pendulum(Experiment):
    energy = PendulumEnergy()

    def dq(self, p, m, t, length, **kwargs):
        return p / (m * length ** 2)

    def dp(self, q, m, t, length, g, **kwargs):
        return -m * g * length * torch.sin(q)


class SpringMass(Experiment):
    energy = SpringMassEnergy()

    def dp(self, q, m, t, length, k, **kwargs):
        return -k * (q - length)


class BasePairPotential(Experiment):
    def dp(self, q, m, t, **kwargs):
        disp = self.displacement(q, **kwargs)
        r = self.distance(disp, **kwargs)

        return self.force(r, disp, **kwargs).sum(axis=-2)

    def displacement(self, q, **kwargs):
        a = torch.unsqueeze(q, -2)
        b = torch.unsqueeze(q, -3)

        return a - b

    def distance(self, disp, **kwargs):
        return disp.norm(dim=-1)


class ThreeBodySpringMass(BasePairPotential):
    energy = ThreeBodySpringMassEnergy()

    def force(self, r, disp, k, length, **kwargs):
        num_particles = k.size(1)
        r_prime = (r + torch.eye(num_particles, num_particles, device=r.device)).unsqueeze(-1)
        direction = disp / r_prime

        offset = r - length

        return -2 * (k * offset).unsqueeze(-1) * direction


class LennardJones(BasePairPotential):
    def force(self, r, disp, eps, sigma, **kwargs):
        prefix = 24 * eps
        sigma12 = sigma ** 12
        sigma6 = sigma ** 6

        r = r.fill_diagonal_(1)

        r13 = r ** 13
        r7 = r ** 7

        inner = 2 * sigma12 / r13 - sigma6 / r7

        num_particles = r.size(1)
        r_prime = (r + torch.eye(num_particles, num_particles, device=r.device)).unsqueeze(-1)
        direction = disp / r_prime

        return -prefix * inner.fill_diagonal_(0).unsqueeze(-1) * direction

