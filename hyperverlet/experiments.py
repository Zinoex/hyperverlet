import abc

import numpy as np
import torch
from torch import nn

from hyperverlet.energy import PendulumEnergy, SpringMassEnergy, ThreeBodySpringMassEnergy


class Experiment(nn.Module, abc.ABC):
    def forward(self, q, p, m, t, **kwargs):
        return self.dq(p, m, t, **kwargs), self.dp(q, m, t, **kwargs)

    def dq(self, p, m, t, **kwargs):
        return p / m

    @abc.abstractmethod
    def dp(self, q, m, t, **kwargs):
        raise NotImplementedError()

    def shift(self, q, **kwargs):
        return q


class Pendulum(Experiment):
    energy = PendulumEnergy()

    def dq(self, p, m, t, length, **kwargs):
        return p / (m * length ** 2)

    def dp(self, q, m, t, length, g, **kwargs):
        if torch.is_tensor(q):
            sin_q = torch.sin(q)
        else:
            sin_q = np.sin(q)
        return -m * g * length * sin_q


class SpringMass(Experiment):
    energy = SpringMassEnergy()

    def dp(self, q, m, t, length, k, **kwargs):
        return -k * (q - length)


class BasePairPotential(Experiment):
    def dp(self, q, m, t, **kwargs):
        disp = self.displacement(q, **kwargs)
        r = self.distance(disp, **kwargs)

        return self.force(r, disp, m, **kwargs).sum(axis=-2)

    def displacement(self, q, **kwargs):
        a = torch.unsqueeze(q, -2)
        b = torch.unsqueeze(q, -3)

        return a - b

    def distance(self, disp, **kwargs):
        return disp.norm(dim=-1)


class ThreeBodySpringMass(BasePairPotential):
    energy = ThreeBodySpringMassEnergy()

    def force(self, r, disp, m, k, length, **kwargs):
        num_particles = k.size(1)
        r_prime = (r + torch.eye(num_particles, num_particles, device=r.device)).unsqueeze(-1)
        direction = disp / r_prime

        offset = r - length

        return -(k * offset).unsqueeze(-1) * direction


class ThreeBodyGravity(BasePairPotential):

    def force(self, r, disp, m, G, **kwargs):
        num_planets = m.size(1)
        r_prime = (r ** 2 + torch.eye(num_planets, num_planets, device=r.device)).unsqueeze(-1)
        direction = disp / r_prime

        m1 = torch.unsqueeze(m, -2)
        m2 = torch.unsqueeze(m, -3)

        m = m1 * m2

        sizes = [1 for _ in range(m.dim())]
        sizes[0] = -1

        return G.view(*sizes) * (m / r_prime) * direction


class LennardJones(BasePairPotential):
    def force(self, r, disp, m, eps, sigma, **kwargs):
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

