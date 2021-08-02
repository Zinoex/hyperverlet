import abc

import numpy as np
import torch
from torch import nn

from hyperverlet.energy import PendulumEnergy, SpringMassEnergy, ThreeBodySpringMassEnergy


class Experiment(nn.Module, abc.ABC):
    def forward(self, q, p, m, t, **kwargs):
        return self.dq(q, p, m, t, **kwargs), self.dp(q, p, m, t, **kwargs)

    def dq(self, q, p, m, t, **kwargs):
        return p / m

    @abc.abstractmethod
    def dp(self, q, p, m, t, **kwargs):
        raise NotImplementedError()

    def shift(self, q, **kwargs):
        return q


class Pendulum(Experiment):
    energy = PendulumEnergy()

    def dq(self, q, p, m, t, length, **kwargs):
        return p / (m * length ** 2)

    def dp(self, q, p, m, t, length, g, **kwargs):
        if torch.is_tensor(q):
            sin_q = torch.sin(q)
        else:
            sin_q = np.sin(q)
        return -m * g * length * sin_q


class DoublePendulum(Experiment):
    energy = PendulumEnergy()

    def dq(self, q, p, m, t, length, **kwargs):

        l1, l2 = length.split(1, dim=-1)
        q1, q2 = q.split(1, dim=-1)
        p1, p2 = p.split(1, dim=-1)
        m1, m2 = m.split(1, dim=-1)

        total_mass = m1 + m2
        angle_diff = q1 - q2
        common_denom = m1 + m2 * angle_diff.sin() ** 2

        dq1 = (l2 * p1 - l1 * p2 * angle_diff.cos()) / (l1 ** 2 * l2 * common_denom)
        dq2 = (l1 * total_mass * p2 - l2 * m2 * p1 * angle_diff.cos()) / (l1 * l2 ** 2 * m2 * common_denom)

        return torch.cat([dq1, dq2], dim=-1)

    def dp(self, q, p, m, t, length, g, **kwargs):

        l1, l2 = length.split(1, dim=-1)
        q1, q2 = q.split(1, dim=-1)
        p1, p2 = p.split(1, dim=-1)
        m1, m2 = m.split(1, dim=-1)

        total_mass = m1 + m2
        angle_diff = q1 - q2
        common_denom = m1 + m2 * angle_diff.sin() ** 2

        c1 = (p1 * p2 * angle_diff.sin()) / (l1 * l2 * common_denom)
        c2 = (l2 ** 2 * m2 * p1 ** 2 + l1 ** 2 * total_mass * p2 ** 2 - l1 * l2 * m2 * p1 * p2 * angle_diff.cos()) * \
             (2 * angle_diff).sin() / (2 * l1 ** 2 * l2 ** 2 * common_denom ** 2)

        dp1 = -total_mass * g * l1 * q1.sin() - c1 + c2
        dp2 = - m2 * g * l2 * q2.sin() + c1 - c2

        return torch.cat([dp1, dp2], dim=-1)


class SpringMass(Experiment):
    energy = SpringMassEnergy()

    def dp(self, q, p, m, t, length, k, **kwargs):
        return -k * (q - length)


class BasePairPotential(Experiment):
    def dp(self, q, p, m, t, **kwargs):
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

