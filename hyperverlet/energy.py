from abc import ABC, abstractmethod, ABCMeta
import numpy as np
import torch


class Energy(ABC):
    @abstractmethod
    def kinetic_energy(self, m, p, **kwargs):
        return

    @abstractmethod
    def potential_energy(self, m, q, **kwargs):
        return

    def total_energy(self, ke, pe):
        return ke + pe

    def all_energies(self, m, q, p, **kwargs):
        ke = self.kinetic_energy(m, p, **kwargs)
        pe = self.potential_energy(m, q, **kwargs)
        te = self.total_energy(ke, pe)

        return ke, pe, te

    def trajectory_sum(self, e):
        sumaxis = tuple(range(1, len(e.shape)))

        if torch.is_tensor(e):
            return e.sum(dim=sumaxis)
        else:
            return e.sum(axis=sumaxis)


class CartesianKineticEnergy(Energy, metaclass=ABCMeta):
    def kinetic_energy(self, m, p, **kwargs):
        v = p / m
        return self.trajectory_sum(0.5 * m * v ** 2)


class ThreeBodySpringMassEnergy(CartesianKineticEnergy):
    def displacement(self, q):
        if torch.is_tensor(q):
            a = torch.unsqueeze(q, -2)
            b = torch.unsqueeze(q, -3)
        else:
            a = np.expand_dims(q, -2)
            b = np.expand_dims(q, -3)
        return a - b

    def distance(self, disp):
        if torch.is_tensor(disp):
            return disp.norm(dim=-1)
        else:
            return np.linalg.norm(disp, axis=-1)

    def potential_energy(self, m, q, k, length, **kwargs):
        disp = self.displacement(q)
        dist = self.distance(disp)
        x = dist - length

        return self.trajectory_sum(0.25 * k * x ** 2)


class SpringMassEnergy(CartesianKineticEnergy):
    def potential_energy(self, m, q, k, length, **kwargs):
        x = q - length

        return self.trajectory_sum(0.5 * k * x ** 2)


class PendulumEnergy(Energy):
    def kinetic_energy(self, m, p, length, **kwargs):
        v = p / (m * length)
        return self.trajectory_sum(0.5 * m * v ** 2)

    def potential_energy(self, m, q, g, length, **kwargs):
        return self.trajectory_sum(m * g * length * (1 - np.cos(q)))
