from abc import ABC, abstractmethod, ABCMeta
import numpy as np
import torch
from numpy import inf

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

    def energy_difference(self, m, pred_q, pred_p, gt_q, gt_p, **kwargs):
        def fix_inf(x):
            x[x == -inf] = 0
            return x
        pred_ke, pred_pe, pred_te = self.all_energies(m, pred_q, pred_p, **kwargs)
        gt_ke, gt_pe, gt_te = self.all_energies(m, gt_q, gt_p, **kwargs)
        return fix_inf(np.log(np.abs(gt_ke - pred_ke))), fix_inf(np.log(np.abs(gt_pe - pred_pe))), fix_inf(np.log(np.abs(gt_te - pred_te)))

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
