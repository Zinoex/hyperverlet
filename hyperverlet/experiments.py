import abc

import numpy as np
import torch
from torch import nn

from hyperverlet.energy import PendulumEnergy, SpringMassEnergy


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


class SpringMass(Experiment):
    energy = SpringMassEnergy()

    def dp(self, q, p, m, t, length, k, **kwargs):
        return -k * (q - length)
