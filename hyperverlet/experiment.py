import torch
from torch import nn


class Pendulum(nn.Module):
    def __init__(self, l, g=9.807):
        super().__init__()

        self.l = l
        self.g = g

    def forward(self, q, p, m, t):
        return p / (m * self.l ** 2), - m * self.g * self.l * torch.sin(q)
