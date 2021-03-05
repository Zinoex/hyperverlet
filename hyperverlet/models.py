import torch
from torch import nn


class PendulumResidual(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(features, features),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.model(x)


class PendulumMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.h_dim = 128

        self.model = nn.Sequential(
            nn.Linear(4, self.h_dim),
            nn.PReLU(),
            PendulumResidual(self.h_dim),
            nn.Linear(self.h_dim, 2)
        )

    def forward(self, q, p, dq, dp, m, t, dt):
        x = torch.cat([p, q, dq, dp], dim=0)
        x = self.model(x)
        x = x.split(1, dim=0)

        return x
