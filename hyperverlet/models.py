import torch
from torch import nn


class PendulumBlock(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(features, features),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.model(x)


class PendulumMLP(nn.Sequential):
    def __init__(self, h_dim):
        super().__init__(
            nn.Linear(2, h_dim),
            nn.PReLU(),
            PendulumBlock(h_dim),
            PendulumBlock(h_dim),
            PendulumBlock(h_dim),
            nn.Linear(h_dim, 1)
        )


class PendulumModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.h_dim = 64

        self.model_q = PendulumMLP(self.h_dim)
        self.model_p = PendulumMLP(self.h_dim)

    def forward(self, q, p, dq, dp, m, t, dt, include_q=True, include_p=True, **kwargs):
        if include_q:
            hq = torch.cat([q, dq], dim=-1)
            hq = self.model_q(hq)
        else:
            hq = None

        if include_p:
            hp = torch.cat([p, dp], dim=-1)
            hp = self.model_p(hp)
        else:
            hp = None

        return hq, hp


class LennardJonesMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.h_dim = 64

        self.model_q = nn.Sequential(
            nn.Linear(12, self.h_dim),
            nn.PReLU(),
            PendulumBlock(self.h_dim),
            PendulumBlock(self.h_dim),
            PendulumBlock(self.h_dim),
            nn.Linear(self.h_dim, 3)
        )

        self.model_p = nn.Sequential(
            nn.Linear(12, self.h_dim),
            nn.PReLU(),
            PendulumBlock(self.h_dim),
            PendulumBlock(self.h_dim),
            PendulumBlock(self.h_dim),
            nn.Linear(self.h_dim, 3)
        )

        # TODO: Try a GNN model for interactions and permutation equivariance

    def forward(self, q, p, dq, dp, m, t, dt, include_q=True, include_p=True, **kwargs):
        if include_q:
            hq = torch.cat([q, dq, p, dp], dim=-1)
            hq = self.model_q(hq)
        else:
            hq = None

        if include_p:
            hp = torch.cat([q, dq, p, dp], dim=-1)
            hp = self.model_p(hp)
        else:
            hp = None

        return hq, hp