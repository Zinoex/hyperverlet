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


class PendulumMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.h_dim = 64

        self.model_q = nn.Sequential(
            nn.Linear(4, self.h_dim),
            nn.PReLU(),
            PendulumBlock(self.h_dim),
            PendulumBlock(self.h_dim),
            PendulumBlock(self.h_dim),
            nn.Linear(self.h_dim, 1)
        )

        self.model_p = nn.Sequential(
            nn.Linear(4, self.h_dim),
            nn.PReLU(),
            PendulumBlock(self.h_dim),
            PendulumBlock(self.h_dim),
            PendulumBlock(self.h_dim),
            nn.Linear(self.h_dim, 1)
        )

    def forward(self, q, p, dq, dp, m, t, dt, include_q=True, include_p=True):
        if include_q:
            hq = torch.cat([q, dq, p, dp], dim=0)
            hq = self.model_q(hq)
        else:
            hq = None

        if include_p:
            hp = torch.cat([q, dq, p, dp], dim=0)
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

    def forward(self, q, p, dq, dp, m, t, dt, include_q=True, include_p=True):
        if include_q:
            q = torch.cat([q, dq, p, dp], dim=-1)
            q = self.model_q(q)
        else:
            q = None

        if include_p:
            p = torch.cat([q, dq, p, dp], dim=-1)
            p = self.model_p(p)
        else:
            p = None

        return q, p