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
            nn.Linear(2, self.h_dim),
            nn.PReLU(),
            PendulumBlock(self.h_dim),
            PendulumBlock(self.h_dim),
            PendulumBlock(self.h_dim),
            nn.Linear(self.h_dim, 1)
        )

        self.model_p = nn.Sequential(
            nn.Linear(2, self.h_dim),
            nn.PReLU(),
            PendulumBlock(self.h_dim),
            PendulumBlock(self.h_dim),
            PendulumBlock(self.h_dim),
            nn.Linear(self.h_dim, 1)
        )

    def forward(self, q, p, dq, dp, m, t, dt, include_q=True, include_p=True):
        if include_q:
            q = torch.cat([q, dq], dim=0)
            q = self.model_q(q)
        else:
            q = None

        if include_p:
            p = torch.cat([p, dp], dim=0)
            p = self.model_p(p)
        else:
            p = None

        return q, p


class LennardJonesMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.h_dim = 64

        self.model_q = nn.Sequential(
            nn.Linear(2, self.h_dim),
            nn.PReLU(),
            PendulumBlock(self.h_dim),
            PendulumBlock(self.h_dim),
            nn.Linear(self.h_dim, 1)
        )

        self.model_p = nn.Sequential(
            nn.Linear(2, self.h_dim),
            nn.PReLU(),
            PendulumBlock(self.h_dim),
            PendulumBlock(self.h_dim),
            nn.Linear(self.h_dim, 1)
        )

    def forward(self, q, p, dq, dp, m, t, dt, include_q=True, include_p=True):
        if include_q:
            q = torch.stack([q, dq], dim=-1)
            q = self.model_q(q).squeeze(-1)
        else:
            q = None

        if include_p:
            p = torch.stack([p, dp], dim=-1)
            p = self.model_p(p).squeeze(-1)
        else:
            p = None

        return q, p