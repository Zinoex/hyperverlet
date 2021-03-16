import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(features, features),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.model(x)


class SingleAxisMLP(nn.Sequential):
    def __init__(self, input_dim, h_dim):
        super().__init__(
            nn.Linear(input_dim, h_dim),
            nn.PReLU(),
            Block(h_dim),
            Block(h_dim),
            Block(h_dim),
            nn.Linear(h_dim, 1)
        )


class PendulumModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.h_dim = 16

        self.model_q = SingleAxisMLP(6, self.h_dim)
        self.model_p = SingleAxisMLP(6, self.h_dim)

    def forward(self, q, p, dq, dp, m, t, dt, length, include_q=True, include_p=True, **kwargs):
        if include_q:
            hq = torch.cat([q, dq, p, dp, m, length], dim=-1)
            hq = self.model_q(hq)
        else:
            hq = None

        if include_p:
            hp = torch.cat([q, dq, p, dp, m, length], dim=-1)
            hp = self.model_p(hp)
        else:
            hp = None

        return hq, hp


class SpringMassModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.h_dim = 16

        self.model_q = SingleAxisMLP(7, self.h_dim)
        self.model_p = SingleAxisMLP(7, self.h_dim)

    def forward(self, q, p, dq, dp, m, t, dt, length, k, include_q=True, include_p=True, **kwargs):
        if include_q:
            hq = torch.cat([q, dq, p, dp, m, length, k], dim=-1)
            hq = self.model_q(hq)
        else:
            hq = None

        if include_p:
            hp = torch.cat([q, dq, p, dp, m, length, k], dim=-1)
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
            Block(self.h_dim),
            Block(self.h_dim),
            Block(self.h_dim),
            nn.Linear(self.h_dim, 3)
        )

        self.model_p = nn.Sequential(
            nn.Linear(12, self.h_dim),
            nn.PReLU(),
            Block(self.h_dim),
            Block(self.h_dim),
            Block(self.h_dim),
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